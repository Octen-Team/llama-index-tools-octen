"""Live calls against the Octen API. Billed as usual.

Skipped unless OCTEN_LIVE=1 and OCTEN_API_KEY are set. Each test makes at
most one billed request with the smallest result count the endpoint allows,
except the search_premier, extract, image_search and design_search tests, which
make two to four (options under test and same-input controls) with a small count
or one to three URLs.
"""

import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import pytest
from octen import OctenAuthenticationError

from llama_index.tools.octen import OctenToolSpec

pytestmark = pytest.mark.skipif(
    os.environ.get("OCTEN_LIVE") != "1" or not os.environ.get("OCTEN_API_KEY"),
    reason="live tests need OCTEN_LIVE=1 and OCTEN_API_KEY",
)


@pytest.fixture(autouse=True)
def _single_attempt_clients(monkeypatch: Any) -> None:
    # Live calls are billed: no SDK retries here, so each request bills at most once.
    from octen import Octen

    def get_client(self: OctenToolSpec) -> Octen:
        return Octen(api_key=self._api_key, retry_after_send=False, max_retries=0)

    monkeypatch.setattr(OctenToolSpec, "_get_client", get_client)


def _spec(**kwargs: Any) -> OctenToolSpec:
    return OctenToolSpec(verbose=False, **kwargs)


def _check(name: str, result: Dict[str, Any]) -> None:
    assert "error" not in result, result.get("error")
    assert result["data"], f"{name}: empty data"
    assert result["request_id"], f"{name}: no request_id"
    print(f"\n[live] {name} request_id={result['request_id']} warning={result.get('warning')}")


def test_live_search_ignores_search_type(mocker: Any) -> None:
    # The old function returns a list without request_id, so read it off the
    # SDK response; search_type used to raise TypeError on octen>=0.5.0.
    import octen.resources.search as sdk_search

    spy = mocker.spy(sdk_search.SearchResource, "search")
    with pytest.warns(FutureWarning, match="search_type"):
        results = _spec().search("Octen web search API", num_results=1, search_type="auto")
    assert results and results[0].get("url")
    assert "search_type" not in spy.call_args.kwargs
    request_id = (spy.spy_return.model_extra or {}).get("request_id")
    assert request_id
    print(f"\n[live] search request_id={request_id}")


def test_live_broad_search() -> None:
    spec = _spec(broad_search_options={"count": 1, "max_queries": 1})
    result = spec.broad_search("Python packaging tools comparison")
    _check("broad_search", result)
    groups = result["data"]["search_results"]
    assert any(group.get("results") for group in groups), groups


def test_live_news_search() -> None:
    spec = _spec(
        news_search_options={
            "count": 1,
            "subjects": {"enable": True, "count": 1, "max_sub_news": 1},
        }
    )
    result = spec.news_search("semiconductor industry", time_range="week")
    _check("news_search", result)
    assert result["data"]["results"]
    subjects = result["data"].get("subjects") or []
    assert any(subject.get("name") for subject in subjects), subjects


def test_live_business_search() -> None:
    spec = _spec(business_search_options={"count": 1, "entities": {"count": 1}})
    result = spec.business_search("NVIDIA company overview")
    _check("business_search", result)
    entities = result["data"].get("entities") or []
    assert any(entity.get("name") for entity in entities), entities


def test_live_pulse_search() -> None:
    spec = _spec(pulse_search_options={"count": 1})
    result = spec.pulse_search("github docs")
    _check("pulse_search", result)
    assert result["data"]["results"]


def test_live_place_search() -> None:
    spec = _spec(place_search_options={"places": {"count": 1}})
    result = spec.place_search(
        "coffee shops", latitude=35.6812, longitude=139.7671, radius=1000
    )
    _check("place_search", result)
    if result.get("warning") == "places_unavailable":
        pytest.skip("place engine unavailable (warning places_unavailable)")
    places = result["data"]["places"]
    assert places
    assert all(place.get("name") for place in places), places


def test_live_bad_key_raises() -> None:
    with pytest.raises(OctenAuthenticationError):
        OctenToolSpec(api_key="invalid-key", verbose=False).pulse_search("github docs")


def test_live_developer_count_out_of_range_raises() -> None:
    # Rejected when the spec is built, before any request is sent.
    with pytest.raises(ValueError, match="news_search_options.*count"):
        _spec(news_search_options={"count": 1000})


# The SDK's own Premier live tests use this query; its results span several years.
TIME_QUERY = "electric vehicle battery"
YEAR_2025 = ("2025-01-01T00:00:00Z", "2025-12-31T23:59:59Z")
# Single-bound checks need unfiltered results dated on both sides of the window; on
# 2026-11-21 they ran from 2022 to 2026-01-30, so the window ends before the latest.
BOUND_WINDOW = ("2024-01-01T00:00:00Z", "2025-06-30T23:59:59Z")
_STAMP = re.compile(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(?:\.(\d+))?(Z|[+-]\d\d:\d\d)?")


def _ts(value: Any) -> Optional[datetime]:
    """None when absent; raises on a value that does not parse, so it cannot pass unchecked."""
    if value is None or value == "":
        return None
    match = _STAMP.fullmatch(value) if isinstance(value, str) else None
    if match is None:
        raise ValueError(f"unparseable timestamp {value!r}")
    base, frac, tz = match.groups()
    # fromisoformat before 3.11 takes neither Z nor fractions other than 3 or 6 digits.
    frac = "." + (frac + "000000")[:6] if frac else ""
    return datetime.fromisoformat(base + frac + ("+00:00" if tz in (None, "Z") else tz))


def _published(result: Dict[str, Any]) -> List[Optional[datetime]]:
    return [_ts(r.get("time_published")) for r in result["data"]["results"]]


def _hosts(result: Dict[str, Any]) -> List[str]:
    return [urlparse(r["url"]).hostname or "" for r in result["data"]["results"]]


def _on(host: str, domain: str) -> bool:
    return host == domain or host.endswith("." + domain)


def test_live_search_premier_include_domains() -> None:
    # Same query with and without the model's domain filter.
    spec = _spec(search_premier_options={"count": 5})
    control = spec.search_premier("Apache Kafka")
    filtered = spec.search_premier("Apache Kafka", include_domains=["wikipedia.org"])
    _check("search_premier control", control)
    _check("search_premier include_domains", filtered)
    print(f"control hosts={_hosts(control)} filtered hosts={_hosts(filtered)}")
    assert any(not _on(h, "wikipedia.org") for h in _hosts(control)), control
    assert _hosts(filtered), filtered
    assert all(_on(h, "wikipedia.org") for h in _hosts(filtered)), filtered


def test_live_search_premier_developer_options(monkeypatch: Any) -> None:
    # Same query with server defaults and with full content plus dynamic results.
    from octen.resources.search import SearchResource

    responses: List[Any] = []
    original = SearchResource.search_premier

    def record(self: SearchResource, *args: Any, **kwargs: Any) -> Any:
        responses.append(original(self, *args, **kwargs))
        return responses[-1]

    monkeypatch.setattr(SearchResource, "search_premier", record)
    query = "Kafka default retention period for topics"
    control = _spec(search_premier_options={"count": 2}).search_premier(query)
    options = {
        "count": 2,
        "result_type": "dynamic",
        "full_content": {"enable": True, "max_tokens": 200},
    }
    result = _spec(search_premier_options=options).search_premier(query)
    _check("search_premier control", control)
    _check("search_premier options", result)
    # meta is not part of the function result; read result_type off the SDK responses.
    result_types = [r.meta.get("result_type") for r in responses]
    print(f"meta result_type={result_types}")
    assert result_types == ["standard", "dynamic"]
    assert 0 < len(control["data"]["results"]) <= 2, control
    assert 0 < len(result["data"]["results"]) <= 2, result
    assert not any(r.get("full_content") for r in control["data"]["results"]), control
    assert all(r.get("full_content") for r in result["data"]["results"]), result


def _premier(**options: Any) -> Dict[str, Any]:
    spec = _spec(
        search_premier_options={"count": 5, "time_basis": "published", **options}
    )
    return spec.search_premier(TIME_QUERY)


def test_live_search_premier_time_window() -> None:
    # The filtered call first, then the same query without bounds as the control.
    start, end = _ts(YEAR_2025[0]), _ts(YEAR_2025[1])
    assert start and end
    result = _premier(start_time=YEAR_2025[0], end_time=YEAR_2025[1])
    _check("search_premier 2025", result)
    published = _published(result)
    print(f"2025 time_published={published}")
    assert published, result
    assert all(t is not None and start <= t <= end for t in published), published

    control = _premier()
    _check("search_premier control", control)
    print(f"control time_published={_published(control)}")
    assert any(t and not start <= t <= end for t in _published(control)), control


def test_live_search_premier_each_time_bound() -> None:
    # The unfiltered control must hold dated results before start and after end, so a
    # single-bound call that the server ignored would fail its own bound check.
    start, end = _ts(BOUND_WINDOW[0]), _ts(BOUND_WINDOW[1])
    assert start and end
    control = _premier(count=10)
    _check("search_premier unbounded", control)
    print(f"unbounded time_published={_published(control)}")
    control_dated = [t for t in _published(control) if t]
    stale = "the live index changed: pick a new TIME_QUERY / BOUND_WINDOW with results on both sides"
    assert any(t < start for t in control_dated), f"no result before start; {stale}: {control_dated}"
    assert any(t > end for t in control_dated), f"no result after end; {stale}: {control_dated}"

    # Start-only first: dropping start_time then fails on the second call.
    after = _premier(count=10, start_time=BOUND_WINDOW[0])
    _check("search_premier start-only", after)
    print(f"start-only time_published={_published(after)}")
    after_dated = [t for t in _published(after) if t]
    assert after_dated and all(t >= start for t in after_dated), after
    assert any(t > end for t in after_dated), after

    before = _premier(count=10, end_time=BOUND_WINDOW[1])
    _check("search_premier end-only", before)
    print(f"end-only time_published={_published(before)}")
    before_dated = [t for t in _published(before) if t]
    assert before_dated and all(t <= end for t in before_dated), before
    assert any(t < start for t in before_dated), before


def test_live_search_premier_model_cannot_pass_count() -> None:
    tools = {t.metadata.name: t for t in _spec().to_tool_list()}
    with pytest.raises(TypeError, match="count"):
        tools["search_premier"].call(query="Apache Kafka", count=50)


def test_live_search_premier_bad_option_raises_at_construction() -> None:
    with pytest.raises(ValueError, match="search_premier_options.*result_type"):
        _spec(search_premier_options={"result_type": "smart"})


EXTRACT_DOC = "https://docs.octen.ai/api-reference/extract"
# No other test fetches it, so its cached copy is usually older than 300 s.
MAX_AGE_URL = "https://en.wikipedia.org/wiki/Web_crawler"
# The .invalid TLD never resolves (RFC 2606); failed URLs are not billed.
FAILING_URL = "https://nonexistent.invalid/"


def _items(name: str, result: Dict[str, Any]) -> List[Dict[str, Any]]:
    _check(name, result)
    items = result["data"]["results"]
    print(f"{name}: " + ", ".join(f"{i.get('url')}={i.get('status')}" for i in items))
    return items


def test_live_extract_content_and_per_url_failure() -> None:
    urls = ["https://example.com", EXTRACT_DOC, FAILING_URL]
    items = _items("extract mixed", _spec().extract(urls))
    # Matched by url, not position: the API reference does not promise an order.
    by_url = {i["url"]: i for i in items}
    assert sorted(by_url) == sorted(urls) and len(items) == len(urls), items
    example, doc, failed = (by_url[u] for u in urls)
    assert example["status"] == doc["status"] == "success"
    assert "documentation examples" in example["full_content"]
    assert "Failed URLs are not billed" in doc["full_content"]
    assert failed["status"] == "failed" and failed.get("error_message"), failed
    assert not failed.get("full_content")
    print(f"failed error_message={failed['error_message']!r}")


def test_live_extract_query_highlights_with_control() -> None:
    control = _items("extract control", _spec().extract([EXTRACT_DOC]))[0]
    focused = _items(
        "extract query",
        _spec().extract([EXTRACT_DOC], query="maximum number of URLs per request"),
    )[0]
    assert control["status"] == focused["status"] == "success"
    assert control.get("full_content") and not control.get("highlights"), control
    assert focused.get("highlights") and not focused.get("full_content"), focused
    print(f"highlights={len(focused['highlights'])} first={focused['highlights'][0][:80]!r}")


def test_live_extract_format_and_links_with_control() -> None:
    # The capped call goes first, so a failure there costs one billed URL.
    options = {"format": "text", "include_links": {"max_links": 3}}
    result = _items(
        "extract text+links", _spec(extract_options=options).extract([EXTRACT_DOC])
    )[0]
    print(f"capped links={len(result.get('links') or [])}")
    assert len(result.get("links") or []) == 3, result.get("links")
    heading = re.compile(r"^#{1,6} ", re.M)
    assert not heading.search(result["full_content"]), result["full_content"][:300]

    # Control: links on with the server default, markdown by default.
    control = _items(
        "extract links control",
        _spec(extract_options={"include_links": {}}).extract([EXTRACT_DOC]),
    )[0]
    print(f"control links={len(control.get('links') or [])}")
    assert len(control.get("links") or []) > 3, control.get("links")
    assert heading.search(control["full_content"]), control["full_content"][:300]


def test_live_extract_mode_advanced_with_control(monkeypatch: Any) -> None:
    from octen.resources.extract import ExtractResource

    responses: List[Any] = []
    original = ExtractResource.extract

    def record(self: ExtractResource, *args: Any, **kwargs: Any) -> Any:
        responses.append(original(self, *args, **kwargs))
        return responses[-1]

    monkeypatch.setattr(ExtractResource, "extract", record)
    url = ["https://example.com"]
    control = _items("extract standard", _spec().extract(url))[0]
    # No client timeout set: it is derived as url_timeout + 90 seconds.
    options = {"mode": "advanced", "url_timeout": 60}
    advanced = _items("extract advanced", _spec(extract_options=options).extract(url))[0]
    by_mode = [r.successful_by_mode for r in responses]
    print(f"resolved_mode={control.get('resolved_mode')},{advanced.get('resolved_mode')} "
          f"successful_by_mode={by_mode}")
    # octen documents that an advanced request may resolve to standard: re-check before
    # calling a failure here a forwarding bug. It is the only live signal that mode is sent.
    assert control["resolved_mode"] == "standard"
    assert advanced["resolved_mode"] == "advanced"
    # Billing agrees with the resolved_mode each item reports.
    for response, item in zip(responses, (control, advanced)):
        billed = response.successful_by_mode
        assert billed[f"{item['resolved_mode']}_urls"] == 1, billed
        assert sum(billed.values()) == 1, billed


def test_live_extract_include_images_with_control() -> None:
    # The flagged call goes first, so a failure there costs one billed URL.
    url = ["https://octen.ai/"]
    flagged = _items(
        "extract images", _spec(extract_options={"include_images": True}).extract(url)
    )[0]
    images = flagged.get("images") or []
    print(f"images={len(images)} first={images[:1]}")
    assert images and all(isinstance(i, dict) and i.get("url") for i in images), flagged
    cover = flagged.get("cover_image") or {}
    print(f"cover_image={cover}")
    assert isinstance(cover, dict) and cover.get("url"), flagged
    control = _items("extract images control", _spec().extract(url))[0]
    assert control["status"] == "success" and not control.get("images"), control
    assert not control.get("cover_image"), control
    # No include_links either, so links stay off: they are opt-in.
    assert not control.get("links"), control


def test_live_extract_max_age_seconds_with_control() -> None:
    # time_last_crawled is when Octen last crawled the URL. The default-cache control
    # runs first; only a stale cached copy makes the max_age_seconds check conclusive.
    url = [MAX_AGE_URL]
    margin = 60  # clock skew between this machine and the server
    control = _items("extract cache control", _spec().extract(url))[0]
    cached = _ts(control.get("time_last_crawled"))
    assert cached is not None, control
    age = (datetime.now(timezone.utc) - cached).total_seconds()
    print(f"control time_last_crawled={cached} age={age:.0f}s")
    if age <= 300 + margin:
        pytest.skip(f"inconclusive: cached copy is only {age:.0f}s old, not older than 300 s")

    sent = datetime.now(timezone.utc)
    fresh = _items(
        "extract max_age 300", _spec(extract_options={"max_age_seconds": 300}).extract(url)
    )[0]
    received = datetime.now(timezone.utc)
    crawled = _ts(fresh.get("time_last_crawled"))
    print(f"max_age time_last_crawled={crawled} sent={sent} received={received}")
    assert crawled is not None, fresh
    assert (sent - crawled).total_seconds() <= 300 + margin, (crawled, sent)
    assert (crawled - received).total_seconds() <= margin, (crawled, received)
    assert crawled > cached, (crawled, cached)


def test_live_extract_cap_rejects_without_request(mocker: Any) -> None:
    from octen.resources.extract import ExtractResource

    spy = mocker.spy(ExtractResource, "extract")
    spec = _spec(extract_options={"max_urls": 2})
    result = spec.extract([EXTRACT_DOC, "https://example.com", "https://octen.ai"])
    assert set(result) == {"error"}
    assert "at most 2" in result["error"]
    spy.assert_not_called()


def test_live_extract_model_cannot_pass_mode() -> None:
    tools = {t.metadata.name: t for t in _spec().to_tool_list()}
    with pytest.raises(TypeError, match="mode"):
        tools["extract"].call(urls=["https://example.com"], mode="advanced")


def test_live_extract_bad_option_raises_at_construction() -> None:
    with pytest.raises(ValueError, match="extract_options.*max_urls"):
        _spec(extract_options={"max_urls": 21})


# The URL does not name its subject. Per https://unsplash.com/photos/Q0-fOL2nqZc (title and
# alt text "Eiffel tower during daytime", checked 2026-11-22) it shows the Eiffel Tower.
REFERENCE_IMAGE = "https://images.unsplash.com/photo-1511739001486-6bfe10ce785f?w=800&fm=jpg"
IMAGE_QUERY = "golden retriever puppy"


def _image_hosts(items: List[Dict[str, Any]]) -> List[str]:
    return [urlparse(i.get("source_page", "")).hostname or "" for i in items]


def _check_images(name: str, result: Dict[str, Any]) -> List[Dict[str, Any]]:
    _check(name, result)
    items = result["data"]["results"]
    print(f"{name}: {len(items)} results, hosts={_image_hosts(items)}")
    for item in items:
        assert item.get("url", "").startswith("http"), item
    return items


def test_live_image_search_by_text_count_and_domains() -> None:
    control = _check_images("image control", _spec().image_search(IMAGE_QUERY))
    assert len(control) > 2, control  # the API default count is 5
    assert all(i.get("source_page") for i in control), control

    capped = _check_images(
        "image count=2", _spec(image_search_options={"count": 2}).image_search(IMAGE_QUERY)
    )
    assert 0 < len(capped) <= 2, capped

    # Exclude the most common source host of the control; the filtered call drops it.
    hosts = _image_hosts(control)
    top = max(set(hosts), key=hosts.count)
    domain = top[4:] if top.startswith("www.") else top
    excluded = _check_images(
        "image exclude_domains", _spec().image_search(IMAGE_QUERY, exclude_domains=[domain])
    )
    assert excluded and not any(_on(h, domain) for h in _image_hosts(excluded)), excluded

    included = _check_images(
        "image include_domains",
        _spec().image_search(IMAGE_QUERY, include_domains=["wikimedia.org"]),
    )
    assert any(not _on(h, "wikimedia.org") for h in hosts), control
    assert included and all(_on(h, "wikimedia.org") for h in _image_hosts(included)), included


def test_live_image_search_by_url_and_safesearch(mocker: Any) -> None:
    from octen.core.http_client import HTTPClient

    spy = mocker.spy(HTTPClient, "request")
    spec = _spec(image_search_options={"count": 3, "safesearch": "off"})
    items = _check_images("image by url", spec.image_search(image_url=REFERENCE_IMAGE))
    assert 0 < len(items) <= 3, items
    # Results about the pictured subject show the image was used, not ignored.
    print(f"titles={[i.get('title') for i in items]}")
    texts = [f"{i.get('title') or ''} {i.get('description') or ''}" for i in items]
    assert any(re.search(r"eiffel", t, re.I) for t in texts), texts
    body = spy.call_args.kwargs["json"]
    print(f"wire body={body}")
    assert body["inputs"] == [{"type": "image", "url": REFERENCE_IMAGE}]
    assert body["safesearch"] == "off"


def test_live_image_search_violations_send_nothing(mocker: Any) -> None:
    from octen.resources.image_search import ImageSearchResource

    spy = mocker.spy(ImageSearchResource, "search")
    both = _spec().image_search("cats", image_url=REFERENCE_IMAGE)
    neither = _spec().image_search()
    assert "not both" in both["error"] and "Got neither" in neither["error"]
    tools = {t.metadata.name: t for t in _spec().to_tool_list()}
    with pytest.raises(TypeError, match="count"):
        tools["image_search"].call(query="cats", count=10)
    with pytest.raises(ValueError, match="image_search_options.*count"):
        _spec(image_search_options={"count": 11})
    spy.assert_not_called()


DESIGN_QUERY = "pricing section for a SaaS landing page"


def _check_design(name: str, result: Dict[str, Any]) -> Dict[str, Any]:
    _check(name, result)
    data = result["data"]
    images, web = data.get("images"), data.get("results")
    print(f"{name}: images={None if images is None else len(images)} "
          f"web={None if web is None else len(web)} "
          f"groups={[(i.get('group') or {}).get('group_id', '')[:24] for i in images or []]}")
    for image in images or []:
        assert image.get("url", "").startswith("http") and image.get("type"), image
    return data


def _base_url(image: Dict[str, Any]) -> str:
    # Image URLs carry a signed query string that may differ between calls.
    return str(image.get("url", "")).split("?")[0]


def test_live_design_search_options_and_group_follow_up() -> None:
    # Same query and image count; the control turns design facts and snippets on. The web
    # options differ only in full_content.
    on = {"images": {"count": 3, "design_facts": True, "snippet": {"enable": True}}}
    control = _check_design(
        "design control", _spec(design_search_options=on).design_search(DESIGN_QUERY)
    )
    print(f"control snippet={[bool(i.get('snippet')) for i in control['images']]} "
          f"design_facts={[bool(i.get('design_facts')) for i in control['images']]}")
    assert not any(r.get("full_content") for r in control["results"]), control["results"]
    assert 0 < len(control["images"]) <= 3, control

    off = {
        "images": {"count": 3, "design_facts": False, "snippet": {"enable": False}},
        "web_search": {"full_content": {"enable": True, "max_tokens": 200}},
    }
    turned_off = _check_design(
        "design options", _spec(design_search_options=off).design_search(DESIGN_QUERY)
    )
    # Matched by URL without the signed query string, not by position.
    by_url = {_base_url(i): i for i in control["images"]}
    matched = [(by_url[_base_url(i)], i) for i in turned_off["images"] if _base_url(i) in by_url]
    print(f"matched images={len(matched)} of {len(turned_off['images'])}")
    assert 0 < len(turned_off["images"]) <= 3, turned_off
    assert any(before.get("snippet") for before, _ in matched), matched
    assert any(before.get("design_facts") for before, _ in matched), matched
    assert not any(i.get("design_facts") or i.get("snippet") for i in turned_off["images"])
    assert turned_off["results"], turned_off
    assert all(r.get("full_content") for r in turned_off["results"]), turned_off["results"]

    grouped = [i for i in control["images"] if (i.get("group") or {}).get("total", 0) >= 3]
    assert grouped, "no image from a group of 3 or more: pick another DESIGN_QUERY"
    seed = grouped[0]
    group_id, position, total = (seed["group"][k] for k in ("group_id", "position", "total"))
    requested = 100
    follow = _check_design(
        "design group_id",
        _spec(design_search_options={"images": {"count": requested}}).design_search(
            group_id=group_id
        ),
    )
    members = follow["images"]
    # The ordered prefix of the group, up to the requested image count.
    expected = min(total, requested)
    ids = [i["group"]["group_id"] for i in members]
    positions = [i["group"]["position"] for i in members]
    print(f"group total={total} seed position={position} positions={positions}")
    assert ids == [group_id] * expected, ids
    assert positions == list(range(1, expected + 1)), positions
    if position <= expected:
        same_slot = [i for i in members if i["group"]["position"] == position]
        assert [_base_url(i) for i in same_slot] == [_base_url(seed)], (same_slot, seed)


def _design_web(name: str, web_search: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    options: Dict[str, Any] = {"images": {"count": 1}}
    if web_search is not None:
        options["web_search"] = web_search
    return _check_design(name, _spec(design_search_options=options).design_search(DESIGN_QUERY))


def test_live_design_search_web_toggle_cap_and_domains(mocker: Any) -> None:
    from octen.core.http_client import HTTPClient

    off = _design_web("design web off", {"enable": False})
    assert not off.get("results"), off
    assert off["images"], off

    # Each call below differs from the control in one web option only.
    control = _design_web("design web control")
    hosts = [urlparse(r["url"]).hostname or "" for r in control["results"]]
    capped = _design_web("design web count=2", {"count": 2})
    print(f"control web={len(control['results'])} capped web={len(capped['results'])}")
    assert len(control["results"]) > 2, control
    assert 0 < len(capped["results"]) <= 2, capped

    assert any(r.get("highlight") for r in control["results"]), control["results"]
    spy = mocker.spy(HTTPClient, "request")
    plain = _design_web("design web highlight off", {"highlight": {"enable": False}})
    print(f"highlight off body={spy.call_args.kwargs['json']['web_search']}")
    assert spy.call_args.kwargs["json"]["web_search"]["highlight"]["enable"] is False
    assert plain["results"] and not any(r.get("highlight") for r in plain["results"]), plain

    domain = hosts[0][4:] if hosts[0].startswith("www.") else hosts[0]
    filtered = _design_web("design web exclude_domains", {"exclude_domains": [domain]})
    filtered_hosts = [urlparse(r["url"]).hostname or "" for r in filtered["results"]]
    print(f"excluded {domain}: control={hosts} filtered={filtered_hosts}")
    assert filtered_hosts and not any(_on(h, domain) for h in filtered_hosts), filtered_hosts


def test_live_design_search_violations_send_nothing(mocker: Any) -> None:
    from octen.resources.design_search import DesignSearchResource

    spy = mocker.spy(DesignSearchResource, "search")
    assert "Got neither" in _spec().design_search(query="  ")["error"]
    tools = {t.metadata.name: t for t in _spec().to_tool_list()}
    with pytest.raises(TypeError, match="images"):
        tools["design_search"].call(query="q", images={"count": 100})
    with pytest.raises(ValueError, match="images.group_id"):
        _spec(design_search_options={"images": {"group_id": "pg_x"}})
    spy.assert_not_called()
