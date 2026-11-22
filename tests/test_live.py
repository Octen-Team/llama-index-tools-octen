"""Live calls against the Octen API. Billed as usual.

Skipped unless OCTEN_LIVE=1 and OCTEN_API_KEY are set. Each test makes at
most one billed request with the smallest result count the endpoint allows,
except the search_premier and extract tests, which make two or three (options
under test and same-input controls) with a small count or one to three URLs.
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
    assert [i["url"] for i in items] == urls, items
    example, doc, failed = items
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
