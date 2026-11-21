"""Live calls against the Octen API. Billed as usual.

Skipped unless OCTEN_LIVE=1 and OCTEN_API_KEY are set. Each test makes at
most one billed request with the smallest result count the endpoint allows,
except the search_premier tests, which make two (a filtered call and a
same-query control) with a small count.
"""

import os
import re
from datetime import datetime
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
# Single-bound controls use a window with content on both sides. With only end_time=2025-12-31
# the server returned mostly undated results and nothing before 2025 (run of 2026-11-21).
H1_2026 = ("2026-01-01T00:00:00Z", "2026-06-30T23:59:59Z")
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
    # Without start_time a result before the window comes back, and without end_time one after.
    start, end = _ts(H1_2026[0]), _ts(H1_2026[1])
    assert start and end
    before = _premier(count=10, end_time=H1_2026[1])
    after = _premier(count=10, start_time=H1_2026[0])
    _check("search_premier end-only", before)
    _check("search_premier start-only", after)
    print(f"end-only time_published={_published(before)}")
    print(f"start-only time_published={_published(after)}")
    assert any(t and t < start for t in _published(before)), before
    assert any(t and t > end for t in _published(after)), after


def test_live_search_premier_model_cannot_pass_count() -> None:
    tools = {t.metadata.name: t for t in _spec().to_tool_list()}
    with pytest.raises(TypeError, match="count"):
        tools["search_premier"].call(query="Apache Kafka", count=50)


def test_live_search_premier_bad_option_raises_at_construction() -> None:
    with pytest.raises(ValueError, match="search_premier_options.*result_type"):
        _spec(search_premier_options={"result_type": "smart"})
