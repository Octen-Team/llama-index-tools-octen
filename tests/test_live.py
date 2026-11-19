"""Live calls against the Octen API. Billed as usual.

Skipped unless OCTEN_LIVE=1 and OCTEN_API_KEY are set. Each test makes at
most one billed request with the smallest result count the endpoint allows.
"""

import os
from typing import Any, Dict

import pytest
from octen import OctenAuthenticationError

from llama_index.tools.octen import OctenToolSpec

pytestmark = pytest.mark.skipif(
    os.environ.get("OCTEN_LIVE") != "1" or not os.environ.get("OCTEN_API_KEY"),
    reason="live tests need OCTEN_LIVE=1 and OCTEN_API_KEY",
)


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
    spec = _spec(news_search_options={"count": 1, "subjects": {"count": 1, "max_sub_news": 1}})
    result = spec.news_search("semiconductor industry", time_range="week")
    _check("news_search", result)
    assert result["data"]["results"]


def test_live_business_search() -> None:
    spec = _spec(business_search_options={"count": 1, "entities": {"count": 1}})
    result = spec.business_search("NVIDIA company overview")
    _check("business_search", result)
    assert result["data"].get("results") or result["data"].get("entities")


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
