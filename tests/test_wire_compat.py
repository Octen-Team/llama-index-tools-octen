"""The requests the existing functions send stay unchanged across releases.

EXPECTED was captured by running _capture() on llama-index-tools-octen 0.2.0
(origin/main 4a023bc) with octen 0.12.0, its minimum supported version; this branch
runs the same inputs with octen 0.13.0. A difference means a 0.2.0 function now sends
a different request, whether from this package or from the octen upgrade.

EXPECTED_PREMIER was captured the same way on 58148b3, before extract was added.
Every capture runs twice: as given, and with extract_options set as well.
"""

import json
from typing import Any, Dict, List, Tuple
from unittest.mock import patch

import pytest

from llama_index.tools.octen import OctenToolSpec

CANNED = {
    "code": 0,
    "msg": "success",
    "request_id": "rid",
    "data": {
        "results": [{"title": "T", "url": "https://example.com", "highlight": "h"}]
    },
    "meta": {},
}

Call = Tuple[str, str, Dict[str, Any], Any]


def _calls() -> List[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    old_args: Dict[str, Any] = {
        "num_results": 3,
        "include_domains": ["a.com"],
        "exclude_domains": ["b.com"],
        "start_time": "2025-01-01",
        "end_time": "2025-06-01",
        "time_basis": "published",
        "include_text": ["python"],
        "exclude_text": ["java"],
        "safesearch": "off",
        "format": "markdown",
    }
    return [
        ("search", {"timeout": 9.0}, old_args),
        ("search_and_retrieve_documents", {"max_characters": 1500}, old_args),
        (
            "search_and_retrieve_highlights",
            {},
            {**old_args, "highlight_max_tokens": 300},
        ),
        (
            "broad_search",
            {
                "timeout": 9.0,
                "broad_search_options": {
                    "count": 2,
                    "max_queries": 3,
                    "highlight": {"enable": True, "max_tokens": 300},
                    "full_content": {"enable": False},
                },
            },
            {"include_domains": ["a.com"], "time_range": "week", "language": ["en"]},
        ),
        (
            "news_search",
            {
                "news_search_options": {
                    "count": 2,
                    "subjects": {"enable": True, "count": 1},
                }
            },
            {"exclude_domains": ["b.com"], "time_range": "month"},
        ),
        (
            "business_search",
            {"business_search_options": {"entities": {"count": 1}, "timeout": 20}},
            {"language": ["en"]},
        ),
        (
            "pulse_search",
            {"pulse_search_options": {"count": 1, "mode": "standard"}},
            {},
        ),
        (
            "place_search",
            {"place_search_options": {"places": {"count": 1}}},
            {"latitude": 35.0, "longitude": 139.0, "radius": 500, "city": "Tokyo"},
        ),
    ]


# Captured from 0.2.0 (4a023bc) with octen 0.12.0; see the module docstring.
EXPECTED: Dict[str, Call] = {
    "search": (
        "POST",
        "/search",
        {
            "count": 3,
            "include_domains": ["a.com"],
            "exclude_domains": ["b.com"],
            "time_basis": "published",
            "start_time": "2025-01-01",
            "end_time": "2025-06-01",
            "include_text": ["python"],
            "exclude_text": ["java"],
            "format": "markdown",
            "safesearch": "off",
            "query": "q",
        },
        9.0,
    ),
    "search_and_retrieve_documents": (
        "POST",
        "/search",
        {
            "count": 3,
            "include_domains": ["a.com"],
            "exclude_domains": ["b.com"],
            "time_basis": "published",
            "start_time": "2025-01-01",
            "end_time": "2025-06-01",
            "full_content": {"enable": True, "max_tokens": 1500},
            "include_text": ["python"],
            "exclude_text": ["java"],
            "format": "markdown",
            "safesearch": "off",
            "query": "q",
        },
        None,
    ),
    "search_and_retrieve_highlights": (
        "POST",
        "/search",
        {
            "count": 3,
            "include_domains": ["a.com"],
            "exclude_domains": ["b.com"],
            "time_basis": "published",
            "start_time": "2025-01-01",
            "end_time": "2025-06-01",
            "highlight": {"enable": True, "max_tokens": 300},
            "include_text": ["python"],
            "exclude_text": ["java"],
            "format": "markdown",
            "safesearch": "off",
            "query": "q",
        },
        None,
    ),
    "broad_search": (
        "POST",
        "/broad-search",
        {
            "query": "q",
            "max_queries": 3,
            "search_options": {
                "count": 2,
                "include_domains": ["a.com"],
                "language": ["en"],
                "time_range": "week",
                "highlight": {"enable": True, "max_tokens": 300},
                "full_content": {"enable": False, "max_tokens": 2048},
            },
        },
        9.0,
    ),
    "news_search": (
        "POST",
        "/news-search",
        {
            "count": 2,
            "exclude_domains": ["b.com"],
            "time_range": "month",
            "query": "q",
            "subjects": {"enable": True, "count": 1},
        },
        None,
    ),
    "business_search": (
        "POST",
        "/business-search",
        {"language": ["en"], "query": "q", "entities": {"count": 1}},
        20,
    ),
    "pulse_search": (
        "POST",
        "/pulse-search",
        {"query": "q", "mode": "standard", "count": 1},
        None,
    ),
    "place_search": (
        "POST",
        "/place-search",
        {
            "query": "q",
            "latitude": 35.0,
            "longitude": 139.0,
            "radius": 500,
            "user_location": {"type": "approximate", "city": "Tokyo"},
            "places": {"count": 1},
        },
        None,
    ),
}


PREMIER_CALL = (
    "search_premier",
    {
        "timeout": 9.0,
        "search_premier_options": {
            "count": 3,
            "result_type": "dynamic",
            "highlight": {"enable": True, "max_tokens": 600},
            "full_content": {"enable": True, "max_tokens": 1000},
            "include_text": ["kafka"],
            "exclude_text": ["ads"],
            "language": ["en", "de"],
            "time_basis": "published",
            "start_time": "2025-01-01T00:00:00Z",
            "end_time": "2025-06-30T23:59:59+08:00",
            "format": "markdown",
            "safesearch": "off",
            "include_images": True,
            "timeout": 20,
        },
    },
    {"include_domains": ["a.com"], "exclude_domains": ["b.com"]},
)

# Captured from 58148b3; see the module docstring.
EXPECTED_PREMIER: Dict[str, Call] = {
    "search_premier": (
        "POST",
        "/search-premier",
        {
            "count": 3,
            "include_domains": ["a.com"],
            "exclude_domains": ["b.com"],
            "language": ["en", "de"],
            "time_basis": "published",
            "start_time": "2025-01-01T00:00:00Z",
            "end_time": "2025-06-30T23:59:59+08:00",
            "highlight": {"enable": True, "max_tokens": 600},
            "full_content": {"enable": True, "max_tokens": 1000},
            "query": "q",
            "result_type": "dynamic",
            "include_text": ["kafka"],
            "exclude_text": ["ads"],
            "format": "markdown",
            "safesearch": "off",
            "include_images": True,
        },
        20,
    ),
}

EXTRACT_OPTIONS = {
    "mode": "advanced",
    "format": "text",
    "max_age_seconds": 600,
    "timeout": 60,
    "include_images": True,
    "include_videos": True,
    "include_audio": True,
    "include_links": {"max_links": 10},
    "max_urls": 20,
    "request_timeout": 120,
}


def _capture(
    calls_in: List[Tuple[str, Dict[str, Any], Dict[str, Any]]],
    extra: Dict[str, Any],
) -> Dict[str, Call]:
    calls: Dict[str, Call] = {}
    for function, spec_kwargs, args in calls_in:
        seen: List[Call] = []

        def fake(
            self: Any,
            method: str,
            endpoint: str,
            timeout: Any = None,
            json: Any = None,
            **kwargs: Any,
        ) -> Dict[str, Any]:
            seen.append((method, endpoint, json, timeout))
            return dict(CANNED)

        spec = OctenToolSpec(api_key="k", verbose=False, **spec_kwargs, **extra)
        with patch("octen.core.http_client.HTTPClient.request", fake):
            getattr(spec, function)("q", **args)
        assert len(seen) == 1, (function, seen)
        calls[function] = seen[0]
    return calls


@pytest.mark.parametrize("extra", [{}, {"extract_options": EXTRACT_OPTIONS}])
def test_existing_functions_send_the_same_requests(extra: Dict[str, Any]) -> None:
    # json.dumps keeps key order, so a reordered body fails too.
    assert json.dumps(_capture(_calls(), extra)) == json.dumps(EXPECTED)


@pytest.mark.parametrize("extra", [{}, {"extract_options": EXTRACT_OPTIONS}])
def test_search_premier_sends_the_same_request(extra: Dict[str, Any]) -> None:
    captured = _capture([PREMIER_CALL], extra)
    assert json.dumps(captured) == json.dumps(EXPECTED_PREMIER)
