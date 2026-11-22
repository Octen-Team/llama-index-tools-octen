"""Unit tests for the extract function.

Limits and field names follow https://docs.octen.ai/api-reference/extract.md and
octen 0.13.0 (octen/resources/extract.py, octen/models/extract.py).
"""

import inspect
import logging
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import octen
import pytest
from octen import (
    OctenAPIError,
    OctenAuthenticationError,
    OctenConnectionError,
    OctenRateLimitError,
    OctenTimeoutError,
    OctenValidationError,
)

from llama_index.tools.octen import OctenToolSpec

URLS = ["https://example.com", "https://docs.octen.ai/overview/welcome"]
ALL_OPTIONS: Dict[str, Any] = {
    "mode": "auto",
    "format": "text",
    "max_age_seconds": 3600,
    "timeout": 45,
    "include_images": True,
    "include_videos": True,
    "include_audio": True,
    "include_links": {"scope": "prefer_external", "max_links": 50},
    "max_urls": 10,
    "request_timeout": 120.0,
}
DEVELOPER_ONLY = [k for k in ALL_OPTIONS]


def _response(
    results: List[Dict[str, Any]], meta: Any = None, request_id: str = "rid-x"
) -> Any:
    return octen.ExtractResponse(
        code=0,
        msg="success",
        data={"results": results},
        meta={} if meta is None else meta,
        request_id=request_id,
    )


@pytest.fixture
def octen_cls(mocker: Any) -> MagicMock:
    cls = mocker.patch("octen.Octen")
    cls.return_value.extract.extract.return_value = _response([])
    return cls


def _sdk(octen_cls: MagicMock) -> MagicMock:
    return octen_cls.return_value.extract.extract


def _spec(**kwargs: Any) -> OctenToolSpec:
    return OctenToolSpec(api_key="k", verbose=False, **kwargs)


def _tools(**kwargs: Any) -> Dict[str, Any]:
    return {t.metadata.name: t for t in _spec(**kwargs).to_tool_list()}


# --- metadata the model sees ---


def test_extract_listed_before_current_date() -> None:
    assert OctenToolSpec.spec_functions[-2:] == ["extract", "current_date"]
    assert "extract" in _tools()


def test_extract_schema() -> None:
    schema = _tools()["extract"].metadata.fn_schema
    assert schema is not None
    assert list(schema.model_fields) == ["urls", "query"]
    json_schema = schema.model_json_schema()
    assert json_schema["required"] == ["urls"]
    assert json_schema["properties"]["urls"]["type"] == "array"
    for field, prop in json_schema["properties"].items():
        assert prop.get("description"), field


def test_extract_schema_matches_function_signature() -> None:
    schema = _tools()["extract"].metadata.fn_schema
    assert schema is not None
    params = inspect.signature(OctenToolSpec.extract).parameters
    assert list(schema.model_fields) == [p for p in params if p != "self"]
    for field, info in schema.model_fields.items():
        assert info.annotation == params[field].annotation, field
        assert info.default == params[field].default or info.is_required(), field


@pytest.mark.parametrize(("options", "cap"), [({}, 5), ({"max_urls": 12}, 12)])
def test_description_states_the_cap(options: Dict[str, Any], cap: int) -> None:
    description = _tools(extract_options=options)["extract"].metadata.description
    assert f"at most {cap} URLs per call" in description
    for phrase in [
        "already have the URLs",
        "search or broad_search",
        "full_content",
        "highlights instead of full_content",
        'status "failed"',
        "error_message",
    ]:
        assert phrase in description, phrase


@pytest.mark.parametrize("cap", [1, 5, 20])
def test_description_passes_openai_length_check(cap: int) -> None:
    _tools(extract_options={"max_urls": cap})["extract"].metadata.to_openai_tool()


# --- what reaches the SDK and the wire ---


def test_model_inputs_reach_sdk(octen_cls: MagicMock) -> None:
    _spec().extract(URLS, query="pricing")
    _sdk(octen_cls).assert_called_once_with(urls=URLS, query="pricing")
    octen_cls.assert_called_once_with(api_key="k", retry_after_send=False)


def test_unset_options_not_sent(octen_cls: MagicMock) -> None:
    _spec().extract(URLS)
    _sdk(octen_cls).assert_called_once_with(urls=URLS)


def test_developer_options_forwarded(octen_cls: MagicMock) -> None:
    _spec(extract_options=ALL_OPTIONS).extract(URLS, query="q")
    expected = {k: v for k, v in ALL_OPTIONS.items() if k != "max_urls"}
    _sdk(octen_cls).assert_called_once_with(urls=URLS, query="q", **expected)


def _wire(spec: OctenToolSpec, *args: Any, **kwargs: Any) -> List[Any]:
    seen: List[Any] = []

    def fake(
        self: Any, method: str, endpoint: str, timeout: Any = None, json: Any = None, **kw: Any
    ) -> Dict[str, Any]:
        seen.append((method, endpoint, json, timeout))
        return {"code": 0, "msg": "success", "request_id": "r", "data": {"results": []}, "meta": {}}

    with patch("octen.core.http_client.HTTPClient.request", fake):
        result = spec.extract(*args, **kwargs)
    assert "error" not in result, result
    return seen


def test_wire_body_with_defaults() -> None:
    assert _wire(_spec(), URLS) == [("POST", "/extract", {"urls": URLS}, None)]


def test_wire_body_with_every_option() -> None:
    seen = _wire(_spec(extract_options=ALL_OPTIONS), URLS, query="pricing")
    assert seen == [
        (
            "POST",
            "/extract",
            {
                "urls": URLS,
                "format": "text",
                "max_age_seconds": 3600,
                "query": "pricing",
                "timeout": 45,
                "include_images": True,
                "include_videos": True,
                "include_audio": True,
                "mode": "auto",
                "include_links": {"scope": "prefer_external", "max_links": 50},
            },
            120.0,
        )
    ]


def test_socket_timeout_falls_back_to_spec_timeout(octen_cls: MagicMock) -> None:
    _spec(timeout=10).extract(URLS)
    _spec(timeout=10, extract_options={"request_timeout": 90}).extract(URLS)
    _spec(timeout=10, extract_options={"timeout": 60}).extract(URLS)
    calls = [c.kwargs for c in _sdk(octen_cls).call_args_list]
    assert calls == [
        {"urls": URLS, "request_timeout": 10},
        {"urls": URLS, "request_timeout": 90},
        # The upstream per-URL timeout never stands in for the socket one, or back.
        {"urls": URLS, "timeout": 60, "request_timeout": 10},
    ]


def test_model_cannot_pass_developer_options(octen_cls: MagicMock) -> None:
    tool = _tools()["extract"]
    for key in DEVELOPER_ONLY:
        with pytest.raises(TypeError, match=key):
            tool.call(urls=URLS, **{key: ALL_OPTIONS[key]})
    octen_cls.assert_not_called()


def test_tool_call_through_llamaindex(octen_cls: MagicMock) -> None:
    tool = _tools()["extract"]
    assert tool.call(urls=URLS).raw_output == {"data": {"results": []}, "request_id": "rid-x"}
    _sdk(octen_cls).assert_called_once_with(urls=URLS)


def test_options_deep_copied(octen_cls: MagicMock) -> None:
    links = {"max_links": 10}
    options = {"include_links": links, "max_urls": 2}
    spec = _spec(extract_options=options)
    links["max_links"] = 5000
    options["max_urls"] = 20
    spec.extract(URLS)
    assert _sdk(octen_cls).call_args.kwargs["include_links"] == {"max_links": 10}
    assert "error" in spec.extract(URLS * 2)


def test_sdk_links_object_accepted(octen_cls: MagicMock) -> None:
    links = octen.ExtractLinksOptions(max_links=7)
    _spec(extract_options={"include_links": links}).extract(URLS)
    assert _sdk(octen_cls).call_args.kwargs["include_links"].max_links == 7


def test_names_match_real_sdk_signature() -> None:
    # No mocks: everything forwarded must be a parameter of the SDK method.
    from octen.resources.extract import ExtractResource

    from llama_index.tools.octen.base import _ALLOWED_OPTIONS

    accepted = set(inspect.signature(ExtractResource.extract).parameters) - {"self"}
    options = _ALLOWED_OPTIONS["extract_options"] - {"max_urls"}
    model_args = set(inspect.signature(OctenToolSpec.extract).parameters) - {"self"}
    assert options <= accepted, options - accepted
    assert model_args <= accepted, model_args - accepted


# --- URL cap and model-argument checks: no request is sent ---


def test_more_urls_than_default_cap_rejected_without_call(octen_cls: MagicMock) -> None:
    urls = [f"https://example.com/{i}" for i in range(6)]
    result = _spec().extract(urls)
    assert set(result) == {"error"}
    message = result["error"]
    assert "6" in message and "at most 5" in message
    assert "split" in message.lower()
    assert "nothing was fetched" in message.lower()
    octen_cls.assert_not_called()


def test_default_cap_accepts_five(octen_cls: MagicMock) -> None:
    urls = [f"https://example.com/{i}" for i in range(5)]
    assert "error" not in _spec().extract(urls)
    _sdk(octen_cls).assert_called_once_with(urls=urls)


@pytest.mark.parametrize("cap", [1, 2, 20])
def test_custom_cap(octen_cls: MagicMock, cap: int) -> None:
    spec = _spec(extract_options={"max_urls": cap})
    urls = [f"https://example.com/{i}" for i in range(cap + 1)]
    result = spec.extract(urls)
    assert set(result) == {"error"}
    assert f"at most {cap}" in result["error"]
    octen_cls.assert_not_called()
    assert "error" not in spec.extract(urls[:cap])
    _sdk(octen_cls).assert_called_once_with(urls=urls[:cap])


def test_empty_url_list_rejected_without_call(octen_cls: MagicMock) -> None:
    result = _spec().extract([])
    assert set(result) == {"error"}
    assert "urls" in result["error"] and "at most 5" in result["error"]
    octen_cls.assert_not_called()


@pytest.mark.parametrize(
    ("args", "field"),
    [
        ({"urls": ["https://example.com/" + "a" * 2029]}, "urls"),
        ({"urls": ["https://example.com", ""]}, "urls"),
        ({"urls": "https://example.com"}, "urls"),
        ({"urls": [1]}, "urls"),
        ({"urls": URLS, "query": "x" * 501}, "query"),
    ],
    ids=["url-too-long", "empty-url", "not-a-list", "not-strings", "query-too-long"],
)
def test_invalid_model_args_returned_without_call(
    octen_cls: MagicMock, args: Dict[str, Any], field: str
) -> None:
    result = _spec().extract(**args)
    assert set(result) == {"error"}
    assert field in result["error"]
    octen_cls.assert_not_called()


def test_longest_allowed_url_and_query_sent(octen_cls: MagicMock) -> None:
    url = "https://example.com/" + "a" * 2028
    assert len(url) == 2048
    assert "error" not in _spec().extract([url], query="x" * 500)


# --- construction checks for developer options ---


@pytest.mark.parametrize(
    ("options", "mentions"),
    [
        ({"max_urls": 0}, "max_urls"),
        ({"max_urls": 21}, "max_urls"),
        ({"max_urls": True}, "max_urls"),
        ({"max_urls": "5"}, "max_urls"),
        ({"max_urls": 5.0}, "max_urls"),
        ({"timeout": 0}, "timeout"),
        ({"timeout": 61}, "timeout"),
        ({"timeout": True}, "timeout"),
        ({"timeout": 5.5}, "timeout"),
        ({"timeout": "30"}, "timeout"),
        ({"max_age_seconds": 299}, "max_age_seconds"),
        ({"max_age_seconds": 31_536_001}, "max_age_seconds"),
        ({"max_age_seconds": True}, "max_age_seconds"),
        ({"mode": "fast"}, "mode"),
        ({"format": "html"}, "format"),
        ({"include_images": "maybe"}, "include_images"),
        ({"include_videos": [True]}, "include_videos"),
        ({"include_audio": "sometimes"}, "include_audio"),
        ({"include_links": True}, "include_links"),
        ({"include_links": {"max_links": 0}}, "include_links"),
        ({"include_links": {"max_links": 1001}}, "include_links"),
        ({"include_links": {"scope": "all"}}, "include_links"),
        ({"include_links": {"limit": 5}}, "include_links"),
        ({"request_timeout": 0}, "request_timeout"),
        ({"request_timeout": float("nan")}, "request_timeout"),
        ({"request_timeout": True}, "request_timeout"),
        ({"request_timeout": "90"}, "request_timeout"),
    ],
    ids=lambda v: v if isinstance(v, str) else None,
)
def test_invalid_option_raises_at_construction(
    octen_cls: MagicMock, options: Dict[str, Any], mentions: str
) -> None:
    with pytest.raises(ValueError, match=f"extract_options.*{mentions}"):
        _spec(extract_options=options)
    octen_cls.assert_not_called()


@pytest.mark.parametrize(
    "options",
    [
        {"max_urls": 1},
        {"max_urls": 20},
        {"timeout": 1},
        {"timeout": 60},
        {"max_age_seconds": 300},
        {"max_age_seconds": 31_536_000},
        {"mode": "standard"},
        {"mode": "advanced"},
        {"mode": "auto"},
        {"format": "markdown"},
        {"format": "text"},
        {"include_links": {}},
        {"include_links": {"scope": "prefer_internal", "max_links": 1}},
        {"include_links": {"max_links": 1000}},
        {"include_images": False, "include_videos": False, "include_audio": False},
        {"request_timeout": 0.5},
    ],
)
def test_option_bounds_accepted(options: Dict[str, Any]) -> None:
    _spec(extract_options=options)


@pytest.mark.parametrize(
    "key", ["urls", "query", "count", "highlight", "full_content", "socket_timeout"]
)
def test_model_or_unknown_key_rejected_as_option(key: str) -> None:
    with pytest.raises(ValueError, match=f"extract_options does not accept '{key}'"):
        _spec(extract_options={key: "x"})


# --- return shape ---


MIXED = [
    {
        "url": "https://example.com",
        "status": "success",
        "resolved_mode": "standard",
        "title": "Example Domain",
        "full_content": "# Example Domain\n\n" + "body " * 5000,
        "highlights": None,
        "time_published": None,
        "time_last_crawled": "2026-11-20T10:00:00Z",
        "page_structure": {"primary": "Content Page", "secondary": None},
        "category": {"primary": "Computers, Electronics & Technology", "secondary": None},
        "favicon": "https://example.com/favicon.ico",
    },
    {
        "url": "https://nonexistent.invalid/",
        "status": "failed",
        "error_message": "fetch failed",
    },
    {
        "url": "https://docs.octen.ai/overview/welcome",
        "status": "success",
        "resolved_mode": "advanced",
        "title": "Welcome",
        "highlights": ["first", "second"],
        "time_published": "2025-01-01T00:00:00Z",
        "links": [{"url": "https://octen.ai", "anchor_text": "Home", "is_external": True}],
    },
]


def test_mixed_batch_keeps_every_item_untruncated(octen_cls: MagicMock) -> None:
    meta = {
        "usage": {
            "total_urls": 3,
            "successful_urls": 2,
            "successful_by_mode": {"standard_urls": 1, "advanced_urls": 1},
        },
        "latency": 1200,
        "warning": "",
    }
    _sdk(octen_cls).return_value = _response(MIXED, meta=meta, request_id="rid-mixed")
    result = _spec().extract([r["url"] for r in MIXED])
    assert result == {"data": {"results": MIXED}, "request_id": "rid-mixed"}
    assert result["data"]["results"][0]["full_content"] == MIXED[0]["full_content"]


def test_warning_kept(octen_cls: MagicMock) -> None:
    _sdk(octen_cls).return_value = _response(MIXED[:1], meta={"warning": "slow; cached"})
    assert _spec().extract(URLS)["warning"] == "slow; cached"


def test_verbose_prints_request_id_and_usage(octen_cls: MagicMock, capsys: Any) -> None:
    usage = {"total_urls": 2, "successful_urls": 1,
             "successful_by_mode": {"standard_urls": 0, "advanced_urls": 1}}
    _sdk(octen_cls).return_value = _response(MIXED[1:], meta={"usage": usage})
    OctenToolSpec(api_key="k").extract(URLS)
    out = capsys.readouterr().out
    assert "extract request_id=rid-x" in out
    assert "2 URLs" in out
    assert "'advanced_urls': 1" in out


# --- errors ---


@pytest.mark.parametrize(
    "exc",
    [
        OctenRateLimitError(),
        OctenAPIError("upstream failed", status_code=503),
        OctenAPIError("bad request", status_code=400),
        OctenTimeoutError(timeout=30, request_sent=False),
        OctenConnectionError(request_sent=False),
    ],
)
def test_transient_errors_returned_to_model(octen_cls: MagicMock, exc: Exception) -> None:
    _sdk(octen_cls).side_effect = exc
    assert _spec().extract(URLS) == {"error": str(exc)}
    octen_cls.return_value.close.assert_called_once()


@pytest.mark.parametrize(
    "exc",
    [
        OctenTimeoutError(timeout=30, request_sent=True),
        OctenConnectionError(request_sent=None),
        OctenAPIError("gateway", status_code=504),
    ],
)
def test_possibly_billed_errors_tell_model_not_to_retry(
    octen_cls: MagicMock, exc: Exception
) -> None:
    _sdk(octen_cls).side_effect = exc
    error = _spec().extract(URLS)["error"]
    assert error.startswith(str(exc))
    assert "may already have reached the server and been billed" in error
    assert "do not retry it automatically" in error


@pytest.mark.parametrize(
    "exc",
    [
        OctenAuthenticationError(),
        OctenAPIError("forbidden", status_code=403),
        OctenAPIError("not found", status_code=404),
        OctenValidationError("bad option"),
    ],
)
def test_config_errors_raise_and_are_logged(
    octen_cls: MagicMock, caplog: Any, exc: Exception
) -> None:
    _sdk(octen_cls).side_effect = exc
    with caplog.at_level(logging.ERROR, logger="llama_index.tools.octen"):
        with pytest.raises(type(exc)):
            _spec().extract(URLS)
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1 and "extract raised" in errors[0].getMessage()
    octen_cls.return_value.close.assert_called_once()
