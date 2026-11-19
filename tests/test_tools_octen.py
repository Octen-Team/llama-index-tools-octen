from typing import Any, Optional
from unittest.mock import MagicMock

import pytest
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from octen import (
    OctenAPIError,
    OctenAuthenticationError,
    OctenConnectionError,
    OctenRateLimitError,
    OctenTimeoutError,
    OctenValidationError,
)

from llama_index.tools.octen import OctenToolSpec
from llama_index.tools.octen.base import _classify


@pytest.fixture
def octen_cls(mocker: Any) -> MagicMock:
    cls = mocker.patch("octen.Octen")
    cls.return_value.search.search.return_value = MagicMock(results=[])
    return cls


def test_class() -> None:
    names_of_base_classes = [b.__name__ for b in OctenToolSpec.__mro__]
    assert BaseToolSpec.__name__ in names_of_base_classes


def test_api_key_explicit(octen_cls: MagicMock, monkeypatch: Any) -> None:
    monkeypatch.setenv("OCTEN_API_KEY", "env-key")
    OctenToolSpec(api_key="explicit-key", verbose=False).search("q")
    assert octen_cls.call_args.kwargs["api_key"] == "explicit-key"


def test_api_key_env_fallback(octen_cls: MagicMock, monkeypatch: Any) -> None:
    monkeypatch.setenv("OCTEN_API_KEY", "env-key")
    OctenToolSpec(verbose=False).search("q")
    assert octen_cls.call_args.kwargs["api_key"] == "env-key"


@pytest.mark.parametrize("explicit", [None, ""])
def test_missing_api_key_raises(monkeypatch: Any, explicit: Optional[str]) -> None:
    monkeypatch.delenv("OCTEN_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OCTEN_API_KEY"):
        OctenToolSpec(api_key=explicit)


def test_client_disables_retry_after_send(octen_cls: MagicMock) -> None:
    OctenToolSpec(api_key="k", verbose=False).search("q")
    octen_cls.assert_called_once_with(api_key="k", retry_after_send=False)
    octen_cls.return_value.close.assert_called_once()


@pytest.mark.parametrize(
    ("kwarg", "options"),
    [
        ("news_search_options", {"query": "x"}),
        ("news_search_options", {"time_range": "day"}),
        ("broad_search_options", {"subjects": {"enable": True}}),
        ("business_search_options", {"max_queries": 3}),
        ("pulse_search_options", {"include_domains": ["a.com"]}),
        ("place_search_options", {"count": 3}),
    ],
)
def test_unknown_option_key_rejected(kwarg: str, options: dict) -> None:
    bad_key = next(iter(options))
    with pytest.raises(ValueError, match=bad_key):
        OctenToolSpec(api_key="k", **{kwarg: options})


@pytest.mark.parametrize(
    ("kwarg", "options"),
    [
        (
            "broad_search_options",
            {"count": 3, "highlight": {}, "full_content": {}, "max_queries": 3, "timeout": 60},
        ),
        (
            "news_search_options",
            {"count": 3, "highlight": {}, "full_content": {}, "subjects": {}, "timeout": 5},
        ),
        (
            "business_search_options",
            {"count": 3, "highlight": {}, "full_content": {}, "entities": {}, "timeout": 5},
        ),
        (
            "pulse_search_options",
            {"count": 3, "highlight": {}, "full_content": {}, "mode": "lite", "timeout": 5},
        ),
        ("place_search_options", {"places": {}, "web_search": {}, "timeout": 5}),
    ],
)
def test_allowed_option_keys_accepted(kwarg: str, options: dict) -> None:
    OctenToolSpec(api_key="k", **{kwarg: options})


@pytest.mark.parametrize(
    ("kwarg", "options", "bad_option"),
    [
        ("news_search_options", {"count": 1000}, "count"),
        ("news_search_options", {"subjects": {"count": 9}}, "subjects"),
        ("news_search_options", {"subjects": {"bogus": 1}}, "subjects"),
        ("broad_search_options", {"max_queries": 99}, "max_queries"),
        ("broad_search_options", {"count": 0}, "count"),
        ("broad_search_options", {"highlight": {"max_tokens": 5}}, "highlight"),
        ("business_search_options", {"entities": {"count": 50}}, "entities"),
        ("business_search_options", {"full_content": {"max_tokens": 10**7}}, "full_content"),
        ("pulse_search_options", {"mode": "turbo"}, "mode"),
        ("pulse_search_options", {"count": 11}, "count"),
        ("place_search_options", {"places": {"count": 0}}, "places"),
        ("place_search_options", {"web_search": {"count": 101}}, "web_search"),
        ("place_search_options", {"web_search": {"time_range": "last week"}}, "web_search"),
        ("news_search_options", {"timeout": 0}, "timeout"),
        ("place_search_options", {"timeout": "soon"}, "timeout"),
        ("pulse_search_options", {"timeout": True}, "timeout"),
        ("broad_search_options", {"timeout": float("nan")}, "timeout"),
        ("business_search_options", {"timeout": float("inf")}, "timeout"),
    ],
)
def test_invalid_option_value_rejected(kwarg: str, options: dict, bad_option: str) -> None:
    with pytest.raises(ValueError, match=f"{kwarg}.*{bad_option}"):
        OctenToolSpec(api_key="k", **{kwarg: options})


@pytest.mark.parametrize("mode", ["lite", "standard"])
def test_pulse_modes_accepted(mode: str) -> None:
    OctenToolSpec(api_key="k", pulse_search_options={"mode": mode})


@pytest.mark.parametrize(
    ("exc", "to_model"),
    [
        (OctenAuthenticationError(), False),
        (OctenAPIError("x", status_code=403), False),
        (OctenValidationError("x"), False),
        (OctenAPIError("x", status_code=404), False),
        (OctenRateLimitError(), True),
        (OctenAPIError("x", status_code=503), True),
        (OctenAPIError("x", status_code=400), True),
        (OctenAPIError("x", status_code=422), True),
        (OctenTimeoutError(request_sent=False), True),
        (OctenConnectionError(request_sent=False), True),
        (OctenAPIError("Request Timeout", status_code=408), True),
        # An envelope that reports an auth or access failure stays a config error.
        (OctenAPIError("bad key", status_code=200, response_data={"code": 401}), False),
        (OctenAPIError("no balance", status_code=200, response_data={"code": 403}), False),
        (RuntimeError("bug"), False),
    ],
)
def test_classify(exc: Exception, to_model: bool) -> None:
    message = _classify(exc)
    if to_model:
        assert message == str(exc)
    else:
        assert message is None


@pytest.mark.parametrize(
    "method",
    ["search", "search_and_retrieve_documents", "search_and_retrieve_highlights"],
)
def test_search_type_ignored_with_warning(octen_cls: MagicMock, method: str) -> None:
    spec = OctenToolSpec(api_key="k", verbose=False)
    with pytest.warns(FutureWarning, match="search_type"):
        getattr(spec, method)("q", search_type="semantic")
    kwargs = octen_cls.return_value.search.search.call_args.kwargs
    assert "search_type" not in kwargs
    assert kwargs["query"] == "q"


@pytest.mark.parametrize(
    "exc",
    [
        OctenTimeoutError(timeout=30, request_sent=True),
        OctenTimeoutError(timeout=30, request_sent=None),
        OctenConnectionError(request_sent=True),
        OctenConnectionError(request_sent=None),
    ],
)
def test_classify_warns_model_when_request_may_have_been_sent(exc: Exception) -> None:
    message = _classify(exc)
    assert message is not None
    assert message.startswith(str(exc))
    assert "may already have reached the server and been billed" in message
    assert "do not retry" in message.lower()


@pytest.mark.parametrize(
    "exc", [OctenTimeoutError(request_sent=False), OctenConnectionError(request_sent=False)]
)
def test_classify_unsent_failure_keeps_plain_message(exc: Exception) -> None:
    assert _classify(exc) == str(exc)


@pytest.mark.parametrize(
    "exc",
    [
        OctenAPIError("malformed successful response", status_code=200),
        OctenAPIError("upstream busy", status_code=200, response_data={"code": 503}),
        OctenAPIError("partial", status_code=206, response_data=[]),
    ],
)
def test_classify_failed_2xx_warns_model_not_to_retry(exc: Exception) -> None:
    message = _classify(exc)
    assert message is not None
    assert message.startswith(str(exc))
    assert "may already have reached the server and been billed" in message
