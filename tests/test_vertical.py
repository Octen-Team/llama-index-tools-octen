"""Unit tests for the search_premier, broad, news, business, pulse and place functions."""

import inspect
import logging
from typing import Any, Dict
from unittest.mock import MagicMock

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

# function name -> attribute path of the SDK method it calls
SDK_METHOD = {
    "broad_search": "search.broad_search",
    "news_search": "search.news_search",
    "business_search": "search.business_search",
    "pulse_search": "search.pulse_search",
    "place_search": "place_search.search",
    "search_premier": "search.search_premier",
}
NEW_FUNCTIONS = list(SDK_METHOD)
FILTERED = ["broad_search", "news_search", "business_search"]
PREMIER_OPTIONS: Dict[str, Any] = {
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
}


def _sdk_method(octen_cls: MagicMock, function: str) -> MagicMock:
    target = octen_cls.return_value
    for part in SDK_METHOD[function].split("."):
        target = getattr(target, part)
    return target


def _response(data: Dict[str, Any], meta: Any = None, request_id: str = "rid-1") -> Any:
    # Real SDK response models, so request_id sits where octen-py puts it.
    return octen.SearchResponse(
        code=0, msg="success", data=data, meta=meta or {}, request_id=request_id
    )


@pytest.fixture
def octen_cls(mocker: Any) -> MagicMock:
    cls = mocker.patch("octen.Octen")
    for function in NEW_FUNCTIONS:
        _sdk_method(cls, function).return_value = _response({"results": []})
    return cls


def _spec(**kwargs: Any) -> OctenToolSpec:
    return OctenToolSpec(api_key="k", verbose=False, **kwargs)


def test_spec_functions_list() -> None:
    assert OctenToolSpec.spec_functions == [
        "search",
        "search_and_retrieve_documents",
        "search_and_retrieve_highlights",
        "broad_search",
        "news_search",
        "business_search",
        "pulse_search",
        "place_search",
        "search_premier",
        "extract",
        "image_search",
        "design_search",
        "current_date",
    ]


def test_tool_list_schemas() -> None:
    tools = {t.metadata.name: t for t in _spec().to_tool_list()}
    filtered = {"query", "include_domains", "exclude_domains", "time_range", "language"}
    expected = {
        "broad_search": filtered,
        "news_search": filtered,
        "business_search": filtered,
        "pulse_search": {"query"},
        "place_search": {
            "query", "latitude", "longitude", "radius",
            "city", "region", "country", "timezone",
        },
        "search_premier": {"query", "include_domains", "exclude_domains"},
    }
    for name, fields in expected.items():
        schema = tools[name].metadata.fn_schema
        assert schema is not None
        assert set(schema.model_fields) == fields, name
        json_schema = schema.model_json_schema()
        assert json_schema["required"] == ["query"], name
        # Check the JSON schema the model receives, not model_fields: some
        # llama-index-core versions set descriptions that never reach it.
        for field, prop in json_schema["properties"].items():
            assert prop.get("description"), f"{name}.{field} has no description"


def test_schema_matches_function_signature() -> None:
    tools = {t.metadata.name: t for t in _spec().to_tool_list()}
    for name in NEW_FUNCTIONS:
        schema = tools[name].metadata.fn_schema
        assert schema is not None
        params = inspect.signature(getattr(OctenToolSpec, name)).parameters
        assert list(schema.model_fields) == [p for p in params if p != "self"], name
        for field, info in schema.model_fields.items():
            assert info.annotation == params[field].annotation, (name, field)
            assert info.default == params[field].default or info.is_required(), (name, field)


def test_new_tools_pass_openai_length_check() -> None:
    for tool in _spec().to_tool_list():
        if tool.metadata.name in NEW_FUNCTIONS:
            tool.metadata.to_openai_tool()  # default 1024-char check


def test_old_tools_metadata_unchanged() -> None:
    from llama_index.core.tools import FunctionTool

    spec = _spec()
    tools = {t.metadata.name: t for t in spec.to_tool_list()}
    for name in ["search", "search_and_retrieve_documents",
                 "search_and_retrieve_highlights", "current_date"]:
        expected = FunctionTool.from_defaults(fn=getattr(spec, name)).metadata
        assert tools[name].metadata.description == expected.description, name


@pytest.mark.parametrize(
    ("function", "options", "model_args", "expected"),
    [
        (
            "broad_search",
            {"count": 2, "max_queries": 3, "highlight": {"enable": True}},
            {"time_range": "week", "include_domains": ["arxiv.org"]},
            {"count": 2, "max_queries": 3, "highlight": {"enable": True},
             "time_range": "week", "include_domains": ["arxiv.org"]},
        ),
        (
            "news_search",
            {"count": 3, "subjects": {"enable": True}},
            {"language": ["en"], "exclude_domains": ["example.com"]},
            {"count": 3, "subjects": {"enable": True},
             "language": ["en"], "exclude_domains": ["example.com"]},
        ),
        (
            "business_search",
            {"entities": {"count": 2}, "full_content": {"enable": True}},
            {},
            {"entities": {"count": 2}, "full_content": {"enable": True}},
        ),
        (
            "pulse_search",
            {"mode": "standard", "count": 1},
            {},
            {"mode": "standard", "count": 1},
        ),
        (
            "place_search",
            {"places": {"count": 1}, "web_search": {"enable": True}},
            {"latitude": 35.68, "longitude": 139.76, "radius": 500},
            {"places": {"count": 1}, "web_search": {"enable": True},
             "latitude": 35.68, "longitude": 139.76, "radius": 500},
        ),
        (
            "search_premier",
            PREMIER_OPTIONS,
            {"include_domains": ["apache.org"], "exclude_domains": ["example.com"]},
            {
                **PREMIER_OPTIONS,
                "include_domains": ["apache.org"],
                "exclude_domains": ["example.com"],
            },
        ),
    ],
)
def test_developer_options_forwarded(
    octen_cls: MagicMock,
    function: str,
    options: Dict[str, Any],
    model_args: Dict[str, Any],
    expected: Dict[str, Any],
) -> None:
    spec = _spec(**{f"{function}_options": options})
    getattr(spec, function)("q", **model_args)
    _sdk_method(octen_cls, function).assert_called_once_with(query="q", **expected)
    octen_cls.assert_called_once_with(api_key="k", retry_after_send=False)


@pytest.mark.parametrize("function", FILTERED)
def test_list_filter_generators_reach_the_sdk_as_lists(
    octen_cls: MagicMock, function: str
) -> None:
    # A one-shot iterable must not be used up by validation and forwarded empty.
    getattr(_spec(), function)(
        "q",
        include_domains=(d for d in ["a.com", "b.com"]),
        exclude_domains=iter(["c.com"]),
        language=(code for code in ["en"]),
    )
    _sdk_method(octen_cls, function).assert_called_once_with(
        query="q", include_domains=["a.com", "b.com"], exclude_domains=["c.com"],
        language=["en"],
    )


def test_premier_domain_generators_reach_the_sdk_as_lists(octen_cls: MagicMock) -> None:
    _spec().search_premier(
        "q", include_domains=(d for d in ["a.com"]), exclude_domains=iter(["b.com"])
    )
    _sdk_method(octen_cls, "search_premier").assert_called_once_with(
        query="q", include_domains=["a.com"], exclude_domains=["b.com"]
    )


@pytest.mark.parametrize("function", NEW_FUNCTIONS)
def test_unset_options_not_sent(octen_cls: MagicMock, function: str) -> None:
    getattr(_spec(), function)("q")
    _sdk_method(octen_cls, function).assert_called_once_with(query="q")


def test_timeout_spec_level_and_per_endpoint_override(octen_cls: MagicMock) -> None:
    spec = _spec(timeout=10, broad_search_options={"timeout": 90})
    spec.news_search("q")
    spec.broad_search("q")
    assert _sdk_method(octen_cls, "news_search").call_args.kwargs["timeout"] == 10
    assert _sdk_method(octen_cls, "broad_search").call_args.kwargs["timeout"] == 90


def test_returns_data_request_id_warning(octen_cls: MagicMock) -> None:
    data = {"query": "q", "results": [{"title": "t", "url": "u"}], "subjects": [{"name": "s"}]}
    _sdk_method(octen_cls, "news_search").return_value = _response(
        data, meta={"warning": "something to know"}, request_id="rid-news"
    )
    assert _spec().news_search("q") == {
        "data": data,
        "request_id": "rid-news",
        "warning": "something to know",
    }


def test_place_unavailable_warning_kept(octen_cls: MagicMock) -> None:
    data = {"places": [], "results": []}
    _sdk_method(octen_cls, "place_search").return_value = octen.PlaceSearchResponse(
        code=0, msg="success", data=data, meta={"warning": "places_unavailable"},
        request_id="rid-place",
    )
    result = _spec().place_search("coffee in Shinjuku")
    assert result == {"data": data, "request_id": "rid-place", "warning": "places_unavailable"}


@pytest.mark.parametrize("meta", [{"warning": ""}, {"warning": None}, {}])
def test_empty_warning_omitted(octen_cls: MagicMock, meta: Dict[str, Any]) -> None:
    _sdk_method(octen_cls, "pulse_search").return_value = _response({"results": []}, meta=meta)
    assert _spec().pulse_search("github docs") == {"data": {"results": []}, "request_id": "rid-1"}


def test_broad_response_without_meta(octen_cls: MagicMock) -> None:
    # /broad-search may omit meta entirely (BroadSearchResponse.meta is optional).
    data = {"query": "q", "queries": ["a"], "search_results": []}
    _sdk_method(octen_cls, "broad_search").return_value = octen.BroadSearchResponse(
        data=data, request_id="rid-broad"
    )
    assert _spec().broad_search("q") == {"data": data, "request_id": "rid-broad"}


@pytest.mark.parametrize("function", NEW_FUNCTIONS)
@pytest.mark.parametrize(
    "exc",
    [
        OctenRateLimitError(),
        OctenAPIError("upstream failed", status_code=503),
        OctenAPIError("bad filter", status_code=400),
        OctenAPIError("unprocessable", status_code=422),
        OctenTimeoutError(timeout=30, request_sent=False),
        OctenConnectionError(request_sent=False),
    ],
)
def test_transient_errors_returned_as_error_dict(
    octen_cls: MagicMock, function: str, exc: Exception
) -> None:
    _sdk_method(octen_cls, function).side_effect = exc
    assert getattr(_spec(), function)("q") == {"error": str(exc)}
    octen_cls.return_value.close.assert_called_once()


@pytest.mark.parametrize("function", NEW_FUNCTIONS)
@pytest.mark.parametrize(
    "exc",
    [
        OctenAuthenticationError(),
        OctenAPIError("forbidden", status_code=403),
        OctenAPIError("not found", status_code=404),
        OctenValidationError("count out of range"),
    ],
)
def test_config_errors_raise(octen_cls: MagicMock, function: str, exc: Exception) -> None:
    _sdk_method(octen_cls, function).side_effect = exc
    with pytest.raises(type(exc)):
        getattr(_spec(), function)("q")
    octen_cls.return_value.close.assert_called_once()


def test_developer_option_out_of_range_raises_at_construction(octen_cls: MagicMock) -> None:
    with pytest.raises(ValueError, match="news_search_options.*count"):
        _spec(news_search_options={"count": 1000})
    octen_cls.assert_not_called()


@pytest.mark.parametrize("function", NEW_FUNCTIONS)
def test_raised_errors_are_logged(
    octen_cls: MagicMock, caplog: Any, function: str
) -> None:
    # Agent workflows turn any tool exception into text for the model, so the
    # log is where a developer sees a configuration error.
    _sdk_method(octen_cls, function).side_effect = OctenAPIError("forbidden", status_code=403)
    with caplog.at_level(logging.ERROR, logger="llama_index.tools.octen"):
        with pytest.raises(OctenAPIError):
            getattr(_spec(), function)("q")
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1
    assert errors[0].name == "llama_index.tools.octen"
    assert "OctenAPIError" in errors[0].getMessage()
    assert "forbidden | Status: 403" in errors[0].getMessage()


def test_errors_returned_to_model_are_not_logged_as_errors(
    octen_cls: MagicMock, caplog: Any
) -> None:
    _sdk_method(octen_cls, "news_search").side_effect = OctenRateLimitError()
    with caplog.at_level(logging.DEBUG, logger="llama_index.tools.octen"):
        assert "error" in _spec().news_search("q")
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


def test_time_range_schema_is_an_enum() -> None:
    tools = {t.metadata.name: t for t in _spec().to_tool_list()}
    for name in FILTERED:
        schema = tools[name].metadata.fn_schema
        assert schema is not None
        prop = schema.model_json_schema()["properties"]["time_range"]
        enums = [branch["enum"] for branch in prop["anyOf"] if "enum" in branch]
        assert enums == [["day", "week", "month", "year", "d", "w", "m", "y"]], name


def test_place_empty_location_fields_dropped(octen_cls: MagicMock) -> None:
    _spec().place_search("ramen", city="Tokyo", region="", country="", timezone="")
    _sdk_method(octen_cls, "place_search").assert_called_once_with(
        query="ramen", user_location={"type": "approximate", "city": "Tokyo"}
    )


def test_place_all_empty_location_fields_send_no_user_location(octen_cls: MagicMock) -> None:
    _spec().place_search("ramen in Shinjuku", city="", region="", country="", timezone="")
    _sdk_method(octen_cls, "place_search").assert_called_once_with(query="ramen in Shinjuku")


def _signature_params(method: Any) -> set:
    return set(inspect.signature(method).parameters) - {"self"}


def test_names_match_real_sdk_signatures() -> None:
    # No mocks: every developer option and model arg must be a parameter of
    # the SDK method it is forwarded to, or the call fails with TypeError.
    from octen.resources.place_search import PlaceSearchResource
    from octen.resources.search import SearchResource

    from llama_index.tools.octen.base import _ALLOWED_OPTIONS

    sdk = {
        "broad_search": SearchResource.broad_search,
        "news_search": SearchResource.news_search,
        "business_search": SearchResource.business_search,
        "pulse_search": SearchResource.pulse_search,
        "place_search": PlaceSearchResource.search,
        "search_premier": SearchResource.search_premier,
    }
    # place_search assembles user_location from its flat location args.
    renamed = {"city": "user_location", "region": "user_location",
               "country": "user_location", "timezone": "user_location"}
    assert set(sdk) == set(NEW_FUNCTIONS)
    for function, method in sdk.items():
        accepted = _signature_params(method)
        assert "bogus_param" not in accepted
        options = _ALLOWED_OPTIONS[f"{function}_options"] - {"timeout"}
        model_args = {renamed.get(a, a) for a in _signature_params(getattr(OctenToolSpec, function))}
        assert options <= accepted, (function, options - accepted)
        assert model_args <= accepted, (function, model_args - accepted)
        assert "timeout" in accepted, function


@pytest.mark.parametrize("function", FILTERED)
@pytest.mark.parametrize("time_range", ["last week", "7d", "WEEK"])
def test_bad_time_range_returns_error_dict_without_call(
    octen_cls: MagicMock, function: str, time_range: str
) -> None:
    result = getattr(_spec(), function)("q", time_range=time_range)
    assert set(result) == {"error"}
    assert "time_range" in result["error"]
    octen_cls.assert_not_called()


@pytest.mark.parametrize("function", FILTERED)
def test_valid_time_ranges_accepted(octen_cls: MagicMock, function: str) -> None:
    for value in ["day", "week", "month", "year", "d", "w", "m", "y"]:
        assert "error" not in getattr(_spec(), function)("q", time_range=value)


@pytest.mark.parametrize("function", NEW_FUNCTIONS)
def test_query_too_long_returns_error_dict_without_call(
    octen_cls: MagicMock, function: str
) -> None:
    result = getattr(_spec(), function)("x" * 501)
    assert set(result) == {"error"}
    assert "query" in result["error"]
    octen_cls.assert_not_called()


@pytest.mark.parametrize("function", FILTERED)
def test_wrong_filter_type_returns_error_dict_without_call(
    octen_cls: MagicMock, function: str
) -> None:
    result = getattr(_spec(), function)("q", include_domains="arxiv.org")
    assert set(result) == {"error"}
    assert "include_domains" in result["error"]
    octen_cls.assert_not_called()


@pytest.mark.parametrize(
    ("args", "mentions"),
    [
        ({"latitude": 35.6}, "longitude"),
        ({"longitude": 139.7}, "latitude"),
    ],
)
def test_place_half_coordinate_returns_error_dict_without_call(
    octen_cls: MagicMock, args: Dict[str, Any], mentions: str
) -> None:
    result = _spec().place_search("coffee", **args)
    assert set(result) == {"error"}
    assert mentions in result["error"]
    octen_cls.assert_not_called()


def test_place_radius_without_point_returns_error_dict_without_call(
    octen_cls: MagicMock,
) -> None:
    result = _spec().place_search("coffee", radius=500)
    assert set(result) == {"error"}
    assert "radius" in result["error"]
    octen_cls.assert_not_called()


@pytest.mark.parametrize(
    ("args", "field"),
    [
        ({"latitude": 91, "longitude": 0}, "latitude"),
        ({"latitude": -91, "longitude": 0}, "latitude"),
        ({"latitude": 0, "longitude": 181}, "longitude"),
        ({"latitude": 0, "longitude": -181}, "longitude"),
        ({"latitude": 0, "longitude": 0, "radius": 0}, "radius"),
        ({"latitude": 0, "longitude": 0, "radius": 50001}, "radius"),
    ],
)
def test_place_out_of_range_returns_error_dict_without_call(
    octen_cls: MagicMock, args: Dict[str, Any], field: str
) -> None:
    result = _spec().place_search("coffee", **args)
    assert set(result) == {"error"}
    assert field in result["error"]
    octen_cls.assert_not_called()


def test_place_bounds_accepted(octen_cls: MagicMock) -> None:
    for args in (
        {"latitude": 90, "longitude": 180, "radius": 50000},
        {"latitude": -90, "longitude": -180, "radius": 1},
    ):
        assert "error" not in _spec().place_search("coffee", **args)


def test_place_location_fields_build_user_location(octen_cls: MagicMock) -> None:
    _spec().place_search("ramen", city="Tokyo", country="JP", timezone="Asia/Tokyo")
    _sdk_method(octen_cls, "place_search").assert_called_once_with(
        query="ramen",
        user_location={
            "type": "approximate",
            "city": "Tokyo",
            "country": "JP",
            "timezone": "Asia/Tokyo",
        },
    )


def test_place_without_location_fields_sends_no_user_location(octen_cls: MagicMock) -> None:
    _spec().place_search("ramen in Shinjuku")
    assert "user_location" not in _sdk_method(octen_cls, "place_search").call_args.kwargs


def test_client_closed_on_error(octen_cls: MagicMock) -> None:
    _sdk_method(octen_cls, "business_search").side_effect = OctenAuthenticationError()
    with pytest.raises(OctenAuthenticationError):
        _spec().business_search("q")
    _sdk_method(octen_cls, "business_search").side_effect = OctenTimeoutError()
    assert "error" in _spec().business_search("q")
    assert octen_cls.return_value.close.call_count == 2


def test_tool_call_through_llamaindex(octen_cls: MagicMock) -> None:
    # The model path: FunctionTool.call with model-supplied kwargs.
    tools = {t.metadata.name: t for t in _spec().to_tool_list()}
    output = tools["news_search"].call(query="q", time_range="last week")
    assert "error" in output.raw_output
    octen_cls.assert_not_called()
    output = tools["news_search"].call(query="q", time_range="week")
    assert output.raw_output == {"data": {"results": []}, "request_id": "rid-1"}


def test_options_deep_copied_at_construction(octen_cls: MagicMock) -> None:
    subjects = {"count": 1}
    highlight = octen.HighlightOptions(enable=True, max_tokens=200)
    options = {"count": 2, "subjects": subjects, "highlight": highlight}
    spec = _spec(news_search_options=options)
    # Caller mutations after construction must not change what is sent,
    # including into values that would have failed validation.
    subjects["count"] = 99
    highlight.max_tokens = 300
    options["count"] = 1000
    spec.news_search("q")
    kwargs = _sdk_method(octen_cls, "news_search").call_args.kwargs
    assert kwargs["count"] == 2
    assert kwargs["subjects"] == {"count": 1}
    assert kwargs["highlight"].max_tokens == 200


@pytest.mark.parametrize(
    ("args", "field"),
    [({"city": False}, "city"), ({"country": 0}, "country"), ({"timezone": []}, "timezone")],
)
def test_place_non_string_location_returns_error_dict_without_call(
    octen_cls: MagicMock, args: Dict[str, Any], field: str
) -> None:
    result = _spec().place_search("ramen", **args)
    assert set(result) == {"error"}
    assert field in result["error"]
    octen_cls.assert_not_called()


def test_timeout_after_send_tells_model_not_to_retry(octen_cls: MagicMock) -> None:
    _sdk_method(octen_cls, "pulse_search").side_effect = OctenTimeoutError(
        timeout=30, request_sent=True
    )
    error = _spec().pulse_search("q")["error"]
    assert "may already have reached the server and been billed" in error


def test_failed_2xx_tells_model_not_to_retry(octen_cls: MagicMock) -> None:
    _sdk_method(octen_cls, "news_search").side_effect = OctenAPIError(
        "malformed successful response", status_code=200
    )
    error = _spec().news_search("q")["error"]
    assert error.startswith("malformed successful response")
    assert "may already have reached the server and been billed" in error


def test_model_visible_description_keeps_instructions() -> None:
    # The model sees ToolMetadata.description; it must carry the docstring's
    # instructions, not only the summary line.
    tools = {t.metadata.name: t.metadata.description for t in _spec().to_tool_list()}
    expected = {
        "news_search": [
            "narrow the ranked articles only",
            "answer from the articles, not the storylines",
            "use business_search",
        ],
        "business_search": ["narrow the ranked sources only, never the cards",
                            "use news_search"],
        "broad_search": ["do not pre-split it", "Do not call it twice",
                         "when in doubt, prefer search"],
        "pulse_search": ['prepend "https://"', "sees no fresh content"],
        "place_search": ["always together", "only with both coordinates",
                         "Pass the caller's location"],
        "search_premier": ["deeper retrieval and reranking", "use search",
                           "include_domains"],
    }
    for name, phrases in expected.items():
        for phrase in phrases:
            assert phrase in tools[name], (name, phrase)


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf"), True, "30"])
def test_shared_timeout_rejected_at_construction(bad: Any) -> None:
    with pytest.raises(ValueError, match="timeout"):
        OctenToolSpec(api_key="k", timeout=bad)


def test_shared_timeout_none_and_positive_accepted() -> None:
    OctenToolSpec(api_key="k", timeout=None)
    OctenToolSpec(api_key="k", timeout=2.5)


def test_huge_int_shared_timeout_rejected() -> None:
    # math.isfinite raises OverflowError on ints too large for a float.
    with pytest.raises(ValueError, match="OctenToolSpec: timeout"):
        OctenToolSpec(api_key="k", timeout=10**400)


def test_huge_int_option_timeout_rejected() -> None:
    with pytest.raises(ValueError, match="news_search_options: timeout"):
        OctenToolSpec(api_key="k", news_search_options={"timeout": 10**400})



@pytest.mark.parametrize(
    ("options", "mentions"),
    [
        ({"count": 0}, "count"),
        ({"count": 101}, "count"),
        ({"result_type": "smart"}, "result_type"),
        ({"highlight": {"max_tokens": 99}}, "highlight"),
        ({"full_content": {"max_tokens": 100001}}, "full_content"),
        ({"full_content": {"enabled": True}}, "full_content"),
        ({"full_content": {1: True, "enabled": True}}, "full_content"),
        ({"highlight": {None: 1}}, "highlight"),
        (
            {"highlight": octen.HighlightOptions(enable=True, maxtokens=300)},
            "highlight",
        ),
        ({"include_text": ["a"] * 6}, "include_text"),
        ({"exclude_text": ["x" * 31]}, "exclude_text"),
        ({"language": ["english"]}, "language"),
        ({"language": "en"}, "language"),
        ({"language": [["en"]]}, "language"),
        ({"time_basis": "indexed"}, "time_basis"),
        ({"time_range": "last week"}, "time_range"),
        ({"start_time": "last week"}, "start_time"),
        ({"end_time": "2025-13-01T00:00:00Z"}, "end_time"),
        ({"start_time": 20250101}, "start_time"),
        ({"format": "html"}, "format"),
        ({"safesearch": "moderate"}, "safesearch"),
        ({"timeout": 0}, "timeout"),
    ],
    ids=lambda v: v if isinstance(v, str) else None,
)
def test_premier_invalid_option_raises_at_construction(
    octen_cls: MagicMock, options: Dict[str, Any], mentions: str
) -> None:
    with pytest.raises(ValueError, match=f"search_premier_options.*{mentions}"):
        _spec(search_premier_options=options)
    octen_cls.assert_not_called()


@pytest.mark.parametrize(
    "key",
    ["query", "include_domains", "exclude_domains", "topic", "num_results", "mode"],
)
def test_premier_model_or_unknown_key_rejected_as_option(key: str) -> None:
    with pytest.raises(
        ValueError, match=f"search_premier_options does not accept '{key}'"
    ):
        _spec(search_premier_options={key: "x"})


# RFC 3339 date-time, the "format: date-time" of
# https://docs.octen.ai/api-reference/search-premier.md
RFC3339_VALID = [
    "2025-01-01T00:00:00Z",
    "2025-01-01T00:00:00.1Z",
    "2025-01-01T00:00:00.123456Z",
    "2025-06-30T23:59:59+08:00",
    "2025-01-01T00:00:00-05:30",
    "2024-02-29T12:00:00Z",
]
RFC3339_INVALID = [
    "2025-01-01",
    "2025-01-01T00:00:00",
    "2025-01-01X00:00:00Z",
    "2025-01-01 00:00:00Z",
    "2025-13-01T00:00:00Z",
    "2025-02-29T00:00:00Z",
    "2025-01-01T24:00:00Z",
    "2025-01-01T00:00:00+24:00",
    "2025-01-01T00:00:00.Z",
    "2025-01-01T00:00:00Z\n",
    "\uff12025-01-01T00:00:00Z",
    "20250101T000000Z",
    "last week",
    "2025-01-01t00:00:00Z",
    "2025-01-01T00:00:00z",
    "2016-12-31T23:59:60Z",
]


@pytest.mark.parametrize("value", RFC3339_VALID)
def test_premier_iso_times_accepted(octen_cls: MagicMock, value: str) -> None:
    _spec(
        search_premier_options={"start_time": value, "end_time": value}
    ).search_premier("q")
    kwargs = _sdk_method(octen_cls, "search_premier").call_args.kwargs
    assert (kwargs["start_time"], kwargs["end_time"]) == (value, value)


@pytest.mark.parametrize("field", ["start_time", "end_time"])
@pytest.mark.parametrize("value", RFC3339_INVALID)
def test_premier_non_rfc3339_time_rejected(value: str, field: str) -> None:
    with pytest.raises(ValueError, match=f"search_premier_options.*{field}"):
        _spec(search_premier_options={field: value})


def test_premier_sdk_option_objects_accepted(octen_cls: MagicMock) -> None:
    highlight = octen.HighlightOptions(enable=True, max_tokens=300)
    full_content = octen.FullContentOptions(enable=True, max_tokens=500)
    spec = _spec(
        search_premier_options={"highlight": highlight, "full_content": full_content}
    )
    spec.search_premier("q")
    kwargs = _sdk_method(octen_cls, "search_premier").call_args.kwargs
    assert kwargs["highlight"].max_tokens == 300
    assert kwargs["full_content"].max_tokens == 500


@pytest.mark.parametrize(
    "extra",
    [
        {"count": 50},
        {"result_type": "dynamic"},
        {"full_content": {"enable": True}},
        {"time_range": "day"},
        {"language": ["en"]},
        {"safesearch": "off"},
        {"timeout": 1},
    ],
    ids=lambda e: next(iter(e)),
)
def test_premier_model_cannot_pass_developer_options(
    octen_cls: MagicMock, extra: Dict[str, Any]
) -> None:
    tools = {t.metadata.name: t for t in _spec().to_tool_list()}
    with pytest.raises(TypeError, match=next(iter(extra))):
        tools["search_premier"].call(query="q", **extra)
    octen_cls.assert_not_called()


@pytest.mark.parametrize(
    ("args", "field"),
    [
        ({"include_domains": ["a" * 61 + ".com"]}, "include_domains"),
        ({"exclude_domains": "example.com"}, "exclude_domains"),
    ],
)
def test_premier_bad_domain_returns_error_dict_without_call(
    octen_cls: MagicMock, args: Dict[str, Any], field: str
) -> None:
    result = _spec().search_premier("q", **args)
    assert set(result) == {"error"}
    assert field in result["error"]
    octen_cls.assert_not_called()


def test_premier_returns_data_request_id(octen_cls: MagicMock) -> None:
    data = {"query": "q", "results": [{"title": "t", "url": "u", "highlight": "h"}]}
    _sdk_method(octen_cls, "search_premier").return_value = _response(
        data, meta={"result_type": "dynamic", "warning": ""}, request_id="rid-premier"
    )
    assert _spec().search_premier("q") == {"data": data, "request_id": "rid-premier"}
    octen_cls.return_value.search.search.assert_not_called()


def test_premier_timeout_spec_level_and_override(octen_cls: MagicMock) -> None:
    _spec(timeout=10).search_premier("q")
    _spec(timeout=10, search_premier_options={"timeout": 60}).search_premier("q")
    calls = _sdk_method(octen_cls, "search_premier").call_args_list
    assert [c.kwargs["timeout"] for c in calls] == [10, 60]


def test_premier_options_deep_copied(octen_cls: MagicMock) -> None:
    include_text = ["kafka"]
    highlight = {"enable": True, "max_tokens": 300}
    spec = _spec(
        search_premier_options={"include_text": include_text, "highlight": highlight}
    )
    include_text.append("later")
    highlight["max_tokens"] = 5
    spec.search_premier("q")
    kwargs = _sdk_method(octen_cls, "search_premier").call_args.kwargs
    assert kwargs["include_text"] == ["kafka"]
    assert kwargs["highlight"] == {"enable": True, "max_tokens": 300}
