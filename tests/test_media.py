"""Unit tests for the image_search and design_search functions.

Limits and field names follow https://docs.octen.ai/api-reference/image-search.md,
https://docs.octen.ai/api-reference/design-search.md and octen 0.13.0
(octen/resources/{image,design}_search.py, octen/models/{image,design}_search.py).
"""

import inspect
import json
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

FUNCTIONS = ["image_search", "design_search"]
# A minimal valid model call for each function.
MODEL_CALL: Dict[str, Dict[str, Any]] = {
    "image_search": {"query": "q"},
    "design_search": {"query": "q"},
}
GROUP_ID = "deck_3b7f0e92c4a1d85f6e2b9c07a3d4f1e8b5c26a9d0f7e3b1c4a8d2e6f9b0c5a7d"
IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/a/a9/Example.jpg"


def _sdk(octen_cls: MagicMock, function: str) -> MagicMock:
    return getattr(octen_cls.return_value, function).search


def _image_response(results: List[Any], meta: Any = None, request_id: str = "rid-img") -> Any:
    return octen.ImageSearchResponse(
        code=0, msg="success", data={"results": results}, meta=meta, request_id=request_id
    )


def _design_response(data: Dict[str, Any], meta: Any = None, request_id: str = "rid-des") -> Any:
    return octen.DesignSearchResponse(
        code=0, msg="success", data=data, meta=meta or {}, request_id=request_id
    )


@pytest.fixture
def octen_cls(mocker: Any) -> MagicMock:
    cls = mocker.patch("octen.Octen")
    _sdk(cls, "image_search").return_value = _image_response([])
    _sdk(cls, "design_search").return_value = _design_response({"images": [], "results": []})
    return cls


def _spec(**kwargs: Any) -> OctenToolSpec:
    return OctenToolSpec(api_key="k", verbose=False, **kwargs)


def _tools(**kwargs: Any) -> Dict[str, Any]:
    return {t.metadata.name: t for t in _spec(**kwargs).to_tool_list()}


# --- tool metadata -----------------------------------------------------------


def test_tool_count_and_names() -> None:
    names = [t.metadata.name for t in _spec().to_tool_list()]
    assert len(names) == 13
    assert names[-3:] == ["image_search", "design_search", "current_date"]


def test_schemas() -> None:
    tools = _tools()
    expected = {
        "image_search": ["query", "image_url", "include_domains", "exclude_domains"],
        "design_search": ["query", "group_id"],
    }
    for name, fields in expected.items():
        schema = tools[name].metadata.fn_schema
        assert schema is not None
        assert list(schema.model_fields) == fields, name
        json_schema = schema.model_json_schema()
        # Neither field is required alone: the rules are checked in the function.
        assert "required" not in json_schema, name
        for field, prop in json_schema["properties"].items():
            assert prop.get("description"), f"{name}.{field} has no description"
        params = inspect.signature(getattr(OctenToolSpec, name)).parameters
        assert list(schema.model_fields) == [p for p in params if p != "self"], name
        for field, info in schema.model_fields.items():
            assert info.annotation == params[field].annotation, (name, field)
            assert info.default == params[field].default, (name, field)


def test_schema_carries_api_limits() -> None:
    tools = _tools()
    props = tools["image_search"].metadata.fn_schema.model_json_schema()["properties"]
    assert props["query"]["anyOf"][0]["maxLength"] == 500
    assert props["include_domains"]["anyOf"][0]["maxItems"] == 10
    assert props["exclude_domains"]["anyOf"][0]["maxItems"] == 10
    props = tools["design_search"].metadata.fn_schema.model_json_schema()["properties"]
    assert props["query"]["anyOf"][0]["maxLength"] == 500


def test_descriptions_pass_openai_length_check_and_keep_rules() -> None:
    tools = _tools()
    for name in FUNCTIONS:
        tools[name].metadata.to_openai_tool()  # default 1024-char check
    image = tools["image_search"].metadata.description
    for phrase in ["exactly one of query", "never both", "use design_search",
                   "cite its source_page", "at most 10 each"]:
        assert phrase in image, phrase
    design = tools["design_search"].metadata.description
    for phrase in ["call again with that group_id", "query may then be left out",
                   "use image_search", "cite its source_page"]:
        assert phrase in design, phrase


# --- outgoing request ---------------------------------------------------------


def test_image_text_query_with_options(octen_cls: MagicMock) -> None:
    spec = _spec(image_search_options={"count": 3, "safesearch": "off", "timeout": 20})
    spec.image_search(
        "red bicycle", include_domains=["unsplash.com"], exclude_domains=["pinterest.com"]
    )
    _sdk(octen_cls, "image_search").assert_called_once_with(
        query="red bicycle",
        include_domains=["unsplash.com"],
        exclude_domains=["pinterest.com"],
        count=3,
        safesearch="off",
        timeout=20,
    )
    octen_cls.assert_called_once_with(api_key="k", retry_after_send=False)


def test_image_url_input(octen_cls: MagicMock) -> None:
    _spec().image_search(image_url=IMAGE_URL)
    _sdk(octen_cls, "image_search").assert_called_once_with(image_url=IMAGE_URL)


def test_image_unset_options_not_sent(octen_cls: MagicMock) -> None:
    _spec().image_search("q")
    _sdk(octen_cls, "image_search").assert_called_once_with(query="q")


def test_design_query_with_options(octen_cls: MagicMock) -> None:
    images = {"count": 8, "design_facts": False, "snippet": {"enable": False}}
    web_search = {"enable": True, "count": 2, "language": ["en"]}
    spec = _spec(design_search_options={"images": images, "web_search": web_search})
    spec.design_search("pricing section")
    _sdk(octen_cls, "design_search").assert_called_once_with(
        query="pricing section", images=images, web_search=web_search
    )


def test_design_unset_options_not_sent(octen_cls: MagicMock) -> None:
    _spec().design_search("q")
    _sdk(octen_cls, "design_search").assert_called_once_with(query="q")


def test_design_group_id_alone(octen_cls: MagicMock) -> None:
    _spec().design_search(group_id=GROUP_ID)
    _sdk(octen_cls, "design_search").assert_called_once_with(images={"group_id": GROUP_ID})


def test_design_group_id_merged_into_developer_images(octen_cls: MagicMock) -> None:
    images = {"count": 30, "design_facts": False}
    spec = _spec(design_search_options={"images": images})
    spec.design_search("deck", group_id=GROUP_ID)
    _sdk(octen_cls, "design_search").assert_called_once_with(
        query="deck", images={"count": 30, "design_facts": False, "group_id": GROUP_ID}
    )
    # The stored options are not changed by the merge.
    spec.design_search("deck")
    assert _sdk(octen_cls, "design_search").call_args.kwargs["images"] == images


def test_design_group_id_with_sdk_images_model(octen_cls: MagicMock) -> None:
    images = octen.DesignImageOptions(count=4, snippet=octen.DesignSnippetOptions(enable=False))
    _spec(design_search_options={"images": images}).design_search(group_id=GROUP_ID)
    _sdk(octen_cls, "design_search").assert_called_once_with(
        images={"count": 4, "snippet": {"enable": False}, "group_id": GROUP_ID}
    )


def test_blank_inputs_are_not_sent(octen_cls: MagicMock) -> None:
    _spec().image_search(query="  ", image_url=IMAGE_URL)
    _sdk(octen_cls, "image_search").assert_called_once_with(image_url=IMAGE_URL)
    _spec().design_search(query="", group_id=GROUP_ID)
    _sdk(octen_cls, "design_search").assert_called_once_with(images={"group_id": GROUP_ID})


def test_timeout_spec_level_and_override(octen_cls: MagicMock) -> None:
    _spec(timeout=10).image_search("q")
    _spec(timeout=10, image_search_options={"timeout": 60}).image_search("q")
    _spec(timeout=10).design_search("q")
    _spec(timeout=10, design_search_options={"timeout": 60}).design_search("q")
    image = [c.kwargs["timeout"] for c in _sdk(octen_cls, "image_search").call_args_list]
    design = [c.kwargs["timeout"] for c in _sdk(octen_cls, "design_search").call_args_list]
    assert image == design == [10, 60]


CANNED = {"code": 0, "msg": "success", "request_id": "rid", "data": {"results": []}, "meta": {}}


def _wire(spec: OctenToolSpec, function: str, **args: Any) -> List[Any]:
    seen: List[Any] = []

    def fake(self: Any, method: str, endpoint: str, timeout: Any = None,
             json: Any = None, **kwargs: Any) -> Dict[str, Any]:
        seen.append((method, endpoint, json, timeout))
        return dict(CANNED)

    with patch("octen.core.http_client.HTTPClient.request", fake):
        result = getattr(spec, function)(**args)
    assert "error" not in result, result
    return seen


def test_image_wire_bodies() -> None:
    spec = _spec(image_search_options={"count": 2, "safesearch": "off"})
    by_text = _wire(spec, "image_search", query="q", include_domains=["a.com"])
    by_url = _wire(spec, "image_search", image_url=IMAGE_URL, exclude_domains=["b.com"])
    assert json.dumps(by_text) == json.dumps([(
        "POST", "/image-search",
        {"inputs": [{"type": "text", "data": "q"}], "count": 2,
         "include_domains": ["a.com"], "safesearch": "off"},
        None,
    )])
    assert json.dumps(by_url) == json.dumps([(
        "POST", "/image-search",
        {"inputs": [{"type": "image", "url": IMAGE_URL}], "count": 2,
         "exclude_domains": ["b.com"], "safesearch": "off"},
        None,
    )])


def test_unset_options_absent_from_wire() -> None:
    assert _wire(_spec(), "image_search", query="q")[0][2] == {
        "inputs": [{"type": "text", "data": "q"}]
    }
    assert _wire(_spec(), "design_search", group_id=GROUP_ID)[0][2] == {
        "images": {"group_id": GROUP_ID}
    }


def test_design_wire_body() -> None:
    options = {
        "images": {"count": 30, "design_facts": False, "snippet": {"enable": True, "max_tokens": 500}},
        "web_search": {"enable": False},
        "timeout": 15,
    }
    seen = _wire(_spec(design_search_options=options), "design_search",
                 query="deck", group_id=GROUP_ID)
    assert json.dumps(seen) == json.dumps([(
        "POST", "/design-search",
        {"query": "deck",
         "images": {"count": 30, "design_facts": False,
                    "snippet": {"enable": True, "max_tokens": 500}, "group_id": GROUP_ID},
         "web_search": {"enable": False}},
        15,
    )])


# --- model-facing rules ---------------------------------------------------------


@pytest.mark.parametrize(
    ("args", "mentions"),
    [
        ({"query": "cats", "image_url": IMAGE_URL}, "not both"),
        ({}, "Got neither"),
        ({"query": " ", "image_url": ""}, "Got neither"),
        ({"image_url": "ftp://example.com/a.jpg"}, "http(s)"),
        ({"image_url": "data:image/png;base64,AAAA"}, "http(s)"),
        ({"image_url": "example.com/a.jpg"}, "http(s)"),
        ({"query": "x" * 501}, "query"),
        ({"query": "q", "include_domains": ["a.com"] * 11}, "include_domains"),
        ({"query": "q", "exclude_domains": ["a.com"] * 11}, "exclude_domains"),
        ({"query": "q", "include_domains": "a.com"}, "include_domains"),
        ({"query": 5}, "query"),
        ({"image_url": ["https://example.com/a.jpg"]}, "image_url"),
    ],
)
def test_image_bad_model_args_return_error_without_request(
    octen_cls: MagicMock, args: Dict[str, Any], mentions: str
) -> None:
    result = _spec().image_search(**args)
    assert set(result) == {"error"}
    assert mentions in result["error"]
    octen_cls.assert_not_called()


def test_image_limits_accepted(octen_cls: MagicMock) -> None:
    spec = _spec()
    assert "error" not in spec.image_search("x" * 500, include_domains=["a.com"] * 10)
    assert "error" not in spec.image_search(image_url="HTTPS://example.com/a.png")
    assert "error" not in spec.image_search(image_url="http://example.com/a.png",
                                            exclude_domains=["b.com"] * 10)


@pytest.mark.parametrize(
    ("args", "mentions"),
    [
        ({}, "Got neither"),
        ({"query": "   "}, "Got neither"),
        ({"query": "", "group_id": " "}, "Got neither"),
        ({"query": "x" * 501}, "query"),
        ({"group_id": 7}, "group_id"),
        ({"query": ["a"]}, "query"),
    ],
)
def test_design_bad_model_args_return_error_without_request(
    octen_cls: MagicMock, args: Dict[str, Any], mentions: str
) -> None:
    result = _spec().design_search(**args)
    assert set(result) == {"error"}
    assert mentions in result["error"]
    octen_cls.assert_not_called()


def test_design_query_limit_accepted(octen_cls: MagicMock) -> None:
    assert "error" not in _spec().design_search("x" * 500)


@pytest.mark.parametrize(
    ("function", "extra"),
    [
        ("image_search", {"count": 10}),
        ("image_search", {"safesearch": "off"}),
        ("image_search", {"image_data": "AAAA"}),
        ("image_search", {"topic": "design"}),
        ("image_search", {"timeout": 1}),
        ("design_search", {"images": {"count": 100}}),
        ("design_search", {"web_search": {"full_content": {"enable": True}}}),
        ("design_search", {"count": 100}),
        ("design_search", {"design_facts": True}),
        ("design_search", {"timeout": 1}),
    ],
    ids=lambda v: v if isinstance(v, str) else next(iter(v)),
)
def test_model_cannot_pass_developer_options(
    octen_cls: MagicMock, function: str, extra: Dict[str, Any]
) -> None:
    tools = _tools()
    with pytest.raises(TypeError, match=next(iter(extra))):
        tools[function].call(**MODEL_CALL[function], **extra)
    octen_cls.assert_not_called()


def test_tool_call_through_llamaindex(octen_cls: MagicMock) -> None:
    tools = _tools()
    output = tools["image_search"].call(query="cats", image_url=IMAGE_URL)
    assert "not both" in output.raw_output["error"]
    output = tools["design_search"].call()
    assert "Got neither" in output.raw_output["error"]
    octen_cls.assert_not_called()
    assert tools["image_search"].call(image_url=IMAGE_URL).raw_output == {
        "data": {"results": []}, "request_id": "rid-img"
    }
    assert tools["design_search"].call(group_id=GROUP_ID).raw_output == {
        "data": {"images": [], "results": []}, "request_id": "rid-des"
    }


# --- construction validation ----------------------------------------------------


@pytest.mark.parametrize(
    ("options", "mentions"),
    [
        ({"count": 0}, "count"),
        ({"count": 11}, "count"),
        ({"count": True}, "count"),
        ({"count": 5.0}, "count"),
        ({"count": "5"}, "count"),
        ({"safesearch": "moderate"}, "safesearch"),
        ({"safesearch": "OFF"}, "safesearch"),
        ({"timeout": 0}, "timeout"),
        ({"timeout": float("inf")}, "timeout"),
    ],
    ids=repr,
)
def test_image_invalid_option_raises(
    octen_cls: MagicMock, options: Dict[str, Any], mentions: str
) -> None:
    with pytest.raises(ValueError, match=f"image_search_options.*{mentions}"):
        _spec(image_search_options=options)
    octen_cls.assert_not_called()


@pytest.mark.parametrize(
    "options",
    [{"count": 1}, {"count": 10}, {"safesearch": "off"}, {"safesearch": "strict"},
     {"timeout": 0.5}],
    ids=repr,
)
def test_image_valid_option_accepted(octen_cls: MagicMock, options: Dict[str, Any]) -> None:
    _spec(image_search_options=options).image_search("q")
    kwargs = _sdk(octen_cls, "image_search").call_args.kwargs
    assert {k: kwargs[k] for k in options} == options


@pytest.mark.parametrize(
    "key",
    ["query", "image_url", "image_data", "inputs", "include_domains", "exclude_domains",
     "topic", "html_snippet", "num_results"],
)
def test_image_model_or_unknown_key_rejected(key: str) -> None:
    with pytest.raises(ValueError, match=f"image_search_options does not accept '{key}'"):
        _spec(image_search_options={key: "x"})


@pytest.mark.parametrize(
    ("options", "mentions"),
    [
        ({"images": {"count": 0}}, "images.count"),
        ({"images": {"count": 101}}, "images.count"),
        ({"images": {"count": True}}, "images.count"),
        ({"images": {"design_facts": "yes"}}, "images.design_facts"),
        ({"images": {"enable": 1}}, "images.enable"),
        ({"images": {"snippet": {"max_tokens": 99}}}, "images.snippet.max_tokens"),
        ({"images": {"snippet": {"max_tokens": 100001}}}, "images.snippet.max_tokens"),
        ({"images": {"snippet": {"enabled": True}}}, "images.snippet.enabled"),
        ({"images": {"counts": 5}}, "images.counts"),
        ({"images": ["count"]}, "images"),
        ({"web_search": {"count": 0}}, "web_search.count"),
        ({"web_search": {"count": 101}}, "web_search.count"),
        ({"web_search": {"enable": "no"}}, "web_search.enable"),
        ({"web_search": {"highlight": {"max_tokens": 99}}}, "web_search.highlight"),
        ({"web_search": {"highlight": {"max_tokens": 20001}}}, "web_search.highlight"),
        ({"web_search": {"highlight": {"enabled": True}}}, "web_search.highlight"),
        ({"web_search": {"full_content": {"max_tokens": 99}}}, "web_search.full_content"),
        ({"web_search": {"full_content": {"max_tokens": 100001}}}, "web_search.full_content"),
        ({"web_search": {"include_domains": ["a" * 61]}}, "web_search.include_domains"),
        ({"web_search": {"exclude_domains": "a.com"}}, "web_search.exclude_domains"),
        ({"web_search": {"language": ["english"]}}, "web_search.language"),
        ({"web_search": {"time_basis": "indexed"}}, "web_search.time_basis"),
        ({"web_search": {"time_range": "last week"}}, "web_search.time_range"),
        ({"web_search": {"start_time": "2025-01-01"}}, "web_search.start_time"),
        ({"web_search": {"end_time": "2025-13-01T00:00:00Z"}}, "web_search.end_time"),
        ({"web_search": {"query": "x"}}, "web_search.query"),
        ({"timeout": 0}, "timeout"),
    ],
    ids=repr,
)
def test_design_invalid_option_raises(
    octen_cls: MagicMock, options: Dict[str, Any], mentions: str
) -> None:
    with pytest.raises(ValueError, match=f"design_search_options.*{mentions}"):
        _spec(design_search_options=options)
    octen_cls.assert_not_called()


@pytest.mark.parametrize(
    "options",
    [
        {"images": {"count": 1}},
        {"images": {"count": 100}},
        {"images": {"enable": False, "design_facts": True}},
        {"images": {"snippet": {"enable": True, "max_tokens": 100}}},
        {"images": {"snippet": {"max_tokens": 100000}}},
        {"web_search": {"count": 1, "enable": True}},
        {"web_search": {"count": 100}},
        {"web_search": {"highlight": {"enable": True, "max_tokens": 100}}},
        {"web_search": {"highlight": {"max_tokens": 20000}}},
        {"web_search": {"full_content": {"enable": True, "max_tokens": 100000}}},
        {"web_search": {"include_domains": ["a" * 60], "exclude_domains": ["b.com"]}},
        {"web_search": {"language": ["en", "zh"], "time_basis": "published",
                        "time_range": "month"}},
        {"web_search": {"start_time": "2025-01-01T00:00:00Z",
                        "end_time": "2025-06-30T23:59:59+08:00"}},
    ],
    ids=repr,
)
def test_design_valid_option_accepted(octen_cls: MagicMock, options: Dict[str, Any]) -> None:
    _spec(design_search_options=options).design_search("q")
    kwargs = _sdk(octen_cls, "design_search").call_args.kwargs
    assert {k: kwargs[k] for k in options} == options


def test_design_sdk_option_models_accepted(octen_cls: MagicMock) -> None:
    images = octen.DesignImageOptions(count=3, design_facts=False)
    web_search = octen.DesignWebSearchOptions(
        count=2, highlight=octen.HighlightOptions(enable=True, max_tokens=300)
    )
    _spec(design_search_options={"images": images, "web_search": web_search}).design_search("q")
    kwargs = _sdk(octen_cls, "design_search").call_args.kwargs
    assert kwargs["images"].count == 3
    assert kwargs["web_search"].highlight.max_tokens == 300


def test_design_sdk_model_with_unknown_highlight_key_rejected() -> None:
    web_search = octen.DesignWebSearchOptions(
        highlight=octen.HighlightOptions(enable=True, maxtokens=300)
    )
    with pytest.raises(ValueError, match="design_search_options.*web_search.highlight"):
        _spec(design_search_options={"web_search": web_search})


@pytest.mark.parametrize(
    "images",
    [{"group_id": GROUP_ID}, octen.DesignImageOptions(group_id=GROUP_ID)],
    ids=["dict", "model"],
)
def test_design_developer_group_id_rejected(images: Any) -> None:
    with pytest.raises(ValueError, match="does not accept 'images.group_id'"):
        _spec(design_search_options={"images": images})


@pytest.mark.parametrize("key", ["query", "group_id", "count", "design_facts", "snippet"])
def test_design_model_or_unknown_key_rejected(key: str) -> None:
    with pytest.raises(ValueError, match=f"design_search_options does not accept '{key}'"):
        _spec(design_search_options={key: "x"})


def test_options_deep_copied(octen_cls: MagicMock) -> None:
    images = {"count": 2, "snippet": {"enable": False}}
    image_options = {"count": 3}
    spec = _spec(design_search_options={"images": images}, image_search_options=image_options)
    images["count"] = 1000
    images["snippet"]["enable"] = True
    image_options["count"] = 1000
    spec.design_search("q")
    spec.image_search("q")
    assert _sdk(octen_cls, "design_search").call_args.kwargs["images"] == {
        "count": 2, "snippet": {"enable": False}
    }
    assert _sdk(octen_cls, "image_search").call_args.kwargs["count"] == 3


# --- output -----------------------------------------------------------------------

IMAGE_RESULT = {
    "title": "Minimalist Interior",
    "url": "https://i.example.com/a.jpg",
    "source_page": "https://example.com/pin/1",
    "description": "A minimalist interior.",
    "width": 736,
    "height": 981,
    "thumbnail": "https://i.example.com/a-thumb.jpg",
    "unknown_future_field": "x",
}


def test_image_output_trimmed(octen_cls: MagicMock) -> None:
    sparse = {"url": "https://i.example.com/b.jpg", "title": None}
    _sdk(octen_cls, "image_search").return_value = _image_response(
        [IMAGE_RESULT, sparse], meta={"warning": "partial", "usage": {"num_search_queries": 1}},
        request_id="rid-7",
    )
    assert _spec().image_search("interior") == {
        "data": {"results": [
            {k: IMAGE_RESULT[k] for k in
             ("title", "url", "source_page", "description", "width", "height")},
            {"url": "https://i.example.com/b.jpg"},
        ]},
        "request_id": "rid-7",
        "warning": "partial",
    }


@pytest.mark.parametrize("meta", [None, {}, {"warning": ""}, {"warning": None}])
def test_image_output_without_warning(octen_cls: MagicMock, meta: Any) -> None:
    _sdk(octen_cls, "image_search").return_value = _image_response([], meta=meta)
    assert _spec().image_search("q") == {"data": {"results": []}, "request_id": "rid-img"}


def test_image_output_without_results_key(octen_cls: MagicMock) -> None:
    _sdk(octen_cls, "image_search").return_value = octen.ImageSearchResponse(
        data={}, request_id="r"
    )
    assert _spec().image_search("q") == {"data": {"results": []}, "request_id": "r"}


DESIGN_IMAGE = {
    "type": "slide",
    "title": "Market opportunity",
    "description": "Three nested circles.",
    "url": "https://cdn.example.com/p-005.png",
    "thumbnail": "https://cdn.example.com/t.jpg",
    "source_page": "https://example.com/seed-deck.pptx",
    "width": 1920,
    "height": 1080,
    "group": {"group_id": GROUP_ID, "type": "deck", "name": "Seed Deck", "position": 5,
              "total": 14, "master": {"majorFont": "Calibri Light"}},
    "design_facts": {"styleFacts": {"fontFamilies": ["Calibri"]}},
    "snippet": {"format": "html", "code": "<section></section>"},
}
DESIGN_WEB = {
    "title": "Pricing Page Design",
    "url": "https://blog.example.com/pricing",
    "highlight": "Three plans side by side.",
    "full_content": "Full text.",
    "authors": "Example Blog",
    "time_published": "2026-09-08T00:00:00Z",
    "time_last_crawled": "2026-11-02T06:15:00Z",
    "favicon": "https://blog.example.com/favicon.ico",
    "cover_image": {"url": "https://img.example.com/cover.png"},
    "images": [{"url": "https://img.example.com/1.png"}],
}


def test_design_output_trimmed(octen_cls: MagicMock) -> None:
    data = {"query": "deck", "images": [DESIGN_IMAGE], "results": [DESIGN_WEB]}
    _sdk(octen_cls, "design_search").return_value = _design_response(
        data, meta={"warning": "w", "latency": 3}, request_id="rid-9"
    )
    expected_image = {k: v for k, v in DESIGN_IMAGE.items() if k != "thumbnail"}
    expected_web = {k: DESIGN_WEB[k] for k in
                    ("title", "url", "highlight", "full_content", "authors", "time_published")}
    assert _spec().design_search("deck") == {
        "data": {"query": "deck", "images": [expected_image], "results": [expected_web]},
        "request_id": "rid-9",
        "warning": "w",
    }


def test_design_output_keeps_only_returned_parts(octen_cls: MagicMock) -> None:
    # images.enable / web_search.enable false: the API leaves that list out.
    _sdk(octen_cls, "design_search").return_value = _design_response({"images": [DESIGN_IMAGE]})
    result = _spec().design_search(group_id=GROUP_ID)
    assert set(result["data"]) == {"images"}
    assert result["data"]["images"][0]["group"]["group_id"] == GROUP_ID
    _sdk(octen_cls, "design_search").return_value = _design_response({"results": None})
    assert _spec().design_search("q")["data"] == {"results": []}


# --- errors -----------------------------------------------------------------------


@pytest.mark.parametrize("function", FUNCTIONS)
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
    _sdk(octen_cls, function).side_effect = exc
    assert getattr(_spec(), function)(**MODEL_CALL[function]) == {"error": str(exc)}
    octen_cls.return_value.close.assert_called_once()


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize(
    "exc",
    [
        OctenTimeoutError(timeout=30, request_sent=True),
        OctenConnectionError(request_sent=None),
        OctenAPIError("bad gateway", status_code=502),
        OctenAPIError("malformed successful response", status_code=200),
    ],
)
def test_maybe_billed_errors_tell_model_not_to_retry(
    octen_cls: MagicMock, function: str, exc: Exception
) -> None:
    _sdk(octen_cls, function).side_effect = exc
    error = getattr(_spec(), function)(**MODEL_CALL[function])["error"]
    assert error.startswith(str(exc))
    assert "may already have reached the server and been billed" in error


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize(
    "exc",
    [
        OctenAuthenticationError(),
        OctenAPIError("forbidden", status_code=403),
        OctenAPIError("not found", status_code=404),
        OctenValidationError("count out of range"),
    ],
)
def test_config_errors_raise_and_log(
    octen_cls: MagicMock, caplog: Any, function: str, exc: Exception
) -> None:
    _sdk(octen_cls, function).side_effect = exc
    with caplog.at_level(logging.ERROR, logger="llama_index.tools.octen"):
        with pytest.raises(type(exc)):
            getattr(_spec(), function)(**MODEL_CALL[function])
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1 and errors[0].getMessage().startswith(f"{function} raised")
    octen_cls.return_value.close.assert_called_once()


def test_names_match_real_sdk_signatures() -> None:
    # No mocks: every developer option and model arg must be a parameter of the SDK method.
    from octen.resources.design_search import DesignSearchResource
    from octen.resources.image_search import ImageSearchResource

    from llama_index.tools.octen.base import _ALLOWED_OPTIONS

    image = set(inspect.signature(ImageSearchResource.search).parameters)
    design = set(inspect.signature(DesignSearchResource.search).parameters)
    model_image = set(inspect.signature(OctenToolSpec.image_search).parameters) - {"self"}
    assert _ALLOWED_OPTIONS["image_search_options"] | model_image <= image
    # group_id travels inside images.
    assert _ALLOWED_OPTIONS["design_search_options"] | {"query"} <= design
    assert "group_id" in octen.DesignImageOptions.model_fields
