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
from tests.test_vertical import RFC3339_INVALID, RFC3339_VALID

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


@pytest.mark.parametrize(
    ("options", "limit"),
    [
        ({}, 5),
        ({"design_search_options": {"images": {"design_facts": False}}}, 5),
        ({"design_search_options": {"images": {"count": 30}}}, 30),
        ({"design_search_options": {"images": octen.DesignImageOptions(count=100)}}, 100),
    ],
)
def test_design_group_follow_up_states_the_image_cap(options: Dict[str, Any], limit: int) -> None:
    # A follow-up returns the group's images up to images.count (API default 5), not all.
    metadata = _tools(**options)["design_search"].metadata
    assert f"at most {limit} per call" in metadata.description
    group_id = metadata.fn_schema.model_json_schema()["properties"]["group_id"]["description"]
    for text in (metadata.description, group_id):
        assert "every image" not in text
    metadata.to_openai_tool()


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


def test_image_domain_generators_reach_the_sdk_as_lists(octen_cls: MagicMock) -> None:
    # A one-shot iterable must not be used up by validation and forwarded empty.
    _spec().image_search(
        "q",
        include_domains=(d for d in ["a.com", "b.com"]),
        exclude_domains=iter(["c.com"]),
    )
    _sdk(octen_cls, "image_search").assert_called_once_with(
        query="q", include_domains=["a.com", "b.com"], exclude_domains=["c.com"]
    )


def test_image_domain_generator_over_the_limit_rejected(octen_cls: MagicMock) -> None:
    result = _spec().image_search("q", include_domains=(f"d{i}.com" for i in range(11)))
    assert "include_domains" in result["error"]
    octen_cls.assert_not_called()


def test_image_endless_domain_iterator_read_only_past_the_limit(octen_cls: MagicMock) -> None:
    import itertools

    read: List[int] = []

    def endless() -> Any:
        for i in itertools.count():
            read.append(i)
            if i > 100:  # a cap for this test only; the tool must stop long before
                raise AssertionError("read past the limit")
            yield f"d{i}.com"

    result = _spec().image_search("q", exclude_domains=endless())
    assert "exclude_domains" in result["error"]
    assert len(read) == 11
    octen_cls.assert_not_called()


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


def test_image_url_with_surrounding_ascii_whitespace_rejected(octen_cls: MagicMock) -> None:
    # The value sent is always the string given, so it is never trimmed first.
    for url in (f"  {IMAGE_URL}", f"{IMAGE_URL}\n", f"\t{IMAGE_URL}"):
        assert "https URL with a domain name" in _spec().image_search(image_url=url)["error"]
    octen_cls.assert_not_called()


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/photo.png?label=50%",
        "https://example.com/photo.png?q=%zz&r=%2",
        "https://example.com/photo.png?q=%zz#ok",
        "https://example.com/photo.png#50%25",
    ],
)
def test_bare_percent_in_query_sent_unchanged(octen_cls: MagicMock, url: str) -> None:
    # Go's net/url, as the API uses it, keeps only the query raw (checked 2026-11-22).
    _spec().image_search(image_url=url)
    _sdk(octen_cls, "image_search").assert_called_once_with(image_url=url)


@pytest.mark.parametrize(
    "url",
    [
        "https://exa%zzmple.com/photo.png",
        "https://example.com/50%/photo.png",
        "https://example.com/photo%2.png?label=ok",
        "https://example.com/photo.png;v=%zz",
        "https://example.com/photo.png#50%",
        "https://example.com/photo.png?ok=1#50%",
        "https://example.com/photo%;ab",
        "https://example.com/a%zz.png",
        "https://ex%zzample.com/a.png",
        "https://example.com/a.png#frag?q=%zz",
    ],
)
def test_bad_percent_outside_the_query_rejected(octen_cls: MagicMock, url: str) -> None:
    assert "https URL with a domain name" in _spec().image_search(image_url=url)["error"]
    octen_cls.assert_not_called()


def test_non_ascii_path_characters_left_to_the_parser(octen_cls: MagicMock) -> None:
    url = "https://example.com/caf\u00e9\u00a0menu.png"
    _spec().image_search(image_url=url)
    _sdk(octen_cls, "image_search").assert_called_once_with(image_url=url)


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com/a.png",
        "HTTP://example.com/a.png",
        "https://1.2.3.4/a.png",
        "https://192.0.2.1:8080/a.png",
        "https://u:p@1.2.3.4/a.png",
        "https://[2001:db8::1]/a.png",
        "https://u:p@[2001:db8::1]:443/a.png",
        "https://[::ffff:192.0.2.1]/a.png",
    ],
)
def test_image_url_needs_https_and_a_domain_name(octen_cls: MagicMock, url: str) -> None:
    error = _spec().image_search(image_url=url)["error"]
    assert "https URL with a domain name (no IP address)" in error
    octen_cls.assert_not_called()


def test_verbose_output_omits_the_image_url(octen_cls: MagicMock, capsys: Any) -> None:
    signed = "https://cdn.example.com/a.jpg?sig=SECRET-TOKEN&se=2026"
    OctenToolSpec(api_key="k", verbose=True).image_search(image_url=signed)
    out = capsys.readouterr().out
    assert "image URL" in out and "request_id=rid-img" in out
    assert "SECRET-TOKEN" not in out and "cdn.example.com" not in out


IMAGES_OFF = {"design_search_options": {"images": {"enable": False}}}


def test_design_images_off_description_offers_no_group_lookup() -> None:
    description = _tools(**IMAGES_OFF)["design_search"].metadata.description
    assert "call again with that group_id" not in description
    assert "screenshot" not in description
    assert "group_id is not available" in description
    _tools(**IMAGES_OFF)["design_search"].metadata.to_openai_tool()
    assert "call again with that group_id" in _tools()["design_search"].metadata.description


@pytest.mark.parametrize(
    "images", [{"enable": False}, octen.DesignImageOptions(enable=False, count=3)]
)
def test_design_images_off_rejects_group_only_call(octen_cls: MagicMock, images: Any) -> None:
    result = _spec(design_search_options={"images": images}).design_search(group_id=GROUP_ID)
    assert set(result) == {"error"}
    assert "query" in result["error"] and "turned off" in result["error"]
    octen_cls.assert_not_called()


def test_design_images_off_schema_requires_query_without_group_id() -> None:
    schema = _tools(**IMAGES_OFF)["design_search"].metadata.fn_schema
    json_schema = schema.model_json_schema()
    assert list(json_schema["properties"]) == ["query"]
    assert json_schema["required"] == ["query"]
    assert json_schema["properties"]["query"]["maxLength"] == 500
    assert json_schema["properties"]["query"]["description"]
    # The default spec keeps both, neither required.
    default = _tools()["design_search"].metadata.fn_schema.model_json_schema()
    assert list(default["properties"]) == ["query", "group_id"] and "required" not in default


def test_design_images_off_does_not_forward_group_id(octen_cls: MagicMock) -> None:
    _spec(**IMAGES_OFF).design_search("pricing page", group_id=GROUP_ID)
    _sdk(octen_cls, "design_search").assert_called_once_with(
        query="pricing page", images={"enable": False}
    )


def test_design_images_off_query_still_searches(octen_cls: MagicMock) -> None:
    _spec(**IMAGES_OFF).design_search("pricing page")
    _sdk(octen_cls, "design_search").assert_called_once_with(
        query="pricing page", images={"enable": False}
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
        ({"image_url": "ftp://example.com/a.jpg"}, "https URL with a domain name"),
        ({"image_url": "data:image/png;base64,AAAA"}, "https URL with a domain name"),
        ({"image_url": "example.com/a.jpg"}, "https URL with a domain name"),
        ({"image_url": "https://"}, "https URL with a domain name"),
        ({"image_url": "https://:443/x.png"}, "https URL with a domain name"),
        ({"image_url": "https://[example.com/x.jpg"}, "https URL with a domain name"),
        ({"image_url": "http://user@/x.png"}, "https URL with a domain name"),
        ({"image_url": "https://example.com:abc/x.png"}, "https URL with a domain name"),
        ({"image_url": "https://example.com:70000/x.png"}, "https URL with a domain name"),
        ({"image_url": "https://exa mple.com/photo.png"}, "https URL with a domain name"),
        ({"image_url": "https://example.com/my photo.png"}, "https URL with a domain name"),
        ({"image_url": "https://example.com/a.png\tb"}, "https URL with a domain name"),
        ({"image_url": "\x00https://example.com/a.png"}, "https URL with a domain name"),
        ({"image_url": "https://example.com/a\x7f.png"}, "https URL with a domain name"),
        ({"image_url": "https://example.com/a.png\x1f"}, "https URL with a domain name"),
        ({"image_url": "https:///example.com/a.png"}, "https URL with a domain name"),
        ({"image_url": "https://\\example.com/a.png"}, "https URL with a domain name"),
        ({"image_url": "https://exa\\mple.com/a.png"}, "https URL with a domain name"),
        ({"image_url": "https://example.com\\a.png"}, "https URL with a domain name"),
        ({"image_url": "https://ex<am>ple.com/a.png"}, "https URL with a domain name"),
        ({"image_url": 'https://exa"mple.com/a.png'}, "https URL with a domain name"),
        ({"image_url": "https://exa%zzmple.com/a.png"}, "https URL with a domain name"),
        ({"image_url": "https://exa%2mple.com/a.png"}, "https URL with a domain name"),
        ({"image_url": "https://example.com/a%2.png"}, "https URL with a domain name"),
        ({"image_url": "https://a@b@example.com/a.png"}, "https URL with a domain name"),
        ({"image_url": "https://u[1]@example.com/a.png"}, "https URL with a domain name"),
        ({"image_url": "https://[::1/a.png"}, "https URL with a domain name"),
        ({"image_url": "https://[zz::1]/a.png"}, "https URL with a domain name"),
        ({"image_url": "https://[1:2:3]/a.png"}, "https URL with a domain name"),
        ({"image_url": "https://[12345::]/a.png"}, "https URL with a domain name"),
        ({"image_url": "https://[::1]x/a.png"}, "https URL with a domain name"),
        ({"image_url": "https://ex]ample.com/a.png"}, "https URL with a domain name"),
        ({"image_url": "https://example.com:80:90/a.png"}, "https URL with a domain name"),
        ({"image_url": "https://exa:mple.com:80/a.png"}, "https URL with a domain name"),
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
    assert "error" not in spec.image_search(image_url="https://example.com/a.png",
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


def test_image_output_is_data_unchanged(octen_cls: MagicMock) -> None:
    # Same as the other functions: data as the API returns it, nothing dropped.
    sparse = {"url": "https://i.example.com/b.jpg", "title": None}
    data = {"results": [IMAGE_RESULT, sparse], "unknown_top_level": 1}
    _sdk(octen_cls, "image_search").return_value = octen.ImageSearchResponse(
        code=0, msg="success", data=data,
        meta={"warning": "partial", "usage": {"num_search_queries": 1}}, request_id="rid-7",
    )
    assert _spec().image_search("interior") == {
        "data": data, "request_id": "rid-7", "warning": "partial"
    }


@pytest.mark.parametrize("meta", [None, {}, {"warning": ""}, {"warning": None}])
def test_image_output_without_warning(octen_cls: MagicMock, meta: Any) -> None:
    _sdk(octen_cls, "image_search").return_value = _image_response([], meta=meta)
    assert _spec().image_search("q") == {"data": {"results": []}, "request_id": "rid-img"}


def test_image_output_without_results_key(octen_cls: MagicMock) -> None:
    _sdk(octen_cls, "image_search").return_value = octen.ImageSearchResponse(
        data={}, request_id="r"
    )
    assert _spec().image_search("q") == {"data": {}, "request_id": "r"}


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


def test_design_output_is_data_unchanged(octen_cls: MagicMock) -> None:
    data = {"query": "deck", "images": [DESIGN_IMAGE], "results": [DESIGN_WEB],
            "specs": {"future": True}}
    _sdk(octen_cls, "design_search").return_value = _design_response(
        data, meta={"warning": "w", "latency": 3}, request_id="rid-9"
    )
    assert _spec().design_search("deck") == {
        "data": data, "request_id": "rid-9", "warning": "w"
    }


def test_design_output_keeps_only_returned_parts(octen_cls: MagicMock) -> None:
    # images.enable / web_search.enable false: the API leaves that list out.
    _sdk(octen_cls, "design_search").return_value = _design_response({"images": [DESIGN_IMAGE]})
    result = _spec().design_search(group_id=GROUP_ID)
    assert result["data"] == {"images": [DESIGN_IMAGE]}
    _sdk(octen_cls, "design_search").return_value = _design_response({"results": None})
    assert _spec().design_search("q")["data"] == {"results": None}


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


@pytest.mark.parametrize(
    ("status", "mentions"),
    [(413, "smaller image"), (415, "supported format")],
)
def test_image_413_415_returned_to_model_with_guidance(
    octen_cls: MagicMock, caplog: Any, status: int, mentions: str
) -> None:
    exc = OctenAPIError("upstream-zq9", status_code=status, request_id="rid-413")
    _sdk(octen_cls, "image_search").side_effect = exc
    with caplog.at_level(logging.ERROR, logger="llama_index.tools.octen"):
        result = _spec().image_search(image_url=IMAGE_URL)
    assert set(result) == {"error"}
    assert f"HTTP {status}" in result["error"] and "rid-413" in result["error"]
    assert mentions in result["error"]
    assert "upstream-zq9" not in result["error"]
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


SIGNED_URL = "https://cdn.example.com/a.jpg?sig=SECRET-TOKEN&se=2026"


@pytest.mark.parametrize(
    "exc",
    [
        # Upstream text echoing a truncated or normalized copy of the URL.
        OctenAPIError("Payload too large for https://cdn.example.com/a.jpg?sig=SECRET-TO",
                      status_code=413, request_id="rid-x"),
        OctenAPIError("unsupported: HTTPS://CDN.EXAMPLE.COM/A.JPG?SIG=SECRET-TOKEN",
                      status_code=415),
        OctenAPIError("cannot read cdn.example.com/a.jpg%3Fsig%3DSECRET-TOKEN", status_code=422),
        OctenAPIError("upstream failed for SECRET-TOKEN", status_code=503),
        OctenAPIError("bad gateway SECRET-TOKEN", status_code=502),
        OctenRateLimitError("slow down SECRET-TOKEN"),
        OctenTimeoutError("timed out fetching SECRET-TOKEN", timeout=30, request_sent=True),
        OctenConnectionError("reset while sending SECRET-TOKEN", request_sent=False),
        OctenAPIError("malformed body with SECRET-TOKEN", status_code=200),
    ],
)
def test_image_url_errors_never_carry_upstream_text(
    octen_cls: MagicMock, exc: Exception
) -> None:
    _sdk(octen_cls, "image_search").side_effect = exc
    error = _spec().image_search(image_url=SIGNED_URL)["error"]
    assert "SECRET" not in error.upper() and "cdn.example.com" not in error.lower(), error
    assert "Image Search" in error


def test_authority_rejects_bracketed_hosts() -> None:
    # The API takes only domain-name hosts, so no bracketed (IPv6) host is valid.
    from llama_index.tools.octen.base import _valid_authority

    assert _valid_authority("u:p@example.com:443")
    for netloc in ("[2001:db8::1]:443", "u@[::1]", "[1:2:3]", "[::1", "[]",
                   "[2001:4860::1]junk:443", "u@[::1]@x"):
        assert not _valid_authority(netloc), netloc


# An Arabic label ending in an ASCII digit: valid under IDNA2008, rejected by Python's
# IDNA2003 "idna" codec (BIDI rule 3), and its punycode form.
ARABIC_LABEL = "\u0627\u062e\u062a\u0628\u0627\u0631" + "1"
ARABIC_PUNYCODE = "xn--1-ymcadjxy"


def test_userinfo_allows_rfc3986_userinfo_characters(octen_cls: MagicMock) -> None:
    for url in ("https://user:pass'word@example.com/image.png",
                "https://u-._~!$&'()*+,;=:%41@example.com/image.png"):
        assert "error" not in _spec().image_search(image_url=url), url
    for url in ("https://us<er@example.com/a.png", "https://us%zzer@example.com/a.png",
                'https://us"er@example.com/a.png', "https://us[er@example.com/a.png"):
        assert "https URL with a domain name" in _spec().image_search(image_url=url)["error"], url


def test_host_validation_does_not_reject_valid_hosts(octen_cls: MagicMock) -> None:
    for host in (f"{ARABIC_LABEL}.example", f"{ARABIC_PUNYCODE}.example",
                 "cafe\u0301.example", "ex{a}mple.com", "a|b^c.example", "a_b~c.example"):
        url = f"https://{host}/a.png"
        assert "error" not in _spec().image_search(image_url=url), url
    assert _sdk(octen_cls, "image_search").call_count == 6


@pytest.mark.parametrize(
    "host", ["exa\\mple.com", "ex<a>mple.com", 'exa"mple.com', "exa'mple.com",
             "exa`mple.com", "exa%2mple.com", "ex]ample.com", "ex[ample.com"],
)
def test_host_validation_rejects_characters_that_cannot_be_in_a_host(
    octen_cls: MagicMock, host: str
) -> None:
    result = _spec().image_search(image_url=f"https://{host}/a.png")
    assert "https URL with a domain name" in result["error"]
    octen_cls.assert_not_called()


def test_sanitized_error_drops_notes_and_other_attributes(octen_cls: MagicMock) -> None:
    import traceback

    exc = OctenAPIError("forbidden", status_code=403, request_id="rid-403")
    # add_note (3.11+) stores notes in __notes__; set it directly to cover 3.10.
    exc.__notes__ = [f"while fetching {SIGNED_URL}"]
    exc.code = 403
    exc.extra_url = SIGNED_URL
    _sdk(octen_cls, "image_search").side_effect = exc
    with pytest.raises(OctenAPIError) as raised:
        _spec().image_search(image_url=SIGNED_URL)
    error = raised.value
    assert (error.status_code, error.request_id, error.code) == (403, "rid-403", 403)
    assert not hasattr(error, "__notes__") and not hasattr(error, "extra_url")
    formatted = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    for text in (str(error), repr(error), formatted, str(vars(error))):
        assert "SECRET" not in text and "cdn.example.com" not in text, text


@pytest.mark.parametrize(
    ("start", "end"),
    [
        ("2025-06-01T00:00:00Z", "2025-01-01T00:00:00Z"),
        ("2025-01-01T08:00:00+08:00", "2024-12-31T23:59:59Z"),
        ("2025-01-01T00:00:00.5Z", "2025-01-01T00:00:00.4Z"),
        # Differs below a microsecond.
        ("2025-01-01T00:00:00.0000002Z", "2025-01-01T00:00:00.0000001Z"),
        ("2025-01-01T08:00:00.1234567891+08:00", "2025-01-01T00:00:00.123456789Z"),
        ("9999-12-31T23:59:59-05:00", "9999-12-31T23:59:59Z"),
        ("0001-01-01T00:00:00Z", "0001-01-01T00:00:00+00:01"),
    ],
)
def test_design_reversed_time_range_rejected(start: str, end: str) -> None:
    with pytest.raises(ValueError, match="design_search_options.*end_time.*before.*start_time"):
        _spec(design_search_options={"web_search": {"start_time": start, "end_time": end}})


@pytest.mark.parametrize(
    ("start", "end"),
    [
        ("2025-01-01T00:00:00Z", "2025-01-01T00:00:00Z"),
        ("2025-01-01T08:00:00+08:00", "2025-01-01T00:00:00Z"),
        ("2025-01-01T00:00:00Z", "2025-06-30T23:59:59+08:00"),
        ("2025-01-01T00:00:00.1Z", "2025-01-01T00:00:00.100000000Z"),
        ("2025-01-01T00:00:00.0000001Z", "2025-01-01T00:00:00.0000002Z"),
        ("2025-01-01T08:00:00.123456789+08:00", "2025-01-01T00:00:00.1234567891Z"),
        # Near the ends of the calendar: applying the offset must not overflow.
        ("0001-01-01T00:00:00+05:00", "9999-12-31T23:59:59-05:00"),
        ("0001-01-01T00:00:00+23:59", "0001-01-01T00:00:00+23:59"),
    ],
)
def test_design_ordered_time_range_accepted(start: str, end: str) -> None:
    _spec(design_search_options={"web_search": {"start_time": start, "end_time": end}})


WEB_OFF = {"design_search_options": {"web_search": {"enable": False}}}


def test_design_description_follows_web_switch() -> None:
    on = _tools()["design_search"].metadata.description
    assert "Web results on the same query come back alongside" in on
    off = _tools(**WEB_OFF)["design_search"].metadata
    assert "web result" not in off.description.lower()
    assert "call again with that group_id" in off.description
    off.to_openai_tool()
    images_off = _tools(**IMAGES_OFF)["design_search"].metadata.description
    assert "screenshot" not in images_off and "group_id is not available" in images_off


@pytest.mark.parametrize(
    "options",
    [
        {"images": {"enable": False}, "web_search": {"enable": False}},
        {"images": octen.DesignImageOptions(enable=False),
         "web_search": octen.DesignWebSearchOptions(enable=False)},
    ],
)
def test_design_both_switches_off_rejected(options: Dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="design_search_options.*images.*web_search"):
        _spec(design_search_options=options)


def test_image_url_accepts_authority_forms(octen_cls: MagicMock) -> None:
    for url in ("https://user:pw@example.com:8443/a.png", "HTTPS://example.com/a.png",
                "https://1.2.3.example/a.png", "https://example.com:/a.png",
                "https://us%40er@example.com/a.png", "https://ex%41mple.com/a.png",
                "https://xn--bcher-kva.example/a.png", "https://b\u00fccher.example/a.png",
                "https://example.com/a%20b.png?x=1#f"):
        assert "error" not in _spec().image_search(image_url=url), url


@pytest.mark.parametrize(
    "exc",
    [
        OctenAuthenticationError("bad key for https://cdn.example.com/a.jpg?sig=SECRET-TOKEN",
                                 request_id="rid-401",
                                 response_data={"msg": "SECRET-TOKEN"}),
        OctenAPIError("forbidden: https://cdn.example.com/a.jpg?sig=SECRET-TOKEN",
                      status_code=403, request_id="rid-403",
                      response_data={"msg": "SECRET-TOKEN"}),
        OctenAPIError("not found SECRET-TOKEN", status_code=404),
        OctenValidationError("image search request failed validation SECRET-TOKEN",
                             errors=[{"input": SIGNED_URL}]),
    ],
)
def test_image_url_raised_errors_are_sanitized(
    octen_cls: MagicMock, caplog: Any, exc: Exception
) -> None:
    _sdk(octen_cls, "image_search").side_effect = exc
    with caplog.at_level(logging.DEBUG, logger="llama_index.tools.octen"):
        with pytest.raises(type(exc)) as raised:
            _spec().image_search(image_url=SIGNED_URL)
    error = raised.value
    assert type(error) is type(exc)
    assert getattr(error, "status_code", None) == getattr(exc, "status_code", None)
    assert getattr(error, "request_id", None) == getattr(exc, "request_id", None)
    assert getattr(error, "response_data", None) is None
    assert error.__cause__ is None and error.__context__ is None
    rendered = [str(error), repr(error), str(error.args), caplog.text,
                str(getattr(error, "errors", ""))]
    for text in rendered:
        assert "SECRET" not in text and "cdn.example.com" not in text, text
    assert "image URL" in str(error)


class _NeedsArgs(Exception):
    def __new__(cls, first: str, second: str) -> "_NeedsArgs":
        return super().__new__(cls, first, second)


class _Slotted(Exception):
    __slots__ = ()

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(name)


class _ArgsOnly(Exception):
    def __setattr__(self, name: str, value: Any) -> None:
        if name != "args":
            raise AttributeError(name)
        super().__setattr__(name, value)


def _exotic_errors() -> List[Exception]:
    errors: List[Exception] = [
        _NeedsArgs(f"bad {SIGNED_URL}", "x"), _Slotted(SIGNED_URL), _ArgsOnly(SIGNED_URL)
    ]
    group = getattr(__builtins__ if isinstance(__builtins__, dict) else __builtins__.__dict__,
                    "get")("ExceptionGroup")
    if group is not None:  # Python 3.11+
        errors.append(group(f"failed {SIGNED_URL}", [ValueError(SIGNED_URL)]))
    return errors


@pytest.mark.parametrize("exc", _exotic_errors(), ids=lambda e: type(e).__name__)
def test_sanitizing_an_exotic_exception_falls_back_cleanly(
    octen_cls: MagicMock, caplog: Any, exc: Exception
) -> None:
    import traceback

    _sdk(octen_cls, "image_search").side_effect = exc
    with caplog.at_level(logging.DEBUG, logger="llama_index.tools.octen"):
        with pytest.raises(Exception) as raised:
            _spec().image_search(image_url=SIGNED_URL)
    error = raised.value
    assert isinstance(error, octen.OctenError), type(error)
    assert error.__cause__ is None and error.__context__ is None
    formatted = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    for text in (str(error), repr(error), formatted, caplog.text):
        assert "SECRET" not in text and "cdn.example.com" not in text, text
    assert type(exc).__name__ in str(error)


def test_text_query_raised_errors_unchanged(octen_cls: MagicMock) -> None:
    exc = OctenAPIError("forbidden for query", status_code=403)
    _sdk(octen_cls, "image_search").side_effect = exc
    with pytest.raises(OctenAPIError) as raised:
        _spec().image_search("cats")
    assert raised.value is exc


@pytest.mark.parametrize(
    "exc",
    [
        OctenTimeoutError(timeout=30, request_sent=True),
        OctenConnectionError(request_sent=None),
        OctenAPIError("bad gateway", status_code=502),
        OctenAPIError("malformed successful response", status_code=200),
    ],
)
def test_billing_warning_never_comes_with_a_retry_hint(
    octen_cls: MagicMock, exc: Exception
) -> None:
    _sdk(octen_cls, "image_search").side_effect = exc
    for kwargs in ({"image_url": IMAGE_URL}, {"query": "cats"}):
        error = _spec().image_search(**kwargs)["error"]
        assert "may already have reached the server and been billed" in error
        assert "try again" not in error.lower() and "try another" not in error.lower(), error


def test_image_url_errors_keep_billing_warning_and_status(octen_cls: MagicMock) -> None:
    _sdk(octen_cls, "image_search").side_effect = OctenTimeoutError(timeout=30, request_sent=True)
    assert "may already have reached the server and been billed" in _spec().image_search(
        image_url=IMAGE_URL)["error"]
    _sdk(octen_cls, "image_search").side_effect = OctenAPIError(
        "x", status_code=503, request_id="rid-503")
    error = _spec().image_search(image_url=IMAGE_URL)["error"]
    assert "HTTP 503" in error and "rid-503" in error
    assert "may already have" not in error
    assert "Try again later" in error


def test_text_query_errors_keep_upstream_text(octen_cls: MagicMock) -> None:
    exc = OctenAPIError("bad filter", status_code=400)
    _sdk(octen_cls, "image_search").side_effect = exc
    assert _spec().image_search("cats")["error"] == str(exc)


@pytest.mark.parametrize(
    "exc",
    [
        OctenAuthenticationError(),
        OctenAPIError("forbidden", status_code=403),
        OctenAPIError("payload too large", status_code=413),
        OctenAPIError("unsupported media type", status_code=415),
    ],
)
def test_design_413_415_and_auth_still_raise(octen_cls: MagicMock, exc: Exception) -> None:
    # Only Image Search takes an image from the model, so only it hands these back.
    _sdk(octen_cls, "design_search").side_effect = exc
    with pytest.raises(type(exc)):
        _spec().design_search("q")


@pytest.mark.parametrize(
    "exc", [OctenAuthenticationError(), OctenAPIError("forbidden", status_code=403)]
)
def test_image_auth_errors_still_raise(octen_cls: MagicMock, exc: Exception) -> None:
    _sdk(octen_cls, "image_search").side_effect = exc
    with pytest.raises(type(exc)):
        _spec().image_search(image_url=IMAGE_URL)


@pytest.mark.parametrize("field", ["start_time", "end_time"])
@pytest.mark.parametrize("value", RFC3339_INVALID)
def test_design_web_search_non_rfc3339_time_rejected(value: str, field: str) -> None:
    with pytest.raises(ValueError, match=f"design_search_options.*web_search.{field}"):
        _spec(design_search_options={"web_search": {field: value}})


@pytest.mark.parametrize("value", RFC3339_VALID)
def test_design_web_search_rfc3339_time_accepted(octen_cls: MagicMock, value: str) -> None:
    web_search = {"start_time": value, "end_time": value}
    _spec(design_search_options={"web_search": web_search}).design_search("q")
    assert _sdk(octen_cls, "design_search").call_args.kwargs["web_search"] == web_search


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
