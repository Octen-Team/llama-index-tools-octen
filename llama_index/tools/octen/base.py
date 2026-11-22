"""Octen tool spec - web search API for AI."""

import copy
import datetime
import decimal
import ipaddress
import itertools
import logging
import math
import os
import re
import warnings
from typing import Any, Dict, FrozenSet, Iterator, List, Literal, Optional, Tuple, Type
from urllib.parse import urlparse

from llama_index.core.bridge.pydantic import BaseModel, Field, ValidationError
from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import SPEC_FUNCTION_TYPE, BaseToolSpec
from llama_index.core.tools.types import ToolMetadata

logger = logging.getLogger("llama_index.tools.octen")

TimeRange = Literal["day", "week", "month", "year", "d", "w", "m", "y"]

# Model-facing descriptions and argument schemas for the newer functions.
# Supplied explicitly because llama-index-core versions differ in how much of
# a docstring reaches the model; each description fits the 1024-char limit
# that ToolMetadata.to_openai_tool() checks by default.

_TIME_RANGE_DESC = (
    'Relative time window: "day", "week", "month" or "year" (or "d", "w", "m", "y").'
)
_LANGUAGE_DESC = (
    'ISO 639-1 codes to filter by, e.g. ["en", "ja"]. Supported: ar, de, en, es, fr, '
    "hi, id, it, ja, ko, nl, pl, pt, ru, th, tr, vi, zh."
)


class _BroadSearchArgs(BaseModel):
    query: str = Field(
        description="One natural-language question, max 500 chars, pronouns resolved. "
        "Do not pre-split it into sub-queries."
    )
    include_domains: Optional[List[str]] = Field(
        default=None, description='Only return results from these domains, e.g. ["arxiv.org"].'
    )
    exclude_domains: Optional[List[str]] = Field(
        default=None, description="Drop results from these domains."
    )
    time_range: Optional[TimeRange] = Field(default=None, description=_TIME_RANGE_DESC)
    language: Optional[List[str]] = Field(default=None, description=_LANGUAGE_DESC)


class _NewsSearchArgs(BaseModel):
    query: str = Field(description="What to find news about, max 500 chars.")
    include_domains: Optional[List[str]] = Field(
        default=None,
        description='Only return articles from these domains, e.g. ["reuters.com"]. '
        "Does not narrow the storylines.",
    )
    exclude_domains: Optional[List[str]] = Field(
        default=None,
        description="Drop articles from these domains. Does not narrow the storylines.",
    )
    time_range: Optional[TimeRange] = Field(default=None, description=_TIME_RANGE_DESC)
    language: Optional[List[str]] = Field(default=None, description=_LANGUAGE_DESC)


class _BusinessSearchArgs(BaseModel):
    query: str = Field(
        description="The company, person or comparison to look up, max 500 chars."
    )
    include_domains: Optional[List[str]] = Field(
        default=None, description="Only return ranked sources from these domains."
    )
    exclude_domains: Optional[List[str]] = Field(
        default=None, description="Drop ranked sources from these domains."
    )
    time_range: Optional[TimeRange] = Field(
        default=None, description=_TIME_RANGE_DESC + " Applies to the ranked sources only."
    )
    language: Optional[List[str]] = Field(
        default=None, description=_LANGUAGE_DESC + " Applies to the ranked sources only."
    )


class _SearchPremierArgs(BaseModel):
    query: str = Field(description="The question or search query, max 500 chars.")
    include_domains: Optional[List[str]] = Field(
        default=None,
        description='Only return results from these domains, e.g. ["arxiv.org"].',
    )
    exclude_domains: Optional[List[str]] = Field(
        default=None, description="Drop results from these domains."
    )


class _PulseSearchArgs(BaseModel):
    query: str = Field(
        description='What to find the address of, e.g. "github docs". Max 500 chars.'
    )


class _PlaceSearchArgs(BaseModel):
    query: str = Field(
        description='What to look for, e.g. "coffee shops" or "ramen in Shinjuku". '
        "Max 500 chars."
    )
    latitude: Optional[float] = Field(
        default=None,
        description="WGS84 latitude to search around, -90 to 90. Must be sent with longitude.",
    )
    longitude: Optional[float] = Field(
        default=None,
        description="WGS84 longitude to search around, -180 to 180. Must be sent with latitude.",
    )
    radius: Optional[int] = Field(
        default=None,
        description="Keep only places within this many metres of the point, 1 to 50000. "
        "Requires latitude and longitude.",
    )
    city: Optional[str] = Field(
        default=None, description="Approximate location when coordinates are unknown: city."
    )
    region: Optional[str] = Field(
        default=None, description="Approximate location: state, province or region."
    )
    country: Optional[str] = Field(
        default=None, description="Approximate location: country name or code."
    )
    timezone: Optional[str] = Field(
        default=None, description='Approximate location: IANA timezone, e.g. "Asia/Tokyo".'
    )


_MODEL_FACING: Dict[str, Tuple[str, Type[BaseModel]]] = {
    "broad_search": (
        "Search the web across many angles in one call: expands one question into "
        "several sub-queries, runs them concurrently and returns results grouped per "
        "sub-query (not deduplicated; a URL may appear more than once). Use for "
        "comparisons across many sources, literature or market surveys, open-ended "
        '"what are the options for X" questions, questions with 3+ independent parts, '
        "and multi-angle questions about recent events. Not for a single fact or page "
        "(use search), a company or person (business_search), news on one topic "
        "(news_search), a place (place_search) or a site's URL (pulse_search). It costs "
        "roughly one search per sub-query and is slower; when in doubt, prefer search. "
        "Pass the question as one query; do not pre-split it. Do not call it twice "
        "after a disappointing result; follow up with a targeted search on the gaps. "
        "The filters apply to every sub-query.",
        _BroadSearchArgs,
    ),
    "news_search": (
        "Search recent news: ranked articles (title, url, snippet) and normally "
        "subjects, the coverage grouped into storylines, each with a summary and time "
        "span. Use for what is being reported now: breaking events, announcements, "
        "coverage of a company or person over recent days or weeks. include_domains "
        "and exclude_domains narrow the ranked articles only; storylines are built "
        "from the open web either way, so when the question restricts its sources "
        '(e.g. "what is Reuters saying about X"), answer from the articles, not the '
        "storylines. For a company or person profile use business_search; for "
        "non-news pages use search; for a multi-angle question use broad_search "
        "instead of looping this tool.",
        _NewsSearchArgs,
    ),
    "business_search": (
        "Look up a company or a person: returns structured cards plus the ranked "
        "sources behind them. A company card has the profile, identifiers, financial "
        "and stock metrics, key people and recent announcements; a person card has "
        "the current position, career history and recent activity. Name two "
        "companies in one query to get a card for each, ready to compare. Use for any "
        "question about a company or the people in one, including the numbers "
        "(revenue, net income, share price, funding rounds, headcount), what it does, "
        "due diligence and comparisons; one call here replaces reading many pages "
        "with search. Not for breaking coverage of an event (use news_search) or "
        "general web pages (use search). The domain, time and language filters "
        "narrow the ranked sources only, never the cards.",
        _BusinessSearchArgs,
    ),
    "pulse_search": (
        "Find the web address of a site or page you could almost name: returns ranked "
        "candidate URLs with titles. Use for official docs and API references "
        '("github docs", "python list methods"), help centres ("how do I return an '
        'amazon order") and known sites ("youtube"); it answers in about 10ms for a '
        "fraction of the cost of search. Not for news or anything needing freshness "
        "(use news_search), and not for questions that need real retrieval, filters "
        "or a time window (use search): it sees no fresh content. URLs may come back "
        'without a scheme (e.g. "github.com/docs"); prepend "https://" before using '
        "one as a link.",
        _PulseSearchArgs,
    ),
    "place_search": (
        "Find real-world places, each returned as a card: name, category, address, "
        "coordinates, phone, website, opening hours, status, rating and review count. "
        'Use for anything with a location: "coffee shops near me", "museums in '
        'Lisbon", "is this place open on Sunday". Pass latitude and longitude, always '
        "together, to search around a point; each card then has its distance in "
        "metres, and radius (only with both coordinates) caps how far. When only the "
        "area is known, pass city, region, country or timezone instead; no distance "
        "comes back that way. Pass the caller's location when the question is "
        'relative to them ("near me", "nearest") or names no area; a query that names '
        'its own area ("ramen in Shinjuku") needs neither.',
        _PlaceSearchArgs,
    ),
    "search_premier": (
        "Search the live web with deeper retrieval and reranking than search, and "
        "return ranked results (title, url, highlight). It searches more broadly and "
        "ranks more carefully, so the page that holds the answer ranks higher, and each "
        "highlight is the passage most relevant to the query from the whole page. Use "
        "for questions that need the exact page holding the answer: a specific fact, a "
        "detail deep inside a page, a niche or long-tail document, or a named report, "
        "filing or document. Pass the question as is. Not for quick, low-latency "
        "lookups (use search); for a multi-part question use broad_search. Narrow with "
        "include_domains and exclude_domains.",
        _SearchPremierArgs,
    ),
}


class _ExtractArgs(BaseModel):
    urls: List[str] = Field(description="The page URLs to read, each at most 2048 characters.")
    query: Optional[str] = Field(
        default=None,
        description="Optional; leave unset in the normal case. When set (max 500 chars), "
        "each result returns highlights, the passages relevant to it, instead of the "
        "full page content.",
    )


# Based on the hosted MCP server's extract tool; {max_urls} is the developer's cap.
_EXTRACT_DESCRIPTION = (
    "Read web pages by URL and return their clean content (markdown or text). Match "
    "each result to its input by the result's url field, not by position. Use it when "
    "you already have the URLs; to find "
    "pages first, use search or broad_search. Send at most {max_urls} URLs per call; "
    "for more, split them across several calls. Without query, each result has the "
    "page's full_content, which is what you want in almost all cases. Only pass query "
    "when you want relevance-ranked snippets on a specific topic: each result then has "
    "highlights instead of full_content, so the page body is partial. Results may also "
    "carry a category (topic) and page_structure (page type) classification. A URL "
    'that could not be read comes back with status "failed" and an error_message '
    "while the other URLs still return their content."
)

class _ImageSearchArgs(BaseModel):
    query: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Text describing the images to find, max 500 chars. "
        "Exactly one of query or image_url, never both.",
    )
    image_url: Optional[str] = Field(
        default=None,
        description="Public https URL with a domain name (no IP address) of an image "
        "already on the web, to find "
        "visually similar images. Exactly one of query or image_url, never both.",
    )
    include_domains: Optional[List[str]] = Field(
        default=None,
        max_length=10,
        description='Only return images whose source page is on these domains, e.g. '
        '["unsplash.com"]. At most 10.',
    )
    exclude_domains: Optional[List[str]] = Field(
        default=None,
        max_length=10,
        description="Drop images whose source page is on these domains. At most 10.",
    )


class _DesignSearchArgs(BaseModel):
    query: Optional[str] = Field(
        default=None,
        max_length=500,
        description='What to find design references for, e.g. "pricing section for a '
        'SaaS landing page". Max 500 chars. Required unless group_id is set.',
    )
    group_id: Optional[str] = Field(
        default=None,
        description="The group.group_id of an image result from an earlier design_search "
        "call, to get that web page's or deck's images in group order instead of "
        "searching by query.",
    )


# Based on the hosted MCP server's image_search tool and the Image Search API reference.
_IMAGE_SEARCH_DESCRIPTION = (
    "Find images on the web by a text query or by a reference image: returns ranked "
    "images, each with its title, image url, source_page (the page hosting it), a "
    "description of its content, and its width and height. Pass exactly one of query "
    "(text describing the images) or image_url (a public https URL, with a domain name, "
    "of a picture already on the web, to find visually similar ones), never both. Use it when the user wants "
    "pictures, photos, diagrams, screenshots or other visual references; not for "
    "general text web search (use search) or UI, web page and slide design references "
    "(use design_search). Show an image by its url and cite its source_page. "
    "include_domains and exclude_domains (at most 10 each) are matched against each "
    "result's source_page."
)

# Based on the Design Search API reference.
_DESIGN_SEARCH_DESCRIPTION = (
    "Find design references: each image result is a screenshot of a web page, page "
    "section, app screen or slide (its type), with a title, a description, the image "
    "url, source_page (not for app screens) and, when returned, its design values "
    "(design_facts: colors, fonts, font sizes) and reusable HTML code (snippet).{web} "
    "Use it when designing or restyling "
    "a UI, a landing page section or a slide deck and you want real examples to follow; "
    "for general photos or pictures use image_search. A web_section or slide result "
    "carries a group (group_id, name, position, total) for its web page or deck: call "
    "again with that group_id to get that page's or deck's images in group order, at "
    "most {limit} per call (total says how many the group has); query may then be left "
    "out. Show an image by its url and cite its source_page."
)
class _DesignWebOnlyArgs(BaseModel):
    query: str = Field(
        max_length=500,
        description='What to find design references and articles for, e.g. "pricing '
        'section for a SaaS landing page". Max 500 chars.',
    )


# Used when the developer sets images.enable to False: only web results come back.
_DESIGN_WEB_ONLY_DESCRIPTION = (
    "Search the web for design references and articles on a design question: returns web "
    "results (title, url and, when returned, a highlight) on the query. Image results "
    "are turned off for this tool, so group_id is not available; always pass query. Use "
    "it when designing or restyling a UI, a landing page section or a slide deck; for "
    "general photos or pictures use image_search. Cite each result by its url."
)
# The API's default images.count, documented at https://docs.octen.ai/api-reference/design-search.md
_DESIGN_DEFAULT_IMAGE_COUNT = 5

def _materialized(value: Any, limit: Optional[int] = None) -> Any:
    """A list from a one-shot iterator (e.g. a generator); other values unchanged.

    With limit, at most that many items are read, so an endless iterator ends.
    """
    if isinstance(value, Iterator):
        return list(itertools.islice(value, limit))
    return value


def _blank(value: Optional[str]) -> bool:
    return value is None or not value.strip()


# RFC 3986 authority: [userinfo "@"] host [":" port]. The API takes only a domain-name
# host, so there is no bracketed IP form; non-ASCII is allowed for internationalized names.
_NAME = r"(?:[^ \x00-\x1f\x7f\\<>\"'`@\[\]:/?#%]|%[0-9A-Fa-f]{2})"
# RFC 3986 userinfo: unreserved, sub-delims (incl. "'"), ":" and %-escapes; non-ASCII too.
_USERINFO = r"(?:[A-Za-z0-9\-._~!$&'()*+,;=:\u0080-\U0010FFFF]|%[0-9A-Fa-f]{2})*"
_AUTHORITY = re.compile(
    rf"(?:{_USERINFO}@)?{_NAME}+(?::[0-9]*)?"
)
_BAD_PERCENT = re.compile(r"%(?![0-9A-Fa-f]{2})")


def _valid_authority(netloc: str) -> bool:
    return _AUTHORITY.fullmatch(netloc) is not None


def _is_ip_address(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


def _is_https_url(value: str) -> bool:
    """The API's rule: an absolute https URL whose host is a domain name, not an IP."""
    # urlparse silently drops tabs, newlines and leading control characters, so ASCII
    # whitespace and controls are checked first; anything else is left to the parser.
    if any(c == " " or ord(c) < 0x20 or ord(c) == 0x7F for c in value):
        return False
    try:
        parsed = urlparse(value)
        # Escapes are checked in the host and path only: the API's URL parser keeps the
        # query and fragment raw, so a bare % there is accepted.
        if _BAD_PERCENT.search(parsed.netloc + parsed.path + parsed.params):
            return False
        parsed.port  # raises on a non-numeric or out-of-range port
        return (
            parsed.scheme.lower() == "https"
            and bool(parsed.hostname)
            and _valid_authority(parsed.netloc)
            and not _is_ip_address(str(parsed.hostname))
        )
    except ValueError:  # e.g. "Invalid IPv6 URL" for an unclosed bracket
        return False


def _image_rejected(exc: Exception) -> bool:
    from octen import OctenAPIError

    return isinstance(exc, OctenAPIError) and exc.status_code in (413, 415)


def _image_url_error(exc: Exception, rejected: bool, message: Optional[str]) -> str:
    """A fixed message for an image-URL call: upstream text may echo the (signed) URL."""
    status = getattr(exc, "status_code", None)
    request_id = getattr(exc, "request_id", None)
    what = f"HTTP {status}" if status else type(exc).__name__
    if request_id:
        what += f", request_id {request_id}"
    if rejected:
        return f"Image Search rejected the image ({what}). {_IMAGE_REJECTED}"
    text = f"{_image_url_failure(what)}."
    if message is not None and message.endswith(_MAY_HAVE_BEEN_SENT):
        # No retry hint next to a do-not-retry warning.
        return f"{text} {_MAY_HAVE_BEEN_SENT}"
    return f"{text} Try again later, try another image URL, or search by text."


def _image_url_failure(what: str) -> str:
    return (
        f"Image Search by image URL failed ({what}). The upstream message is left out "
        "because it may contain the image URL"
    )


# Metadata kept on a sanitized exception; free text never is.
_SAFE_ERROR_FIELDS = ("status_code", "request_id", "code", "timeout", "request_sent",
                      "retry_after")


def _sanitized(exc: Exception) -> Exception:
    """The same exception class, status and request id, without upstream text or body."""
    from octen import OctenAPIError, OctenError, OctenValidationError

    status = getattr(exc, "status_code", None)
    request_id = getattr(exc, "request_id", None)
    what = type(exc).__name__ + (f", HTTP {status}" if status else "")
    if request_id:
        what += f", request_id {request_id}"
    message = _image_url_failure(what)
    try:
        # A new instance, not a copy: notes and other attributes may carry the URL.
        clean = type(exc).__new__(type(exc))
        clean.args = (message,)
        clean.message = message  # type: ignore[attr-defined]
        for name in _SAFE_ERROR_FIELDS:
            value = getattr(exc, name, None)
            if value is None or isinstance(value, (bool, int, float, str)):
                if hasattr(exc, name):
                    setattr(clean, name, value)
        if isinstance(exc, OctenAPIError):
            clean.response_data = None  # type: ignore[attr-defined]
        if isinstance(exc, OctenValidationError):
            clean.errors = []  # type: ignore[attr-defined]
    except Exception:
        # A class that cannot be rebuilt (e.g. its __new__ needs arguments).
        return OctenError(message)
    return clean


# Image Search's 413 / 415 come from the image the model chose, so the model can retry.
_IMAGE_REJECTED = (
    "The image could not be used. Try another image: a smaller image (at most 5MB) "
    "in a supported format (JPEG, PNG, WEBP, BMP, TIFF, ICO, DIB, ICNS or SGI), "
    "or search by text instead."
)


# Documented at https://docs.octen.ai/api-reference/extract.md
_EXTRACT_DEFAULT_MAX_URLS = 5
_EXTRACT_DEFAULT_URL_TIMEOUT = 30
# Client headroom over the per-URL budget, as in the hosted MCP server.
_EXTRACT_CLIENT_HEADROOM = 90
_EXTRACT_INT_BOUNDS = (
    ("max_urls", 1, 20),
    ("url_timeout", 1, 60),
    ("max_age_seconds", 300, 31_536_000),
)


_SEARCH_TYPE_DEPRECATION = (
    "search_type is deprecated and ignored: the Octen API no longer takes it. "
    "It will be removed in a future release."
)

_COMMON_OPTIONS = frozenset({"count", "highlight", "full_content", "timeout"})

# Developer-owned options per endpoint. Everything else is either model-facing
# or not exposed, so it is rejected at construction.
_ALLOWED_OPTIONS: Dict[str, FrozenSet[str]] = {
    "broad_search_options": _COMMON_OPTIONS | {"max_queries"},
    "news_search_options": _COMMON_OPTIONS | {"subjects"},
    "business_search_options": _COMMON_OPTIONS | {"entities"},
    "pulse_search_options": _COMMON_OPTIONS | {"mode"},
    "place_search_options": frozenset({"places", "web_search", "timeout"}),
    "search_premier_options": _COMMON_OPTIONS
    | {
        "result_type",
        "include_text",
        "exclude_text",
        "language",
        "time_basis",
        "time_range",
        "start_time",
        "end_time",
        "format",
        "safesearch",
        "include_images",
    },
    # url_timeout is the API's per-URL budget, sent as "timeout" in the request.
    "extract_options": frozenset(
        {
            "mode",
            "format",
            "max_age_seconds",
            "url_timeout",
            "include_images",
            "include_videos",
            "include_audio",
            "include_links",
            "max_urls",
            "timeout",
        }
    ),
    "image_search_options": frozenset({"count", "safesearch", "timeout"}),
    # images.group_id is model-facing and rejected here.
    "design_search_options": frozenset({"images", "web_search", "timeout"}),
}

# RFC 3339 date-time, the "format: date-time" of the Web Search Premier API reference.
# Stricter on purpose: uppercase T and Z only, as in the reference's example, and no leap
# second (:60), which Python's datetime cannot represent.
_RFC3339 = re.compile(
    r"(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})"
    r"(?:\.\d+)?(?:Z|[+-](\d{2}):(\d{2}))",
    re.ASCII,
)


def _is_rfc3339(value: Any) -> bool:
    match = _RFC3339.fullmatch(value) if isinstance(value, str) else None
    if match is None:
        return False
    year, month, day, hour, minute, second, off_hour, off_minute = match.groups()
    try:
        datetime.datetime(
            int(year), int(month), int(day), int(hour), int(minute), int(second)
        )
    except ValueError:
        return False
    return off_hour is None or (int(off_hour) <= 23 and int(off_minute) <= 59)


# Values documented at https://docs.octen.ai/api-reference/search-premier.md
_LANGUAGES = frozenset("ar de en es fr hi id it ja ko nl pl pt ru th tr vi zh".split())


_MAY_HAVE_BEEN_SENT = (
    "The request may already have reached the server and been billed; "
    "do not retry it automatically."
)


def _classify(exc: Exception) -> Optional[str]:
    """Return the message to hand to the model, or None if exc must be raised.

    Configuration errors (bad key, no access or balance, invalid developer
    options) go to the developer; transient errors and server 400/422, which
    almost always come from a model-supplied filter, go to the model.
    """
    from octen import (
        OctenAPIError,
        OctenConnectionError,
        OctenRateLimitError,
        OctenTimeoutError,
    )

    if isinstance(exc, (OctenTimeoutError, OctenConnectionError)):
        # request_sent is True or None when the request may have been served.
        if getattr(exc, "request_sent", None) is False:
            return str(exc)
        return f"{exc}. {_MAY_HAVE_BEEN_SENT}"
    if isinstance(exc, OctenRateLimitError):
        return str(exc)
    # 401 (OctenAuthenticationError), 403 and other 4xx fall through to raise.
    if isinstance(exc, OctenAPIError):
        status = exc.status_code
        if status is None:
            return None
        if 200 <= status < 300:
            # Envelope error or malformed body behind a 2xx. An envelope that
            # reports an auth or access failure is still a configuration error.
            data = exc.response_data
            code = data.get("code") if isinstance(data, dict) else None
            return None if code in (401, 403) else f"{exc}. {_MAY_HAVE_BEEN_SENT}"
        if status in (502, 504):
            # Already retried by the SDK; a gateway can send these after the
            # backend has finished and billed the request.
            return f"{exc}. {_MAY_HAVE_BEEN_SENT}"
        if status in (400, 408, 422) or status >= 500:
            return str(exc)
    return None


def _describe(exc: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(p) for p in err['loc']) or 'arguments'}: {err['msg']}"
        for err in exc.errors()
    )


def _check_timeout(label: str, timeout: Any) -> None:
    """Reject a timeout that is not a finite, positive number of seconds."""
    if timeout is None:
        return
    try:
        valid = (
            not isinstance(timeout, bool)
            and isinstance(timeout, (int, float))
            and math.isfinite(timeout)
            and timeout > 0
        )
    except OverflowError:  # an int too large for a float
        valid = False
    if not valid:
        raise ValueError(f"{label}: timeout must be a positive number of seconds, got {timeout!r}")


def _validate_option_values(name: str, options: Dict[str, Any]) -> None:
    """Check developer options with the SDK request model the call would build."""
    import octen

    if name == "extract_options":
        _check_extract_options(options)
        return
    if name == "image_search_options":
        _check_image_options(options)
        return
    if name == "design_search_options":
        _check_design_options(options)
        return
    _check_timeout(name, options.get("timeout"))

    fields = {k: v for k, v in options.items() if k != "timeout"}
    query = "placeholder"
    try:
        if name == "broad_search_options":
            max_queries = fields.pop("max_queries", None)
            octen.SearchOptions(**fields)
            if max_queries is not None:
                octen.BroadSearchRequest(query=query, max_queries=max_queries)
        else:
            request_model = {
                "news_search_options": octen.NewsSearchRequest,
                "business_search_options": octen.BusinessSearchRequest,
                "pulse_search_options": octen.PulseSearchRequest,
                "place_search_options": octen.PlaceSearchRequest,
                "search_premier_options": octen.PremierSearchRequest,
            }[name]
            if name == "search_premier_options":
                _check_premier_options(fields)
            request_model(query=query, **fields)
    except ValidationError as exc:
        raise ValueError(f"{name} has an invalid value: {_describe(exc)}") from exc
    except ValueError as exc:
        raise ValueError(f"{name} has an invalid value: {exc}") from None


def _check_premier_options(options: Dict[str, Any]) -> None:
    """Checks octen's PremierSearchRequest leaves out; each would fail on every call."""
    for key in ("highlight", "full_content"):
        value = options.get(key)
        # octen keeps unknown keys in these, so a typo like "enabled" would do nothing.
        if isinstance(value, BaseModel):
            given = set(value.model_extra or {})
        else:
            given = set(value or {}) if isinstance(value, dict) else set()
        # Keys may be of any type, so compare and report them via repr.
        unknown = sorted(given - {"enable", "max_tokens"}, key=repr)
        if unknown:
            names = ", ".join(repr(k) for k in unknown)
            raise ValueError(f"{key}: unknown option {names}")
    language = options.get("language")
    if language is not None:
        if not isinstance(language, list) or not all(
            isinstance(code, str) and code in _LANGUAGES for code in language
        ):
            codes = ", ".join(sorted(_LANGUAGES))
            raise ValueError(f"language must be a list of: {codes}")
    for key in ("start_time", "end_time"):
        value = options.get(key)
        if value is None:
            continue
        if not _is_rfc3339(value):
            raise ValueError(
                f"{key} must be an RFC 3339 time, e.g. 2025-01-01T00:00:00Z"
            )


def _check_extract_options(options: Dict[str, Any]) -> None:
    """Check extract options; integers are strict so that True or 5.5 is not coerced."""
    import octen

    name = "extract_options"
    _check_timeout(name, options.get("timeout"))
    for key, low, high in _EXTRACT_INT_BOUNDS:
        value = options.get(key)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise ValueError(
                f"{name} has an invalid value: {key} must be an integer from "
                f"{low} to {high}, got {value!r}"
            )
    fields = _extract_sdk_options(options)
    fields.pop("request_timeout", None)
    try:
        octen.ExtractRequest(urls=["https://example.com"], **fields)
    except ValidationError as exc:
        raise ValueError(f"{name} has an invalid value: {_describe(exc)}") from exc


def _strict(name: str, prefix: str, model: Any, value: Any) -> Any:
    """Validate with an SDK model in strict mode, so True or 5.0 is not coerced."""
    if isinstance(value, BaseModel):
        value = value.model_dump(exclude_none=True)
    try:
        return model.model_validate(value, strict=True)
    except ValidationError as exc:
        problems = "; ".join(
            f"{prefix}{'.'.join(str(p) for p in err['loc'])}: {err['msg']}"
            for err in exc.errors()
        )
        raise ValueError(f"{name} has an invalid value: {problems}") from exc


def _check_image_options(options: Dict[str, Any]) -> None:
    import octen

    name = "image_search_options"
    _check_timeout(name, options.get("timeout"))
    fields = {k: v for k, v in options.items() if k != "timeout"}
    placeholder = {"inputs": [{"type": "text", "data": "placeholder"}]}
    _strict(name, "", octen.ImageSearchRequest, {**placeholder, **fields})


def _check_design_options(options: Dict[str, Any]) -> None:
    import octen

    name = "design_search_options"
    _check_timeout(name, options.get("timeout"))
    images = options.get("images")
    if images is not None:
        parsed = _strict(name, "images.", octen.DesignImageOptions, images)
        if parsed.group_id is not None:
            raise ValueError(
                f"{name} does not accept 'images.group_id': the model sets it per call"
            )
    web_search = options.get("web_search")
    if web_search is not None:
        _strict(name, "web_search.", octen.DesignWebSearchOptions, web_search)
        if isinstance(web_search, BaseModel):
            web_search = web_search.model_dump(exclude_none=True)
        try:
            _check_premier_options(web_search)
        except ValueError as exc:
            raise ValueError(f"{name} has an invalid value: web_search.{exc}") from None
        start, end = web_search.get("start_time"), web_search.get("end_time")
        if start is not None and end is not None and _instant(end) < _instant(start):
            raise ValueError(
                f"{name} has an invalid value: web_search.end_time {end!r} is before "
                f"start_time {start!r}"
            )
    if (
        _as_dict(images).get("enable") is False
        and _as_dict(web_search).get("enable") is False
    ):
        raise ValueError(
            f"{name} turns off both images and web_search, so design_search would "
            "return nothing; leave at least one of them enabled"
        )


def _instant(value: str) -> Tuple[int, decimal.Decimal]:
    """An RFC 3339 time already checked by _is_rfc3339, as (UTC whole second, fraction).

    Plain integers, so an offset near year 1 or 9999 cannot overflow datetime, and the
    fraction is kept in full rather than cut to microseconds.
    """
    match = _RFC3339.fullmatch(value)
    if match is None:
        raise ValueError(f"not an RFC 3339 time: {value!r}")
    year, month, day, hour, minute, second, off_hour, off_minute = match.groups()
    fraction = re.search(r"\.(\d+)", value[19:])
    if off_hour is None:
        offset = datetime.timedelta(0)
    else:
        sign = -1 if value[-6] == "-" else 1
        offset = sign * datetime.timedelta(hours=int(off_hour), minutes=int(off_minute))
    days = datetime.date(int(year), int(month), int(day)).toordinal()
    whole = days * 86400 + int(hour) * 3600 + int(minute) * 60 + int(second)
    whole -= int(offset.total_seconds())
    return whole, decimal.Decimal("0." + fraction.group(1)) if fraction else decimal.Decimal(0)


def _as_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, BaseModel):
        return value.model_dump(exclude_none=True)
    return dict(value or {})


def _extract_sdk_options(options: Dict[str, Any]) -> Dict[str, Any]:
    """Map extract_options to the SDK's names: url_timeout -> timeout, timeout -> request_timeout."""
    renamed = {"url_timeout": "timeout", "timeout": "request_timeout"}
    return {renamed.get(k, k): v for k, v in options.items() if k != "max_urls"}


def _invalid_model_args(request_model: Any, model_args: Dict[str, Any]) -> Optional[str]:
    """Validate model-supplied args with the SDK's own request model.

    octen-py raises OctenValidationError for these locally, which would reach
    the developer; a model mistake has to go back to the model instead.
    """
    try:
        request_model(**model_args)
    except ValidationError as exc:
        return f"Invalid arguments: {_describe(exc)}"
    return None


class OctenToolSpec(BaseToolSpec):
    """Octen tool spec - fast, real-time web search API for AI.

    Get an API key at https://octen.ai
    """

    spec_functions = [
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

    def __init__(
        self,
        api_key: Optional[str] = None,
        verbose: bool = True,
        max_results: int = 5,
        max_characters: int = 2000,
        timeout: Optional[float] = None,
        *,
        broad_search_options: Optional[Dict[str, Any]] = None,
        news_search_options: Optional[Dict[str, Any]] = None,
        business_search_options: Optional[Dict[str, Any]] = None,
        pulse_search_options: Optional[Dict[str, Any]] = None,
        place_search_options: Optional[Dict[str, Any]] = None,
        search_premier_options: Optional[Dict[str, Any]] = None,
        extract_options: Optional[Dict[str, Any]] = None,
        image_search_options: Optional[Dict[str, Any]] = None,
        design_search_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize with parameters.

        Args:
            api_key: Octen API key from https://octen.ai. Defaults to the
                OCTEN_API_KEY environment variable.
            verbose: Whether to print search metadata.
            max_results: Default number of results to return.
            max_characters: Max characters for full content retrieval.
            timeout: Request timeout in seconds.
            broad_search_options: Options sent on every broad_search call:
                count, highlight, full_content, max_queries, timeout.
            news_search_options: Options sent on every news_search call:
                count, highlight, full_content, subjects, timeout.
            business_search_options: Options sent on every business_search
                call: count, highlight, full_content, entities, timeout.
            pulse_search_options: Options sent on every pulse_search call:
                count, highlight, full_content, mode, timeout.
            place_search_options: Options sent on every place_search call:
                places, web_search, timeout.
            search_premier_options: Options sent on every search_premier call:
                count, result_type, highlight, full_content, include_text,
                exclude_text, language, time_basis, time_range, start_time,
                end_time, format, safesearch, include_images, timeout.
            extract_options: Options for every extract call: mode, format,
                max_age_seconds, url_timeout (the API's per-URL budget, 1-60),
                include_images, include_videos, include_audio, include_links,
                max_urls (URLs the model may send per call, 1-20, default 5)
                and timeout (the HTTP request timeout per attempt; defaults to
                the spec-level timeout, else (url_timeout or 30) + 90 seconds).
            image_search_options: Options sent on every image_search call:
                count, safesearch, timeout.
            design_search_options: Options sent on every design_search call:
                images (all keys but group_id, which the model sets), web_search,
                timeout.

        Raises:
            ValueError: if no API key is available, or an options dict holds
                a key that is not allowed for its endpoint or an invalid value.
        """
        resolved_key = api_key if api_key is not None else os.environ.get("OCTEN_API_KEY")
        if not resolved_key:
            raise ValueError(
                "No Octen API key: pass api_key or set the OCTEN_API_KEY environment variable."
            )
        _check_timeout("OctenToolSpec", timeout)
        given = {
            "broad_search_options": broad_search_options,
            "news_search_options": news_search_options,
            "business_search_options": business_search_options,
            "pulse_search_options": pulse_search_options,
            "place_search_options": place_search_options,
            "search_premier_options": search_premier_options,
            "extract_options": extract_options,
            "image_search_options": image_search_options,
            "design_search_options": design_search_options,
        }
        self._options: Dict[str, Dict[str, Any]] = {}
        for name, options in given.items():
            # Deep copy: later changes to the caller's dicts or SDK option
            # objects must not alter what is sent or bypass validation.
            options = copy.deepcopy(dict(options or {}))
            # Keys may be of any type, so compare and report them via repr.
            unknown = sorted((k for k in options if k not in _ALLOWED_OPTIONS[name]), key=repr)
            if unknown:
                raise ValueError(
                    f"{name} does not accept {', '.join(repr(k) for k in unknown)}; "
                    f"allowed keys: {', '.join(sorted(_ALLOWED_OPTIONS[name]))}"
                )
            _validate_option_values(name, options)
            self._options[name] = options

        self._api_key = resolved_key
        self._verbose = verbose
        self._max_results = max_results
        self._max_characters = max_characters
        self._timeout = timeout

    def get_metadata_from_fn_name(
        self, fn_name: str, spec_functions: Optional[List[SPEC_FUNCTION_TYPE]] = None
    ) -> Optional[ToolMetadata]:
        """Return explicit metadata for the newer functions.

        The older functions keep the metadata llama-index-core derives from
        their signatures and docstrings.
        """
        if fn_name == "extract":
            description = _EXTRACT_DESCRIPTION.format(max_urls=self._extract_max_urls())
            return ToolMetadata(name=fn_name, description=description, fn_schema=_ExtractArgs)
        if fn_name == "image_search":
            return ToolMetadata(
                name=fn_name, description=_IMAGE_SEARCH_DESCRIPTION, fn_schema=_ImageSearchArgs
            )
        if fn_name == "design_search" and self._design_images_off():
            return ToolMetadata(
                name=fn_name,
                description=_DESIGN_WEB_ONLY_DESCRIPTION,
                fn_schema=_DesignWebOnlyArgs,
            )
        if fn_name == "design_search":
            images = _as_dict(self._options["design_search_options"].get("images"))
            limit = images.get("count") or _DESIGN_DEFAULT_IMAGE_COUNT
            web_off = _as_dict(self._options["design_search_options"].get("web_search")).get(
                "enable"
            ) is False
            web = "" if web_off else " Web results on the same query come back alongside."
            description = _DESIGN_SEARCH_DESCRIPTION.format(limit=limit, web=web)
            return ToolMetadata(name=fn_name, description=description, fn_schema=_DesignSearchArgs)
        model_facing = _MODEL_FACING.get(fn_name)
        if model_facing is None:
            return super().get_metadata_from_fn_name(fn_name, spec_functions=spec_functions)
        description, fn_schema = model_facing
        return ToolMetadata(name=fn_name, description=description, fn_schema=fn_schema)

    def _get_client(self) -> Any:
        """Create a new Octen client.

        Retries after the request is sent are off: such a request may already
        have been served and billed.
        """
        from octen import Octen

        return Octen(api_key=self._api_key, retry_after_send=False)

    def _build_kwargs(
        self,
        query: str,
        num_results: Optional[int] = None,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        time_basis: Optional[str] = None,
        include_text: Optional[List[str]] = None,
        exclude_text: Optional[List[str]] = None,
        safesearch: Optional[str] = None,
        format: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build kwargs for the Octen SDK search call."""
        kwargs: Dict[str, Any] = {
            "query": query,
            "count": num_results or self._max_results,
        }
        if include_domains:
            kwargs["include_domains"] = include_domains
        if exclude_domains:
            kwargs["exclude_domains"] = exclude_domains
        if start_time is not None:
            kwargs["start_time"] = start_time
        if end_time is not None:
            kwargs["end_time"] = end_time
        if time_basis is not None:
            kwargs["time_basis"] = time_basis
        if include_text:
            kwargs["include_text"] = include_text
        if exclude_text:
            kwargs["exclude_text"] = exclude_text
        if safesearch is not None:
            kwargs["safesearch"] = safesearch
        if format is not None:
            kwargs["format"] = format
        if self._timeout is not None:
            kwargs["timeout"] = self._timeout
        return kwargs

    def search(
        self,
        query: str,
        num_results: Optional[int] = None,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        search_type: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        time_basis: Optional[str] = None,
        include_text: Optional[List[str]] = None,
        exclude_text: Optional[List[str]] = None,
        safesearch: Optional[str] = None,
        format: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search the live web for one focused lookup (one fact, page or document); for news use news_search, a company or person business_search, a site's URL pulse_search, a place place_search, a multi-part question broad_search.

        Returns ranked results (title, url, snippet). A sequence of search calls is slower and gives worse coverage than one broad_search fan-out.

        Args:
            query: A natural language search query.
            num_results: Number of results to return. Defaults to max_results.
            include_domains: A list of domains to restrict results to, e.g. ["arxiv.org"].
            exclude_domains: A list of domains to exclude from results.
            search_type: Deprecated and ignored.
            start_time: Start date filter, e.g. "2024-01-01". Use `current_date` to get today's date.
            end_time: End date filter.
            time_basis: Time filter basis - "auto", "published", or "crawled".
            include_text: Text that must appear in results.
            exclude_text: Text to exclude from results.
            safesearch: Safe search mode - "off" or "strict".
            format: Result format - "text" or "markdown".
        """
        if search_type is not None:
            warnings.warn(_SEARCH_TYPE_DEPRECATION, FutureWarning, stacklevel=2)
        kwargs = self._build_kwargs(
            query,
            num_results=num_results,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
            start_time=start_time,
            end_time=end_time,
            time_basis=time_basis,
            include_text=include_text,
            exclude_text=exclude_text,
            safesearch=safesearch,
            format=format,
        )
        client = self._get_client()
        try:
            response = client.search.search(**kwargs)
        finally:
            client.close()

        results = []
        for r in response.results:
            entry: Dict[str, Any] = {}
            if isinstance(r, dict):
                entry = {k: v for k, v in r.items() if v is not None}
            else:
                for field in ("title", "url", "highlight", "authors", "time_published"):
                    val = getattr(r, field, None)
                    if val is not None:
                        entry[field] = val
            results.append(entry)

        if self._verbose:
            print(f"[Octen Tool] Found {len(results)} results for: {query}")
        return results

    def search_and_retrieve_documents(
        self,
        query: str,
        num_results: Optional[int] = None,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        search_type: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        time_basis: Optional[str] = None,
        include_text: Optional[List[str]] = None,
        exclude_text: Optional[List[str]] = None,
        safesearch: Optional[str] = None,
        format: Optional[str] = None,
    ) -> List[Document]:
        """Search and retrieve full page content as Documents.

        Args:
            query: A natural language search query.
            num_results: Number of results to return.
            include_domains: Domains to restrict results to.
            exclude_domains: Domains to exclude.
            search_type: Deprecated and ignored.
            start_time: Start date filter, e.g. "2024-01-01".
            end_time: End date filter.
            time_basis: "auto", "published", or "crawled".
            include_text: Text that must appear in results.
            exclude_text: Text to exclude from results.
            safesearch: Safe search mode - "off" or "strict".
            format: Result format - "text" or "markdown".
        """
        from octen.models.search import FullContentOptions

        if search_type is not None:
            warnings.warn(_SEARCH_TYPE_DEPRECATION, FutureWarning, stacklevel=2)
        kwargs = self._build_kwargs(
            query,
            num_results=num_results,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
            start_time=start_time,
            end_time=end_time,
            time_basis=time_basis,
            include_text=include_text,
            exclude_text=exclude_text,
            safesearch=safesearch,
            format=format,
        )
        kwargs["full_content"] = FullContentOptions(
            enable=True, max_tokens=self._max_characters
        )

        client = self._get_client()
        try:
            response = client.search.search(**kwargs)
        finally:
            client.close()

        if self._verbose:
            print(f"[Octen Tool] Retrieved {len(response.results)} documents for: {query}")

        documents = []
        for r in response.results:
            if isinstance(r, dict):
                text = r.get("full_content") or r.get("highlight") or ""
                title = r.get("title", "")
                url = r.get("url", "")
            else:
                text = getattr(r, "full_content", None) or getattr(r, "highlight", None) or ""
                title = getattr(r, "title", "")
                url = getattr(r, "url", "")
            documents.append(
                Document(text=str(text), metadata={"title": title, "url": url})
            )
        return documents

    def search_and_retrieve_highlights(
        self,
        query: str,
        num_results: Optional[int] = None,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        search_type: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        time_basis: Optional[str] = None,
        include_text: Optional[List[str]] = None,
        exclude_text: Optional[List[str]] = None,
        safesearch: Optional[str] = None,
        format: Optional[str] = None,
        highlight_max_tokens: Optional[int] = 200,
    ) -> List[Document]:
        """Search and retrieve highlighted snippets as Documents.

        Args:
            query: A natural language search query.
            num_results: Number of results to return.
            include_domains: Domains to restrict results to.
            exclude_domains: Domains to exclude.
            search_type: Deprecated and ignored.
            start_time: Start date filter.
            end_time: End date filter.
            time_basis: "auto", "published", or "crawled".
            include_text: Text that must appear in results.
            exclude_text: Text to exclude from results.
            safesearch: Safe search mode - "off" or "strict".
            format: Result format - "text" or "markdown".
            highlight_max_tokens: Max tokens for highlighted snippets.
        """
        from octen.models.search import HighlightOptions

        if search_type is not None:
            warnings.warn(_SEARCH_TYPE_DEPRECATION, FutureWarning, stacklevel=2)
        kwargs = self._build_kwargs(
            query,
            num_results=num_results,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
            start_time=start_time,
            end_time=end_time,
            time_basis=time_basis,
            include_text=include_text,
            exclude_text=exclude_text,
            safesearch=safesearch,
            format=format,
        )
        kwargs["highlight"] = HighlightOptions(
            enable=True, max_tokens=highlight_max_tokens or 200
        )

        client = self._get_client()
        try:
            response = client.search.search(**kwargs)
        finally:
            client.close()

        if self._verbose:
            print(f"[Octen Tool] Retrieved {len(response.results)} highlights for: {query}")

        documents = []
        for r in response.results:
            if isinstance(r, dict):
                text = r.get("highlight") or ""
                title = r.get("title", "")
                url = r.get("url", "")
            else:
                text = getattr(r, "highlight", None) or ""
                title = getattr(r, "title", "")
                url = getattr(r, "url", "")
            documents.append(
                Document(text=str(text), metadata={"title": title, "url": url})
            )
        return documents

    # Endpoints below expose only the query and non-billing filters to the
    # model; count, content options and the like come from the constructor.

    def broad_search(
        self,
        query: str,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        time_range: Optional[TimeRange] = None,
        language: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Broad Search: fan one question out into concurrent sub-queries.

        The description and argument schema the model sees are in
        _MODEL_FACING. Developer options come from broad_search_options.

        Args:
            query: The question, max 500 chars.
            include_domains: Only return results from these domains.
            exclude_domains: Drop results from these domains.
            time_range: Relative time window.
            language: ISO 639-1 codes to filter by.

        Returns:
            {"data", "request_id", "warning"?}, or {"error": ...} for errors
            handed to the model.
        """
        from octen import SearchRequest

        return self._vertical_search(
            "broad_search",
            SearchRequest,
            query=query,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
            time_range=time_range,
            language=language,
        )

    def news_search(
        self,
        query: str,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        time_range: Optional[TimeRange] = None,
        language: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """News Search: ranked articles plus storylines.

        The description and argument schema the model sees are in
        _MODEL_FACING. Developer options come from news_search_options.

        Args:
            query: What to find news about, max 500 chars.
            include_domains: Only return articles from these domains.
            exclude_domains: Drop articles from these domains.
            time_range: Relative time window.
            language: ISO 639-1 codes to filter by.

        Returns:
            {"data", "request_id", "warning"?}, or {"error": ...} for errors
            handed to the model.
        """
        from octen import NewsSearchRequest

        return self._vertical_search(
            "news_search",
            NewsSearchRequest,
            query=query,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
            time_range=time_range,
            language=language,
        )

    def business_search(
        self,
        query: str,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        time_range: Optional[TimeRange] = None,
        language: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Business Search: company and person cards plus ranked sources.

        The description and argument schema the model sees are in
        _MODEL_FACING. Developer options come from business_search_options.

        Args:
            query: The company, person or comparison, max 500 chars.
            include_domains: Only return ranked sources from these domains.
            exclude_domains: Drop ranked sources from these domains.
            time_range: Relative time window for the ranked sources.
            language: ISO 639-1 codes to filter the ranked sources by.

        Returns:
            {"data", "request_id", "warning"?}, or {"error": ...} for errors
            handed to the model.
        """
        from octen import BusinessSearchRequest

        return self._vertical_search(
            "business_search",
            BusinessSearchRequest,
            query=query,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
            time_range=time_range,
            language=language,
        )

    def pulse_search(self, query: str) -> Dict[str, Any]:
        """Pulse Search: ranked candidate URLs for a site or page.

        The description and argument schema the model sees are in
        _MODEL_FACING. Developer options come from pulse_search_options.

        Args:
            query: What to find the address of, max 500 chars.

        Returns:
            {"data", "request_id", "warning"?}, or {"error": ...} for errors
            handed to the model.
        """
        from octen import PulseSearchRequest

        return self._vertical_search("pulse_search", PulseSearchRequest, query=query)

    def place_search(
        self,
        query: str,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        radius: Optional[int] = None,
        city: Optional[str] = None,
        region: Optional[str] = None,
        country: Optional[str] = None,
        timezone: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Place Search: real-world places as cards.

        The description and argument schema the model sees are in
        _MODEL_FACING. Developer options come from place_search_options.

        Args:
            query: What to look for, max 500 chars.
            latitude: WGS84 latitude; must be sent with longitude.
            longitude: WGS84 longitude; must be sent with latitude.
            radius: Metres around the point, 1-50000; needs both coordinates.
            city: Approximate location, sent as user_location.
            region: Approximate location, sent as user_location.
            country: Approximate location, sent as user_location.
            timezone: Approximate location (IANA), sent as user_location.

        Returns:
            {"data", "request_id", "warning"?}, or {"error": ...} for errors
            handed to the model. A "places_unavailable" warning means the
            place engine did not answer.
        """
        from octen import PlaceSearchRequest

        location = {
            key: value
            for key, value in (
                ("city", city),
                ("region", region),
                ("country", country),
                ("timezone", timezone),
            )
            if value is not None and value != ""
        }
        return self._vertical_search(
            "place_search",
            PlaceSearchRequest,
            query=query,
            latitude=latitude,
            longitude=longitude,
            radius=radius,
            user_location={"type": "approximate", **location} if location else None,
        )

    def search_premier(
        self,
        query: str,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Web Search Premier: web search with deeper retrieval and reranking.

        The description and argument schema the model sees are in
        _MODEL_FACING. Developer options come from search_premier_options.

        Args:
            query: The question or search query, max 500 chars.
            include_domains: Only return results from these domains.
            exclude_domains: Drop results from these domains.

        Returns:
            {"data", "request_id", "warning"?}, or {"error": ...} for errors
            handed to the model.
        """
        from octen import PremierSearchRequest

        return self._vertical_search(
            "search_premier",
            PremierSearchRequest,
            query=query,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
        )

    def extract(self, urls: List[str], query: Optional[str] = None) -> Dict[str, Any]:
        """Extract: read web pages by URL.

        The description and argument schema the model sees are built in
        get_metadata_from_fn_name. Developer options come from extract_options.

        Args:
            urls: The page URLs, at most max_urls of them, each up to 2048 chars.
            query: Optional; returns highlights instead of full content.

        Returns:
            {"data", "request_id", "warning"?}, or {"error": ...} for errors
            handed to the model. Failed URLs stay in data with their error.
        """
        from octen import ExtractRequest

        max_urls = self._extract_max_urls()
        # Read at most max_urls + 1 entries from an unsized iterable, so an endless one ends.
        unsized = not hasattr(urls, "__len__") and hasattr(urls, "__iter__")
        if unsized:
            urls = list(itertools.islice(urls, max_urls + 1))
        # Normalize once (a tuple or generator becomes a list), then cap and send that list.
        try:
            urls = _ExtractArgs(urls=urls, query=query).urls
        except ValidationError as exc:
            return {"error": f"Invalid arguments: {_describe(exc)}"}
        if not urls:
            return {"error": f"urls is empty: pass 1 to at most {max_urls} page URLs."}
        if len(urls) > max_urls:
            got = f"more than {max_urls}" if unsized else str(len(urls))
            return {
                "error": f"Got {got} URLs, but this tool reads at most {max_urls} "
                "URLs per call. Nothing was fetched. Split the URLs into several calls "
                f"of at most {max_urls} each, or keep only the ones you need."
            }
        model_args = {k: v for k, v in (("urls", urls), ("query", query)) if v is not None}
        problem = _invalid_model_args(ExtractRequest, model_args)
        if problem is not None:
            return {"error": problem}

        kwargs = _extract_sdk_options(self._options["extract_options"])
        if kwargs.get("request_timeout") is None:
            # A batch can outlast the per-URL budget, so the client waits longer.
            url_timeout = kwargs.get("timeout") or _EXTRACT_DEFAULT_URL_TIMEOUT
            kwargs["request_timeout"] = (
                self._timeout if self._timeout is not None
                else url_timeout + _EXTRACT_CLIENT_HEADROOM
            )
        kwargs.update(model_args)
        return self._call("extract", kwargs, f"{len(urls)} URLs")

    def image_search(
        self,
        query: Optional[str] = None,
        image_url: Optional[str] = None,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Image Search: find images by a text query or a reference image URL.

        The description and argument schema the model sees are set in
        get_metadata_from_fn_name. Developer options come from image_search_options.

        Args:
            query: Text describing the images, max 500 chars.
            image_url: Public https URL, with a domain name, of a reference image.
            include_domains: Only images whose source page is on these domains.
            exclude_domains: Drop images whose source page is on these domains.

        Returns:
            {"data": {"results": [...]}, "request_id", "warning"?}, or {"error": ...}
            for errors handed to the model. Exactly one of query and image_url.
        """
        try:
            args = _ImageSearchArgs(
                query=query,
                image_url=image_url,
                include_domains=_materialized(include_domains, limit=11),
                exclude_domains=_materialized(exclude_domains, limit=11),
            )
        except ValidationError as exc:
            return {"error": f"Invalid arguments: {_describe(exc)}"}
        # Forward the validated lists: a one-shot iterable is used up by validation.
        include_domains, exclude_domains = args.include_domains, args.exclude_domains
        has_query, has_url = not _blank(query), not _blank(image_url)
        if has_query and has_url:
            return {
                "error": "Got both query and image_url, but this tool takes exactly one "
                "input. Nothing was searched. Pass query to search by text, or image_url "
                "to search by image, not both."
            }
        if not has_query and not has_url:
            return {
                "error": "Got neither query nor image_url. Nothing was searched. Pass query "
                "(text describing the images) or image_url (a public image URL), exactly one."
            }
        url = str(image_url) if has_url else None  # sent exactly as given
        if url is not None and not _is_https_url(url):
            return {
                "error": "image_url must be a public https URL with a domain name (no IP "
                "address) of an image. Nothing was searched."
            }
        model_args = {
            k: v
            for k, v in (
                ("query", query if has_query else None),
                ("image_url", url),
                ("include_domains", include_domains),
                ("exclude_domains", exclude_domains),
            )
            if v is not None
        }
        kwargs = self._developer_kwargs("image_search")
        kwargs.update(model_args)
        # A signed image URL can carry a token, so it is never printed.
        subject = query if has_query else "image URL"
        return self._call("image_search", kwargs, str(subject))

    def design_search(
        self, query: Optional[str] = None, group_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Design Search: design references as screenshots with their design values.

        The description and argument schema the model sees are set in
        get_metadata_from_fn_name. Developer options come from design_search_options.

        Args:
            query: What to find design references for, max 500 chars.
                Required unless group_id is set.
            group_id: The group.group_id of an earlier image result; returns
                that web page's or deck's images in group order, up to images.count.

        Returns:
            {"data": {"query"?, "images", "results"}, "request_id", "warning"?},
            or {"error": ...} for errors handed to the model.
        """
        try:
            _DesignSearchArgs(query=query, group_id=group_id)
        except ValidationError as exc:
            return {"error": f"Invalid arguments: {_describe(exc)}"}
        has_query, has_group = not _blank(query), not _blank(group_id)
        if has_group and self._design_images_off():
            if not has_query:
                return {
                    "error": "Image results are turned off for this tool, so group_id cannot "
                    "be used. Nothing was searched. Pass query instead."
                }
            has_group = False  # meaningless without image results, so not sent
        if not has_query and not has_group:
            return {
                "error": "Got neither query nor group_id. Nothing was searched. Pass query "
                "(what to find design references for), or group_id from an earlier "
                "result's group to get that web page's or deck's images."
            }
        kwargs = self._developer_kwargs("design_search")
        if has_group:
            kwargs["images"] = {**_as_dict(kwargs.get("images")), "group_id": group_id}
        if has_query:
            kwargs["query"] = query
        subject = query if has_query else f"group {group_id}"
        return self._call("design_search", kwargs, str(subject))

    def _design_images_off(self) -> bool:
        images = _as_dict(self._options["design_search_options"].get("images"))
        return images.get("enable") is False

    def _developer_kwargs(self, endpoint: str) -> Dict[str, Any]:
        """Developer options for one call, with the timeout resolved."""
        kwargs = dict(self._options[f"{endpoint}_options"])
        timeout = kwargs.pop("timeout", None)
        if timeout is None:
            timeout = self._timeout
        if timeout is not None:
            kwargs["timeout"] = timeout
        return kwargs

    def _extract_max_urls(self) -> int:
        max_urls = self._options["extract_options"].get("max_urls")
        return _EXTRACT_DEFAULT_MAX_URLS if max_urls is None else max_urls

    def _vertical_search(
        self, endpoint: str, request_model: Any, **model_args: Any
    ) -> Dict[str, Any]:
        """Call one of the newer endpoints and shape the result for the model."""
        # A one-shot iterable would be used up by validation and forwarded empty.
        model_args = {k: _materialized(v) for k, v in model_args.items() if v is not None}
        problem = _invalid_model_args(request_model, model_args)
        if problem is not None:
            return {"error": problem}

        kwargs = self._developer_kwargs(endpoint)
        kwargs.update(model_args)
        return self._call(endpoint, kwargs, model_args["query"])

    def _call(self, endpoint: str, kwargs: Dict[str, Any], subject: str) -> Dict[str, Any]:
        """Send one request and shape the result for the model."""
        sanitized: Optional[Exception] = None
        client = self._get_client()
        try:
            if endpoint == "place_search":
                response = client.place_search.search(**kwargs)
            elif endpoint == "extract":
                response = client.extract.extract(**kwargs)
            elif endpoint in ("image_search", "design_search"):
                response = getattr(client, endpoint).search(**kwargs)
            else:
                response = getattr(client.search, endpoint)(**kwargs)
        except Exception as exc:
            message = _classify(exc)
            if endpoint == "image_search":
                rejected = _image_rejected(exc)
                if "image_url" in kwargs and (rejected or message is not None):
                    message = _image_url_error(exc, rejected, message)
                elif rejected:
                    message = f"{exc}. {_IMAGE_REJECTED}"
            if message is None and endpoint == "image_search" and "image_url" in kwargs:
                sanitized = _sanitized(exc)
                logger.error("%s raised %s: %s", endpoint, type(sanitized).__name__, sanitized)
            elif message is None:
                # Agent workflows turn any tool exception into text for the
                # model, so log it where the developer will see it.
                logger.error("%s raised %s: %s", endpoint, type(exc).__name__, exc)
                raise
            else:
                return {"error": message}
        finally:
            client.close()
        if sanitized is not None:
            # Raised outside the handler, so neither __cause__ nor __context__ holds
            # the original exception and its upstream text.
            raise sanitized from None

        extra = getattr(response, "model_extra", None) or {}
        request_id = extra.get("request_id", getattr(response, "request_id", None))
        result: Dict[str, Any] = {"data": response.data, "request_id": request_id}
        warning = getattr(response, "warning", None)
        if warning:
            result["warning"] = warning
        if self._verbose:
            line = f"[Octen Tool] {endpoint} request_id={request_id} for: {subject}"
            if endpoint == "extract":
                # Per-mode billing counts are for the developer, not the model.
                line += f" usage={(response.meta or {}).get('usage')}"
            print(line)
        return result

    def current_date(self) -> str:
        """Return today's date.

        Call this before other functions that take date arguments like start_time or end_time.
        """
        return str(datetime.date.today())
