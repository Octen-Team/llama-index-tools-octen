"""Octen tool spec - web search API for AI."""

import copy
import datetime
import logging
import math
import os
import warnings
from typing import Any, Dict, FrozenSet, List, Literal, Optional, Tuple, Type

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
}


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
}


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
            }[name]
            request_model(query=query, **fields)
    except ValidationError as exc:
        raise ValueError(f"{name} has an invalid value: {_describe(exc)}") from exc


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

    def _vertical_search(
        self, endpoint: str, request_model: Any, **model_args: Any
    ) -> Dict[str, Any]:
        """Call one of the newer endpoints and shape the result for the model."""
        model_args = {k: v for k, v in model_args.items() if v is not None}
        problem = _invalid_model_args(request_model, model_args)
        if problem is not None:
            return {"error": problem}

        kwargs = dict(self._options[f"{endpoint}_options"])
        timeout = kwargs.pop("timeout", None)
        if timeout is None:
            timeout = self._timeout
        if timeout is not None:
            kwargs["timeout"] = timeout
        kwargs.update(model_args)

        client = self._get_client()
        try:
            if endpoint == "place_search":
                response = client.place_search.search(**kwargs)
            else:
                response = getattr(client.search, endpoint)(**kwargs)
        except Exception as exc:
            message = _classify(exc)
            if message is None:
                # Agent workflows turn any tool exception into text for the
                # model, so log it where the developer will see it.
                logger.error("%s raised %s: %s", endpoint, type(exc).__name__, exc)
                raise
            return {"error": message}
        finally:
            client.close()

        extra = getattr(response, "model_extra", None) or {}
        request_id = extra.get("request_id", getattr(response, "request_id", None))
        result: Dict[str, Any] = {"data": response.data, "request_id": request_id}
        warning = getattr(response, "warning", None)
        if warning:
            result["warning"] = warning
        if self._verbose:
            print(f"[Octen Tool] {endpoint} request_id={request_id} for: {model_args['query']}")
        return result

    def current_date(self) -> str:
        """Return today's date.

        Call this before other functions that take date arguments like start_time or end_time.
        """
        return str(datetime.date.today())
