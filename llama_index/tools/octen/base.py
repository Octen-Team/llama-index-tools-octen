"""Octen tool spec - web search API for AI."""

import datetime
import logging
import os
import warnings
from typing import Any, Dict, FrozenSet, List, Literal, Optional

from llama_index.core.bridge.pydantic import ValidationError
from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec

logger = logging.getLogger("llama_index.tools.octen")

TimeRange = Literal["day", "week", "month", "year", "d", "w", "m", "y"]

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

    if isinstance(exc, (OctenRateLimitError, OctenTimeoutError, OctenConnectionError)):
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
            return None if code in (401, 403) else str(exc)
        if status in (400, 408, 422) or status >= 500:
            return str(exc)
    return None


def _describe(exc: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(p) for p in err['loc']) or 'arguments'}: {err['msg']}"
        for err in exc.errors()
    )


def _validate_option_values(name: str, options: Dict[str, Any]) -> None:
    """Check developer options with the SDK request model the call would build."""
    import octen

    timeout = options.get("timeout")
    if timeout is not None and (
        isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0
    ):
        raise ValueError(f"{name}: timeout must be a positive number of seconds, got {timeout!r}")

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
        given = {
            "broad_search_options": broad_search_options,
            "news_search_options": news_search_options,
            "business_search_options": business_search_options,
            "pulse_search_options": pulse_search_options,
            "place_search_options": place_search_options,
        }
        self._options: Dict[str, Dict[str, Any]] = {}
        for name, options in given.items():
            options = dict(options or {})
            unknown = sorted(set(options) - _ALLOWED_OPTIONS[name])
            if unknown:
                raise ValueError(
                    f"{name} does not accept {', '.join(unknown)}; "
                    f"allowed keys: {', '.join(sorted(_ALLOWED_OPTIONS[name]))}"
                )
            _validate_option_values(name, options)
            self._options[name] = options

        self._api_key = resolved_key
        self._verbose = verbose
        self._max_results = max_results
        self._max_characters = max_characters
        self._timeout = timeout

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
        """Search the live web and return ranked results (title, url, snippet) for one focused lookup: one fact, one page, one document you do not already know the address of.

        For news use news_search; for anything about a company or the people in one (what it does, who runs it, its revenue or share price, what it has announced) use business_search; when all you need is the URL of a site or docs page you could almost name, use pulse_search. If the question spans several independent subtopics, use broad_search instead: a sequence of search calls is slower and gives worse coverage than one fan-out.

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
        """Search the web across many angles in one call, for comparisons, research, surveys and questions with several distinct parts. Expands the question into multiple sub-queries and runs them concurrently.

        USE WHEN the question has multiple distinct parts or entities that one search cannot cover: comparing vendors, products or pricing across many sources; literature reviews, market or landscape surveys; open-ended "what are the options for X" or "how do people solve Y"; a question that clearly decomposes into 3+ independent sub-questions; multi-angle questions about recent events (do NOT loop news_search).

        DO NOT USE for anything another tool answers directly, since fanning out costs several times more and returns pages where those return the answer: a single fact, entity or document (use search); a company or the people in one, including its financials (use business_search); news or recent events on a single topic (use news_search); a real-world place, address or opening hours (use place_search); the URL of a site or docs page you could almost name (use pulse_search); a straight A-vs-B comparison of two known entities (two targeted search calls are cheaper). Do NOT call broad_search twice after a disappointing result; follow up with a targeted search on the specific gaps.

        COST: roughly one search per sub-query, and notably higher latency than a single search. When in doubt, prefer search.

        Results are grouped per sub-query and NOT deduplicated: the same URL may appear under several sub-queries. The filters apply to every sub-query.

        Args:
            query (str): One natural-language question, max 500 chars. Resolve pronouns from the conversation first; do NOT pre-split it into sub-queries.
            include_domains (Optional[List[str]]): Only return results from these domains, e.g. ["arxiv.org"].
            exclude_domains (Optional[List[str]]): Drop results from these domains.
            time_range (Optional[TimeRange]): Relative time window, one of "day", "week", "month", "year" (or "d", "w", "m", "y").
            language (Optional[List[str]]): ISO 639-1 codes to filter results by, e.g. ["en", "ja"]. Supported: ar, de, en, es, fr, hi, id, it, ja, ko, nl, pl, pt, ru, th, tr, vi, zh.

        Returns:
            {"data": ..., "request_id": ..., "warning": ...} with the API data as returned (warning only when present), or {"error": ...} when the call failed and may be retried or adjusted.
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
        """Search recent news and return ranked articles (title, url, snippet): current events, headlines, timely reporting. Alongside the articles it normally returns subjects: the coverage grouped into storylines, each with a summary and the span it covers, so a "what happened with X" question is answerable without reading every article.

        include_domains and exclude_domains narrow the ranked articles only; the storylines are built from the open web either way. When the question restricts its sources ("what is Reuters saying about X"), answer from the articles, not the storylines.

        USE FOR what is being reported right now: breaking events, announcements, coverage of a company or person over the last days or weeks. For a company or person profile (financials, key people, career history) use business_search, which returns structured cards instead of articles. For non-news web pages use search. For a multi-angle question spanning several independent subtopics, use broad_search rather than looping this tool.

        Args:
            query (str): What to find news about, max 500 chars.
            include_domains (Optional[List[str]]): Only return articles from these domains, e.g. ["reuters.com"].
            exclude_domains (Optional[List[str]]): Drop articles from these domains.
            time_range (Optional[TimeRange]): Relative time window, one of "day", "week", "month", "year" (or "d", "w", "m", "y").
            language (Optional[List[str]]): ISO 639-1 codes to filter articles by, e.g. ["en", "ja"]. Supported: ar, de, en, es, fr, hi, id, it, ja, ko, nl, pl, pt, ru, th, tr, vi, zh.

        Returns:
            {"data": ..., "request_id": ..., "warning": ...} with the API data as returned (warning only when present), or {"error": ...} when the call failed and may be retried or adjusted.
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
        """Look up a company or a person and get a structured card back, plus the ranked sources behind it. A company card carries the profile, identifiers, financial and stock metrics, key people and recent announcements; a person card carries the current position, career history and recent activity. Name two companies in one query and you get a card for each, same fields, ready to compare.

        USE FOR any question about a company or the people in one, including the numbers: revenue, net income, gross profit by quarter and year, share price and 52-week range, funding rounds and who led them, headcount and web traffic, key people and their titles, career history, recent announcements. Also "what does this company do", due diligence, and how two of them compare. The card already holds those figures, so "what was their revenue last quarter" is one call here; reaching for search means reading a dozen pages for numbers this returns directly.

        NOT FOR breaking coverage of an event (use news_search) or general web pages (use search). The domain, time and language filters narrow the ranked sources only, never the cards.

        Examples: "NVIDIA company overview", "Lisa Su AMD CEO", "compare Stripe and Adyen payment processing".

        Args:
            query (str): The company, person or comparison to look up, max 500 chars.
            include_domains (Optional[List[str]]): Only return ranked sources from these domains.
            exclude_domains (Optional[List[str]]): Drop ranked sources from these domains.
            time_range (Optional[TimeRange]): Relative time window for the ranked sources, one of "day", "week", "month", "year" (or "d", "w", "m", "y").
            language (Optional[List[str]]): ISO 639-1 codes to filter the ranked sources by, e.g. ["en"]. Supported: ar, de, en, es, fr, hi, id, it, ja, ko, nl, pl, pt, ru, th, tr, vi, zh.

        Returns:
            {"data": ..., "request_id": ..., "warning": ...} with the API data as returned (warning only when present), or {"error": ...} when the call failed and may be retried or adjusted.
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
        """When what is wanted is a web address rather than what is written on the page, this is the tool. It returns ranked candidate URLs with titles: the single most likely one, the top three, or a shortlist to choose from, whichever the question calls for. Official docs and API references ("github docs", "uv documentation", "python list methods"), help centres ("how do I return an amazon order"), a site you know exists ("youtube").

        Prefer it over search for those: one inference predicts the URLs instead of searching and ranking them, so it answers in about 10ms for a fraction of the cost.

        NOT FOR news or anything needing freshness (use news_search). NOT FOR a question that needs real retrieval across the open web, domain or language filters, or a time window: this tool has none of those, sees no fresh content, and is not a cheaper substitute for search.

        URLs may come back without a scheme (e.g. "github.com/docs"); prepend "https://" before using one as a link.

        Args:
            query (str): What to find the destination for, e.g. "github docs". Max 500 chars.

        Returns:
            {"data": ..., "request_id": ..., "warning": ...} with the API data as returned (warning only when present), or {"error": ...} when the call failed and may be retried or adjusted.
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
        """Find real-world places and get each one back as a card: name, category, address, coordinates, phone, website, opening hours, status (open, closed, permanently closed), rating and review count. Pass latitude and longitude to search around a point, and each card then carries its distance in metres; add radius to cap how far. When you only know the city, pass city, region, country or timezone instead (no distance comes back that way). A query that names its own area ("ramen in Shinjuku") works without either.

        USE FOR anything with a location: "coffee shops near me", "family-friendly restaurants", "museums in Lisbon", "is this place open on Sunday", "what's their phone number". The cards answer these directly; search would return pages about places that then have to be read.

        Pass the caller's location when the question is relative to them ("near me", "nearby", "closest") or names no area at all: without it such a query is not limited to anywhere. A query that names its own area does not need it.

        Args:
            query (str): What to look for, e.g. "coffee shops" or "ramen in Shinjuku". Max 500 chars.
            latitude (Optional[float]): WGS84 latitude to search around, -90 to 90. Must be sent with longitude.
            longitude (Optional[float]): WGS84 longitude to search around, -180 to 180. Must be sent with latitude.
            radius (Optional[int]): Keep only places within this many metres of the point, 1 to 50000. Requires latitude and longitude.
            city (Optional[str]): Approximate location when coordinates are unknown: city name.
            region (Optional[str]): Approximate location: state, province or region.
            country (Optional[str]): Approximate location: country name or code.
            timezone (Optional[str]): Approximate location: IANA timezone, e.g. "Asia/Tokyo".

        Returns:
            {"data": ..., "request_id": ..., "warning": ...} with the API data as returned (warning only when present, e.g. "places_unavailable"), or {"error": ...} when the call failed and may be retried or adjusted.
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
            if value
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
