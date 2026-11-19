"""Octen tool spec - web search API for AI."""

import datetime
import os
import warnings
from typing import Any, Dict, FrozenSet, List, Optional

from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec

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
        if status is not None and (status in (400, 422) or status >= 500):
            return str(exc)
    return None


class OctenToolSpec(BaseToolSpec):
    """Octen tool spec - fast, real-time web search API for AI.

    Get an API key at https://octen.ai
    """

    spec_functions = [
        "search",
        "search_and_retrieve_documents",
        "search_and_retrieve_highlights",
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
                a key that is not allowed for its endpoint.
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
        """Search the web using Octen.

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
            warnings.warn(_SEARCH_TYPE_DEPRECATION, DeprecationWarning, stacklevel=2)
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
            warnings.warn(_SEARCH_TYPE_DEPRECATION, DeprecationWarning, stacklevel=2)
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
            warnings.warn(_SEARCH_TYPE_DEPRECATION, DeprecationWarning, stacklevel=2)
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

    def current_date(self) -> str:
        """Return today's date.

        Call this before other functions that take date arguments like start_time or end_time.
        """
        return str(datetime.date.today())
