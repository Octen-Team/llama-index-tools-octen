# LlamaIndex Tools Integration: Octen

This tool connects to [Octen](https://octen.ai), a fast real-time web search API for AI, to enable your agent to search and retrieve content from the Internet.

To begin, you need to obtain an API key at [octen.ai](https://octen.ai). Pass it as `api_key`, or set the `OCTEN_API_KEY` environment variable.

## Installation

```bash
pip install llama-index-tools-octen
```

## Usage

```python
import os
from llama_index.tools.octen import OctenToolSpec
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

octen_tool = OctenToolSpec()  # reads OCTEN_API_KEY
agent = FunctionAgent(
    tools=octen_tool.to_tool_list(),
    llm=OpenAI(model="gpt-4.1"),
)

response = await agent.run(
    "What are the latest developments in AI?"
)
print(response)
```

### Basic Search

```python
from llama_index.tools.octen import OctenToolSpec

tool = OctenToolSpec(api_key=os.environ["OCTEN_API_KEY"])

results = tool.search("latest AI research papers", num_results=5)
for r in results:
    print(r["title"], r["url"])
```

### Search with Domain Filtering

```python
results = tool.search(
    "machine learning",
    include_domains=["arxiv.org", "papers.nips.cc"],
)

results = tool.search(
    "Python tutorial",
    exclude_domains=["youtube.com", "reddit.com"],
)
```

### Search with Time Filter

```python
tool = OctenToolSpec(api_key=os.environ["OCTEN_API_KEY"])

results = tool.search(
    "Python release notes",
    start_time="2025-01-01",
    end_time="2026-03-31",
    time_basis="published",
)
```

### Retrieve Full Page Content

```python
tool = OctenToolSpec(api_key=os.environ["OCTEN_API_KEY"], max_characters=3000)

# Returns List[Document], each Document.text contains the full page content
docs = tool.search_and_retrieve_documents(
    query="Python 3.13 new features",
    num_results=3,
    include_domains=["docs.python.org"],
)

for doc in docs:
    print(doc.metadata["title"], doc.metadata["url"])
    print(doc.text[:200])
```

### Retrieve Highlighted Snippets

```python
# Returns List[Document], each Document.text contains a concise snippet
highlights = tool.search_and_retrieve_highlights(
    query="LlamaIndex agent tutorial",
    num_results=5,
    highlight_max_tokens=300,
)

for doc in highlights:
    print(doc.metadata["title"])
    print(doc.text)
```

### Text Filters and Safe Search

```python
results = tool.search(
    "Python web framework",
    include_text=["async"],
    exclude_text=["deprecated"],
    safesearch="strict",
    format="markdown",
)
```

## Broad, News, Business, Pulse and Place Search

These five functions give the model only the query and filters that do not change cost. Everything that decides how much comes back, and so what a call costs, is set by you when you build the spec:

```python
tool = OctenToolSpec(
    news_search_options={"count": 5, "subjects": {"count": 2}},
    place_search_options={"places": {"count": 3}},
    broad_search_options={"max_queries": 5, "timeout": 90},
)

tool.news_search("EU AI Act enforcement", time_range="week")
tool.place_search("coffee shops", latitude=35.6812, longitude=139.7671, radius=1000)
```

| Function | Arguments the model can set | Developer options (`<function>_options`) |
|---|---|---|
| `broad_search` | `query`, `include_domains`, `exclude_domains`, `time_range`, `language` | `count`, `highlight`, `full_content`, `max_queries`, `timeout` |
| `news_search` | `query`, `include_domains`, `exclude_domains`, `time_range`, `language` | `count`, `highlight`, `full_content`, `subjects`, `timeout` |
| `business_search` | `query`, `include_domains`, `exclude_domains`, `time_range`, `language` | `count`, `highlight`, `full_content`, `entities`, `timeout` |
| `pulse_search` | `query` | `count`, `highlight`, `full_content`, `mode`, `timeout` |
| `place_search` | `query`, `latitude`, `longitude`, `radius`, `city`, `region`, `country`, `timezone` | `places`, `web_search`, `timeout` |

- Options you leave out are not sent, so the API defaults apply. Values are passed to the [`octen`](https://pypi.org/project/octen/) SDK as-is (`dict`s or the SDK's option models). They are checked when the spec is built: a key that is not listed for a function, or a value the SDK would reject (for example `count=1000`), raises `ValueError`. `timeout` in an options dict overrides the spec-level `timeout` for that function.
- `time_range` is one of `day`, `week`, `month`, `year` (or `d`, `w`, `m`, `y`). `language` is a list of ISO 639-1 codes.
- `place_search` needs `latitude` and `longitude` together, and `radius` only with both. `city`, `region`, `country` and `timezone` are sent as an approximate `user_location`, for when coordinates are unknown.

**Return value.** `{"data": ..., "request_id": "...", "warning": "..."}`. `data` is the API response data as returned, including endpoint-specific parts such as `subjects`, `entities`, `places` and `queries`. `warning` is present only when the API sent one, for example `places_unavailable` from `place_search`.

**Errors.**

- Raised to you (configuration problems): invalid API key (401), no access or insufficient balance (403), and other 4xx responses. Each one is also logged at `ERROR` level on the `llama_index.tools.octen` logger.
- Returned to the model as `{"error": "..."}`, so the agent can retry or adjust: rate limits (429), server errors (5xx), 400/408/422 responses, error or malformed bodies behind a 2xx status, timeouts, connection errors, and invalid model arguments (for example `time_range="last week"`, a latitude without a longitude, or a radius without a point). Invalid model arguments are caught before any request is sent.

> **Agent workflows hide raised errors.** LlamaIndex agent workflows (`FunctionAgent`, `ReActAgent`, `AgentWorkflow`) catch every exception a tool raises and pass its text to the model, so an invalid key or an exhausted balance shows up only as a failed tool call. Watch the `llama_index.tools.octen` logger, and check your setup by calling a function directly once before handing the tools to an agent:
>
> ```python
> import logging
> logging.basicConfig(level=logging.INFO)
>
> tool = OctenToolSpec(pulse_search_options={"count": 1})
> print(tool.pulse_search("github docs"))  # raises here if the key or account is not usable
> ```

## Available Tools

- `search`: Search the web using Octen for a list of results matching a natural language query.

- `search_and_retrieve_documents`: Search and retrieve full page content as LlamaIndex Documents.

- `search_and_retrieve_highlights`: Search and retrieve highlighted snippets as LlamaIndex Documents.

- `broad_search`: Fan one question out into several sub-queries for comparisons, research and surveys.

- `news_search`: Recent news articles, grouped into storylines.

- `business_search`: Company and person cards with financials, key people and recent activity, plus ranked sources.

- `pulse_search`: Find the URL of a site or docs page you could almost name.

- `place_search`: Real-world places as cards with address, opening hours, rating and distance.

- `current_date`: Utility for the agent to get today's date (useful for time-filtered searches).

## Configuration

| Parameter | Type | Default | Description |
|---|---|---|---|
| `api_key` | `str` | `OCTEN_API_KEY` | Octen API key; falls back to the environment variable, `ValueError` if neither is set |
| `verbose` | `bool` | `True` | Print search metadata |
| `max_results` | `int` | `5` | Default number of results |
| `max_characters` | `int` | `2000` | Max characters for full content retrieval |
| `timeout` | `float` | `None` | Request timeout in seconds |
| `broad_search_options` | `dict` | `None` | Keyword-only. Options for `broad_search` (see above) |
| `news_search_options` | `dict` | `None` | Keyword-only. Options for `news_search` |
| `business_search_options` | `dict` | `None` | Keyword-only. Options for `business_search` |
| `pulse_search_options` | `dict` | `None` | Keyword-only. Options for `pulse_search` |
| `place_search_options` | `dict` | `None` | Keyword-only. Options for `place_search` |

### Retries

The spec creates its `octen` clients with `retry_after_send=False`, for all functions including `search`, `search_and_retrieve_documents` and `search_and_retrieve_highlights`: a request that timed out or lost its connection after it was sent is not retried, because it may already have been served and billed. Failures before the request is sent, and 408/429/5xx responses, are still retried. If you also call the `octen` SDK directly, consider passing `retry_after_send=False` there too.

### Per-query Parameters

`search`, `search_and_retrieve_documents` and `search_and_retrieve_highlights` support:

| Parameter | Type | Description |
|---|---|---|
| `query` | `str` | Natural language search query |
| `num_results` | `int` | Number of results (overrides default) |
| `include_domains` | `list[str]` | Restrict to these domains |
| `exclude_domains` | `list[str]` | Exclude these domains |
| `search_type` | `str` | Deprecated and ignored; emits a `FutureWarning` |
| `start_time` | `str` | Start date filter (ISO 8601) |
| `end_time` | `str` | End date filter (ISO 8601) |
| `time_basis` | `str` | `auto`, `published`, or `crawled` |
| `include_text` | `list[str]` | Text that must appear in results |
| `exclude_text` | `list[str]` | Text to exclude from results |
| `safesearch` | `str` | `off` or `strict` |
| `format` | `str` | `text` or `markdown` |

`search_and_retrieve_highlights` additionally supports:

| Parameter | Type | Description |
|---|---|---|
| `highlight_max_tokens` | `int` | Max tokens for highlighted snippets (default: 200) |
