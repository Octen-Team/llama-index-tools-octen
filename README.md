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

## Web Search Premier

`search_premier` searches the web with deeper retrieval and reranking, for questions that need the exact page holding the answer: a specific fact, a detail deep inside a page, a niche document or a named report. It takes the same request options and returns the same response shape as Web Search; for low-latency lookups use `search` instead. It follows the same design as the vertical functions below: the model sets only `query`, `include_domains` and `exclude_domains`, and everything else comes from `search_premier_options`.

```python
tool = OctenToolSpec(
    search_premier_options={"count": 5, "highlight": {"enable": True, "max_tokens": 1024}},
)

result = tool.search_premier("Kafka default retention period for topics")
for r in result["data"]["results"]:
    print(r["title"], r["url"])
```

`search_premier_options` accepts `count` (1-100), `result_type` (`"standard"` or `"dynamic"`), `highlight` and `full_content` (only the `enable` and `max_tokens` keys), `include_text` and `exclude_text` (up to 5, 30 characters each), `language` (ISO 639-1 codes), `time_basis`, `time_range`, `start_time` and `end_time` (RFC 3339, e.g. `2025-01-01T00:00:00Z`), `format`, `safesearch`, `include_images` and `timeout`. `result_type="dynamic"` returns only relevant results, up to `count`, and adjusts the highlight length per result. Return value and errors are as described for the vertical functions below.

## Extract

`extract` reads web pages by URL and returns their clean content. Match each result to its input by the item's `url` field, not by position. The model sets only `urls` and, optionally, `query`: with a query each result has `highlights` (the passages relevant to it) instead of `full_content`. Everything that changes cost or response size comes from `extract_options`.

```python
tool = OctenToolSpec(
    extract_options={"max_urls": 10, "mode": "auto", "url_timeout": 60},
)

result = tool.extract(["https://example.com", "https://docs.octen.ai/overview/welcome"])
for item in result["data"]["results"]:
    if item["status"] == "success":
        print(item["url"], item.get("title"), len(item["full_content"]))
    else:
        print(item["url"], "failed:", item["error_message"])
```

| Option | Values | Default (when left out) |
|---|---|---|
| `max_urls` | URLs the model may pass per call, 1-20 | 5 |
| `mode` | `"standard"`, `"advanced"` or `"auto"` | not sent; the API uses `standard` |
| `format` | `"markdown"` or `"text"` | not sent; the API uses `markdown` |
| `max_age_seconds` | cache age before a page is fetched again, 300-31536000 | not sent; the API uses 86400 |
| `url_timeout` | the API's per-URL fetch timeout in seconds, 1-60; sent as `timeout` | not sent; the API uses 30 |
| `include_images`, `include_videos`, `include_audio` | `True` to return media URLs found on each page | not sent |
| `include_links` | `{"scope": "prefer_internal" \| "prefer_external", "max_links": 1-1000}`; `{}` uses the API defaults | not sent |
| `timeout` | HTTP request timeout passed to the SDK, in seconds, per attempt | spec-level `timeout`, else `url_timeout` (or 30) + 90 |

- As in the other options dicts, `timeout` is the HTTP request timeout passed to the SDK and overrides the spec-level `timeout`. The API's per-URL budget is `url_timeout`. When neither `timeout` is set, each request waits up to `url_timeout` (or the API default of 30) plus 90 seconds, since a batch, and `mode="advanced"` or `"auto"`, can take longer than one URL's budget. It is not a deadline for the whole call: the SDK still retries 408, 429, 500, 502, 503 and 504 responses (see [Retries](#retries)), so a call can take longer across attempts.
- A call with more URLs than `max_urls`, or with none, sends no request: the model gets `{"error": ...}` with the limit and is told to split the URLs into several calls. URLs longer than 2048 characters, empty URLs and a `query` over 500 characters are also returned to the model before any request.
- `data` keeps every result the API returns, failed ones included; match results to inputs by each item's `url`, not by position. A URL that could not be read has `status: "failed"` and an `error_message`, and is not billed; the others still return content. Content is not truncated. Each successful item has `resolved_mode`, the mode it was billed at; with `verbose=True` the request's usage counts (`successful_by_mode`) are printed.
- Errors are split as described for the vertical functions below.

## Image Search and Design Search

`image_search` finds images on the web by a text query or by a reference image. The model passes exactly one of `query` (text) or `image_url` (a public https URL with a domain name, not an IP address, of an image already on the web; the API accepts only these, and the URL is sent exactly as given), plus optional `include_domains` and `exclude_domains`, matched against each result's `source_page`. Base64 image input is not offered to the model, as it is too large for a model to produce; call the `octen` SDK's `image_search.search` directly for that.

`design_search` finds design references: each image result is a screenshot of a web page, page section, app screen or slide with its design values, and web results on the same query come back alongside. The model sets `query` and, to page through a web page or deck, `group_id`, taken from the `group` of an earlier result; `query` is required unless `group_id` is set. A `group_id` call returns that group's images in group order, up to `images.count` (the API default is 5): set `design_search_options={"images": {"count": ...}}` high enough for the groups you expect, since a group can have more images than that (its `total`). The tool description tells the model this limit.

```python
tool = OctenToolSpec(
    image_search_options={"count": 5, "safesearch": "strict"},
    design_search_options={
        "images": {"count": 5, "design_facts": True, "snippet": {"enable": False}},
        "web_search": {"count": 3},
    },
)

result = tool.image_search("golden retriever puppy", include_domains=["wikimedia.org"])
for r in result["data"]["results"]:
    print(r["url"], r["source_page"])

result = tool.design_search("pricing section for a SaaS landing page")
grouped = [i for i in result["data"]["images"] if i.get("group")]
if grouped:
    deck = tool.design_search(group_id=grouped[0]["group"]["group_id"])
```

| Option | Values | Default (when left out) |
|---|---|---|
| `image_search_options["count"]` | 1-10 | not sent; the API returns 5 |
| `image_search_options["safesearch"]` | `"off"` or `"strict"` | not sent; the API uses `strict` |
| `design_search_options["images"]` | `enable`, `count` (1-100), `design_facts`, `snippet` (`enable`, `max_tokens` 100-100000); not `group_id`, which the model sets | not sent; the API returns 5 images with `design_facts`, and a `snippet` (up to 5000 tokens) for each image that has reusable code |
| `design_search_options["web_search"]` | `enable`, `count` (1-100), `highlight`, `full_content` (only `enable` and `max_tokens`), `include_domains`, `exclude_domains`, `language`, `time_basis`, `time_range`, `start_time`, `end_time` (RFC 3339) | not sent; the API returns 5 web results with highlights |
| `timeout` (both) | HTTP request timeout passed to the SDK, in seconds | spec-level `timeout` |

- Both inputs to `image_search`, neither, an `image_url` that is not an https URL with a domain name or that holds ASCII whitespace or control characters, a `design_search` call with neither `query` nor `group_id`, a query over 500 characters, a `group_id` over 127 characters, or more than 10 domains is returned to the model as `{"error": ...}` before any request is sent.
- An HTTP 413 (image too large) or 415 (unsupported image format) from Image Search is returned to the model with a suggestion to try a smaller image in a supported format, or a text query; it is not raised. Other errors are split as for the vertical functions below. For a call by `image_url`, every error handed to the model is a fixed message with the status and request id, without the upstream error text, which may echo a signed image URL. Errors raised to you on such a call keep their class, status and request id, but carry the same kind of fixed message, no response body and no chained upstream exception; the log line is sanitized the same way, and the verbose output names only the input type. This covers this package's own error messages, log lines and verbose output only: LlamaIndex callbacks and instrumentation can record the raw tool arguments, including `image_url`, before the function runs, so if you pass signed image URLs, disable or redact input tracing.
- With `design_search_options={"images": {"enable": False}}` only web results come back: the tool description says so, the argument schema requires `query` and has no `group_id`, a call with `group_id` but no `query` is returned to the model, and a `group_id` sent alongside a `query` is not forwarded.
- With `"web_search": {"enable": False}` the description promises no web results. Turning off both `images` and `web_search` raises `ValueError`, since the tool would return nothing.
- Options are checked when the spec is built, with strict types: `True` or `5.0` is not accepted as a count. A `web_search` time range whose `end_time` is before its `start_time` raises `ValueError`.
- **Design Search output is large by default.** Left to the API defaults, each image result carries its `design_facts` and, when it has reusable code, a `snippet` of up to 5000 tokens, and all of it reaches the model. Turn them off with `design_search_options={"images": {"design_facts": False, "snippet": {"enable": False}}}`, or shorten snippets with `"snippet": {"max_tokens": 500}`.
- Return value and errors are as described for the vertical functions below: `data` is the API response data as returned, including thumbnails, each design image's `group` (with the `group_id` the model passes back) and the web results' extra fields.

## Broad, News, Business, Pulse and Place Search

These five functions give the model only the query and filters. Billing-related options and limits, such as result counts, content options and Broad Search's `max_queries`, are set by you when you build the spec. They bound the cost of a call rather than fix it: Broad Search, for example, is billed per sub-query, and `max_queries` is an upper limit; how many sub-queries a call actually runs depends on the query.

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
| `search_premier` | `query`, `include_domains`, `exclude_domains` | `count`, `result_type`, `highlight`, `full_content`, `include_text`, `exclude_text`, `language`, `time_basis`, `time_range`, `start_time`, `end_time`, `format`, `safesearch`, `include_images`, `timeout` |
| `extract` | `urls`, `query` | `max_urls`, `mode`, `format`, `max_age_seconds`, `url_timeout`, `include_images`, `include_videos`, `include_audio`, `include_links`, `timeout` (see [Extract](#extract)) |
| `image_search` | `query` or `image_url` (exactly one), `include_domains`, `exclude_domains` | `count`, `safesearch`, `timeout` (see [Image Search and Design Search](#image-search-and-design-search)) |
| `design_search` | `query`, `group_id` | `images`, `web_search`, `timeout` |

- Options you leave out are not sent, so the API defaults apply. Values are passed to the [`octen`](https://pypi.org/project/octen/) SDK as-is (`dict`s or the SDK's option models). They are checked when the spec is built: a key that is not listed for a function, or a value the SDK would reject (for example `count=1000`), raises `ValueError`. `timeout` in an options dict overrides the spec-level `timeout` for that function.
- `time_range` is one of `day`, `week`, `month`, `year` (or `d`, `w`, `m`, `y`). `language` is a list of ISO 639-1 codes.
- `place_search` needs `latitude` and `longitude` together, and `radius` only with both. `city`, `region`, `country` and `timezone` are sent as an approximate `user_location`, for when coordinates are unknown.

**Return value.** `{"data": ..., "request_id": "...", "warning": "..."}`. `data` is the API response data as returned, including endpoint-specific parts such as `subjects`, `entities`, `places` and `queries`. `warning` is present only when the API sent one, for example `places_unavailable` from `place_search`.

**Errors.**

- Raised to you (configuration problems): invalid API key (401), no access or insufficient balance (403), and other 4xx responses. Each one is also logged at `ERROR` level on the `llama_index.tools.octen` logger.
- Returned to the model as `{"error": "..."}`, so the agent can adjust or decide what to do next: rate limits (429), server errors (5xx), 400/408/422 responses, error or malformed bodies behind a 2xx status, timeouts, connection errors, and invalid model arguments (for example `time_range="last week"`, a latitude without a longitude, or a radius without a point). Invalid model arguments are caught before any request is sent. For a 502 or 504 (which the SDK has already retried), a timeout or connection error after the request may have been sent, or a 2xx response that carried an error or could not be read, the error text tells the model the call may already have been served and billed and should not be retried automatically.

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

- `search_premier`: Web search with deeper retrieval and reranking, for hard-to-find specific facts and pages.

- `extract`: Read web pages by URL and return their clean content, with failed URLs reported per URL.

- `image_search`: Find images on the web by a text query or a reference image URL.

- `design_search`: Design references (web pages, page sections, app screens, slides) with their design values, plus web results.

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
| `search_premier_options` | `dict` | `None` | Keyword-only. Options for `search_premier` |
| `extract_options` | `dict` | `None` | Keyword-only. Options for `extract` |
| `image_search_options` | `dict` | `None` | Keyword-only. Options for `image_search` |
| `design_search_options` | `dict` | `None` | Keyword-only. Options for `design_search` |

### Retries

The spec creates its `octen` clients with `retry_after_send=False`, for all functions including `search`, `search_and_retrieve_documents` and `search_and_retrieve_highlights`. This stops the SDK from retrying a request that timed out or lost its connection after it was sent, which may already have been served and billed. It does not rule out a second charge: failures before the request is sent are still retried, and so are 408, 429, 500, 502, 503 and 504 responses. A gateway can return 502 or 504 after the backend has finished, so a status-code retry can still be billed twice. If you also call the `octen` SDK directly, consider passing `retry_after_send=False` there too.

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
