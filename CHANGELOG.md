# Changelog

## 0.4.1

### Added

- `result_type` as a developer option for Web Search and Broad Search. It
  determines how results are returned: `"standard"` returns results as your
  options specify; `"dynamic"` returns only relevant results, up to `count`, and
  adjusts highlight length per result. With `"dynamic"` a call can return fewer
  results than requested, or none.
  - Web Search: the new keyword-only constructor argument `search_options`
    accepts `result_type` and applies it to `search`,
    `search_and_retrieve_documents` and `search_and_retrieve_highlights`. It is
    unrelated to the `search_options` field of the API's Broad Search request.
  - Broad Search: `broad_search_options` now accepts `result_type`.
  - The model never sees or sets it: no function signature or tool schema
    changes. Any value other than `"standard"` or `"dynamic"` raises
    `ValueError` when the spec is built, and `search_options` accepts no other
    key. The other endpoints' options (news, business, pulse, place) still
    reject it.
  - When it is not set, requests are unchanged.

### Changed

- The `octen` requirement is now `>=0.13.1,<1`, the first release that accepts
  `result_type` in `search()` and `broad_search()`. Upgrading this package also
  upgrades `octen` if you are on 0.13.0.

## 0.4.0

### Upgrading from 0.3.x

Existing functions are unchanged: same parameters, same requests, same return
shapes and errors. The `octen` requirement stays `>=0.13.0,<1`, which already
has both new endpoints. One thing changes:

- **`to_tool_list()` now returns thirteen tools instead of eleven.** An agent
  built with `OctenToolSpec(...).to_tool_list()` also gets `image_search` and
  `design_search`, and the model may call them; they have their own pricing.
  To keep the 0.3.x tool set:

  ```python
  spec = OctenToolSpec()
  tools = spec.to_tool_list(
      spec_functions=[
          f for f in spec.spec_functions if f not in ("image_search", "design_search")
      ]
  )
  ```

### Added

- `image_search`: Image Search, finds images on the web by a text query or by
  a reference image.
  - The model passes exactly one of `query` (text, max 500 characters) or
    `image_url` (a public https URL with a domain name, not an IP address, as
    the API requires), plus optional `include_domains` and `exclude_domains` (at
    most 10 each, matched against each result's source page). Both inputs,
    neither, an `image_url` that is not an https URL with a domain name (or that
    holds ASCII whitespace or control characters), or a value over a limit is
    returned to the model before any request; a valid `image_url` is sent
    exactly as given. Base64 image input is not offered to the model; call the `octen`
    SDK directly for it.
  - `count` (1-10), `safesearch` (`"off"` or `"strict"`) and `timeout` come from
    the new keyword-only constructor argument `image_search_options`, checked
    when the spec is built.
  - An HTTP 413 (image too large) or 415 (unsupported image format) is returned
    to the model with a suggestion to try a smaller image in a supported format
    or a text query, instead of being raised. For a call by `image_url`, errors
    handed to the model are fixed messages with the status and request id, without
    the upstream error text; errors raised to you keep their class, status and
    request id but carry a fixed message, no response body and no chained cause.
    This covers the package's own messages and logs only: LlamaIndex callbacks
    and instrumentation can record the raw `image_url` argument, so disable or
    redact input tracing when passing signed URLs.
- `design_search`: Design Search, finds design references: screenshots of web
  pages, page sections, app screens and slides with their design values, plus
  web results on the same query.
  - The model sets `query` (max 500 characters) and `group_id` (max 127
    characters, as the API requires), taken from an
    earlier result's `group`, to get that web page's or deck's images in group
    order, up to `images.count` (the API default is 5); the tool description
    states that limit. `query` is required unless `group_id` is set; a call with
    neither is returned to the model before any request. With
    `images={"enable": False}` the description and argument schema offer no
    `group_id` and require `query`; a `group_id` call without `query` is returned
    to the model, and one with `query` does not forward `group_id`. With
    `web_search={"enable": False}` the description promises no web results;
    turning off both raises `ValueError`, as does a `web_search` time range that
    ends before it starts.
  - `images` (`enable`, `count` 1-100, `design_facts`, and `snippet` with
    `enable` and `max_tokens` 100-100000; `group_id` is not accepted here),
    `web_search` (`enable`, `count`, `highlight`, `full_content`, domain,
    language and time filters) and `timeout` come from the new keyword-only
    constructor argument `design_search_options`, checked when the spec is
    built. Left out, the API defaults apply, which return `design_facts` and a
    `snippet` of up to 5000 tokens for each image that has reusable code; turn
    them off with `images={"design_facts": False, "snippet": {"enable": False}}`
    or shorten snippets with `snippet.max_tokens`.
- Both return `{"data", "request_id", "warning"?}`, with `data` as the API
  returns it, and split errors as the other functions do. Numbers and booleans
  in both options dicts are checked strictly: `True` or `5.0` is not accepted
  as a count.

### Fixed

- `broad_search`, `news_search`, `business_search` and `search_premier` no
  longer drop `include_domains`, `exclude_domains` or `language` when it is
  passed as a one-shot iterable such as a generator. Validation used to consume
  the iterable, so an empty one was sent and the filter was silently not
  applied. Lists are sent as before.

## 0.3.0

### Upgrading from 0.2.x

Existing functions are unchanged: same parameters, same requests, same return
shapes and errors. Things that do change:

- **`to_tool_list()` now returns eleven tools instead of nine.** An agent built
  with `OctenToolSpec(...).to_tool_list()` also gets `search_premier` and
  `extract`, and the model may call them; they have their own pricing. To keep
  the 0.2.x tool set, pass `spec_functions` to `to_tool_list()` without
  `"search_premier"` and `"extract"`.
- **`octen` is upgraded to 0.13.0 or newer.** If your project pins `octen` below
  0.13.0, dependency resolution fails until you relax that pin.

### Added

- `search_premier`: Web Search Premier, web search with deeper retrieval and
  reranking for hard-to-find specific facts and pages.
  - The model sets only `query`, `include_domains` and `exclude_domains`.
    Result count, `result_type`, highlight / full content, text, language and
    time filters, `format`, `safesearch`, `include_images` and `timeout` come
    from the new keyword-only constructor argument `search_premier_options`,
    checked when the spec is built; `highlight` and `full_content` accept only
    the `enable` and `max_tokens` keys, `language` only the documented ISO
    639-1 codes, and `start_time` / `end_time` must be RFC 3339
    times (e.g. `2025-01-01T00:00:00Z`).
  - Return shape and error handling are the same as the vertical functions.
- `extract`: Extract, reads web pages by URL and returns their clean content.
  - The model sets only `urls` and an optional `query` (highlights instead of
    full content). `mode`, `format`, `max_age_seconds`, `url_timeout` (the
    API's per-URL budget, 1-60), `include_images` / `include_videos` /
    `include_audio`, `include_links`, `timeout` and `max_urls` come from the
    new keyword-only constructor argument `extract_options`, checked when the
    spec is built.
  - `timeout` is the HTTP request timeout passed to the SDK, as in the other
    options dicts. Left unset, it is the spec-level `timeout`, else
    `url_timeout` (or 30) plus 90 seconds. It applies per attempt: the SDK
    still retries 408, 429, 500, 502, 503 and 504 responses, so a call can
    take longer.
  - `max_urls` (1-20, default 5) caps the URLs per call. A call with more, or
    with none, sends no request and tells the model the limit.
  - `data` keeps every result, including failed URLs with their
    `error_message`; match results to inputs by each item's `url`, not by
    position. Content is not truncated. Otherwise the return shape and error
    handling are the same as the vertical functions.

### Changed

- Requires `octen>=0.13.0,<1` (the release that adds `search_premier`).

## 0.2.0

### Upgrading from 0.1.x

Existing code keeps working: the constructor's positional arguments and the
`search`, `search_and_retrieve_documents` and `search_and_retrieve_highlights`
functions have the same parameters, return shapes and errors. Things that do
change:

- **`to_tool_list()` now returns nine tools instead of four.** An agent built
  with `OctenToolSpec(...).to_tool_list()` also gets `broad_search`,
  `news_search`, `business_search`, `pulse_search` and `place_search`, and the
  model may call them; they have their own pricing (Broad Search is billed per
  sub-query). To keep the 0.1.x tool set:

  ```python
  tools = OctenToolSpec().to_tool_list(
      spec_functions=[
          "search",
          "search_and_retrieve_documents",
          "search_and_retrieve_highlights",
          "current_date",
      ]
  )
  ```

- **`octen` is upgraded to 0.12.0 or newer.** If your project pins `octen` below
  0.12.0, dependency resolution fails until you relax that pin.
- **No retry after a request was sent.** A request that timed out or lost its
  connection after it was sent is no longer retried, so it can't be billed a
  second time by the retry. Calls that used to succeed on such a retry now raise
  the SDK error. Retries on status codes (408, 429, 5xx) are unchanged.
- **`search_type` is ignored.** With `octen` 0.5.0 or newer, passing it used to
  raise `TypeError`; it now works and emits a `FutureWarning`. If you relied on
  it with an older `octen`, it no longer has any effect.
- **Stricter constructor checks.** An empty `api_key`, or a `timeout` that is
  not a positive finite number, raises `ValueError` when the spec is built
  instead of failing on the first call. Omitting `api_key` now reads
  `OCTEN_API_KEY`.
- **Reworded `search` description.** Its first line now points the model to
  the new functions, so when it chooses `search` may shift.

### Added

- Five new spec functions: `broad_search`, `news_search`, `business_search`,
  `pulse_search` and `place_search`. The model gets only the query and
  filters; billing-related options and limits (result counts, content
  options, Broad Search's `max_queries` and others) are set by the developer
  through the new `broad_search_options`, `news_search_options`,
  `business_search_options`, `pulse_search_options` and
  `place_search_options` constructor arguments. These bound the cost rather
  than fix it: Broad Search is billed per sub-query, and the number of
  sub-queries it runs, up to `max_queries`, depends on the query.
- Developer options are validated when the spec is built: an unknown key
  or a value the `octen` SDK would reject (for example `count=1000`) raises
  `ValueError`.
- The new functions return `{"data", "request_id", "warning"}` (`warning`
  only when the API sent one). Configuration errors (invalid key, no access
  or balance, other 4xx) are raised and also logged at `ERROR` level on the
  `llama_index.tools.octen` logger. Rate limits, server errors, 400/408/422,
  error or malformed bodies behind a 2xx status, timeouts, connection errors
  and invalid model arguments come back to the model as `{"error": "..."}`.
  For a 502 or 504, a timeout or connection error that may have happened
  after the request was sent, or a 2xx response that carried an error or
  could not be read, that text tells the model the call may already have been
  served and billed and should not be retried automatically.
- LlamaIndex agent workflows catch every exception a tool raises and hand
  its text to the model, so a raised configuration error is not visible to
  you there except through that log. Call a function directly once to check
  your key and account before giving the tools to an agent.
- `api_key` falls back to the `OCTEN_API_KEY` environment variable.

### Changed

- Requires `octen>=0.12.0`.
- Clients are created with `retry_after_send=False`, for the new functions
  and for the existing `search`, `search_and_retrieve_documents` and
  `search_and_retrieve_highlights` alike: a request that timed out or lost
  its connection after it was sent is no longer retried, since it may
  already have been served and billed. Responses with status 408, 429, 500,
  502, 503 or 504 are still retried by the SDK, and a 502 or 504 can arrive
  after the backend finished, so a second charge remains possible.
- The constructor's `timeout` must be a finite number of seconds greater than
  0 (or `None`); other values raise `ValueError` instead of failing on the
  first request.

### Deprecated

- `search_type` on `search`, `search_and_retrieve_documents` and
  `search_and_retrieve_highlights` is accepted but ignored and emits a
  `FutureWarning` (shown by default, unlike `DeprecationWarning`). The Octen
  API no longer takes it, and with
  `octen>=0.5.0` passing it raised `TypeError`. It will be removed in a
  future release.

### Known issues

- The existing functions keep their 0.1.0 behaviour: they still expose
  `num_results`, `start_time` and other options to the model, return lists,
  and raise on errors. They will be aligned with the new functions in a
  future major release.
- Calling a generated tool directly through `FunctionTool.call` with a
  missing `query` or an unknown argument raises `TypeError` instead of
  returning `{"error": ...}`. Agent workflows turn it into text the model
  sees, and the existing functions behave the same way.
- The routing hint in the `search` description names the other functions
  even when only `search` is registered with the agent.
- When octen-py retries a 502 or 504 internally and the final attempt fails
  differently (for example with a 503), the "may already have been billed"
  hint is not added, because the function only sees the final error.
