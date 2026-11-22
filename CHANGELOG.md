# Changelog

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
