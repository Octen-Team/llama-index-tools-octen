# Changelog

## 0.2.0

### Added

- Five new spec functions: `broad_search`, `news_search`, `business_search`,
  `pulse_search` and `place_search`. The model gets only the query and
  filters that do not change cost; result counts, content options and other
  billing-relevant options are set by the developer through the new
  `broad_search_options`, `news_search_options`, `business_search_options`,
  `pulse_search_options` and `place_search_options` constructor arguments.
- Developer options are validated when the spec is built: an unknown key
  or a value the `octen` SDK would reject (for example `count=1000`) raises
  `ValueError`.
- The new functions return `{"data", "request_id", "warning"}` (`warning`
  only when the API sent one). Configuration errors (invalid key, no access
  or balance, other 4xx) are raised and also logged at `ERROR` level on the
  `llama_index.tools.octen` logger. Rate limits, server errors, 400/408/422,
  error or malformed bodies behind a 2xx status, timeouts, connection errors
  and invalid model arguments come back to the model as `{"error": "..."}`.
  When a timeout or connection error may have happened after the request
  was sent, or a 2xx response carried an error or could not be read, that
  text tells the model the call may already have been
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
  already have been served and billed.
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

### Known inconsistency

- The existing functions keep their 0.1.0 behaviour: they still expose
  `num_results`, `start_time` and other options to the model, return lists,
  and raise on errors. They will be aligned with the new functions in a
  future major release.
