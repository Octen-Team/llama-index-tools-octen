# Changelog

## 0.2.0

### Added

- Five new spec functions: `broad_search`, `news_search`, `business_search`,
  `pulse_search` and `place_search`. The model gets only the query and
  filters that do not change cost; result counts, content options and other
  billing-relevant options are set by the developer through the new
  `broad_search_options`, `news_search_options`, `business_search_options`,
  `pulse_search_options` and `place_search_options` constructor arguments.
- The new functions return `{"data", "request_id", "warning"}` (`warning`
  only when the API sent one). Configuration errors (invalid key, no access
  or balance, invalid developer options) are raised; rate limits, server
  errors, timeouts, connection errors and invalid model arguments come back
  to the model as `{"error": "..."}`.
- `api_key` falls back to the `OCTEN_API_KEY` environment variable.

### Changed

- Requires `octen>=0.12.0`.
- Clients are created with `retry_after_send=False`: a request that timed
  out or lost its connection after it was sent is no longer retried, since
  it may already have been served and billed.

### Deprecated

- `search_type` on `search`, `search_and_retrieve_documents` and
  `search_and_retrieve_highlights` is accepted but ignored and emits a
  `DeprecationWarning`. The Octen API no longer takes it, and with
  `octen>=0.5.0` passing it raised `TypeError`. It will be removed in a
  future release.

### Known inconsistency

- The existing functions keep their 0.1.0 behaviour: they still expose
  `num_results`, `start_time` and other options to the model, return lists,
  and raise on errors. They will be aligned with the new functions in a
  future major release.
