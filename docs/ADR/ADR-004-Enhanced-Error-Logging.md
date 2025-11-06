# ADR-004: Enhanced Error Handling & Logging

Status: Accepted (Implementation: beta.5+)

Note: Error type taxonomy and rate-limiting parameters may be refined during implementation based on real-world usage patterns.

Context
- 2.0 currently has working error paths and minimal logs. We want a unified error envelope, structured logging, and consistent HTTP/CLI mapping without overcomplicating local workflows.

Decision
- Implement a unified error envelope and structured logging after beta.3, with opt-in JSON logs and basic redaction. Preserve current defaults for developer ergonomics.

Scope (phase 1)
- Error JSON (CLI/Server): {"status":"error","error":{"type","message","detail"?,"retryable"?}, "data"?}
- Server HTTP mapping: 400/404/503 stable (already in place), graceful SSE error close.
- Logging: INFO/WARN/ERROR (+DEBUG), optional JSON logs via env `MLXK2_LOG_JSON=1`; redact secrets.
- Correlation: `request_id` (UUID4) included in responses and logs.

Out of scope (for now)
- Embeddings/other endpoints, distributed tracing, external log backends.

Open Questions
- Error.type taxonomy and granularity vs. stability.
- Default log format (plain) vs. JSON ergonomics; env/flag naming.
- Rate-limiting repeated errors; scope and counters.

Acceptance (high level)
- Tests assert error.type ↔ HTTP status mapping, presence/shape of `request_id`, SSE error termination, and redaction of tokens.

Specification (phase 1)
- Error envelope (CLI/Server consistent)
  - JSON shape: {"status":"error","error":{"type": <enum>, "message": <str>, "detail": <obj|null>, "retryable": <bool|null>}, ...}
  - Standardized type values: access_denied, model_not_found, ambiguous_match, download_failed, validation_error, push_operation_failed, server_shutdown, internal_error.
  - Correlation: request_id/trace_id (UUID) included in responses and logs.

- Logging (structured, level-based output)
  - Levels: INFO (startup, model switch), WARN (preflight warnings, recoveries), ERROR (unhandled/500), DEBUG (enabled by --verbose).
  - Formats: plain text by default; optional JSON logs via MLXK2_LOG_JSON=1 (fields: ts, level, msg, request_id, route, model, duration_ms).
  - Redaction: filter sensitive data (HF_TOKEN, user-specific paths, access URLs).
  - Rate limiting: suppress duplicate error floods (e.g., max 1/5s with counters).

- Server specifics
  - HTTP mapping: 503 during shutdown (_shutdown_event), 404 on model-load errors, 400 for invalid requests (e.g., multiple prompts in completions).
  - Streaming errors: final SSE chunk carries error field, then [DONE]; interrupts emit a clear marker and close cleanly.
  - Hot-swap logging: "Switching to model", "Model loaded", cleanup results (freed memory, optional).

Rollout plan
- Beta.3: ✅ Keep current behavior; add tests (done) and reduce noisy logs (done).
- Beta.4 (KW 41 2024): Runtime Check (Issue #36) - separate bugfix, not part of ADR-004.
- Beta.5+ (Q4 2024): ADR-004 Phase 1 implementation
  - Add request_id generation and propagation
  - Unified error envelope for HTTP errors
  - Optional JSON logs via env `MLXK2_LOG_JSON=1`
  - Minimal redaction (HF_TOKEN, paths)
- Beta.5+ (follow-up): SSE error finalization parity across endpoints; rate-limit error floods.
- 2.0.0 Final (Q1 2026): Production-ready with full error/logging infrastructure.

- CLI operations
  - Exit codes: success=0; any status:error → 1 (no special codes per type).
  - --verbose: buffer hub/server logs in hf_logs[]; do not mix progress logs into JSON; human mode shows concise summary (+URL/commit with --verbose).
  - Preflight (#30): preflight_warning as data field; WARN log-level; access_denied is a hard error.

- Tests (coverage)
  - Mapping tests: error.type ↔ HTTP status; request_id present; optional JSON logs.
  - Streaming failure scenarios: interrupt and exception → proper finalization/marker.
  - Redaction tests: HF_TOKEN never appears in logs/JSON in cleartext.
