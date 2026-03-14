# Changelog

All notable changes to this project will be documented in this file.

## [1.13.0] - 2026-03-14

### Changed
- **Unified logging across all modules** — replaced 14 `print()` calls with structured `logging` in `synthesize.py`, `vector_store.py`, `ingest.py`, and `evaluate.py`; all output now routes through the logging framework for level filtering, consistent formatting, and log aggregation compatibility
- **Citation regex extracted to module constant** — `CITATION_PATTERN` is now a pre-compiled `re.compile()` constant in `synthesize.py` instead of a raw string re-created on every call; importable for downstream use
- **Lambda-to-function conversion** — `_CLASSIFICATION_MESSAGES` and `_SYNTHESIS_MESSAGES` lambdas replaced with proper named functions (`_classification_messages`, `_synthesis_messages`) for meaningful tracebacks and PEP 8 E731 compliance
- **Type hint completeness** — added `Optional[Set[str]]` on `extract_cited_doc_ids`, return types on `_call_llm`/`_call_llm_async`/`_parse_classification`/`_parse_synthesis`, and `Any` type annotations on LLM response parameters
- **Version sync** — `main.py` version now matches CHANGELOG (was stuck at 1.12.0)
- Updated 3 tests to match refactored code: `capsys` → `caplog` for log assertions (print→logging migration)

## [1.12.2] - 2026-03-13

### Changed
- **Portfolio-grade README rewrite** — restructured for visual impact and scannability:
  - Added badge row (Python, FastAPI, FAISS, test count, security layers)
  - New full-system architecture diagram showing security perimeter around the pipeline
  - Security promoted to first-class section with 14-layer CWE mapping table
  - Test table sorted by count (descending) for quick impact assessment
  - Design decisions refactored into comparison table with rationale and trade-offs
  - API examples consolidated with auth header usage
  - Project structure updated to reflect `ip.py` and JSON metadata migration

## [1.12.1] - 2026-03-13

### Added
- **Resilience test suite** — 16 new tests in `test_resilience.py` targeting production failure modes that happy-path tests miss:
  - `TestLoadIndexCorruption` (5 tests): corrupted JSON metadata, truncated FAISS binary, permission denied, OS-level I/O errors, null JSON edge case
  - `TestSaveIndexFailures` (3 tests): disk-full on FAISS write, permission denied on metadata write, non-serializable metadata detection
  - `TestSearchMetadataEdgeCases` (4 tests): metadata with missing keys, k > index size padding, zero-score results, negative FAISS scores
  - `TestAddDocumentsEdgeCases` (2 tests): metadata count tracking across sequential adds, single-doc index auto-creation
  - `TestSearchEngineSingleton` (2 tests): successful init caching, singleton poisoning prevention on load failure
- 378 tests total across 15 test suites

## [1.12.0] - 2026-03-11

### Security
- **Trusted proxy IP resolution (CWE-348)** — rate limiting previously used `req.client.host` which returns the proxy IP behind a reverse proxy (nginx, ALB, Cloudflare), collapsing all clients into one rate-limit bucket and making rate limiting useless. New `TRUSTED_PROXY_COUNT` env var enables extraction of real client IPs from `X-Forwarded-For` by counting from the right (attacker-resistant). Falls back to socket IP when disabled (default 0) or when header is malformed.
- **Security doc gap table updated** — removed stale entries for API auth and body size limit (both fixed in v1.10.0+)

### Added
- `src/utils/ip.py` — `resolve_client_ip()` with IP validation, proxy count bounds checking, and IPv6 support
- 18 new tests in `test_ip_resolution.py` — proxy disabled/enabled modes, single/double proxy, attacker spoofing resistance, IPv6, invalid IP fallback
- 362 tests total across 14 test suites

## [1.11.0] - 2026-03-10

### Security
- **API key authentication (CWE-862)** — opt-in via `API_KEYS` env var (comma-separated). When set, all `POST /query` requests require a valid key via `Authorization: Bearer <key>` or `X-API-Key: <key>` header. Uses `hmac.compare_digest` for constant-time comparison to prevent timing attacks (CWE-208). Public paths (`/health`, `/docs`) remain unauthenticated.

### Added
- 10 new auth tests in `test_auth.py` — Bearer token acceptance, X-API-Key header, missing/wrong/empty key rejection, public path bypass, auth-disabled passthrough, constant-time comparison structural check
- 344 tests total across 13 test suites

## [1.10.4] - 2026-03-10

### Added
- **Middleware test suite** — 17 new tests in `test_middleware.py` covering the three security-critical middleware layers that had zero dedicated tests:
  - `MaxBodySizeMiddleware` (7 tests): oversized payload rejection (413), invalid Content-Length (400), GET bypass, boundary values at exact limit and limit+1
  - `RequestIDMiddleware` (6 tests): auto-generation when missing, client ID passthrough, oversized/control-char/empty ID rejection, 64-char boundary acceptance
  - Global exception handler (4 tests): stack trace suppression (CWE-209), request ID inclusion in 500s, security headers on error responses, file path leak prevention
- 334 tests total across 12 test suites

## [1.10.3] - 2026-03-10

### Fixed
- **asyncio.Lock() at module level (CWE-362)** — `_async_client_lock` was created at import time outside a running event loop, which raises `DeprecationWarning` on Python 3.10+ and `RuntimeError` on Python 3.12+; now lazily initialized inside `get_async_llm_client()` on first call
- **FAISS dimension mismatch crashes** — `add_documents()` and `search()` now validate embedding/query vector dimensions against `index.d` before calling FAISS, converting cryptic C++ segfaults into clear `ValueError` messages

### Added
- 4 new regression tests: lazy async lock initialization (1), dimension mismatch in add/search (2), matching dimension success (1) — 317 total

## [1.10.2] - 2026-03-10

### Added
- **Unit tests for response builders** — 28 new tests covering `build_citations()`, `build_diagnostics()`, `_sanitize_field()`, `_parse_classification()`, and `_parse_synthesis()` — all previously untested at the unit level
- Tests for: sanitizer application to corpus fields, None/missing source handling, hallucination detection & sorting, citation coverage math, timing rounding, LLM classification fallback on unknown/empty categories, synthesis fallback to top result, None doc_id filtering in synthesis — 313 total

## [1.10.1] - 2026-03-09

### Fixed
- **None doc_id/snippet propagation in LLM context (CWE-20)** — `build_context_str()` now skips search results with `None` or empty `doc_id`/`snippet` instead of formatting them as `[None] None`, which polluted the LLM's context window and could cause the model to cite `[None]` as a source
- **`calculate_confidence` false matches on None doc_ids** — `available_ids` set now filters out `None` entries, preventing a `None` doc_id from matching a hallucinated `[None]` citation and inflating confidence scores
- **`_parse_synthesis` same None-in-set issue** — `available_ids` and `citations_used` filtering now use `.get()` with truthiness check, consistent with the fix already applied to `calculate_citation_coverage` in v1.3.1
- **Confidence denominator inflated by broken metadata** — `citation_ratio` now divides by `len(available_ids)` instead of `len(search_results)`, so entries with `None` doc_ids don't deflate confidence scores for well-cited answers

### Added
- 5 new tests: None doc_id/snippet skipping in `build_context_str` (3), None doc_id filtering in `calculate_confidence` (2) — 285 total

## [1.10.0] - 2026-03-09

### Security
- **Request body size limit (CWE-400)** — new `MaxBodySizeMiddleware` rejects POST/PUT/PATCH requests exceeding `MAX_BODY_BYTES` (default 64 KB) before the body is parsed, preventing memory exhaustion attacks that bypass Pydantic validation and rate limiting
- **Embedding model name validation (CWE-73)** — `EMBEDDING_MODEL` env var is now validated against an allowlist regex at startup; rejects filesystem paths (`../../etc/passwd`), URLs (`http://evil.com/backdoor`), `file://` URIs, home directory references (`~/`), and shell metacharacters to prevent path traversal and SSRF at model load time

### Added
- `MaxBodySizeMiddleware` in `main.py` — early Content-Length check with 413 response
- `_validate_model_name()` in `embed.py` — startup-time model name sanitization
- `MAX_BODY_BYTES` env var (default 65536, min 1024) for configurable body size limit
- 15 new tests: body size enforcement (5), model name validation (10) — 280 total

## [1.9.0] - 2026-03-09

### Changed
- **Extracted `response.py` module** (`src/api/response.py`) — citation assembly and diagnostics computation extracted from the 80-line `query_endpoint` handler into `build_citations()` and `build_diagnostics()` helpers; endpoint now reads as a clean pipeline orchestrator (classify → retrieve → synthesize → format)
- **Removed dead backwards-compat aliases** from `prompt.py` — `RAG_PROMPT_TEMPLATE` and `CLASSIFICATION_PROMPT_TEMPLATE` module-level aliases were unused by any test or import; deleted to reduce confusion

## [1.8.2] - 2026-03-08

### Added
- **Security architecture documentation** (`docs/security.md`) — comprehensive threat model mapping 12 defense layers to specific CWEs, with configuration reference, architecture diagram, and known gaps table; consolidates security context previously scattered across changelog entries
- README security section linking to the new doc

## [1.8.1] - 2026-03-08

### Security
- **Citation field sanitization (CWE-116)** — `doc_id`, `snippet`, and `source` fields in `/query` response citations now pass through `_sanitize_output()` to strip C0 control characters; previously only the LLM answer and reflected query were sanitized, leaving corpus-sourced fields as an unsanitized output channel for terminal injection and ANSI escape attacks

### Added
- 4 new tests: snippet control char stripping, doc_id sanitization, source filename sanitization, None source handling — 265 total

## [1.8.0] - 2026-03-08

### Changed
- **Extracted `safe_int_env()` helper** (`src/utils/env.py`) — consolidates the defensive env-var parsing pattern duplicated across `_parse_llm_timeout()` (synthesize.py) and `_parse_hsts_max_age()` (main.py) into a single reusable function with bounds checking and graceful fallback; now also used for `RATE_LIMIT_RPM`
- **Extracted `_SECURITY_HEADERS` dict** (main.py) — 7 static security headers are now a declarative dict instead of scattered assignments, making them easier to audit and extend; dynamic headers (CSP, HSTS) remain explicit
- **Extracted `_call_llm()` / `_call_llm_async()` wrappers** (synthesize.py) — all OpenAI API calls now route through a single sync/async entry point, further reducing duplication between the sync and async code paths
- **Deleted `_parse_llm_timeout()`** and **`_parse_hsts_max_age()`** — replaced by `safe_int_env()` calls
- Updated 7 existing tests to target `safe_int_env` instead of deleted parser functions

## [1.7.1] - 2026-03-08

### Fixed
- **Race condition in singleton initialization (CWE-362)** — `get_llm_client()`, `get_async_llm_client()`, and `get_search_engine()` used unguarded check-then-act patterns on global singletons; under concurrent access (multiple threads or asyncio tasks), duplicate clients could be created, leaking connections and causing subtle state bugs. Added `threading.Lock` for sync factories and `asyncio.Lock` for the async factory with double-checked locking.

### Changed
- `get_async_llm_client()` is now `async def` — callers must `await` it, matching the concurrency model it protects

### Added
- 3 new race condition tests: concurrent thread safety for sync client (1), concurrent asyncio task safety for async client (1), concurrent thread safety for search engine (1) — 261 total

## [1.7.0] - 2026-03-06

### Security
- **Rate limiter memory exhaustion fix (CWE-400)** — `_rate_store` now evicts stale IP entries when tracked count exceeds `_MAX_TRACKED_IPS` (10,000), preventing unbounded memory growth from IP rotation attacks
- **Global exception handler (CWE-209)** — unhandled exceptions now return a generic `"Internal server error."` with a request ID instead of leaking stack traces, file paths, or module names
- **Exception responses get security headers** — discovered that FastAPI exception handler responses bypass `BaseHTTPMiddleware`; refactored security headers into shared `_apply_security_headers()` helper called from both middleware and exception handler, closing a header gap on 500 responses

### Added
- **Request ID middleware** — every response includes `X-Request-ID` for security incident correlation; clients can pass their own (validated: max 64 chars, no control chars) or one is auto-generated via `uuid4`
- **Request ID in query logs** — `/query` log lines now include `request_id=` for end-to-end traceability
- `_evict_stale_ips()` — garbage collector for expired rate limiter entries
- 9 new hardening tests: rate limiter memory (2), global exception handler (3), request ID middleware (4) — 258 total

## [1.6.0] - 2026-03-06

### Security
- **In-memory rate limiter on `/query`** — sliding-window per-IP rate limiting (default 30 req/min, configurable via `RATE_LIMIT_RPM` env var) prevents budget drain and DoS attacks (CWE-770)
- **LLM output sanitization** — strips C0 control characters from LLM responses before returning to client, preventing terminal injection and ANSI escape attacks while preserving legitimate whitespace (newlines, tabs)
- **Reflected query sanitization** — the echoed `query` field in responses is now sanitized, preventing control character injection via reflected input
- **Safe `LLM_TIMEOUT` parsing** — non-numeric, zero, and negative values now fall back to default 30s instead of crashing (matches `HSTS_MAX_AGE` defensive pattern)

### Added
- `_check_rate_limit()` — sliding-window rate limiter with automatic expiry
- `_sanitize_output()` — control character filter for LLM responses
- `_parse_llm_timeout()` — safe env var parser for timeout configuration
- 11 new hardening tests: rate limiting (3), output sanitization (4), LLM_TIMEOUT parsing (4) — 249 total

## [1.5.0] - 2026-03-06

### Added
- **Retrieval diagnostics** — new opt-in `include_diagnostics` parameter on `/query` returns per-stage latency breakdown (`retrieval_ms`, `synthesis_ms`), `documents_searched` count, `citation_coverage` ratio, and `hallucinated_citations` list for full pipeline observability
- `Diagnostics` Pydantic model in schemas for structured diagnostics response
- 4 new API tests: diagnostics omitted by default, diagnostics included when requested, hallucinated citation detection, empty results edge case — 238 total

## [1.4.1] - 2026-03-06

### Fixed
- **`search.py` TOCTOU race** — replaced `os.path.exists()` + `os.makedirs()` with atomic `os.makedirs(exist_ok=True)`, matching the fix already applied to `ingest.py` in v1.3.2
- **`search.py` singleton poisoning** — if `load_index()` raises (corrupted index, permission denied), `_vector_store` is no longer left in a broken state; subsequent retries re-attempt initialization (mirrors `embed.py` fix from v1.3.3)
- **`prompt.py` double-replacement injection** — `format_rag_prompt` now uses split-and-join instead of chained `.replace()` so documents containing the literal text `{query}` are no longer silently expanded into the user's actual query, preventing context corruption

### Added
- 4 new tests: singleton poisoning (2), double-replacement context corruption (2) — 234 total

## [1.4.0] - 2026-03-06

### Changed
- **Refactored `synthesize.py` to eliminate sync/async duplication** — extracted `_parse_classification()`, `_parse_synthesis()`, and `_SYNTHESIS_ERROR` as shared helpers; sync and async variants are now thin wrappers around the same parsing logic, reducing the file from 259 to 197 lines and ensuring future changes to response handling only need to be made once

## [1.3.7] - 2026-03-05

### Security
- **LLM request timeout** — OpenAI clients now enforce a 30s timeout (configurable via `LLM_TIMEOUT` env var) to prevent resource exhaustion from hung upstream connections (CWE-400)
- **Error message sanitization** — exception handlers log only `type(e).__name__` instead of full error messages, preventing API key and internal path leakage in logs
- **Full control character stripping** — query logging now strips all C0 control chars (`U+0000`–`U+001F`, `U+007F`) via regex, not just `\n`/`\r`, closing CRLF injection gaps (null bytes, tabs, escape sequences)
- **HSTS_MAX_AGE validation** — `_parse_hsts_max_age()` now gracefully handles non-numeric and negative values instead of crashing on `int()` conversion
- **Path traversal guard** — `load_documents()` resolves symlinks via `os.path.realpath()` and rejects files whose canonical path escapes the corpus directory

### Added
- 11 new hardening tests: LLM timeout (3), error sanitization (2), control char stripping (2), HSTS validation (3), path traversal (1) — 230 total

## [1.3.6] - 2026-03-05

### Security
- **Hardened CSP with 4 new directives** — `object-src 'none'` (blocks plugin exploits), `base-uri 'self'` (prevents base tag injection), `form-action 'self'` (blocks cross-origin form submissions), `upgrade-insecure-requests` (auto HTTP→HTTPS)
- **Added `Cache-Control: no-store`** — prevents browsers and proxies from caching sensitive API response data
- **Added `X-DNS-Prefetch-Control: off`** — prevents DNS prefetching information leakage
- **CSP now configurable via `CSP_POLICY` env var** — allows override for environments requiring Swagger UI (`unsafe-inline`) or custom policies
- **HSTS max-age configurable via `HSTS_MAX_AGE` env var** — defaults to 2 years (63072000s)

### Added
- 7 new hardening tests: `object-src`, `base-uri`, `form-action`, `upgrade-insecure-requests`, CSP env override, DNS prefetch control, cache-control (219 total)

## [1.3.5] - 2026-03-05

### Added
- **22 hardening tests** (`test_hardening.py`) covering 4 previously untested areas:
  - **Security headers** (9 tests): verifies all 7 security headers (CSP, HSTS, X-Frame-Options, etc.) on health, query, and error responses
  - **Prompt injection resistance** (5 tests): proves safe `.replace()` templates handle `{curly_braces}`, `%(percent)s`, and template variable names in user input
  - **File I/O error resilience** (4 tests): exercises encoding errors, permission denied, empty files, and mixed bad/good files in `load_documents`
  - **`_client_kwargs` configuration** (4 tests): validates OpenAI client config building — key-only, key+base_url, missing key, empty base_url
- Total test count: 212 across 10 test suites

## [1.3.4] - 2026-02-25

### Added
- **`OPENAI_BASE_URL` env var support** — route all LLM calls through any OpenAI-compatible proxy (LiteLLM, OpenRouter, Ollama, vLLM) to use Anthropic, Qwen, Mistral, or local models as drop-in backends
- **Proxy integration guide** (`docs/proxy-integration.md`) — configuration examples for LiteLLM, OpenRouter, Ollama, and vLLM with model selection tips
- **`_client_kwargs()` helper** in `synthesize.py` — centralizes client config (API key + base URL) shared by sync and async clients, replacing duplicated setup logic

## [1.3.3] - 2026-02-23

### Fixed
- **Embedder singleton poisoning on model load failure** — if `SentenceTransformer()` raises (network error, OOM, missing model), the singleton is no longer left in a broken state; subsequent retries work correctly
- **Pinned `sentence-transformers` upper bound** — added `<4.0.0` cap to prevent silent breaking changes from floating `>=2.2.2`

### Added
- **Configurable embedding model** — `EMBEDDING_MODEL` env var lets you swap models without code changes (default: `all-MiniLM-L6-v2`)
- 3 new tests for singleton failure recovery, retry behavior, and model name configuration (189 total)

## [1.3.2] - 2026-02-19

### Fixed
- **`calculate_citation_coverage` denominator bug** — None doc_ids no longer inflate the denominator, fixing under-reported coverage when metadata is incomplete
- **Format string injection in LLM prompts** — replaced `str.format()` with safe `str.replace()` so user queries containing `{context_str}` or `{query}` no longer cause `KeyError` crashes
- **`classify_query` inconsistent error fallback** — API errors now return `"exploratory"` (safe default) instead of `"factual"`, matching the out-of-vocab fallback behavior
- **`ingest.py` TOCTOU race** — replaced `os.path.exists()` + `os.makedirs()` with atomic `os.makedirs(exist_ok=True)`
- **`ingest.py` single-file abort** — bad corpus files (permission denied, encoding errors) are now skipped with a warning instead of aborting the entire ingest

## [1.3.1] - 2026-02-09

### Fixed
- **`calculate_citation_coverage` TypeError crash** — `None` doc_id values now skipped gracefully instead of raising `TypeError` on `None in 'string'`
- **`estimate_hallucination_rate` placeholder replaced** — implemented word-overlap heuristic that compares answer content words against context; returns 0.0 (grounded) to 1.0 (hallucinated) instead of hardcoded 0.1

### Added
- 10 new tests for the fixed functions (1 regression + 1 mixed-validity test for coverage, 8 tests for hallucination rate)

## [1.3.0] - 2026-02-09

### Security
- **Added `Content-Security-Policy` header** — restricts script/style/image sources to `'self'`, blocks framing via `frame-ancestors 'none'`
- **Added `Strict-Transport-Security` header** — enforces HTTPS with 2-year max-age, includeSubDomains, and preload
- **Dockerfile runs as non-root user** — creates dedicated `appuser` to limit blast radius of container compromise
- **Added `.dockerignore`** — prevents `.env`, `.git`, tests, `__pycache__`, and data artifacts from leaking into Docker image
- **Removed unused dependencies** — dropped `requests` and `click` from requirements.txt (never imported, unnecessary attack surface)

### Changed
- **Migrated `@app.on_event("startup")` to `lifespan` context manager** — replaces deprecated FastAPI pattern, aligns with modern FastAPI lifecycle management

## [1.2.0] - 2026-02-08

### Security
- **Replaced `pickle` with `json`** for metadata serialization in `vector_store.py` — eliminates Remote Code Execution (RCE) risk from malicious pickle files
- **Added security headers middleware** to FastAPI app — `X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`, `Permissions-Policy`, `X-Permitted-Cross-Domain-Policies`
- **Added CORS middleware** — opt-in via `CORS_ORIGINS` env var, locked down by default
- **Sanitized log output** — strips newlines and truncates user queries to prevent log injection attacks
- **Fixed Dockerfile healthcheck** — replaced `curl` (not available in `python:3.11-slim`) with `python urllib`
- **Added `DISABLE_DOCS` env var** — allows disabling Swagger/ReDoc in production
- **Cleaned `.gitignore`** — removed duplicate entries, added `*.pkl` glob pattern

### Fixed
- README Testing section updated to reflect actual test coverage (142 tests across 8 suites, was incorrectly showing 64 tests across 2 suites)

### Added
- `__init__.py` files to all src subdirectories (`api/`, `data/`, `eval/`, `llm/`, `retrieval/`, `utils/`) for proper Python package structure
- CHANGELOG.md

## [0.2.0] - 2025-02-07

### Added
- 142 unit tests across 8 test suites — 100% function coverage
  - `test_core.py` (50): pure function tests
  - `test_llm.py` (14): mock-based sync LLM tests
  - `test_llm_async.py` (14): async LLM tests
  - `test_api.py` (10): FastAPI endpoint tests
  - `test_vector_store.py` (14): FAISS wrapper tests
  - `test_search.py` (14): search orchestration tests
  - `test_integration_gaps.py` (13): environment validation and ingest pipeline tests
  - `test_evaluation.py` (13): evaluation pipeline tests
- Portfolio-quality README with architecture diagram and confidence scoring table

## [0.1.0] - 2025-02-06

### Added
- Initial RAG system with FastAPI + FAISS + OpenAI
- Query classification (factual / exploratory / ambiguous)
- Citation-enforced answer synthesis with `[doc_id]` references
- Real confidence scoring based on citation coverage
- `k` parameter for retrieval depth control (1–20)
- Async LLM support for non-blocking API calls
- Configurable models via environment variables
- Offline evaluation pipeline with CSV export
- Docker support
- Health check endpoint
