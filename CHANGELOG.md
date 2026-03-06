# Changelog

All notable changes to this project will be documented in this file.

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
