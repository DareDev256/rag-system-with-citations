# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

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
