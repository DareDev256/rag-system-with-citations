# Synthesis Pipeline — Citation Analysis & Confidence Scoring

> **Target module:** `src/llm/synthesize.py`
> **Key function:** `analyze_citations()` → `CitationAnalysis`

The synthesis pipeline is the core of this system. Every answer flows through it — classification, retrieval context assembly, LLM synthesis, citation validation, and confidence scoring. This document explains the internals.

---

## Pipeline Flow

```
  query
    │
    ▼
┌──────────────┐     ┌────────────────────┐
│ classify_    │────►│ _parse_            │──► category: str
│ query()      │     │ classification()   │    ("factual" | "exploratory" | "ambiguous")
└──────────────┘     └────────────────────┘

  query + search_results
    │
    ▼
┌──────────────┐     ┌────────────────────┐     ┌─────────────────┐
│ synthesize_  │────►│ _build_synthesis_  │────►│ _parse_         │
│ answer()     │     │ prompt()           │     │ synthesis()     │
└──────────────┘     └────────────────────┘     └────────┬────────┘
                                                         │
                           ┌─────────────────────────────┤
                           │                             │
                           ▼                             ▼
                    ┌──────────────┐           ┌──────────────────┐
                    │ analyze_     │           │ calculate_       │
                    │ citations()  │           │ confidence()     │
                    └──────┬───────┘           └──────────────────┘
                           │
                           ▼
                    CitationAnalysis
                    (frozen dataclass)
```

Both `classify_query` and `synthesize_answer` have async variants (`_async` suffix) with identical contracts.

---

## `analyze_citations(answer, search_results)` — The Core Function

**Location:** `src/llm/synthesize.py:76`

This is the single function that computes every citation metric the pipeline needs. Before this existed, the same set math was duplicated across `_parse_synthesis` and `build_diagnostics` — both independently extracted doc IDs, computed intersections, and calculated coverage. Now it runs once.

### Signature

```python
def analyze_citations(answer: str, search_results: List[Dict]) -> CitationAnalysis
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `answer` | `str` | Raw LLM output text, expected to contain `[doc_id]` citations |
| `search_results` | `List[Dict]` | FAISS retrieval results. Each dict must have `doc_id` (str\|int\|None) and `snippet` (str) |

### Returns `CitationAnalysis`

Frozen dataclass (`slots=True`) — immutable after creation, hashable, zero-copy passable.

| Field | Type | Description |
|-------|------|-------------|
| `available_ids` | `frozenset` | All valid (non-None) doc IDs from search results |
| `all_cited_ids` | `frozenset` | Every `[...]` match in the answer text |
| `valid_cited_ids` | `frozenset` | `all_cited_ids ∩ available_ids` — grounded citations |
| `hallucinated_ids` | `frozenset` | `all_cited_ids − available_ids` — fabricated references |
| `coverage` | `float` | `len(valid) / len(available)`, rounded to 2 decimals. 0.0 when no docs available |

### Internal Steps

1. Extract valid doc IDs from search results — `get_available_doc_ids()` skips None/empty → `{"doc_001", "doc_002", "doc_003"}`
2. Extract ALL `[...]` citations from answer text (unfiltered) → `{"doc_001", "doc_999"}`
3. Set intersection = grounded citations: `all_cited & available` → `{"doc_001"}`
4. Set difference = hallucinated citations: `all_cited - available` → `{"doc_999"}`
5. Coverage ratio: `len(valid) / len(available)` → `0.33`

### Consumers

| Consumer | What it uses | Why |
|----------|-------------|-----|
| `_parse_synthesis()` | `valid_cited_ids` | Filters search results to only cited docs for the response |
| `_parse_synthesis()` | `available_ids` | Passes to `calculate_confidence()` to skip recomputation |
| `build_diagnostics()` | `coverage`, `hallucinated_ids` | Populates the diagnostics response object |
| `build_diagnostics()` | Entire object | Accepts via `citation_analysis` kwarg to avoid duplicate call |

---

## `calculate_confidence()` — The Scoring Function

**Location:** `src/llm/synthesize.py:97`

### Signature

```python
def calculate_confidence(
    answer: str,
    search_results: List[Dict],
    cited_ids: Set[str],
    *,
    _available_ids: Optional[Set[str]] = None,
) -> float
```

### Scoring Logic

The confidence score is a **real metric based on citation grounding**, not LLM self-assessment:

```
 0.0  ─── No search results (pipeline failure)
 0.1  ─── Refusal detected ("cannot answer", "not enough information", etc.)
 0.3  ─── Answer generated but zero valid citations (hallucination risk)
0.6–1.0 ── Cited answer: 0.6 + 0.4 × (valid_citations / available_docs)
```

### The `_available_ids` Optimization

`_parse_synthesis` already computes `available_ids` via `analyze_citations`. The `_available_ids` keyword-only arg lets it pass the pre-computed set, avoiding a second `get_available_doc_ids()` call over the same search results. This is a pure performance optimization — the output is identical.

---

## Citation Regex

```python
CITATION_PATTERN = re.compile(r'\[([^\]]+)\]')
```

Matches any `[non-empty-content]` in the answer text. Design choices:

- **Requires 1+ characters** inside brackets — empty `[]` is ignored
- **Greedy on content** — known edge case: unclosed brackets like `[doc_001` won't match
- **No validation** — raw extraction; filtering against `available_ids` happens in `analyze_citations`

---

## Error Handling: `_safe_llm_call`

Both sync and async paths use a wrapper that catches all exceptions:

```python
def _safe_llm_call(model, messages, parser, fallback, label, **kwargs):
    client = get_llm_client()
    try:
        response = _call_llm(client, model, messages, **kwargs)
        return parser(response)
    except Exception as e:
        logger.error("%s failed: %s", label, type(e).__name__)
        return fallback
```

- **Classification failure** → returns `"exploratory"` (safest default)
- **Synthesis failure** → returns `{"answer": "Error generating answer.", "citations_used": [], "confidence": 0.0}`

No exception propagates to the API layer. The client always gets a response.

---

## Diagnostics Pipeline

When `include_diagnostics=true` is passed to `/query`, the response includes a `diagnostics` object built by `src/api/response.py:build_diagnostics()`:

```
CitationAnalysis.coverage ──────────────────────► citation_coverage
CitationAnalysis.hallucinated_ids ──────────────► hallucinated_citations
answer + context_str → estimate_hallucination_rate() → hallucination_rate
coverage + hallucination_rate + confidence → calculate_answer_quality() → answer_quality_score
```

### Quality Score Formula (`src/eval/metrics.py`)

```python
quality = 0.40 × citation_coverage + 0.35 × (1 - hallucination_rate) + 0.25 × confidence
```

| Weight | Signal | Measures |
|--------|--------|----------|
| 40% | Citation coverage | Are retrieved docs actually cited? |
| 35% | Inverse hallucination | Is the answer grounded in context? |
| 25% | Confidence | Does the citation ratio support the answer? |

---

## Prompt Injection Defense

Context assembly (`src/llm/prompt.py:build_context_str`) applies three defense layers before the LLM sees any corpus text:

1. **Truncation** — each snippet capped at 2000 chars
2. **Pattern neutralization** — injection phrases wrapped in `[BLOCKED INSTRUCTION: ...]`
3. **XML isolation** — context wrapped in `<retrieved_documents>` delimiters

## Extending the Pipeline

To add a new synthesis variant (e.g., streaming), build messages with `_synthesis_messages(_build_synthesis_prompt(query, results))`, parse through `_parse_synthesis(response, search_results)`, and pass `result["_citation_analysis"]` to `build_diagnostics()` to avoid recomputation.
