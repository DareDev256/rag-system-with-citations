"""Pydantic request/response models for the RAG API.

Defines the contract between clients and the /query endpoint. Field
constraints (min_length, ge/le bounds) enforce input validation at the
schema layer before any business logic runs — Pydantic rejects invalid
payloads with a 422 response automatically.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class QueryRequest(BaseModel):
    """Incoming query payload for the /query endpoint.

    Attributes:
        query: The natural-language question (1–1000 chars).
        k: Number of documents to retrieve from FAISS (1–20, default 5).
        include_diagnostics: When True, response includes per-stage latency,
            citation coverage, and hallucination detection results.
    """

    query: str = Field(..., min_length=1, max_length=1000)
    k: int = Field(default=5, ge=1, le=20, description="Number of documents to retrieve")
    include_diagnostics: bool = Field(default=False, description="Include retrieval diagnostics in the response")


class Citation(BaseModel):
    """A single source citation attached to the LLM's answer.

    Attributes:
        doc_id: Unique document identifier referenced as [doc_id] in the answer.
        snippet: Text excerpt from the source document.
        score: FAISS similarity score (higher = more relevant for IndexFlatIP).
        source: Original filename of the source document, if available.
    """

    doc_id: str
    snippet: str
    score: Optional[float] = None
    source: Optional[str] = None


class Diagnostics(BaseModel):
    """Per-stage pipeline metrics returned when include_diagnostics=true.

    Enables clients to identify bottlenecks (retrieval vs. synthesis),
    detect hallucinated citations, and monitor citation grounding quality.
    """

    retrieval_ms: float = Field(description="Time spent on FAISS similarity search")
    synthesis_ms: float = Field(description="Time spent on LLM answer generation")
    documents_searched: int = Field(description="Number of documents retrieved from index")
    citation_coverage: float = Field(description="Fraction of retrieved docs cited in the answer (0.0-1.0)")
    hallucinated_citations: List[str] = Field(default_factory=list, description="Citation IDs in the answer that don't match any retrieved document")


class QueryResponse(BaseModel):
    """Full response from the /query endpoint.

    Contains the LLM answer with citations, a real confidence score based
    on citation coverage (not model self-assessment), and optional diagnostics.
    """

    query: str
    category: str
    answer: str
    citations: List[Citation]
    confidence: float
    latency_ms: float
    diagnostics: Optional[Diagnostics] = Field(default=None, description="Retrieval diagnostics (opt-in via include_diagnostics=true)")
