from pydantic import BaseModel, Field
from typing import List, Optional

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    k: int = Field(default=5, ge=1, le=20, description="Number of documents to retrieve")
    include_diagnostics: bool = Field(default=False, description="Include retrieval diagnostics in the response")

class Citation(BaseModel):
    doc_id: str
    snippet: str
    score: Optional[float] = None
    source: Optional[str] = None

class Diagnostics(BaseModel):
    retrieval_ms: float = Field(description="Time spent on FAISS similarity search")
    synthesis_ms: float = Field(description="Time spent on LLM answer generation")
    documents_searched: int = Field(description="Number of documents retrieved from index")
    citation_coverage: float = Field(description="Fraction of retrieved docs cited in the answer (0.0-1.0)")
    hallucinated_citations: List[str] = Field(default_factory=list, description="Citation IDs in the answer that don't match any retrieved document")

class QueryResponse(BaseModel):
    query: str
    category: str
    answer: str
    citations: List[Citation]
    confidence: float
    latency_ms: float
    diagnostics: Optional[Diagnostics] = Field(default=None, description="Retrieval diagnostics (opt-in via include_diagnostics=true)")
