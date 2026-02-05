from pydantic import BaseModel, Field
from typing import List, Optional

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    k: int = Field(default=5, ge=1, le=20, description="Number of documents to retrieve")

class Citation(BaseModel):
    doc_id: str
    snippet: str
    score: Optional[float] = None
    source: Optional[str] = None

class QueryResponse(BaseModel):
    query: str
    category: str
    answer: str
    citations: List[Citation]
    confidence: float
    latency_ms: float
