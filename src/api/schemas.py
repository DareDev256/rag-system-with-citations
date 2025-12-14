from pydantic import BaseModel
from typing import List, Optional

class QueryRequest(BaseModel):
    query: str

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
