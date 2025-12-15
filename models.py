from pydantic import BaseModel
from typing import List

class ChatRequest(BaseModel):
    question: str
    transactions: List[dict]  # [{"description": "Pizza", "amount": 20, "category": "food"}]

class ChatResponse(BaseModel):
    answer: str

class SearchRequest(BaseModel):
    query: str
    transactions: List[dict]

class SearchResult(BaseModel):
    id: str
    text: str
    score: float

class SearchResponse(BaseModel):
    results: List[SearchResult]
