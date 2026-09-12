"""Recommendation request/response schemas."""
from pydantic import BaseModel
from typing import List, Optional

class ProductRecommendation(BaseModel):
    """Single product recommendation."""
    product_id: str
    name: str
    similarity_score: float
    description: Optional[str] = None

class RecommendationRequest(BaseModel):
    """Request schema for recommendations."""
    query: str
    limit: int = 5
    min_similarity: float = 0.0

class RecommendationResponse(BaseModel):
    """Response schema for recommendations."""
    recommendations: List[ProductRecommendation]
    query: str
    count: int
