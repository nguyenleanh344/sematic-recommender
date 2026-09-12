"""Product data model."""
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class Product:
    """Product data model."""
    product_id: str
    name: str
    description: str
    category: Optional[str] = None
    price: Optional[float] = None
    tags: Optional[List[str]] = None
    embedding: Optional[List[float]] = None
