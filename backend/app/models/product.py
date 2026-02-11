from pydantic import BaseModel
from typing import Optional, Dict

class Product(BaseModel):
    product_id: Optional[str] = None
    title: str
    description: str
    category: Optional[str] = None
    attributes: Optional[Dict] = {}
