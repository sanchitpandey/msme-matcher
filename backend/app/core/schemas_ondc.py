from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict

# --- INCOMING: Search Request ---
class ONDCDescriptor(BaseModel):
    name: Optional[str] = None
    tags: Optional[List[Dict[str, Any]]] = None

class ONDCIntentItem(BaseModel):
    descriptor: Optional[ONDCDescriptor] = None

class ONDCIntent(BaseModel):
    item: Optional[ONDCIntentItem] = None

class ONDCMessage(BaseModel):
    intent: Optional[ONDCIntent] = None

class ONDCContext(BaseModel):
    domain: str = "ONDC:RET10"
    action: str = "search"
    transaction_id: str
    message_id: str
    timestamp: str
    bap_id: Optional[str] = None
    bap_uri: Optional[str] = None
    
    class Config:
        extra = "ignore"

class ONDCSearchRequest(BaseModel):
    context: ONDCContext
    message: ONDCMessage

# --- OUTGOING: Catalog Response ---
class ONDCResponseDescriptor(BaseModel):
    name: str
    short_desc: Optional[str] = ""
    long_desc: Optional[str] = ""

class ONDCItem(BaseModel):
    id: str
    descriptor: Dict[str, str] = Field(default_factory=dict)
    price: Dict[str, str] = Field(default_factory=lambda: {"currency": "INR", "value": "0"})
    quantity: Dict[str, Any] = Field(default_factory=lambda: {"available": {"count": 99}})
    category_id: Optional[str] = None

class ONDCProvider(BaseModel):
    id: str
    descriptor: ONDCResponseDescriptor
    items: List[ONDCItem]

class ONDCCatalog(BaseModel):
    descriptor: Dict[str, str] = Field(default_factory=lambda: {"name": "IndiaAI MSME Registry"})
    providers: List[ONDCProvider]

class ONDCOnSearchMessage(BaseModel):
    catalog: ONDCCatalog

class ONDCOnSearchResponse(BaseModel):
    context: ONDCContext
    message: ONDCOnSearchMessage