import logging
from datetime import datetime
from typing import List

from app.core.schemas_ondc import (
    ONDCSearchRequest, ONDCOnSearchResponse, 
    ONDCProvider, ONDCItem, ONDCCatalog, 
    ONDCOnSearchMessage, ONDCResponseDescriptor
)
from app.services.retrieve import search
from app.services.classify import predict_category
from app.services.rank import re_rank_results

logger = logging.getLogger(__name__)

def process_ondc_search(request: ONDCSearchRequest) -> ONDCOnSearchResponse:
    """
    Translates ONDC Intent -> Internal Search -> ONDC Catalog.
    """
    try:
        # 1. Extract Query from ONDC Intent
        query_text = "General Search"
        
        # Safe extraction of nested ONDC fields
        if (request.message.intent and 
            request.message.intent.item and 
            request.message.intent.item.descriptor):
            query_text = request.message.intent.item.descriptor.name or ""
        
        logger.info(f"ONDC Adapter received query: {query_text}")

        # 2. Run Internal AI Pipeline
        query_cat, _ = predict_category(query_text)
        candidates = search(query_text, top_k=20)
        ranked_results = re_rank_results(query_text, query_cat, candidates)
        
        top_matches = ranked_results[:5]

        # 3. Map Internal Data -> ONDC Provider Format
        providers = []
        for match in top_matches:
            item = ONDCItem(
                id=f"{match['snp_id']}_item_1",
                descriptor={"name": "Custom Job Work / Manufacturing"},
                category_id=match.get('category', 'Manufacturing')
            )
            
            provider = ONDCProvider(
                id=match['snp_id'],
                descriptor=ONDCResponseDescriptor(
                    name=match['name'],
                    short_desc=match.get('category', ''),
                    long_desc=match.get('capability_text', '')
                ),
                items=[item]
            )
            providers.append(provider)

        # 4. Construct ONDC Response Packet
        response_context = request.context.model_copy()
        response_context.action = "on_search"
        response_context.timestamp = datetime.utcnow().isoformat() + "Z"

        return ONDCOnSearchResponse(
            context=response_context,
            message=ONDCOnSearchMessage(
                catalog=ONDCCatalog(
                    providers=providers
                )
            )
        )

    except Exception as e:
        logger.error(f"ONDC Adapter Error: {e}")
        raise e