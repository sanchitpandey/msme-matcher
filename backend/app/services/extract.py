import re
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

# PATTERNS
PATTERNS = {
    "capacity": [
        r"(\d+[\s-]*(?:kg|tonnes|tons|pcs|pieces|mtrs|meters|units)[\s/]*(?:per|/)?[\s-]*(?:day|week|month|shift|daily))",
        r"capacity[\s:-]*(\d+[\s\w]+)"
    ],
    "dimensions": [
        r"(\d+(?:\.\d+)?[\s-]*(?:mm|cm|inch|ft|meter))",
        r"tolerance[\s:-]*([+\-]?\d+(?:\.\d+)?\s*mm)"
    ],
    "material": [
        r"\b(cotton|silk|polyester|viscose|nylon|wool|linen)\b",
        r"\b(mild steel|ms|stainless steel|ss|aluminium|copper|brass|plastic|abs|nylon)\b",
        r"\b(organic|halal|fssai)\b"
    ],
    "machines": [
        r"(\d+[\s-]*(?:machine|loom|spindle|lathe|cnc|vmc))",
    ]
}

def extract_attributes(text: str) -> Dict[str, Any]:
    """
    Parses unstructured text into structured attributes.
    Example: "Steel plates 5mm, 2000kg/day" -> {"material": "Steel", "dims": "5mm", "capacity": "2000kg/day"}
    """
    text = text.lower()
    attributes: Dict[str, List[str]] = {}

    try:
        # Regex Extraction
        for key, regex_list in PATTERNS.items():
            matches = []
            for pattern in regex_list:
                found = re.findall(pattern, text)
                matches.extend(found)
            
            if matches:
                # Deduplicate and clean
                attributes[key] = list(set([m.strip() for m in matches]))

        # Heuristic: Price Tier Guessing
        if "premium" in text or "export quality" in text or "high precision" in text:
            attributes["quality_signal"] = ["High"]
        elif "cheap" in text or "low cost" in text:
            attributes["quality_signal"] = ["Economy"]

    except Exception as e:
        logger.error(f"Attribute extraction error: {e}")

    return attributes

def extract_basic_fields(text: str):
    """
    Extracts basic MSME registration fields from unstructured OCR text.
    Heuristic based extraction for hackathon demo reliability.

    Returns:
        dict with business_name and location if detected.
    """
    result = {}

    try:
        lines = [l.strip() for l in text.split("\n") if l.strip()]

        if lines:
            # first meaningful line often business name
            result["business_name"] = lines[0][:120]

        # location detection (basic)
        cities = [
            "delhi","mumbai","surat","bangalore","pune","hyderabad",
            "chennai","ahmedabad","kolkata","noida","gurgaon",
            "coimbatore","jaipur","lucknow","kanpur","indore"
        ]

        text_lower = text.lower()
        for c in cities:
            if c in text_lower:
                result["location"] = c.title()
                break

    except Exception as e:
        logger.error(f"Basic field extraction failed: {e}")

    return result