import io
import logging
from typing import Dict, Any

from PIL import Image
import pytesseract

from app.services.classify import predict_category
from app.services.extract import extract_attributes, extract_basic_fields

logger = logging.getLogger(__name__)


def extract_text_from_image(file_bytes: bytes) -> str:
    """
    Extract raw text from uploaded image using Tesseract OCR.

    Args:
        file_bytes: Raw image bytes.

    Returns:
        Extracted text string.
    """
    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        text = pytesseract.image_to_string(image)
        logger.info(f"OCR extracted {len(text)} characters.")
        return text.strip()
    except Exception as e:
        logger.error(f"OCR extraction failed: {e}")
        return ""


def build_auto_filled_form(text: str) -> Dict[str, Any]:
    """
    Convert OCR text into structured MSME registration form.

    Pipeline:
    OCR text -> category prediction -> attribute extraction -> basic field detection.

    Returns structured dict ready for frontend or API.
    """
    category, confidence = predict_category(text)
    attributes = extract_attributes(text)
    basic = extract_basic_fields(text)

    return {
        "business_name": basic.get("business_name"),
        "location": basic.get("location"),
        "predicted_category": category,
        "category_confidence": confidence,
        "attributes": attributes,
        "raw_text_preview": text[:500]
    }