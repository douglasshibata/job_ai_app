"""Security utilities: HTML sanitization, Trafilatura markdown conversion, and PII masking."""
import re
import bleach
import trafilatura

ALLOWED_HTML_TAGS = ["p", "br", "ul", "ol", "li", "strong", "em", "b", "i", "h1", "h2", "h3", "h4", "h5", "h6", "code", "pre"]
ALLOWED_HTML_ATTRS = {}

def clean_html(raw_html: str) -> str:
    """Sanitizes raw HTML to safe tags only using bleach."""
    if not raw_html:
        return ""
    return bleach.clean(raw_html, tags=ALLOWED_HTML_TAGS, attributes=ALLOWED_HTML_ATTRS, strip=True)

def sanitize_job_description(content: str) -> str:
    """
    Cleans raw job description using trafilatura extraction if HTML is detected,
    or falls back to bleach text cleaning.
    """
    if not content:
        return ""

    content_str = str(content).strip()

    if "<" in content_str and ">" in content_str:
        extracted = trafilatura.extract(
            content_str,
            include_comments=False,
            include_tables=True,
            include_links=False,
            output_format="txt"
        )
        if extracted and len(extracted.strip()) > 20:
            return extracted.strip()

        cleaned = bleach.clean(content_str, tags=[], strip=True)
        return re.sub(r"\s+", " ", cleaned).strip()

    return re.sub(r"\n{3,}", "\n\n", content_str).strip()

def mask_pii(text: str) -> str:
    """
    Masks sensitive personal information (emails, phone numbers, CPFs).
    """
    if not text:
        return ""

    masked = re.sub(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "[EMAIL_REDACTED]", text)
    masked = re.sub(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b", "[CPF_REDACTED]", masked)
    masked = re.sub(r"(\+?\d{1,3}[-.\s]?)?(\(?\d{2,3}\)?[-.\s]?)?\d{4,5}[-.\s]?\d{4}\b", "[PHONE_REDACTED]", masked)
    return masked
