"""Safe, small document parser strategy for TXT, Markdown, PDF, and DOCX."""
from __future__ import annotations

from io import BytesIO


SUPPORTED_MIME_TYPES = {
    "text/plain": ".txt", "text/markdown": ".md", "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
}


def parse_upload(filename: str, mime_type: str, content: bytes) -> tuple[str, str]:
    suffix = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    expected_suffix = SUPPORTED_MIME_TYPES.get(mime_type)
    if expected_suffix and suffix != expected_suffix.lstrip("."):
        raise ValueError("File extension does not match the supplied content type")
    if suffix in {"txt", "md"}:
        return content.decode("utf-8"), f".{suffix}"
    if suffix == "pdf":
        from pypdf import PdfReader
        return "\n".join(page.extract_text() or "" for page in PdfReader(BytesIO(content)).pages), ".pdf"
    if suffix == "docx":
        from docx import Document as DocxDocument
        return "\n".join(paragraph.text for paragraph in DocxDocument(BytesIO(content)).paragraphs), ".docx"
    raise ValueError("Only TXT, Markdown, PDF, and DOCX files are supported")
