"""
Security utilities and middleware for DocuMind.
"""

import re
from typing import Optional

from app.core.config import settings
from app.core.exceptions import InvalidFileError, FileSizeLimitExceededError


class SecurityHeadersMiddleware:
    """Pure ASGI middleware adding security-hardening HTTP headers without buffering."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_security_headers(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.extend([
                    (b"x-content-type-options", b"nosniff"),
                    (b"x-frame-options", b"sameorigin"),
                    (b"x-xss-protection", b"1; mode=block"),
                    (b"referrer-policy", b"strict-origin-when-cross-origin"),
                    (b"permissions-policy", b"geolocation=(), camera=(), microphone=()"),
                ])
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_with_security_headers)


def sanitize_text(text: Optional[str]) -> str:
    """
    Sanitize text input:
    - Removes null bytes
    - Strips non-printable control characters except standard whitespace (newlines, tabs)
    - Normalizes unicode carriage returns to standard newlines
    """
    if not text:
        return ""
    # Remove null bytes
    sanitized = text.replace("\x00", "")
    # Normalize CRLF / CR to LF
    sanitized = sanitized.replace("\r\n", "\n").replace("\r", "\n")
    # Remove ASCII control characters (0-8, 11-12, 14-31, 127)
    sanitized = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", "", sanitized)
    return sanitized.strip()


def validate_file_upload(filename: Optional[str], file_size: int, content_preview: bytes = b"") -> None:
    """
    Validates uploaded file against allowed extensions, size limits, and magic bytes.
    """
    if not filename:
        raise InvalidFileError("Missing file name.")

    lower_name = filename.lower()
    ext = "." + lower_name.split(".")[-1] if "." in lower_name else ""
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise InvalidFileError(
            f"Unsupported file type '{ext}'. Allowed extensions are: {', '.join(sorted(settings.ALLOWED_EXTENSIONS))}."
        )

    if file_size > settings.MAX_UPLOAD_SIZE_BYTES:
        raise FileSizeLimitExceededError(settings.MAX_UPLOAD_SIZE_BYTES)

    # Check magic bytes for PDF
    if ext == ".pdf" and len(content_preview) >= 4:
        if not content_preview.startswith(b"%PDF"):
            raise InvalidFileError("Uploaded file has .pdf extension but is not a valid PDF document.")
