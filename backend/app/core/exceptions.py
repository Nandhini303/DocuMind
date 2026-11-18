"""
Domain exception definitions for DocuMind.
"""

from typing import Optional


class DocuMindError(Exception):
    """Base exception for DocuMind domain errors."""
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class DocumentParsingError(DocuMindError):
    """Raised when parsing or extracting text from a document fails."""
    def __init__(self, message: str = "Could not parse document", status_code: int = 400):
        super().__init__(message, status_code)


class FileSizeLimitExceededError(DocuMindError):
    """Raised when an uploaded file exceeds the allowed size limit."""
    def __init__(self, max_bytes: int, status_code: int = 413):
        max_mb = max_bytes / (1024 * 1024)
        super().__init__(f"File size exceeds the maximum limit of {max_mb:.1f} MB.", status_code)


class InvalidFileError(DocuMindError):
    """Raised when an uploaded file is invalid or has an unsupported extension."""
    def __init__(self, message: str = "Invalid file or unsupported format.", status_code: int = 400):
        super().__init__(message, status_code)


class MissingAPIKeyError(DocuMindError):
    """Raised when an LLM API call is requested without an API key configured."""
    def __init__(
        self,
        message: str = "Server is missing an API key. Please configure ANTHROPIC_API_KEY in your environment.",
        status_code: int = 500,
    ):
        super().__init__(message, status_code)


class LLMServiceError(DocuMindError):
    """Raised when the LLM provider fails or returns an error."""
    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message, status_code)
