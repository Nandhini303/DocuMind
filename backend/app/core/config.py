"""
Application configuration for DocuMind backend.
Loads settings from environment variables and .env file (checking backend and root).
"""

import os
from pathlib import Path
from typing import Optional, Set
from dotenv import load_dotenv

# Search for .env in current dir, backend/, and project root
CURRENT_DIR = Path(__file__).resolve().parent
POSSIBLE_ENV_PATHS = [
    CURRENT_DIR.parent.parent / ".env",          # DocuMind/backend/.env
    CURRENT_DIR.parent.parent.parent / ".env",   # DocuMind/.env
    Path.cwd() / ".env",
]

for p in POSSIBLE_ENV_PATHS:
    if p.exists():
        load_dotenv(dotenv_path=p)

# Also standard fallback
load_dotenv()


class Settings:
    """Application settings, credentials, and guardrails."""

    APP_NAME: str = "DocuMind AI"
    APP_VERSION: str = "2.0.0"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "production")
    DEBUG: bool = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")

    # Host & Port
    PORT: int = int(os.getenv("PORT", "8080"))
    HOST: str = os.getenv("HOST", "0.0.0.0")

    # Claude / Anthropic Settings
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
    ANTHROPIC_API_URL: str = os.getenv("ANTHROPIC_API_URL", "https://api.anthropic.com/v1/messages")

    # OpenAI Settings (Supports OPENAI_API_KEY or OPENAI_API)
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")
    OPENAI_API_URL: str = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")

    # Gemini Settings (Optional fallback)
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    # Document & RAG Guardrails
    MAX_UPLOAD_SIZE_BYTES: int = int(os.getenv("MAX_UPLOAD_SIZE_BYTES", str(15 * 1024 * 1024)))  # 15 MB
    MAX_DOCUMENT_CHARS: int = int(os.getenv("MAX_DOCUMENT_CHARS", "500000"))  # Up to ~100k words
    DIRECT_PROCESSING_CHAR_LIMIT: int = int(os.getenv("DIRECT_PROCESSING_CHAR_LIMIT", "12000"))
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    TOP_K_CHUNKS: int = int(os.getenv("TOP_K_CHUNKS", "5"))

    # Security & Allowed Extensions
    ALLOWED_EXTENSIONS: Set[str] = {".txt", ".pdf"}

    # Mock mode flag for offline tests or demo without API keys
    ENABLE_MOCK_FALLBACK: bool = os.getenv("ENABLE_MOCK_FALLBACK", "false").lower() in ("true", "1", "yes")

    @property
    def active_provider(self) -> str:
        """
        Determines active provider.
        Prioritizes Anthropic Claude, then OpenAI, then Gemini.
        """
        if self.ANTHROPIC_API_KEY:
            return "anthropic"
        if self.OPENAI_API_KEY:
            return "openai"
        if self.GEMINI_API_KEY:
            return "gemini"
        return "none"

    @property
    def is_api_key_configured(self) -> bool:
        """Returns True if any valid LLM API key is present."""
        return bool(self.ANTHROPIC_API_KEY or self.OPENAI_API_KEY or self.GEMINI_API_KEY)

    @property
    def active_model(self) -> str:
        """Returns the active model name."""
        if self.active_provider == "anthropic":
            return self.ANTHROPIC_MODEL
        if self.active_provider == "openai":
            return self.OPENAI_MODEL
        if self.active_provider == "gemini":
            return self.GEMINI_MODEL
        return self.ANTHROPIC_MODEL


settings = Settings()
