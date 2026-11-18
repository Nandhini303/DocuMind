"""
LLM service for DocuMind.
Provides native async streaming with Anthropic Claude API (primary),
OpenAI API (when configured in .env), optional Gemini fallback,
and mock fallback for offline tests.
"""

import json
import logging
from typing import AsyncGenerator, Optional, Dict, Any
import httpx

from app.core.config import settings
from app.core.exceptions import MissingAPIKeyError, LLMServiceError
from app.models.schemas import AnalysisMode

logger = logging.getLogger("documind.llm")

# System prompts for each analysis mode
SYSTEM_PROMPTS: Dict[AnalysisMode, str] = {
    AnalysisMode.SUMMARIZE: (
        "You are an expert editor. Read the document below and produce a "
        "concise, well-structured summary. Use short paragraphs or bullet "
        "points. Capture the main argument, key facts, and conclusion. "
        "Do not add outside information."
    ),
    AnalysisMode.KEY_POINTS: (
        "You are an analyst. Read the document below and extract the most "
        "important points as a clean bulleted list, ordered by importance. "
        "Each bullet should be one clear sentence. Include a one-line "
        "'Bottom line' at the end."
    ),
    AnalysisMode.SIMPLIFY: (
        "You are a patient teacher. Rewrite the document below in plain, "
        "simple language a general reader (no specialist background) could "
        "understand. Keep it faithful to the original meaning, avoid "
        "jargon, and keep roughly the same length as the original."
    ),
}


class LLMService:
    """Async LLM orchestration service supporting Claude, OpenAI, and Gemini."""

    def __init__(self):
        self.anthropic_key = settings.ANTHROPIC_API_KEY
        self.openai_key = settings.OPENAI_API_KEY
        self.gemini_key = settings.GEMINI_API_KEY

    def get_system_prompt(self, mode: AnalysisMode) -> str:
        """Retrieves system prompt for specified mode."""
        return SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPTS[AnalysisMode.SUMMARIZE])

    async def stream_analysis(
        self,
        mode: AnalysisMode,
        content: str,
        custom_system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Streams generated text deltas from the configured LLM provider.
        Yields raw text tokens (str).
        """
        system_instruction = custom_system_prompt or self.get_system_prompt(mode)

        # Check for mock mode (used in test suite or offline dev)
        if settings.ENABLE_MOCK_FALLBACK:
            async for delta in self._stream_mock_response(mode, content):
                yield delta
            return

        if not settings.is_api_key_configured:
            raise MissingAPIKeyError(
                "Server is missing an API key. Set ANTHROPIC_API_KEY or OPENAI_API in your environment."
            )

        # Provider: Anthropic Claude (Primary requested)
        if settings.active_provider == "anthropic":
            async for delta in self._stream_claude(system_instruction, content):
                yield delta
            return

        # Provider: OpenAI (Detected from user's .env)
        if settings.active_provider == "openai":
            async for delta in self._stream_openai(system_instruction, content):
                yield delta
            return

        # Fallback: Google Gemini
        if settings.active_provider == "gemini":
            async for delta in self._stream_gemini(system_instruction, content):
                yield delta
            return

        raise MissingAPIKeyError("No valid LLM provider configured.")

    async def _stream_claude(
        self,
        system_prompt: str,
        user_content: str,
    ) -> AsyncGenerator[str, None]:
        """
        Streams from Anthropic Claude Messages API using asynchronous HTTP SSE.
        """
        headers = {
            "x-api-key": self.anthropic_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        payload = {
            "model": settings.ANTHROPIC_MODEL,
            "max_tokens": 2500,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": user_content}
            ],
            "stream": True,
        }

        timeout = httpx.Timeout(connect=15.0, read=60.0, write=15.0, pool=15.0)

        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                async with client.stream(
                    "POST",
                    settings.ANTHROPIC_API_URL,
                    headers=headers,
                    json=payload,
                ) as response:
                    if response.status_code != 200:
                        error_bytes = await response.aread()
                        error_msg = error_bytes.decode("utf-8", errors="replace")
                        logger.error(f"Anthropic API returned error {response.status_code}: {error_msg}")
                        try:
                            err_json = json.loads(error_msg)
                            detail = err_json.get("error", {}).get("message", error_msg)
                        except Exception:
                            detail = error_msg
                        raise LLMServiceError(f"Anthropic API error ({response.status_code}): {detail}")

                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            event = json.loads(data_str)
                            event_type = event.get("type")
                            if event_type == "content_block_delta":
                                delta_text = event.get("delta", {}).get("text", "")
                                if delta_text:
                                    yield delta_text
                            elif event_type == "error":
                                err_msg = event.get("error", {}).get("message", "Stream error")
                                raise LLMServiceError(err_msg)
                        except json.JSONDecodeError:
                            continue

            except httpx.RequestError as exc:
                logger.error(f"Network error connecting to Anthropic API: {exc}")
                raise LLMServiceError(f"Network error communicating with Claude API: {exc}")

    async def _stream_openai(
        self,
        system_prompt: str,
        user_content: str,
    ) -> AsyncGenerator[str, None]:
        """
        Streams from OpenAI Chat Completions API using asynchronous HTTP SSE.
        """
        headers = {
            "Authorization": f"Bearer {self.openai_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": settings.OPENAI_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "stream": True,
        }

        timeout = httpx.Timeout(connect=15.0, read=60.0, write=15.0, pool=15.0)

        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                async with client.stream(
                    "POST",
                    settings.OPENAI_API_URL,
                    headers=headers,
                    json=payload,
                ) as response:
                    if response.status_code != 200:
                        error_bytes = await response.aread()
                        error_msg = error_bytes.decode("utf-8", errors="replace")
                        logger.error(f"OpenAI API returned error {response.status_code}: {error_msg}")
                        try:
                            err_json = json.loads(error_msg)
                            detail = err_json.get("error", {}).get("message", error_msg)
                        except Exception:
                            detail = error_msg
                        raise LLMServiceError(f"OpenAI API error ({response.status_code}): {detail}")

                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            event = json.loads(data_str)
                            choices = event.get("choices", [])
                            if choices:
                                delta_text = choices[0].get("delta", {}).get("content", "")
                                if delta_text:
                                    yield delta_text
                        except json.JSONDecodeError:
                            continue

            except httpx.RequestError as exc:
                logger.error(f"Network error connecting to OpenAI API: {exc}")
                raise LLMServiceError(f"Network error communicating with OpenAI API: {exc}")

    async def _stream_gemini(
        self,
        system_prompt: str,
        user_content: str,
    ) -> AsyncGenerator[str, None]:
        """
        Fallback streamer for Google Gemini when GEMINI_API_KEY is configured.
        """
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self.gemini_key)
            response = client.models.generate_content_stream(
                model=settings.GEMINI_MODEL,
                contents=user_content,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    max_output_tokens=2500,
                ),
            )
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as exc:
            logger.error(f"Gemini API error: {exc}")
            raise LLMServiceError(f"Gemini API error: {exc}")

    async def _stream_mock_response(
        self,
        mode: AnalysisMode,
        content: str,
    ) -> AsyncGenerator[str, None]:
        """
        Simulated streamer used for offline development and end-to-end testing.
        """
        words = content.split()[:50]
        preview = " ".join(words)

        if mode == AnalysisMode.SUMMARIZE:
            lines = [
                "### Document Summary\n\n",
                f"**Overview:** The provided document discusses key themes starting with: *\"{preview}...\"*\n\n",
                "- **Core Argument:** The text addresses fundamental domain concepts and requirements.\n",
                "- **Key Findings:** Key evidence and analytical points were successfully extracted.\n",
                "- **Conclusion:** The document provides clear actionable takeaways for implementation.\n",
            ]
        elif mode == AnalysisMode.KEY_POINTS:
            lines = [
                "### Key Points\n\n",
                "1. **Core Insight:** Comprehensive analysis was performed across all sections.\n",
                "2. **Critical Detail:** Document structure and arguments were preserved.\n",
                "3. **Scope & Metrics:** Text content was verified for integrity and clarity.\n\n",
                "**Bottom Line:** The document provides clear, structured insights for immediate use.\n",
            ]
        else:
            lines = [
                "### Simplified Version\n\n",
                "Here is the document explained in everyday terms:\n\n",
                f"Think of this text as explaining: {preview}...\n\n",
                "In plain words, it breaks down complex ideas into manageable, easy-to-understand parts.\n",
            ]

        for line in lines:
            yield line
