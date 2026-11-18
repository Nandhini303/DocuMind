"""
Server-Sent Events (SSE) utilities.
Provides optimized JSON event serialization and safe client disconnection handling.
"""

import asyncio
import json
import logging
from typing import AsyncGenerator
from starlette.requests import Request

logger = logging.getLogger("documind.sse")


def format_sse_delta(delta: str) -> str:
    """Formats a text delta into an SSE data packet."""
    payload = json.dumps({"delta": delta})
    return f"data: {payload}\n\n"


def format_sse_error(error_message: str) -> str:
    """Formats an error message into an SSE data packet."""
    payload = json.dumps({"error": error_message})
    return f"data: {payload}\n\n"


def format_sse_done() -> str:
    """Formats the final completion event into an SSE data packet."""
    payload = json.dumps({"done": True})
    return f"data: {payload}\n\n"


async def is_client_disconnected(request: Request) -> bool:
    """
    Safely checks if the client disconnected without blocking ASGI receive queue.
    """
    try:
        # Non-blocking poll with 1ms timeout to prevent hanging on TestClient or keep-alive
        return await asyncio.wait_for(request.is_disconnected(), timeout=0.002)
    except (asyncio.TimeoutError, Exception):
        return False


async def sse_event_stream(
    generator: AsyncGenerator[str, None],
    request: Request,
) -> AsyncGenerator[str, None]:
    """
    Wraps an async generator of text deltas into formatted SSE events.
    Monitors client connection state and stops generation if client disconnects.
    """
    try:
        async for chunk in generator:
            # Check for client disconnect to avoid wasting LLM tokens
            if await is_client_disconnected(request):
                logger.info("Client disconnected. Aborting LLM stream.")
                break

            if chunk:
                yield format_sse_delta(chunk)

        yield format_sse_done()

    except Exception as exc:
        logger.error(f"Error in SSE stream: {exc}", exc_info=True)
        yield format_sse_error(str(exc))
