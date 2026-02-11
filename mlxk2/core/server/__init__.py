"""
MLX Knife Server Module.

Provides the refactored server structure with modular handlers and streaming.
The monolithic server_base.py has been split into:
- streaming.py: SSE streaming generators
- handlers/: Request handlers (chat, audio, models)
- model_manager.py: Model lifecycle management (Phase 2 refactoring)

This module provides re-exports for the new modular structure.
The original server_base.py imports from here and maintains backwards compatibility.
"""

# Re-export from streaming
from .streaming import (
    generate_completion_stream,
    generate_chat_stream,
    stream_vision_chunks,
    emulate_sse_stream,
)

# Re-export handlers
from .handlers.chat import (
    ChatHandlerContext,
    handle_text_chat_completion,
    handle_vision_chat_completion,
    process_vision_chunks_server,
)
from .handlers.audio import (
    handle_audio_chat_completion,
    handle_transcription,
)
from .handlers.models import handle_list_models

# Re-export ModelManager (Phase 2 refactoring)
from .model_manager import (
    ModelManager,
    get_available_memory_bytes,
    get_memory_pressure,
    wait_for_memory_release,
)

__all__ = [
    # Streaming
    "generate_completion_stream",
    "generate_chat_stream",
    "stream_vision_chunks",
    "emulate_sse_stream",
    # Chat handlers
    "ChatHandlerContext",
    "handle_text_chat_completion",
    "handle_vision_chat_completion",
    "process_vision_chunks_server",
    # Audio handlers
    "handle_audio_chat_completion",
    "handle_transcription",
    # Models handler
    "handle_list_models",
    # ModelManager (Phase 2)
    "ModelManager",
    "get_available_memory_bytes",
    "get_memory_pressure",
    "wait_for_memory_release",
]
