from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Any = None
    tool_calls: Optional[list[dict[str, Any]]] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: Optional[float] = None
    stream: bool = False
    tools: Optional[list[dict[str, Any]]] = None
    tool_choice: Optional[Any] = None
    provider: Optional[str] = None
    alias: Optional[str] = None
    strict_alias: bool = True

    model_config = ConfigDict(extra="allow")



class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None


class Usage(BaseModel):
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Optional[Usage] = None


class ChatCompletionChunkChoiceDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[list[dict[str, Any]]] = None



class ChatCompletionChunkChoice(BaseModel):
    index: int
    delta: ChatCompletionChunkChoiceDelta
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionChunkChoice]


def sse_pack(payload: str) -> str:
    return f"data: {payload}\n\n"
