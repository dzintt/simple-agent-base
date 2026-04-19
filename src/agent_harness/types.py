from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


@dataclass(slots=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Awaitable[Any]] = field(repr=False)
    arguments_model: type[BaseModel] = field(repr=False)
    strict: bool = True


class ToolCallRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    call_id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    raw_arguments: str = "{}"


class ToolExecutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    call_id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    output: str
    raw_output: Any | None = None


ImageDetail = Literal["low", "high", "auto", "original"]


class TextPart(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["text"] = "text"
    text: str

    def __init__(self, text: str | None = None, **data: Any) -> None:
        if text is not None and "text" not in data:
            data["text"] = text
        if "type" not in data:
            data["type"] = "text"
        super().__init__(**data)


class ImagePart(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["image"] = "image"
    image_url: str
    detail: ImageDetail = "auto"

    @classmethod
    def from_url(cls, url: str, detail: ImageDetail = "auto") -> ImagePart:
        if not url.strip():
            raise ValueError("Image URL cannot be empty.")
        return cls(image_url=url, detail=detail)

    @classmethod
    def from_file(cls, path: str, detail: ImageDetail = "auto") -> ImagePart:
        file_path = Path(path)
        if not file_path.exists():
            raise ValueError(f"Image file does not exist: {file_path}")
        if not file_path.is_file():
            raise ValueError(f"Image path is not a file: {file_path}")

        mime_type, _ = mimetypes.guess_type(file_path.name)
        supported_mime_types = {
            "image/png",
            "image/jpeg",
            "image/webp",
            "image/gif",
        }
        if mime_type not in supported_mime_types:
            raise ValueError(
                "Unsupported image file type. Supported formats are PNG, JPEG, WEBP, and GIF."
            )

        try:
            raw_bytes = file_path.read_bytes()
        except OSError as exc:
            raise ValueError(f"Could not read image file: {file_path}") from exc

        encoded = base64.b64encode(raw_bytes).decode("ascii")
        data_url = f"data:{mime_type};base64,{encoded}"
        return cls(image_url=data_url, detail=detail)


ContentPart = Annotated[TextPart | ImagePart, Field(discriminator="type")]


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant", "developer", "system"]
    content: str | list[ContentPart]


MessageInput = ChatMessage | dict[str, object]


class AgentRunResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_text: str
    output_data: BaseModel | None = None
    response_id: str | None = None
    tool_results: list[ToolExecutionResult] = Field(default_factory=list)
    raw_responses: list[dict[str, Any]] = Field(default_factory=list)


class AgentEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal[
        "text_delta",
        "tool_call_started",
        "tool_call_completed",
        "completed",
        "error",
    ]
    delta: str | None = None
    tool_call: ToolCallRequest | None = None
    tool_result: ToolExecutionResult | None = None
    result: AgentRunResult | None = None
    error: str | None = None
