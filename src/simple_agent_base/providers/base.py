from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from simple_agent_base.mcp import MCPApprovalRequest, MCPCallRecord
from simple_agent_base.types import ToolCallRequest

ConversationItem = dict[str, Any]


class ProviderResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    response_id: str | None = None
    output_text: str = ""
    output_data: BaseModel | None = None
    tool_calls: list[ToolCallRequest] = Field(default_factory=list)
    output_items: list[ConversationItem] = Field(default_factory=list)
    raw_response: dict[str, Any] | None = None


class ProviderTextDeltaEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["text_delta"] = "text_delta"
    delta: str


class ProviderCompletedEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["completed"] = "completed"
    response: ProviderResponse


class ProviderMCPCallStartedEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["mcp_call_started"] = "mcp_call_started"
    mcp_call: MCPCallRecord


class ProviderMCPCallCompletedEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["mcp_call_completed"] = "mcp_call_completed"
    mcp_call: MCPCallRecord


class ProviderMCPApprovalRequestedEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["mcp_approval_requested"] = "mcp_approval_requested"
    mcp_approval: MCPApprovalRequest


ProviderEvent = (
    ProviderTextDeltaEvent
    | ProviderCompletedEvent
    | ProviderMCPCallStartedEvent
    | ProviderMCPCallCompletedEvent
    | ProviderMCPApprovalRequestedEvent
)


class Provider(Protocol):
    async def create_response(
        self,
        *,
        input_items: Sequence[ConversationItem],
        tools: Sequence[dict[str, Any]],
        response_model: type[BaseModel] | None = None,
    ) -> ProviderResponse: ...

    async def stream_response(
        self,
        *,
        input_items: Sequence[ConversationItem],
        tools: Sequence[dict[str, Any]],
        response_model: type[BaseModel] | None = None,
    ) -> AsyncIterator[ProviderEvent]: ...

    async def close(self) -> None: ...
