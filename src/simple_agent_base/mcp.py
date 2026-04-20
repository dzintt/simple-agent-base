from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MCPServer(BaseModel):
    """Configuration for a hosted (remote) MCP server.

    Mirrors OpenAI's ``tools=[{"type": "mcp", ...}]`` entry. Exactly one of
    ``server_url`` or ``connector_id`` is required.
    """

    model_config = ConfigDict(extra="forbid")

    server_label: str
    server_url: str | None = None
    connector_id: str | None = None
    authorization: str | None = None
    headers: dict[str, str] | None = None
    allowed_tools: list[str] | dict[str, Any] | None = None
    require_approval: Literal["always", "never"] | dict[str, Any] | None = "never"
    server_description: str | None = None

    @model_validator(mode="after")
    def _validate_source(self) -> MCPServer:
        provided = [value for value in (self.server_url, self.connector_id) if value]
        if len(provided) != 1:
            raise ValueError(
                "MCPServer requires exactly one of server_url or connector_id."
            )
        return self

    def to_tool_param(self) -> dict[str, Any]:
        """Serialize to the OpenAI Responses API ``{"type": "mcp", ...}`` shape."""
        payload: dict[str, Any] = {
            "type": "mcp",
            "server_label": self.server_label,
        }
        if self.server_url is not None:
            payload["server_url"] = self.server_url
        if self.connector_id is not None:
            payload["connector_id"] = self.connector_id
        if self.authorization is not None:
            payload["authorization"] = self.authorization
        if self.headers is not None:
            payload["headers"] = dict(self.headers)
        if self.server_description is not None:
            payload["server_description"] = self.server_description
        if self.allowed_tools is not None:
            if isinstance(self.allowed_tools, list):
                payload["allowed_tools"] = list(self.allowed_tools)
            else:
                payload["allowed_tools"] = dict(self.allowed_tools)
        if self.require_approval is not None:
            if isinstance(self.require_approval, str):
                payload["require_approval"] = self.require_approval
            else:
                payload["require_approval"] = dict(self.require_approval)
        return payload


class MCPApprovalRequest(BaseModel):
    """A request from the model to approve an MCP tool invocation."""

    model_config = ConfigDict(extra="forbid")

    id: str
    server_label: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class MCPCallRecord(BaseModel):
    """A record of an MCP tool invocation performed by the Responses API."""

    model_config = ConfigDict(extra="forbid")

    id: str | None = None
    server_label: str | None = None
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    output: str | None = None
    error: str | None = None


ApprovalHandler = Callable[[MCPApprovalRequest], "bool | Awaitable[bool]"]
