from __future__ import annotations

import socket
import subprocess
import sys
import time
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from mcp import types as mcp_types
from pydantic import BaseModel, ValidationError

from simple_agent_base import Agent, AgentConfig, MCPServer, tool
from simple_agent_base.errors import MCPApprovalRequiredError, ToolExecutionError, ToolRegistrationError
from simple_agent_base.mcp import normalize_mcp_tool_result
from simple_agent_base.providers.base import (
    ConversationItem,
    ProviderCompletedEvent,
    ProviderEvent,
    ProviderResponse,
)

FIXTURE_SERVER = Path(__file__).parent / "fixtures" / "mcp_demo_server.py"


class FakeProvider:
    def __init__(
        self,
        responses: list[ProviderResponse] | None = None,
        stream_sequences: list[list[ProviderEvent]] | None = None,
    ) -> None:
        self.responses = list(responses or [])
        self.stream_sequences = list(stream_sequences or [])
        self.calls: list[dict[str, Any]] = []
        self.stream_calls: list[dict[str, Any]] = []

    async def create_response(
        self,
        *,
        input_items: Sequence[ConversationItem],
        tools: Sequence[dict[str, Any]],
        response_model: type[BaseModel] | None = None,
    ) -> ProviderResponse:
        self.calls.append(
            {
                "input_items": list(input_items),
                "tools": list(tools),
                "response_model": response_model,
            }
        )
        return self.responses.pop(0)

    async def stream_response(
        self,
        *,
        input_items: Sequence[ConversationItem],
        tools: Sequence[dict[str, Any]],
        response_model: type[BaseModel] | None = None,
    ) -> AsyncIterator[ProviderEvent]:
        self.stream_calls.append(
            {
                "input_items": list(input_items),
                "tools": list(tools),
                "response_model": response_model,
            }
        )
        for event in self.stream_sequences.pop(0):
            yield event

    async def close(self) -> None:
        return None


@pytest.fixture
def stdio_server() -> MCPServer:
    return MCPServer.stdio(
        name="demo",
        command=sys.executable,
        args=[str(FIXTURE_SERVER), "stdio"],
    )


@pytest.fixture
def http_server() -> str:
    port = _free_port()
    process = subprocess.Popen(
        [sys.executable, str(FIXTURE_SERVER), "http", str(port)],
        cwd=str(FIXTURE_SERVER.parent.parent.parent),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}/mcp"
    try:
        _wait_for_http_server(url)
        yield url
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def test_mcp_server_stdio_constructor() -> None:
    server = MCPServer.stdio(
        name="demo",
        command="python",
        args=["server.py"],
        env={"TOKEN": "abc"},
        cwd="C:/tmp",
        allowed_tools=["echo"],
        require_approval=True,
    )

    assert server.transport == "stdio"
    assert server.command == "python"
    assert server.args == ["server.py"]
    assert server.env == {"TOKEN": "abc"}
    assert server.cwd == "C:/tmp"
    assert server.allowed_tools == ["echo"]
    assert server.require_approval is True


def test_mcp_server_http_constructor() -> None:
    server = MCPServer.http(
        name="demo-http",
        url="https://example.com/mcp",
        headers={"Authorization": "Bearer 123"},
        allowed_tools=["echo"],
    )

    assert server.transport == "streamable_http"
    assert server.url == "https://example.com/mcp"
    assert server.headers == {"Authorization": "Bearer 123"}
    assert server.allowed_tools == ["echo"]
    assert server.require_approval is False


def test_mcp_server_rejects_invalid_transport_fields() -> None:
    with pytest.raises(ValidationError):
        MCPServer(name="bad", transport="stdio")

    with pytest.raises(ValidationError):
        MCPServer(
            name="bad-http",
            transport="streamable_http",
            url="https://example.com/mcp",
            command="python",
        )


def test_normalize_mcp_tool_result_prefers_text_blocks() -> None:
    result = mcp_types.CallToolResult(
        content=[
            mcp_types.TextContent(type="text", text="one"),
            mcp_types.TextContent(type="text", text="two"),
        ],
        isError=False,
    )

    assert normalize_mcp_tool_result(result) == "one\ntwo"


def test_normalize_mcp_tool_result_uses_structured_content() -> None:
    result = mcp_types.CallToolResult(
        content=[],
        structuredContent={"name": "Ada"},
        isError=False,
    )

    assert normalize_mcp_tool_result(result) == '{"name": "Ada"}'


def test_normalize_mcp_tool_result_falls_back_to_full_payload() -> None:
    result = mcp_types.CallToolResult(content=[], isError=False)

    output = normalize_mcp_tool_result(result)

    assert '"isError": false' in output


@pytest.mark.asyncio
async def test_agent_discovers_and_executes_stdio_mcp_tool(stdio_server: MCPServer) -> None:
    provider = FakeProvider(
        responses=[
            ProviderResponse(
                response_id="resp_1",
                tool_calls=[
                    {
                        "call_id": "call_1",
                        "name": "demo__echo",
                        "arguments": {"message": "hello"},
                        "raw_arguments": '{"message":"hello"}',
                    }
                ],
                output_items=[],
                raw_response={"id": "resp_1"},
            ),
            ProviderResponse(
                response_id="resp_2",
                output_text="done",
                output_items=[],
                raw_response={"id": "resp_2"},
            ),
        ]
    )
    agent = Agent(
        config=AgentConfig(model="gpt-5"),
        provider=provider,
        mcp_servers=[stdio_server],
    )

    try:
        result = await agent.run("Use the MCP tool.")
    finally:
        await agent.aclose()

    assert [tool["name"] for tool in provider.calls[0]["tools"]] == [
        "demo__echo",
        "demo__add",
        "demo__structured_value",
        "demo__fail",
    ]
    assert result.output_text == "done"
    assert result.tool_results[0].output == "echo:hello"
    assert result.mcp_calls[0].server_name == "demo"
    assert result.mcp_calls[0].name == "echo"
    assert result.mcp_calls[0].output == "echo:hello"


@pytest.mark.asyncio
async def test_agent_filters_allowed_mcp_tools(stdio_server: MCPServer) -> None:
    provider = FakeProvider(
        responses=[
            ProviderResponse(
                response_id="resp_1",
                output_text="done",
                output_items=[],
                raw_response={"id": "resp_1"},
            )
        ]
    )
    agent = Agent(
        config=AgentConfig(model="gpt-5"),
        provider=provider,
        mcp_servers=[
            stdio_server.model_copy(update={"allowed_tools": ["echo"]})
        ],
    )

    try:
        await agent.run("List tools.")
    finally:
        await agent.aclose()

    assert [tool["name"] for tool in provider.calls[0]["tools"]] == ["demo__echo"]


@pytest.mark.asyncio
async def test_agent_approval_handler_can_deny_mcp_call(stdio_server: MCPServer) -> None:
    provider = FakeProvider(
        responses=[
            ProviderResponse(
                response_id="resp_1",
                tool_calls=[
                    {
                        "call_id": "call_1",
                        "name": "demo__echo",
                        "arguments": {"message": "hello"},
                        "raw_arguments": '{"message":"hello"}',
                    }
                ],
                output_items=[],
                raw_response={"id": "resp_1"},
            ),
            ProviderResponse(
                response_id="resp_2",
                output_text="handled",
                output_items=[],
                raw_response={"id": "resp_2"},
            ),
        ]
    )
    agent = Agent(
        config=AgentConfig(model="gpt-5"),
        provider=provider,
        mcp_servers=[stdio_server.model_copy(update={"require_approval": True})],
        approval_handler=lambda request: False,
    )

    try:
        result = await agent.run("Use the MCP tool.")
    finally:
        await agent.aclose()

    assert result.output_text == "handled"
    assert result.tool_results[0].output == "MCP tool call denied by approval handler."
    assert result.mcp_calls[0].error == "MCP tool call denied by approval handler."


@pytest.mark.asyncio
async def test_agent_requires_approval_handler_for_gated_mcp_call(stdio_server: MCPServer) -> None:
    provider = FakeProvider(
        responses=[
            ProviderResponse(
                response_id="resp_1",
                tool_calls=[
                    {
                        "call_id": "call_1",
                        "name": "demo__echo",
                        "arguments": {"message": "hello"},
                        "raw_arguments": '{"message":"hello"}',
                    }
                ],
                output_items=[],
                raw_response={"id": "resp_1"},
            )
        ]
    )
    agent = Agent(
        config=AgentConfig(model="gpt-5"),
        provider=provider,
        mcp_servers=[stdio_server.model_copy(update={"require_approval": True})],
    )

    try:
        with pytest.raises(MCPApprovalRequiredError):
            await agent.run("Use the MCP tool.")
    finally:
        await agent.aclose()


@pytest.mark.asyncio
async def test_agent_surfaces_mcp_error_results_as_tool_failures(stdio_server: MCPServer) -> None:
    provider = FakeProvider(
        responses=[
            ProviderResponse(
                response_id="resp_1",
                tool_calls=[
                    {
                        "call_id": "call_1",
                        "name": "demo__fail",
                        "arguments": {"message": "boom"},
                        "raw_arguments": '{"message":"boom"}',
                    }
                ],
                output_items=[],
                raw_response={"id": "resp_1"},
            )
        ]
    )
    agent = Agent(
        config=AgentConfig(model="gpt-5"),
        provider=provider,
        mcp_servers=[stdio_server],
    )

    try:
        with pytest.raises(ToolExecutionError, match="boom"):
            await agent.run("Use the failing MCP tool.")
    finally:
        await agent.aclose()


@pytest.mark.asyncio
async def test_stream_emits_local_mcp_events(stdio_server: MCPServer) -> None:
    provider = FakeProvider(
        stream_sequences=[
            [
                ProviderCompletedEvent(
                    response=ProviderResponse(
                        response_id="resp_1",
                        tool_calls=[
                            {
                                "call_id": "call_1",
                                "name": "demo__echo",
                                "arguments": {"message": "hello"},
                                "raw_arguments": '{"message":"hello"}',
                            }
                        ],
                        output_items=[],
                        raw_response={"id": "resp_1"},
                    )
                )
            ],
            [
                ProviderCompletedEvent(
                    response=ProviderResponse(
                        response_id="resp_2",
                        output_text="done",
                        output_items=[],
                        raw_response={"id": "resp_2"},
                    )
                )
            ],
        ]
    )
    agent = Agent(
        config=AgentConfig(model="gpt-5"),
        provider=provider,
        mcp_servers=[stdio_server.model_copy(update={"require_approval": True})],
        approval_handler=lambda request: True,
    )

    try:
        events = [event async for event in agent.stream("Use the MCP tool.")]
    finally:
        await agent.aclose()

    assert [event.type for event in events] == [
        "tool_call_started",
        "mcp_approval_requested",
        "mcp_call_started",
        "mcp_call_completed",
        "tool_call_completed",
        "completed",
    ]
    approval = events[1].mcp_approval
    assert approval is not None
    assert approval.server_name == "demo"
    assert approval.name == "echo"
    mcp_completed = events[3].mcp_call
    assert mcp_completed is not None
    assert mcp_completed.server_name == "demo"
    assert mcp_completed.output == "echo:hello"


@tool(name="demo__echo")
async def conflicting_local_tool(message: str) -> str:
    """Conflicts with the MCP echo tool."""
    return message


@pytest.mark.asyncio
async def test_agent_rejects_conflicting_local_and_mcp_tool_names(stdio_server: MCPServer) -> None:
    provider = FakeProvider(
        responses=[
            ProviderResponse(
                response_id="resp_1",
                output_text="done",
                output_items=[],
                raw_response={"id": "resp_1"},
            )
        ]
    )
    agent = Agent(
        config=AgentConfig(model="gpt-5"),
        provider=provider,
        tools=[conflicting_local_tool],
        mcp_servers=[stdio_server],
    )

    try:
        with pytest.raises(ToolRegistrationError, match="demo__echo"):
            await agent.run("Hello.")
    finally:
        await agent.aclose()


@pytest.mark.asyncio
async def test_agent_executes_streamable_http_mcp_tool(http_server: str) -> None:
    provider = FakeProvider(
        responses=[
            ProviderResponse(
                response_id="resp_1",
                tool_calls=[
                    {
                        "call_id": "call_1",
                        "name": "demohttp__add",
                        "arguments": {"a": 2, "b": 3},
                        "raw_arguments": '{"a":2,"b":3}',
                    }
                ],
                output_items=[],
                raw_response={"id": "resp_1"},
            ),
            ProviderResponse(
                response_id="resp_2",
                output_text="done",
                output_items=[],
                raw_response={"id": "resp_2"},
            ),
        ]
    )
    agent = Agent(
        config=AgentConfig(model="gpt-5"),
        provider=provider,
        mcp_servers=[MCPServer.http(name="demohttp", url=http_server)],
    )

    try:
        result = await agent.run("Use the HTTP MCP tool.")
    finally:
        await agent.aclose()

    assert result.tool_results[0].output == "5"
    assert result.mcp_calls[0].server_name == "demohttp"
    assert result.mcp_calls[0].name == "add"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_http_server(url: str) -> None:
    host = "127.0.0.1"
    port = int(url.rsplit(":", 1)[1].split("/", 1)[0])
    deadline = time.time() + 10
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return
        except Exception as exc:  # pragma: no cover - best-effort startup polling
            last_error = exc
        time.sleep(0.1)
    raise RuntimeError(f"HTTP MCP fixture did not start: {last_error}")
