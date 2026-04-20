from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from simple_agent_base import (
    Agent,
    AgentConfig,
    MCPApprovalRequest,
    MCPApprovalRequiredError,
    MCPServer,
)
from simple_agent_base.providers.base import (
    ConversationItem,
    ProviderCompletedEvent,
    ProviderEvent,
    ProviderMCPApprovalRequestedEvent,
    ProviderMCPCallCompletedEvent,
    ProviderMCPCallStartedEvent,
    ProviderResponse,
    ProviderTextDeltaEvent,
)


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


# ---------------------------------------------------------------------------
# MCPServer.to_tool_param shape
# ---------------------------------------------------------------------------


def test_mcp_server_to_tool_param_minimal() -> None:
    server = MCPServer(server_label="deepwiki", server_url="https://mcp.deepwiki.com/mcp")

    assert server.to_tool_param() == {
        "type": "mcp",
        "server_label": "deepwiki",
        "server_url": "https://mcp.deepwiki.com/mcp",
    }


def test_mcp_server_to_tool_param_full() -> None:
    server = MCPServer(
        server_label="gh",
        server_url="https://gitmcp.io/owner/repo",
        authorization="Bearer abc",
        headers={"X-Extra": "1"},
        allowed_tools=["list_files", "read_file"],
        require_approval={"always": {"tool_names": ["delete_file"]}},
        server_description="GitHub MCP",
    )

    payload = server.to_tool_param()
    assert payload["type"] == "mcp"
    assert payload["server_label"] == "gh"
    assert payload["server_url"] == "https://gitmcp.io/owner/repo"
    assert payload["authorization"] == "Bearer abc"
    assert payload["headers"] == {"X-Extra": "1"}
    assert payload["allowed_tools"] == ["list_files", "read_file"]
    assert payload["require_approval"] == {"always": {"tool_names": ["delete_file"]}}
    assert payload["server_description"] == "GitHub MCP"


def test_mcp_server_requires_exactly_one_source() -> None:
    with pytest.raises(ValidationError):
        MCPServer(server_label="oops")
    with pytest.raises(ValidationError):
        MCPServer(
            server_label="oops",
            server_url="https://x",
            connector_id="connector_gmail",
        )


def test_mcp_server_accepts_connector_id() -> None:
    server = MCPServer(server_label="gmail", connector_id="connector_gmail")
    payload = server.to_tool_param()
    assert payload["connector_id"] == "connector_gmail"
    assert "server_url" not in payload


# ---------------------------------------------------------------------------
# Agent wiring
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_forwards_mcp_tool_in_tools_kwarg() -> None:
    provider = FakeProvider(
        [
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
            MCPServer(server_label="deepwiki", server_url="https://mcp.deepwiki.com/mcp")
        ],
    )

    await agent.run("Hi.")

    tools = provider.calls[0]["tools"]
    assert len(tools) == 1
    assert tools[0]["type"] == "mcp"
    assert tools[0]["server_label"] == "deepwiki"


@pytest.mark.asyncio
async def test_agent_collects_mcp_calls_from_output_items() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                response_id="resp_1",
                output_text="All done.",
                output_items=[
                    {
                        "type": "mcp_list_tools",
                        "server_label": "deepwiki",
                        "tools": [],
                    },
                    {
                        "type": "mcp_call",
                        "id": "mcp_1",
                        "server_label": "deepwiki",
                        "name": "ask_question",
                        "arguments": '{"q":"what"}',
                        "output": "answer",
                    },
                ],
                raw_response={"id": "resp_1"},
            )
        ]
    )
    agent = Agent(
        config=AgentConfig(model="gpt-5"),
        provider=provider,
        mcp_servers=[MCPServer(server_label="deepwiki", server_url="https://x")],
    )

    result = await agent.run("Hi.")

    assert [c.name for c in result.mcp_calls] == ["ask_question"]
    assert result.mcp_calls[0].arguments == {"q": "what"}
    assert result.mcp_calls[0].output == "answer"


# ---------------------------------------------------------------------------
# Approval loop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_approval_handler_is_called_and_response_item_appended() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                response_id="resp_1",
                output_text="",
                output_items=[
                    {
                        "type": "mcp_approval_request",
                        "id": "appr_1",
                        "server_label": "deepwiki",
                        "name": "ask_question",
                        "arguments": '{"q":"what"}',
                    }
                ],
                raw_response={"id": "resp_1"},
            ),
            ProviderResponse(
                response_id="resp_2",
                output_text="All good.",
                output_items=[],
                raw_response={"id": "resp_2"},
            ),
        ]
    )

    seen: list[MCPApprovalRequest] = []

    def handler(req: MCPApprovalRequest) -> bool:
        seen.append(req)
        return True

    agent = Agent(
        config=AgentConfig(model="gpt-5"),
        provider=provider,
        mcp_servers=[
            MCPServer(
                server_label="deepwiki",
                server_url="https://x",
                require_approval="always",
            )
        ],
        approval_handler=handler,
    )

    result = await agent.run("Hi.")

    assert result.output_text == "All good."
    assert len(seen) == 1
    assert seen[0].id == "appr_1"
    assert seen[0].name == "ask_question"
    assert seen[0].arguments == {"q": "what"}

    second_turn_items = provider.calls[1]["input_items"]
    assert second_turn_items[-1] == {
        "type": "mcp_approval_response",
        "approval_request_id": "appr_1",
        "approve": True,
    }


@pytest.mark.asyncio
async def test_async_approval_handler_is_awaited() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                response_id="resp_1",
                output_items=[
                    {
                        "type": "mcp_approval_request",
                        "id": "appr_1",
                        "server_label": "deepwiki",
                        "name": "ask_question",
                        "arguments": "{}",
                    }
                ],
                raw_response={"id": "resp_1"},
            ),
            ProviderResponse(
                response_id="resp_2",
                output_text="Denied handled.",
                output_items=[],
                raw_response={"id": "resp_2"},
            ),
        ]
    )

    async def handler(req: MCPApprovalRequest) -> bool:
        return False

    agent = Agent(
        config=AgentConfig(model="gpt-5"),
        provider=provider,
        mcp_servers=[
            MCPServer(
                server_label="deepwiki",
                server_url="https://x",
                require_approval="always",
            )
        ],
        approval_handler=handler,
    )

    await agent.run("Hi.")
    assert provider.calls[1]["input_items"][-1] == {
        "type": "mcp_approval_response",
        "approval_request_id": "appr_1",
        "approve": False,
    }


@pytest.mark.asyncio
async def test_missing_approval_handler_raises() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                response_id="resp_1",
                output_items=[
                    {
                        "type": "mcp_approval_request",
                        "id": "appr_1",
                        "server_label": "deepwiki",
                        "name": "ask_question",
                        "arguments": "{}",
                    }
                ],
                raw_response={"id": "resp_1"},
            )
        ]
    )
    agent = Agent(
        config=AgentConfig(model="gpt-5"),
        provider=provider,
        mcp_servers=[
            MCPServer(
                server_label="deepwiki",
                server_url="https://x",
                require_approval="always",
            )
        ],
    )

    with pytest.raises(MCPApprovalRequiredError):
        await agent.run("Hi.")


# ---------------------------------------------------------------------------
# Streaming events
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_emits_mcp_call_events() -> None:
    provider = FakeProvider(
        stream_sequences=[
            [
                ProviderMCPCallStartedEvent(
                    mcp_call={
                        "id": "mcp_1",
                        "server_label": "deepwiki",
                        "name": "ask_question",
                        "arguments": {},
                    }
                ),
                ProviderMCPCallCompletedEvent(
                    mcp_call={
                        "id": "mcp_1",
                        "server_label": "deepwiki",
                        "name": "ask_question",
                        "arguments": {},
                        "output": "answer",
                    }
                ),
                ProviderCompletedEvent(
                    response=ProviderResponse(
                        response_id="resp_1",
                        output_text="",
                        output_items=[
                            {
                                "type": "mcp_call",
                                "id": "mcp_1",
                                "server_label": "deepwiki",
                                "name": "ask_question",
                                "arguments": "{}",
                                "output": "answer",
                            }
                        ],
                        raw_response={"id": "resp_1"},
                    )
                )
            ],
            [
                ProviderTextDeltaEvent(delta="Done"),
                ProviderCompletedEvent(
                    response=ProviderResponse(
                        response_id="resp_2",
                        output_text="Done",
                        output_items=[],
                        raw_response={"id": "resp_2"},
                    )
                ),
            ],
        ]
    )
    agent = Agent(
        config=AgentConfig(model="gpt-5"),
        provider=provider,
        mcp_servers=[MCPServer(server_label="deepwiki", server_url="https://x")],
    )

    events = [event async for event in agent.stream("Hi.")]

    types = [e.type for e in events]
    assert "mcp_call_started" in types
    assert "mcp_call_completed" in types
    assert types[-1] == "completed"
    completed = events[-1]
    assert completed.result is not None
    assert [c.name for c in completed.result.mcp_calls] == ["ask_question"]


@pytest.mark.asyncio
async def test_stream_emits_mcp_approval_requested_event() -> None:
    provider = FakeProvider(
        stream_sequences=[
            [
                ProviderMCPApprovalRequestedEvent(
                    mcp_approval={
                        "id": "appr_1",
                        "server_label": "deepwiki",
                        "name": "ask_question",
                        "arguments": {},
                    }
                ),
                ProviderCompletedEvent(
                    response=ProviderResponse(
                        response_id="resp_1",
                        output_items=[
                            {
                                "type": "mcp_approval_request",
                                "id": "appr_1",
                                "server_label": "deepwiki",
                                "name": "ask_question",
                                "arguments": "{}",
                            }
                        ],
                        raw_response={"id": "resp_1"},
                    )
                )
            ],
            [
                ProviderCompletedEvent(
                    response=ProviderResponse(
                        response_id="resp_2",
                        output_text="Done",
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
        mcp_servers=[
            MCPServer(
                server_label="deepwiki",
                server_url="https://x",
                require_approval="always",
            )
        ],
        approval_handler=lambda req: True,
    )

    events = [event async for event in agent.stream("Hi.")]

    types = [e.type for e in events]
    assert "mcp_approval_requested" in types
    approval_event = next(e for e in events if e.type == "mcp_approval_requested")
    assert approval_event.mcp_approval is not None
    assert approval_event.mcp_approval.id == "appr_1"
