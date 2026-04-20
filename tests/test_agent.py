from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from simple_agent_base import Agent, AgentConfig, ChatMessage, FilePart, ImagePart, TextPart, tool
from simple_agent_base.errors import MaxTurnsExceededError, ProviderError, ToolExecutionError
from simple_agent_base.providers.base import (
    ConversationItem,
    ProviderCompletedEvent,
    ProviderEvent,
    ProviderResponse,
    ProviderTextDeltaEvent,
)


class FakeProvider:
    def __init__(self, responses: list[ProviderResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

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
        if not self.responses:
            raise ProviderError("No more fake responses configured.")
        return self.responses.pop(0)

    async def stream_response(
        self,
        *,
        input_items: Sequence[ConversationItem],
        tools: Sequence[dict[str, Any]],
        response_model: type[BaseModel] | None = None,
    ) -> AsyncIterator[ProviderEvent]:
        raise NotImplementedError

    async def close(self) -> None:
        return None


class ClosableFakeProvider(FakeProvider):
    def __init__(self, responses: list[ProviderResponse]) -> None:
        super().__init__(responses)
        self.closed = False

    async def close(self) -> None:
        self.closed = True


@tool
async def ping(message: str) -> str:
    """Echo a message back."""
    return f"pong: {message}"


@tool
async def uppercase(value: str) -> str:
    """Uppercase a value."""
    return value.upper()


@tool
async def slow_ping(message: str) -> str:
    """Echo a message back after a short delay."""
    await asyncio.sleep(0.05)
    return f"pong: {message}"


@tool
def sync_ping(message: str) -> str:
    """Echo a message back synchronously."""
    return f"sync-pong: {message}"


@tool
def sync_explode(message: str) -> str:
    """Always fail synchronously."""
    raise ValueError(message)


@tool
async def explode(message: str) -> str:
    """Always fail."""
    raise ValueError(message)


class Person(BaseModel):
    name: str
    age: int


class WeatherAnswer(BaseModel):
    city: str
    temperature_f: int
    summary: str


@pytest.mark.asyncio
async def test_run_without_tools_returns_plain_text() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                response_id="resp_1",
                output_text="hello world",
                output_items=[
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "hello world"}],
                    }
                ],
                raw_response={"id": "resp_1"},
            )
        ]
    )
    agent = Agent(config=AgentConfig(model="gpt-5"), provider=provider)

    result = await agent.run("Say hello.")

    assert result.output_text == "hello world"
    assert result.output_data is None
    assert result.tool_results == []


def test_run_sync_returns_plain_text() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                response_id="resp_1",
                output_text="hello world",
                output_items=[],
                raw_response={"id": "resp_1"},
            )
        ]
    )
    agent = Agent(config=AgentConfig(model="gpt-5"), provider=provider)

    result = agent.run_sync("Say hello.")

    assert result.output_text == "hello world"


def test_close_wraps_aclose() -> None:
    provider = ClosableFakeProvider([])
    agent = Agent(config=AgentConfig(model="gpt-5"), provider=provider)

    agent.close()

    assert provider.closed is True


@pytest.mark.asyncio
async def test_run_sync_raises_inside_running_event_loop() -> None:
    agent = Agent(config=AgentConfig(model="gpt-5"), provider=FakeProvider([]))

    with pytest.raises(RuntimeError, match="run_sync\\(\\) cannot be used inside a running event loop"):
        agent.run_sync("Say hello.")


def test_chat_session_run_sync_preserves_history() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                response_id="resp_1",
                output_text="Stored.",
                output_items=[
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Stored."}],
                    }
                ],
                raw_response={"id": "resp_1"},
            ),
            ProviderResponse(
                response_id="resp_2",
                output_text="You said Anson.",
                output_items=[
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "You said Anson."}],
                    }
                ],
                raw_response={"id": "resp_2"},
            ),
        ]
    )
    agent = Agent(config=AgentConfig(model="gpt-5"), provider=provider)
    chat = agent.chat()

    first = chat.run_sync("My name is Anson.")
    second = chat.run_sync("What name did I say?")

    assert first.output_text == "Stored."
    assert second.output_text == "You said Anson."
    assert chat.history == [
        ChatMessage(role="user", content="My name is Anson."),
        ChatMessage(role="assistant", content="Stored."),
        ChatMessage(role="user", content="What name did I say?"),
        ChatMessage(role="assistant", content="You said Anson."),
    ]


@pytest.mark.asyncio
async def test_chat_snapshot_returns_full_session_state() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                response_id="resp_1",
                output_text="Stored.",
                output_items=[
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Stored."}],
                    }
                ],
                raw_response={"id": "resp_1"},
            )
        ]
    )
    agent = Agent(config=AgentConfig(model="gpt-5"), provider=provider)
    chat = agent.chat(system_prompt="You are concise.")

    await chat.run("My name is Anson.")
    snapshot = chat.snapshot()

    assert snapshot.version == "v1"
    assert snapshot.system_prompt == "You are concise."
    assert snapshot.items == [
        {
            "type": "message",
            "role": "user",
            "content": "My name is Anson.",
        },
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Stored."}],
        },
    ]


@pytest.mark.asyncio
async def test_chat_from_snapshot_restores_conversation_state() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                response_id="resp_1",
                output_text="You said Anson.",
                output_items=[
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "You said Anson."}],
                    }
                ],
                raw_response={"id": "resp_1"},
            )
        ]
    )
    agent = Agent(config=AgentConfig(model="gpt-5"), provider=provider)
    restored = agent.chat_from_snapshot(
        {
            "version": "v1",
            "items": [
                {
                    "type": "message",
                    "role": "user",
                    "content": "My name is Anson.",
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Stored."}],
                },
            ],
            "system_prompt": "You are concise.",
        }
    )

    result = await restored.run("What name did I say?")

    assert result.output_text == "You said Anson."
    assert provider.calls[0]["input_items"] == [
        {
            "type": "message",
            "role": "developer",
            "content": "You are concise.",
        },
        {
            "type": "message",
            "role": "user",
            "content": "My name is Anson.",
        },
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Stored."}],
        },
        {
            "type": "message",
            "role": "user",
            "content": "What name did I say?",
        },
    ]


@pytest.mark.asyncio
async def test_restored_chat_uses_snapshot_system_prompt_over_agent_default() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                response_id="resp_1",
                output_text="Done.",
                output_items=[
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Done."}],
                    }
                ],
                raw_response={"id": "resp_1"},
            )
        ]
    )
    agent = Agent(
        config=AgentConfig(model="gpt-5"),
        provider=provider,
        system_prompt="Agent default prompt",
    )
    restored = agent.chat_from_snapshot(
        {
            "version": "v1",
            "items": [],
            "system_prompt": "Snapshot prompt",
        }
    )

    await restored.run("Hello.")

    assert provider.calls[0]["input_items"][0] == {
        "type": "message",
        "role": "developer",
        "content": "Snapshot prompt",
    }


@pytest.mark.asyncio
async def test_snapshot_does_not_persist_convenience_prompt_as_message_item() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                response_id="resp_1",
                output_text="Stored.",
                output_items=[
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Stored."}],
                    }
                ],
                raw_response={"id": "resp_1"},
            )
        ]
    )
    agent = Agent(config=AgentConfig(model="gpt-5"), provider=provider)
    chat = agent.chat(system_prompt="You are concise.")

    await chat.run("My name is Anson.")
    snapshot = chat.snapshot()

    assert all(item.get("role") != "developer" for item in snapshot.items)


@pytest.mark.asyncio
async def test_chat_from_snapshot_preserves_multimodal_history() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                response_id="resp_1",
                output_text="The image showed a cat.",
                output_items=[
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "The image showed a cat."}],
                    }
                ],
                raw_response={"id": "resp_1"},
            )
        ]
    )
    agent = Agent(config=AgentConfig(model="gpt-5"), provider=provider)
    restored = agent.chat_from_snapshot(
        {
            "version": "v1",
            "items": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Remember this image."},
                        {
                            "type": "input_image",
                            "image_url": "https://example.com/cat.png",
                            "detail": "high",
                        },
                    ],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Stored the image."}],
                },
            ],
            "system_prompt": None,
        }
    )

    result = await restored.run("What was in the image?")

    assert result.output_text == "The image showed a cat."
    assert provider.calls[0]["input_items"] == [
        {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Remember this image."},
                {
                    "type": "input_image",
                    "image_url": "https://example.com/cat.png",
                    "detail": "high",
                },
            ],
        },
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Stored the image."}],
        },
        {
            "type": "message",
            "role": "user",
            "content": "What was in the image?",
        },
    ]


@pytest.mark.asyncio
async def test_chat_from_snapshot_preserves_file_history() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                response_id="resp_1",
                output_text="The file mentioned teal.",
                output_items=[
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "The file mentioned teal."}],
                    }
                ],
                raw_response={"id": "resp_1"},
            )
        ]
    )
    agent = Agent(config=AgentConfig(model="gpt-5"), provider=provider)
    restored = agent.chat_from_snapshot(
        {
            "version": "v1",
            "items": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Remember this PDF."},
                        {
                            "type": "input_file",
                            "file_url": "https://example.com/report.pdf",
                        },
                    ],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Stored the file."}],
                },
            ],
        }
    )

    result = await restored.run("What did the file mention?")

    assert result.output_text == "The file mentioned teal."
    assert provider.calls[0]["input_items"] == [
        {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Remember this PDF."},
                {
                    "type": "input_file",
                    "file_url": "https://example.com/report.pdf",
                },
            ],
        },
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Stored the file."}],
        },
        {
            "type": "message",
            "role": "user",
            "content": "What did the file mention?",
        },
    ]


@pytest.mark.asyncio
async def test_run_prepends_agent_level_system_prompt() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                response_id="resp_1",
                output_text="hello world",
                output_items=[],
                raw_response={"id": "resp_1"},
            )
        ]
    )
    agent = Agent(
        config=AgentConfig(model="gpt-5"),
        provider=provider,
        system_prompt="You are concise.",
    )

    await agent.run("Say hello.")

    assert provider.calls[0]["input_items"] == [
        {
            "type": "message",
            "role": "developer",
            "content": "You are concise.",
        },
        {
            "type": "message",
            "role": "user",
            "content": "Say hello.",
        },
    ]


@pytest.mark.asyncio
async def test_run_level_system_prompt_overrides_agent_default() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                response_id="resp_1",
                output_text="hello world",
                output_items=[],
                raw_response={"id": "resp_1"},
            )
        ]
    )
    agent = Agent(
        config=AgentConfig(model="gpt-5"),
        provider=provider,
        system_prompt="Default prompt",
    )

    await agent.run("Say hello.", system_prompt="Override prompt")

    assert provider.calls[0]["input_items"] == [
        {
            "type": "message",
            "role": "developer",
            "content": "Override prompt",
        },
        {
            "type": "message",
            "role": "user",
            "content": "Say hello.",
        },
    ]


@pytest.mark.asyncio
async def test_run_accepts_multimodal_message() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                response_id="resp_1",
                output_text="That looks like a cat.",
                output_items=[],
                raw_response={"id": "resp_1"},
            )
        ]
    )
    agent = Agent(config=AgentConfig(model="gpt-5"), provider=provider)

    result = await agent.run(
        [
            ChatMessage(
                role="user",
                content=[
                    TextPart("What is in this image?"),
                    ImagePart.from_url("https://example.com/cat.png"),
                ],
            )
        ]
    )

    assert result.output_text == "That looks like a cat."
    assert provider.calls[0]["input_items"] == [
        {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "What is in this image?"},
                {
                    "type": "input_image",
                    "image_url": "https://example.com/cat.png",
                    "detail": "auto",
                },
            ],
        }
    ]


@pytest.mark.asyncio
async def test_run_accepts_file_message() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                response_id="resp_1",
                output_text="The file says hello.",
                output_items=[],
                raw_response={"id": "resp_1"},
            )
        ]
    )
    agent = Agent(config=AgentConfig(model="gpt-5"), provider=provider)

    result = await agent.run(
        [
            ChatMessage(
                role="user",
                content=[
                    TextPart("What does this file say?"),
                    FilePart.from_url("https://example.com/report.pdf"),
                ],
            )
        ]
    )

    assert result.output_text == "The file says hello."
    assert provider.calls[0]["input_items"] == [
        {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "What does this file say?"},
                {
                    "type": "input_file",
                    "file_url": "https://example.com/report.pdf",
                },
            ],
        }
    ]


@pytest.mark.asyncio
async def test_run_accepts_multiple_messages() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                response_id="resp_1",
                output_text="I remember the earlier messages.",
                output_items=[],
                raw_response={"id": "resp_1"},
            )
        ]
    )
    agent = Agent(config=AgentConfig(model="gpt-5"), provider=provider)

    result = await agent.run(
        [
            ChatMessage(role="system", content="You are concise."),
            ChatMessage(role="user", content="My name is Anson."),
            ChatMessage(role="assistant", content="Noted."),
            ChatMessage(role="user", content="What's my name?"),
        ]
    )

    assert result.output_text == "I remember the earlier messages."
    assert provider.calls[0]["input_items"] == [
        {
            "type": "message",
            "role": "system",
            "content": "You are concise.",
        },
        {
            "type": "message",
            "role": "user",
            "content": "My name is Anson.",
        },
        {
            "type": "message",
            "role": "assistant",
            "content": "Noted.",
        },
        {
            "type": "message",
            "role": "user",
            "content": "What's my name?",
        },
    ]


@pytest.mark.asyncio
async def test_run_preserves_explicit_history_alongside_system_prompt() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                response_id="resp_1",
                output_text="I remember the earlier messages.",
                output_items=[],
                raw_response={"id": "resp_1"},
            )
        ]
    )
    agent = Agent(
        config=AgentConfig(model="gpt-5"),
        provider=provider,
        system_prompt="Default prompt",
    )

    await agent.run(
        [
            ChatMessage(role="system", content="Explicit system."),
            ChatMessage(role="user", content="Hello."),
        ]
    )

    assert provider.calls[0]["input_items"] == [
        {
            "type": "message",
            "role": "developer",
            "content": "Default prompt",
        },
        {
            "type": "message",
            "role": "system",
            "content": "Explicit system.",
        },
        {
            "type": "message",
            "role": "user",
            "content": "Hello.",
        },
    ]


@pytest.mark.asyncio
async def test_run_executes_one_tool_then_returns_final_response() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                response_id="resp_1",
                output_text="",
                tool_calls=[
                    {
                        "call_id": "call_1",
                        "name": "ping",
                        "arguments": {"message": "hello"},
                        "raw_arguments": '{"message":"hello"}',
                    }
                ],
                output_items=[
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "ping",
                        "arguments": '{"message":"hello"}',
                    }
                ],
                raw_response={"id": "resp_1"},
            ),
            ProviderResponse(
                response_id="resp_2",
                output_text="The tool said pong: hello",
                output_items=[
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "The tool said pong: hello"}],
                    }
                ],
                raw_response={"id": "resp_2"},
            ),
        ]
    )
    agent = Agent(config=AgentConfig(model="gpt-5"), tools=[ping], provider=provider)

    result = await agent.run("Use the tool.")

    assert result.output_text == "The tool said pong: hello"
    assert len(result.tool_results) == 1
    assert result.tool_results[0].output == "pong: hello"
    assert provider.calls[1]["input_items"][-1] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": "pong: hello",
    }


@pytest.mark.asyncio
async def test_run_executes_multiple_sequential_tool_calls() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                response_id="resp_1",
                tool_calls=[
                    {
                        "call_id": "call_1",
                        "name": "ping",
                        "arguments": {"message": "hello"},
                        "raw_arguments": '{"message":"hello"}',
                    },
                    {
                        "call_id": "call_2",
                        "name": "uppercase",
                        "arguments": {"value": "world"},
                        "raw_arguments": '{"value":"world"}',
                    },
                ],
                output_items=[
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "ping",
                        "arguments": '{"message":"hello"}',
                    },
                    {
                        "type": "function_call",
                        "call_id": "call_2",
                        "name": "uppercase",
                        "arguments": '{"value":"world"}',
                    },
                ],
                raw_response={"id": "resp_1"},
            ),
            ProviderResponse(
                response_id="resp_2",
                output_text="pong: hello / WORLD",
                output_items=[],
                raw_response={"id": "resp_2"},
            ),
        ]
    )
    agent = Agent(config=AgentConfig(model="gpt-5"), tools=[ping, uppercase], provider=provider)

    result = await agent.run("Use both tools.")

    assert [tool_result.output for tool_result in result.tool_results] == ["pong: hello", "WORLD"]
    second_turn_items = provider.calls[1]["input_items"]
    assert second_turn_items[-2:] == [
        {"type": "function_call_output", "call_id": "call_1", "output": "pong: hello"},
        {"type": "function_call_output", "call_id": "call_2", "output": "WORLD"},
    ]


@pytest.mark.asyncio
async def test_parallel_tool_batch_executes_concurrently_when_enabled() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                response_id="resp_1",
                tool_calls=[
                    {
                        "call_id": "call_1",
                        "name": "slow_ping",
                        "arguments": {"message": "alpha"},
                        "raw_arguments": '{"message":"alpha"}',
                    },
                    {
                        "call_id": "call_2",
                        "name": "slow_ping",
                        "arguments": {"message": "beta"},
                        "raw_arguments": '{"message":"beta"}',
                    },
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
        config=AgentConfig(model="gpt-5", parallel_tool_calls=True),
        tools=[slow_ping],
        provider=provider,
    )

    start = time.perf_counter()
    result = await agent.run("Use the slow tool twice.")
    elapsed = time.perf_counter() - start

    assert result.output_text == "done"
    assert [tool_result.output for tool_result in result.tool_results] == ["pong: alpha", "pong: beta"]
    assert provider.calls[1]["input_items"][-2:] == [
        {"type": "function_call_output", "call_id": "call_1", "output": "pong: alpha"},
        {"type": "function_call_output", "call_id": "call_2", "output": "pong: beta"},
    ]
    assert elapsed < 0.09


@pytest.mark.asyncio
async def test_parallel_tool_batch_remains_sequential_by_default() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                response_id="resp_1",
                tool_calls=[
                    {
                        "call_id": "call_1",
                        "name": "slow_ping",
                        "arguments": {"message": "alpha"},
                        "raw_arguments": '{"message":"alpha"}',
                    },
                    {
                        "call_id": "call_2",
                        "name": "slow_ping",
                        "arguments": {"message": "beta"},
                        "raw_arguments": '{"message":"beta"}',
                    },
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
        tools=[slow_ping],
        provider=provider,
    )

    start = time.perf_counter()
    result = await agent.run("Use the slow tool twice.")
    elapsed = time.perf_counter() - start

    assert result.output_text == "done"
    assert [tool_result.output for tool_result in result.tool_results] == ["pong: alpha", "pong: beta"]
    assert elapsed >= 0.09


@pytest.mark.asyncio
async def test_max_turns_exceeded_raises_error() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                response_id="resp_1",
                tool_calls=[
                    {
                        "call_id": "call_1",
                        "name": "ping",
                        "arguments": {"message": "again"},
                        "raw_arguments": '{"message":"again"}',
                    }
                ],
                output_items=[],
                raw_response={"id": "resp_1"},
            )
        ]
    )
    agent = Agent(
        config=AgentConfig(model="gpt-5", max_turns=1),
        tools=[ping],
        provider=provider,
    )

    with pytest.raises(MaxTurnsExceededError):
        await agent.run("Loop forever.")


@pytest.mark.asyncio
async def test_tool_errors_surface_as_tool_execution_errors() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                response_id="resp_1",
                tool_calls=[
                    {
                        "call_id": "call_1",
                        "name": "explode",
                        "arguments": {"message": "boom"},
                        "raw_arguments": '{"message":"boom"}',
                    }
                ],
                output_items=[],
                raw_response={"id": "resp_1"},
            )
        ]
    )
    agent = Agent(config=AgentConfig(model="gpt-5"), tools=[explode], provider=provider)

    with pytest.raises(ToolExecutionError):
        await agent.run("Fail.")


@pytest.mark.asyncio
async def test_sync_tool_definition_executes_successfully() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                response_id="resp_1",
                output_text="",
                tool_calls=[
                    {
                        "call_id": "call_1",
                        "name": "sync_ping",
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
    agent = Agent(config=AgentConfig(model="gpt-5"), tools=[sync_ping], provider=provider)

    result = await agent.run("Use the sync tool.")

    assert [tool_result.output for tool_result in result.tool_results] == ["sync-pong: hello"]


@pytest.mark.asyncio
async def test_sync_tool_errors_surface_as_tool_execution_errors() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                response_id="resp_1",
                tool_calls=[
                    {
                        "call_id": "call_1",
                        "name": "sync_explode",
                        "arguments": {"message": "boom"},
                        "raw_arguments": '{"message":"boom"}',
                    }
                ],
                output_items=[],
                raw_response={"id": "resp_1"},
            )
        ]
    )
    agent = Agent(config=AgentConfig(model="gpt-5"), tools=[sync_explode], provider=provider)

    with pytest.raises(ToolExecutionError):
        await agent.run("Fail with sync tool.")


@pytest.mark.asyncio
async def test_run_returns_structured_output_without_tools() -> None:
    person = Person(name="Sarah", age=29)
    provider = FakeProvider(
        [
            ProviderResponse(
                response_id="resp_1",
                output_text='{"name":"Sarah","age":29}',
                output_data=person,
                output_items=[],
                raw_response={"id": "resp_1"},
            )
        ]
    )
    agent = Agent(config=AgentConfig(model="gpt-5"), provider=provider)

    result = await agent.run(
        "Extract the person from: Sarah is 29 years old.",
        response_model=Person,
    )

    assert result.output_data == person
    assert result.output_text == '{"name":"Sarah","age":29}'
    assert provider.calls[0]["response_model"] is Person


@pytest.mark.asyncio
async def test_run_supports_structured_output_with_image_input() -> None:
    person = Person(name="Sarah", age=29)
    provider = FakeProvider(
        [
            ProviderResponse(
                response_id="resp_1",
                output_text='{"name":"Sarah","age":29}',
                output_data=person,
                output_items=[],
                raw_response={"id": "resp_1"},
            )
        ]
    )
    agent = Agent(config=AgentConfig(model="gpt-5"), provider=provider)

    result = await agent.run(
        [
            ChatMessage(
                role="user",
                content=[
                    TextPart("Extract the person in this image."),
                    ImagePart.from_url("https://example.com/person.png"),
                ],
            )
        ],
        response_model=Person,
    )

    assert result.output_data == person
    assert provider.calls[0]["response_model"] is Person
    assert provider.calls[0]["input_items"] == [
        {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Extract the person in this image."},
                {
                    "type": "input_image",
                    "image_url": "https://example.com/person.png",
                    "detail": "auto",
                },
            ],
        }
    ]


@pytest.mark.asyncio
async def test_run_returns_structured_output_after_tool_call() -> None:
    weather = WeatherAnswer(city="San Francisco", temperature_f=65, summary="Foggy")
    provider = FakeProvider(
        [
            ProviderResponse(
                response_id="resp_1",
                tool_calls=[
                    {
                        "call_id": "call_1",
                        "name": "ping",
                        "arguments": {"message": "weather"},
                        "raw_arguments": '{"message":"weather"}',
                    }
                ],
                output_items=[
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "ping",
                        "arguments": '{"message":"weather"}',
                    }
                ],
                raw_response={"id": "resp_1"},
            ),
            ProviderResponse(
                response_id="resp_2",
                output_text='{"city":"San Francisco","temperature_f":65,"summary":"Foggy"}',
                output_data=weather,
                output_items=[],
                raw_response={"id": "resp_2"},
            ),
        ]
    )
    agent = Agent(config=AgentConfig(model="gpt-5"), tools=[ping], provider=provider)

    result = await agent.run(
        "Use the tool and return a structured weather answer.",
        response_model=WeatherAnswer,
    )

    assert result.output_data == weather
    assert [tool_result.output for tool_result in result.tool_results] == ["pong: weather"]
    assert provider.calls[0]["response_model"] is WeatherAnswer
    assert provider.calls[1]["response_model"] is WeatherAnswer


@pytest.mark.asyncio
async def test_parallel_tool_batch_supports_structured_output_after_tools() -> None:
    weather = WeatherAnswer(city="San Francisco", temperature_f=65, summary="Foggy")
    provider = FakeProvider(
        [
            ProviderResponse(
                response_id="resp_1",
                tool_calls=[
                    {
                        "call_id": "call_1",
                        "name": "slow_ping",
                        "arguments": {"message": "alpha"},
                        "raw_arguments": '{"message":"alpha"}',
                    },
                    {
                        "call_id": "call_2",
                        "name": "slow_ping",
                        "arguments": {"message": "beta"},
                        "raw_arguments": '{"message":"beta"}',
                    },
                ],
                output_items=[],
                raw_response={"id": "resp_1"},
            ),
            ProviderResponse(
                response_id="resp_2",
                output_text='{"city":"San Francisco","temperature_f":65,"summary":"Foggy"}',
                output_data=weather,
                output_items=[],
                raw_response={"id": "resp_2"},
            ),
        ]
    )
    agent = Agent(
        config=AgentConfig(model="gpt-5", parallel_tool_calls=True),
        tools=[slow_ping],
        provider=provider,
    )

    result = await agent.run(
        "Use the tools and return a structured weather answer.",
        response_model=WeatherAnswer,
    )

    assert result.output_data == weather
    assert [tool_result.output for tool_result in result.tool_results] == ["pong: alpha", "pong: beta"]
    assert provider.calls[0]["response_model"] is WeatherAnswer
    assert provider.calls[1]["response_model"] is WeatherAnswer


@pytest.mark.asyncio
async def test_parallel_tool_batch_failure_fails_run() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                response_id="resp_1",
                tool_calls=[
                    {
                        "call_id": "call_1",
                        "name": "slow_ping",
                        "arguments": {"message": "alpha"},
                        "raw_arguments": '{"message":"alpha"}',
                    },
                    {
                        "call_id": "call_2",
                        "name": "explode",
                        "arguments": {"message": "boom"},
                        "raw_arguments": '{"message":"boom"}',
                    },
                ],
                output_items=[],
                raw_response={"id": "resp_1"},
            )
        ]
    )
    agent = Agent(
        config=AgentConfig(model="gpt-5", parallel_tool_calls=True),
        tools=[slow_ping, explode],
        provider=provider,
    )

    with pytest.raises(ToolExecutionError):
        await agent.run("Fail with parallel tools.")


@pytest.mark.asyncio
async def test_run_supports_tool_calls_with_system_prompt() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                response_id="resp_1",
                output_text="",
                tool_calls=[
                    {
                        "call_id": "call_1",
                        "name": "ping",
                        "arguments": {"message": "hello"},
                        "raw_arguments": '{"message":"hello"}',
                    }
                ],
                output_items=[
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "ping",
                        "arguments": '{"message":"hello"}',
                    }
                ],
                raw_response={"id": "resp_1"},
            ),
            ProviderResponse(
                response_id="resp_2",
                output_text="The tool said pong: hello",
                output_items=[],
                raw_response={"id": "resp_2"},
            ),
        ]
    )
    agent = Agent(
        config=AgentConfig(model="gpt-5"),
        tools=[ping],
        provider=provider,
        system_prompt="Use tools when helpful.",
    )

    result = await agent.run("Use the tool.")

    assert result.output_text == "The tool said pong: hello"
    assert provider.calls[0]["input_items"][0] == {
        "type": "message",
        "role": "developer",
        "content": "Use tools when helpful.",
    }
    assert provider.calls[1]["input_items"][0] == {
        "type": "message",
        "role": "developer",
        "content": "Use tools when helpful.",
    }


@pytest.mark.asyncio
async def test_run_surfaces_structured_provider_failures() -> None:
    class FailingStructuredProvider(FakeProvider):
        async def create_response(
            self,
            *,
            input_items: Sequence[ConversationItem],
            tools: Sequence[dict[str, Any]],
            response_model: type[BaseModel] | None = None,
        ) -> ProviderResponse:
            raise ProviderError("structured parse failed")

    agent = Agent(
        config=AgentConfig(model="gpt-5"),
        provider=FailingStructuredProvider([]),
    )

    with pytest.raises(ProviderError, match="structured parse failed"):
        await agent.run("Return structured data.", response_model=Person)


@pytest.mark.asyncio
async def test_chat_session_preserves_conversation_history() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                response_id="resp_1",
                output_text="Your name is Anson.",
                output_items=[
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Your name is Anson."}],
                    }
                ],
                raw_response={"id": "resp_1"},
            ),
            ProviderResponse(
                response_id="resp_2",
                output_text="You told me your name is Anson.",
                output_items=[
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "You told me your name is Anson."}],
                    }
                ],
                raw_response={"id": "resp_2"},
            ),
        ]
    )
    agent = Agent(config=AgentConfig(model="gpt-5"), provider=provider)
    chat = agent.chat()

    first = await chat.run("My name is Anson.")
    second = await chat.run("What name did I give you?")

    assert first.output_text == "Your name is Anson."
    assert second.output_text == "You told me your name is Anson."
    assert provider.calls[1]["input_items"] == [
        {
            "type": "message",
            "role": "user",
            "content": "My name is Anson.",
        },
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Your name is Anson."}],
        },
        {
            "type": "message",
            "role": "user",
            "content": "What name did I give you?",
        },
    ]
    assert chat.history == [
        ChatMessage(role="user", content="My name is Anson."),
        ChatMessage(role="assistant", content="Your name is Anson."),
        ChatMessage(role="user", content="What name did I give you?"),
        ChatMessage(role="assistant", content="You told me your name is Anson."),
    ]


@pytest.mark.asyncio
async def test_chat_session_uses_default_system_prompt_without_leaking_into_history() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                response_id="resp_1",
                output_text="Stored.",
                output_items=[
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Stored."}],
                    }
                ],
                raw_response={"id": "resp_1"},
            ),
            ProviderResponse(
                response_id="resp_2",
                output_text="You said Anson.",
                output_items=[
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "You said Anson."}],
                    }
                ],
                raw_response={"id": "resp_2"},
            ),
        ]
    )
    agent = Agent(
        config=AgentConfig(model="gpt-5"),
        provider=provider,
        system_prompt="You are concise.",
    )
    chat = agent.chat()

    await chat.run("My name is Anson.")
    await chat.run("What name did I say?")

    assert provider.calls[0]["input_items"][0] == {
        "type": "message",
        "role": "developer",
        "content": "You are concise.",
    }
    assert provider.calls[1]["input_items"][0] == {
        "type": "message",
        "role": "developer",
        "content": "You are concise.",
    }
    assert chat.history == [
        ChatMessage(role="user", content="My name is Anson."),
        ChatMessage(role="assistant", content="Stored."),
        ChatMessage(role="user", content="What name did I say?"),
        ChatMessage(role="assistant", content="You said Anson."),
    ]


@pytest.mark.asyncio
async def test_chat_session_run_level_system_prompt_overrides_for_one_call_only() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                response_id="resp_1",
                output_text="First",
                output_items=[
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "First"}],
                    }
                ],
                raw_response={"id": "resp_1"},
            ),
            ProviderResponse(
                response_id="resp_2",
                output_text="Second",
                output_items=[
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Second"}],
                    }
                ],
                raw_response={"id": "resp_2"},
            ),
        ]
    )
    agent = Agent(config=AgentConfig(model="gpt-5"), provider=provider)
    chat = agent.chat(system_prompt="Session prompt")

    await chat.run("Hello.", system_prompt="Override prompt")
    await chat.run("Hello again.")

    assert provider.calls[0]["input_items"][0] == {
        "type": "message",
        "role": "developer",
        "content": "Override prompt",
    }
    assert provider.calls[1]["input_items"][0] == {
        "type": "message",
        "role": "developer",
        "content": "Session prompt",
    }


@pytest.mark.asyncio
async def test_chat_session_preserves_multimodal_history() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                response_id="resp_1",
                output_text="Stored the image.",
                output_items=[
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Stored the image."}],
                    }
                ],
                raw_response={"id": "resp_1"},
            ),
            ProviderResponse(
                response_id="resp_2",
                output_text="The image showed a cat.",
                output_items=[
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "The image showed a cat."}],
                    }
                ],
                raw_response={"id": "resp_2"},
            ),
        ]
    )
    agent = Agent(config=AgentConfig(model="gpt-5"), provider=provider)
    chat = agent.chat()

    await chat.run(
        [
            ChatMessage(
                role="user",
                content=[
                    TextPart("Remember this image."),
                    ImagePart.from_url("https://example.com/cat.png", detail="high"),
                ],
            )
        ]
    )
    second = await chat.run("What was in the image?")

    assert second.output_text == "The image showed a cat."
    assert provider.calls[1]["input_items"] == [
        {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Remember this image."},
                {
                    "type": "input_image",
                    "image_url": "https://example.com/cat.png",
                    "detail": "high",
                },
            ],
        },
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Stored the image."}],
        },
        {
            "type": "message",
            "role": "user",
            "content": "What was in the image?",
        },
    ]
    assert chat.history == [
        ChatMessage(
            role="user",
            content=[
                TextPart("Remember this image."),
                ImagePart.from_url("https://example.com/cat.png", detail="high"),
            ],
        ),
        ChatMessage(role="assistant", content="Stored the image."),
        ChatMessage(role="user", content="What was in the image?"),
        ChatMessage(role="assistant", content="The image showed a cat."),
    ]


def test_image_part_from_file_creates_data_url(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\nfakepng")

    part = ImagePart.from_file(str(image_path), detail="high")

    assert part.detail == "high"
    assert part.image_url.startswith("data:image/png;base64,")


def test_file_part_from_file_creates_data_url(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.txt"
    file_path.write_text("hello world")

    part = FilePart.from_file(str(file_path))

    assert part.filename == "sample.txt"
    assert part.file_data is not None
    assert part.file_data.startswith("data:text/plain;base64,")


def test_file_part_from_file_rejects_missing_paths() -> None:
    with pytest.raises(ValueError, match="does not exist"):
        FilePart.from_file("missing.pdf")


def test_file_part_from_file_rejects_unsupported_types(tmp_path: Path) -> None:
    file_path = tmp_path / "archive.zip"
    file_path.write_bytes(b"zip-data")

    with pytest.raises(ValueError, match="Unsupported file type"):
        FilePart.from_file(str(file_path))


def test_image_part_from_file_rejects_missing_paths() -> None:
    with pytest.raises(ValueError, match="does not exist"):
        ImagePart.from_file("missing.png")


def test_image_part_from_file_rejects_unsupported_types(tmp_path: Path) -> None:
    file_path = tmp_path / "notes.txt"
    file_path.write_text("not an image")

    with pytest.raises(ValueError, match="Unsupported image file type"):
        ImagePart.from_file(str(file_path))
