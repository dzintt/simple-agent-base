from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from simple_agent_base import AgentConfig
from simple_agent_base.config import ReasoningEffort
from simple_agent_base.providers.openai import OpenAIResponsesProvider


class FakeSummaryPart(BaseModel):
    text: str
    type: str = "summary_text"


class FakeReasoningItem(BaseModel):
    id: str = "rs_1"
    type: str = "reasoning"
    summary: list[FakeSummaryPart]


class FakeOutputTextItem(BaseModel):
    type: str = "message"
    role: str = "assistant"
    content: list[dict[str, str]]


class FakeResponse(BaseModel):
    id: str = "resp_1"
    output_text: str = ""
    output: list[BaseModel]
    output_parsed: BaseModel | None = None


class FakeStream:
    def __init__(self, events: list[object], final_response: BaseModel) -> None:
        self._events = list(events)
        self._final_response = final_response

    async def __aenter__(self) -> FakeStream:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    def __aiter__(self) -> AsyncIterator[object]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[object]:
        for event in self._events:
            yield event

    async def get_final_response(self) -> BaseModel:
        return self._final_response


class FakeResponsesAPI:
    def __init__(self, stream: FakeStream | None = None) -> None:
        self._stream = stream

    def stream(self, **_: Any) -> FakeStream:
        if self._stream is None:
            raise AssertionError("No fake stream configured.")
        return self._stream


class FakeClient:
    def __init__(self, stream: FakeStream | None = None) -> None:
        self.responses = FakeResponsesAPI(stream=stream)

    async def close(self) -> None:
        return None


REASONING_SUMMARY = "Checked the constraint."


def make_provider(*, reasoning_effort: ReasoningEffort | None = None) -> OpenAIResponsesProvider:
    return OpenAIResponsesProvider(
        AgentConfig(
            model="gpt-5.4",
            api_key="test-key",
            reasoning_effort=reasoning_effort,
        )
    )


def make_reasoning_response(*, summary_text: str = REASONING_SUMMARY) -> FakeResponse:
    return FakeResponse(
        output_text="reasoning-ok",
        output=[
            FakeReasoningItem(summary=[FakeSummaryPart(text=summary_text)]),
            FakeOutputTextItem(content=[{"type": "output_text", "text": "reasoning-ok"}]),
        ],
    )


def test_request_kwargs_include_reasoning_when_reasoning_effort_is_set() -> None:
    provider = make_provider(reasoning_effort="high")

    kwargs = provider._request_kwargs([], [])

    assert kwargs["reasoning"] == {"effort": "high", "summary": "auto"}


def test_request_kwargs_omit_reasoning_when_reasoning_effort_is_none() -> None:
    provider = make_provider()

    kwargs = provider._request_kwargs([], [])

    assert "reasoning" not in kwargs


def test_convert_response_extracts_reasoning_summary_from_reasoning_item() -> None:
    provider = make_provider()
    response = FakeResponse(
        output_text="reasoning-ok",
        output=[
            FakeReasoningItem(summary=[FakeSummaryPart(text="First part. "), FakeSummaryPart(text="Second part.")]),
            FakeOutputTextItem(content=[{"type": "output_text", "text": "reasoning-ok"}]),
        ],
    )

    converted = provider._convert_response(response)

    assert converted.output_text == "reasoning-ok"
    assert converted.reasoning_summary == "First part.Second part."


def test_convert_response_returns_none_when_no_reasoning_item_exists() -> None:
    provider = make_provider()
    response = FakeResponse(
        output_text="reasoning-ok",
        output=[FakeOutputTextItem(content=[{"type": "output_text", "text": "reasoning-ok"}])],
    )

    converted = provider._convert_response(response)

    assert converted.reasoning_summary is None


@pytest.mark.asyncio
async def test_stream_response_emits_reasoning_delta_and_completed_summary() -> None:
    provider = make_provider(reasoning_effort="high")
    final_response = make_reasoning_response()
    provider._client = FakeClient(
        stream=FakeStream(
            events=[
                SimpleNamespace(type="response.reasoning_summary_text.delta", delta="Checked ", item_id="rs_1", summary_index=0),
                SimpleNamespace(type="response.reasoning_summary_text.delta", delta="the constraint.", item_id="rs_1", summary_index=0),
                SimpleNamespace(type="response.output_text.delta", delta="reasoning-ok"),
            ],
            final_response=final_response,
        )
    )

    events = [event async for event in provider.stream_response(input_items=[], tools=[])]

    assert [event.delta for event in events if event.type == "reasoning_delta"] == [
        "Checked ",
        "the constraint.",
    ]
    assert [event.delta for event in events if event.type == "text_delta"] == ["reasoning-ok"]
    assert events[-1].type == "completed"
    assert events[-1].response.reasoning_summary == REASONING_SUMMARY


@pytest.mark.asyncio
async def test_stream_response_uses_done_fallback_when_no_reasoning_delta_is_seen() -> None:
    provider = make_provider(reasoning_effort="high")
    final_response = FakeResponse(
        output_text="reasoning-ok",
        output=[FakeOutputTextItem(content=[{"type": "output_text", "text": "reasoning-ok"}])],
    )
    provider._client = FakeClient(
        stream=FakeStream(
            events=[
                SimpleNamespace(
                    type="response.reasoning_summary_text.done",
                    text=REASONING_SUMMARY,
                    item_id="rs_1",
                    summary_index=0,
                ),
            ],
            final_response=final_response,
        )
    )

    events = [event async for event in provider.stream_response(input_items=[], tools=[])]

    assert [event.delta for event in events if event.type == "reasoning_delta"] == []
    assert events[-1].type == "completed"
    assert events[-1].response.reasoning_summary == REASONING_SUMMARY
