from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import TYPE_CHECKING

from pydantic import BaseModel

from agent_harness.providers.base import ConversationItem
from agent_harness.types import AgentEvent, AgentRunResult, ChatMessage, MessageInput

if TYPE_CHECKING:
    from agent_harness.agent import Agent


class ChatSession:
    def __init__(
        self,
        agent: Agent,
        items: Sequence[ConversationItem] | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self._agent = agent
        self._items = list(items or [])
        self._system_prompt = system_prompt

    @property
    def history(self) -> list[ChatMessage]:
        return self._agent._messages_from_items(self._items)

    @property
    def items(self) -> list[ConversationItem]:
        return list(self._items)

    async def run(
        self,
        input_data: str | Sequence[MessageInput],
        *,
        response_model: type[BaseModel] | None = None,
        system_prompt: str | None = None,
    ) -> AgentRunResult:
        resolved_system_prompt = self._resolve_system_prompt(system_prompt)
        transcript = [*self._items, *self._agent._normalize_input(input_data)]
        transcript = self._agent._prepend_system_prompt(
            transcript,
            system_prompt=resolved_system_prompt,
        )
        result = await self._agent._run_transcript(transcript, response_model=response_model)
        self._items = self._agent._persistable_items(
            self._agent._strip_prepended_system_prompt(
                transcript,
                system_prompt=resolved_system_prompt,
            )
        )
        return result

    async def stream(
        self,
        input_data: str | Sequence[MessageInput],
        *,
        response_model: type[BaseModel] | None = None,
        system_prompt: str | None = None,
    ) -> AsyncIterator[AgentEvent]:
        resolved_system_prompt = self._resolve_system_prompt(system_prompt)
        new_items = self._agent._normalize_input(input_data)
        transcript = [*self._items, *new_items]
        transcript = self._agent._prepend_system_prompt(
            transcript,
            system_prompt=resolved_system_prompt,
        )

        async for event in self._agent._stream_transcript(transcript, response_model=response_model):
            if event.type == "completed":
                self._items = self._agent._persistable_items(
                    self._agent._strip_prepended_system_prompt(
                        transcript,
                        system_prompt=resolved_system_prompt,
                    )
                )
            yield event

    def reset(self) -> None:
        self._items.clear()

    def _resolve_system_prompt(self, system_prompt: str | None) -> str | None:
        cleaned = self._agent._clean_system_prompt(system_prompt)
        if cleaned is not None:
            return cleaned
        return self._system_prompt
