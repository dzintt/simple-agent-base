from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Sequence
from typing import Any

from pydantic import BaseModel

from agent_harness.chat import ChatSession
from agent_harness.config import AgentConfig
from agent_harness.errors import MaxTurnsExceededError
from agent_harness.providers.base import ConversationItem, Provider
from agent_harness.providers.openai import OpenAIResponsesProvider
from agent_harness.tools import ToolRegistry
from agent_harness.types import (
    AgentEvent,
    AgentRunResult,
    ChatMessage,
    ImagePart,
    MessageInput,
    TextPart,
    ToolCallRequest,
    ToolExecutionResult,
)


class Agent:
    def __init__(
        self,
        config: AgentConfig,
        tools: list[Callable[..., Any]] | ToolRegistry | None = None,
        provider: Provider | None = None,
    ) -> None:
        self.config = config
        self.registry = tools if isinstance(tools, ToolRegistry) else ToolRegistry(tools)
        self.provider = provider or OpenAIResponsesProvider(config)

    async def run(
        self,
        input_data: str | Sequence[MessageInput],
        *,
        response_model: type[BaseModel] | None = None,
    ) -> AgentRunResult:
        transcript = self._normalize_input(input_data)
        return await self._run_transcript(transcript, response_model=response_model)

    async def stream(
        self,
        input_data: str | Sequence[MessageInput],
        *,
        response_model: type[BaseModel] | None = None,
    ) -> AsyncIterator[AgentEvent]:
        transcript = self._normalize_input(input_data)
        async for event in self._stream_transcript(transcript, response_model=response_model):
            yield event

    def chat(self, messages: str | Sequence[MessageInput] | None = None) -> ChatSession:
        initial_items: list[ConversationItem] = []
        if messages is not None:
            initial_items = self._normalize_input(messages)
        return ChatSession(self, items=initial_items)

    async def aclose(self) -> None:
        await self.provider.close()

    async def _run_transcript(
        self,
        transcript: list[ConversationItem],
        *,
        response_model: type[BaseModel] | None = None,
    ) -> AgentRunResult:
        tool_results: list[ToolExecutionResult] = []
        raw_responses: list[dict[str, Any]] = []

        for _ in range(self.config.max_turns):
            response = await self.provider.create_response(
                input_items=transcript,
                tools=self.registry.to_openai_tools(),
                response_model=response_model,
            )
            raw_responses.append(response.raw_response or {})
            transcript.extend(response.output_items)

            if not response.tool_calls:
                return AgentRunResult(
                    output_text=response.output_text,
                    output_data=response.output_data,
                    response_id=response.response_id,
                    tool_results=tool_results,
                    raw_responses=raw_responses,
                )

            for call in response.tool_calls:
                result = await self._execute_tool(call)
                tool_results.append(result)
                transcript.append(self._tool_output_item(result))

        raise MaxTurnsExceededError(
            f"Agent exceeded max_turns={self.config.max_turns} before reaching a final response."
        )

    async def _stream_transcript(
        self,
        transcript: list[ConversationItem],
        *,
        response_model: type[BaseModel] | None = None,
    ) -> AsyncIterator[AgentEvent]:
        tool_results: list[ToolExecutionResult] = []
        raw_responses: list[dict[str, Any]] = []

        try:
            for _ in range(self.config.max_turns):
                final_response = None

                async for event in self.provider.stream_response(
                    input_items=transcript,
                    tools=self.registry.to_openai_tools(),
                    response_model=response_model,
                ):
                    if event.type == "text_delta":
                        yield AgentEvent(type="text_delta", delta=event.delta)
                    elif event.type == "completed":
                        final_response = event.response

                if final_response is None:
                    raise MaxTurnsExceededError("Provider stream completed without a final response.")

                raw_responses.append(final_response.raw_response or {})
                transcript.extend(final_response.output_items)

                if not final_response.tool_calls:
                    result = AgentRunResult(
                        output_text=final_response.output_text,
                        output_data=final_response.output_data,
                        response_id=final_response.response_id,
                        tool_results=tool_results,
                        raw_responses=raw_responses,
                    )
                    yield AgentEvent(type="completed", result=result)
                    return

                for call in final_response.tool_calls:
                    yield AgentEvent(type="tool_call_started", tool_call=call)
                    tool_result = await self._execute_tool(call)
                    tool_results.append(tool_result)
                    transcript.append(self._tool_output_item(tool_result))
                    yield AgentEvent(type="tool_call_completed", tool_result=tool_result)

            raise MaxTurnsExceededError(
                f"Agent exceeded max_turns={self.config.max_turns} before reaching a final response."
            )
        except Exception as exc:
            yield AgentEvent(type="error", error=str(exc))

    async def _execute_tool(self, call: ToolCallRequest) -> ToolExecutionResult:
        return await self.registry.execute(call)

    @staticmethod
    def _user_message(prompt: str) -> dict[str, Any]:
        return Agent._message_to_item(ChatMessage(role="user", content=prompt))

    @staticmethod
    def _tool_output_item(result: ToolExecutionResult) -> dict[str, Any]:
        return {
            "type": "function_call_output",
            "call_id": result.call_id,
            "output": result.output,
        }

    def _normalize_input(self, input_data: str | Sequence[MessageInput]) -> list[ConversationItem]:
        if isinstance(input_data, str):
            return [self._user_message(input_data)]

        items: list[ConversationItem] = []
        for message in input_data:
            if isinstance(message, str):
                items.append(self._user_message(message))
            else:
                chat_message = ChatMessage.model_validate(message)
                items.append(self._message_to_item(chat_message))

        return items

    @staticmethod
    def _message_to_item(message: ChatMessage) -> ConversationItem:
        if isinstance(message.content, str):
            content: str | list[dict[str, Any]] = message.content
        else:
            content = [Agent._content_part_to_item(part) for part in message.content]

        return {
            "type": "message",
            "role": message.role,
            "content": content,
        }

    @staticmethod
    def _messages_from_items(items: Sequence[ConversationItem]) -> list[ChatMessage]:
        messages: list[ChatMessage] = []

        for item in items:
            if item.get("type") != "message":
                continue

            role = item.get("role")
            content_value = item.get("content", [])
            if not isinstance(role, str):
                continue

            text_parts: list[str] = []
            content_parts: list[TextPart | ImagePart] = []
            saw_image = False

            if isinstance(content_value, str):
                text_parts.append(content_value)
            elif isinstance(content_value, list):
                for block in content_value:
                    if not isinstance(block, dict):
                        continue

                    block_type = block.get("type")
                    if block_type in {"input_text", "output_text"}:
                        text = block.get("text")
                        if isinstance(text, str):
                            text_parts.append(text)
                            content_parts.append(TextPart(text))
                    elif block_type == "input_image":
                        image_url = block.get("image_url")
                        detail = block.get("detail", "auto")
                        if isinstance(image_url, str) and isinstance(detail, str):
                            content_parts.append(ImagePart(image_url=image_url, detail=detail))
                            saw_image = True

            if saw_image and content_parts:
                messages.append(ChatMessage(role=role, content=content_parts))
            elif text_parts:
                messages.append(ChatMessage(role=role, content="".join(text_parts)))

        return messages

    @staticmethod
    def _content_part_to_item(part: TextPart | ImagePart) -> dict[str, Any]:
        if isinstance(part, TextPart):
            return {
                "type": "input_text",
                "text": part.text,
            }

        return {
            "type": "input_image",
            "image_url": part.image_url,
            "detail": part.detail,
        }
