from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from io import StringIO
from typing import cast

from pydantic import BaseModel
from openai import AsyncOpenAI, DefaultAioHttpClient

from simple_agent_base.config import AgentConfig
from simple_agent_base.errors import ProviderError
from simple_agent_base.types import ConversationItem, JSONObject, ToolCallRequest, UsageMetadata

from .base import (
    ProviderCompletedEvent,
    ProviderEvent,
    ProviderReasoningDeltaEvent,
    ProviderResponse,
    ProviderTextDeltaEvent,
    ProviderToolArgumentsDeltaEvent,
)


@dataclass(slots=True)
class _ReasoningSummaryAccumulator:
    buffer: StringIO = field(default_factory=StringIO)
    seen_keys: set[tuple[str | None, int | None]] = field(default_factory=set)

    def add_delta(self, event: object) -> str:
        self.seen_keys.add(self._key_for(event))
        delta = getattr(event, "delta", "")
        if delta:
            self.buffer.write(delta)
        return delta

    def add_done_fallback(self, event: object) -> None:
        if self._key_for(event) in self.seen_keys:
            return

        text = getattr(event, "text", "")
        if text:
            self.buffer.write(text)

    def build(self) -> str | None:
        summary = self.buffer.getvalue().strip()
        return summary or None

    @staticmethod
    def _key_for(event: object) -> tuple[str | None, int | None]:
        return (getattr(event, "item_id", None), getattr(event, "summary_index", None))


class OpenAIResponsesProvider:
    def __init__(self, config: AgentConfig) -> None:
        self._config = config
        self._client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
            http_client=DefaultAioHttpClient(),
        )

    async def create_response(
        self,
        *,
        input_items: Sequence[ConversationItem],
        tools: Sequence[JSONObject],
        response_model: type[BaseModel] | None = None,
    ) -> ProviderResponse:
        try:
            if response_model is None:
                response = await self._client.responses.create(**self._request_kwargs(input_items, tools))
            else:
                response = await self._client.responses.parse(
                    **self._request_kwargs(input_items, tools, response_model=response_model)
                )
        except Exception as exc:
            raise ProviderError(f"OpenAI response request failed: {exc}") from exc

        return self._convert_response(response)

    async def stream_response(
        self,
        *,
        input_items: Sequence[ConversationItem],
        tools: Sequence[JSONObject],
        response_model: type[BaseModel] | None = None,
    ) -> AsyncIterator[ProviderEvent]:
        reasoning_summary = _ReasoningSummaryAccumulator()
        function_call_meta: dict[str, tuple[str | None, str | None]] = {}

        try:
            async with self._client.responses.stream(
                **self._request_kwargs(input_items, tools, response_model=response_model)
            ) as stream:
                async for event in stream:
                    if event.type == "response.output_text.delta":
                        yield ProviderTextDeltaEvent(delta=event.delta)
                    elif event.type == "response.reasoning_summary_text.delta":
                        yield ProviderReasoningDeltaEvent(delta=reasoning_summary.add_delta(event))
                    elif event.type == "response.reasoning_summary_text.done":
                        reasoning_summary.add_done_fallback(event)
                    elif event.type == "response.output_item.added":
                        item = getattr(event, "item", None)
                        if getattr(item, "type", None) == "function_call":
                            function_call_meta[item.id] = (
                                getattr(item, "call_id", None),
                                getattr(item, "name", None),
                            )
                    elif event.type == "response.function_call_arguments.delta":
                        call_id, name = function_call_meta.get(event.item_id, (None, None))
                        yield ProviderToolArgumentsDeltaEvent(
                            item_id=event.item_id,
                            call_id=call_id,
                            name=name,
                            delta=event.delta,
                        )

                final_response = await stream.get_final_response()
        except Exception as exc:
            raise ProviderError(f"OpenAI streaming response request failed: {exc}") from exc

        response = self._convert_response(final_response)
        if response.reasoning_summary is None:
            response.reasoning_summary = reasoning_summary.build()
        yield ProviderCompletedEvent(response=response)

    async def close(self) -> None:
        await self._client.close()

    def _request_kwargs(
        self,
        input_items: Sequence[ConversationItem],
        tools: Sequence[JSONObject],
        response_model: type[BaseModel] | None = None,
    ) -> JSONObject:
        kwargs: JSONObject = {
            "model": self._config.model,
            "input": list(input_items),
            "parallel_tool_calls": self._config.parallel_tool_calls,
        }

        if tools:
            kwargs["tools"] = list(tools)

        if self._config.reasoning_effort is not None:
            kwargs["reasoning"] = {
                "effort": self._config.reasoning_effort,
                "summary": "auto",
            }

        if self._config.temperature is not None:
            kwargs["temperature"] = self._config.temperature

        if response_model is not None:
            kwargs["text_format"] = response_model

        return kwargs

    def _convert_response(self, response: BaseModel) -> ProviderResponse:
        output_items = [self._to_dict(item) for item in getattr(response, "output", [])]
        tool_calls: list[ToolCallRequest] = []

        for item in getattr(response, "output", []):
            if getattr(item, "type", None) != "function_call":
                continue

            raw_arguments = getattr(item, "arguments", "{}")

            try:
                parsed_arguments = json.loads(raw_arguments) if raw_arguments else {}
            except json.JSONDecodeError as exc:
                raise ProviderError(
                    f"Model returned invalid JSON arguments for tool '{item.name}': {raw_arguments}"
                ) from exc

            if not isinstance(parsed_arguments, dict):
                raise ProviderError(
                    f"Model returned non-object JSON arguments for tool '{item.name}': {raw_arguments}"
                )

            tool_calls.append(
                ToolCallRequest(
                    call_id=item.call_id,
                    name=item.name,
                    arguments=cast(JSONObject, parsed_arguments),
                    raw_arguments=raw_arguments,
                )
            )

        return ProviderResponse(
            response_id=getattr(response, "id", None),
            output_text=getattr(response, "output_text", ""),
            reasoning_summary=self._extract_reasoning_summary(response),
            output_data=cast(BaseModel | None, getattr(response, "output_parsed", None)),
            tool_calls=tool_calls,
            output_items=output_items,
            usage=self._extract_usage(response),
            raw_response=self._to_dict(response),
        )

    def _extract_usage(self, response: BaseModel) -> UsageMetadata | None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return None

        raw_usage = self._to_dict(usage)
        return UsageMetadata(
            input_tokens=self._optional_int(raw_usage.get("input_tokens")),
            output_tokens=self._optional_int(raw_usage.get("output_tokens")),
            total_tokens=self._optional_int(raw_usage.get("total_tokens")),
            input_tokens_details=self._optional_object(raw_usage.get("input_tokens_details")),
            output_tokens_details=self._optional_object(raw_usage.get("output_tokens_details")),
            raw=raw_usage,
        )

    def _extract_reasoning_summary(self, response: BaseModel) -> str | None:
        reasoning_summaries = [
            summary
            for item in getattr(response, "output", [])
            if getattr(item, "type", None) == "reasoning"
            if (summary := self._join_non_empty_texts(getattr(item, "summary", []) or [])) is not None
        ]
        return self._join_non_empty_texts(reasoning_summaries, separator="\n\n")

    @staticmethod
    def _join_non_empty_texts(parts: Sequence[object], *, separator: str = "") -> str | None:
        cleaned_parts: list[str] = []
        for part in parts:
            text = getattr(part, "text", part)
            if not isinstance(text, str):
                continue

            stripped = text.strip()
            if stripped:
                cleaned_parts.append(stripped)

        if not cleaned_parts:
            return None

        summary = separator.join(cleaned_parts).strip()
        return summary or None

    @staticmethod
    def _optional_int(value: object) -> int | None:
        return value if isinstance(value, int) else None

    @staticmethod
    def _optional_object(value: object) -> JSONObject | None:
        return cast(JSONObject, value) if isinstance(value, dict) else None

    @staticmethod
    def _to_dict(value: object) -> JSONObject:
        if isinstance(value, BaseModel):
            return cast(JSONObject, value.model_dump(mode="json", warnings="none"))
        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            return cast(JSONObject, to_dict())
        if isinstance(value, dict):
            return cast(JSONObject, value)
        raise ProviderError(f"Unsupported response payload type: {type(value)!r}")
