from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from typing import Any

from pydantic import BaseModel
from openai import AsyncOpenAI, DefaultAioHttpClient

from agent_harness.config import AgentConfig
from agent_harness.errors import ProviderError
from agent_harness.types import ToolCallRequest

from .base import ConversationItem, ProviderCompletedEvent, ProviderResponse, ProviderTextDeltaEvent


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
        tools: Sequence[dict[str, Any]],
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
        tools: Sequence[dict[str, Any]],
        response_model: type[BaseModel] | None = None,
    ) -> AsyncIterator[ProviderTextDeltaEvent | ProviderCompletedEvent]:
        try:
            async with self._client.responses.stream(
                **self._request_kwargs(input_items, tools, response_model=response_model)
            ) as stream:
                async for event in stream:
                    if event.type == "response.output_text.delta":
                        yield ProviderTextDeltaEvent(delta=event.delta)

                final_response = await stream.get_final_response()
        except Exception as exc:
            raise ProviderError(f"OpenAI streaming response request failed: {exc}") from exc

        yield ProviderCompletedEvent(response=self._convert_response(final_response))

    async def close(self) -> None:
        await self._client.close()

    def _request_kwargs(
        self,
        input_items: Sequence[ConversationItem],
        tools: Sequence[dict[str, Any]],
        response_model: type[BaseModel] | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self._config.model,
            "input": list(input_items),
            "parallel_tool_calls": self._config.parallel_tool_calls,
        }

        if tools:
            kwargs["tools"] = list(tools)

        if self._config.temperature is not None:
            kwargs["temperature"] = self._config.temperature

        if response_model is not None:
            kwargs["text_format"] = response_model

        return kwargs

    def _convert_response(self, response: Any) -> ProviderResponse:
        output_items = [self._to_dict(item) for item in getattr(response, "output", [])]
        tool_calls: list[ToolCallRequest] = []

        for item in getattr(response, "output", []):
            if getattr(item, "type", None) != "function_call":
                continue

            raw_arguments = getattr(item, "arguments", "{}")

            try:
                arguments = json.loads(raw_arguments) if raw_arguments else {}
            except json.JSONDecodeError as exc:
                raise ProviderError(
                    f"Model returned invalid JSON arguments for tool '{item.name}': {raw_arguments}"
                ) from exc

            tool_calls.append(
                ToolCallRequest(
                    call_id=item.call_id,
                    name=item.name,
                    arguments=arguments,
                    raw_arguments=raw_arguments,
                )
            )

        return ProviderResponse(
            response_id=getattr(response, "id", None),
            output_text=getattr(response, "output_text", ""),
            output_data=getattr(response, "output_parsed", None),
            tool_calls=tool_calls,
            output_items=output_items,
            raw_response=self._to_dict(response),
        )

    @staticmethod
    def _to_dict(value: Any) -> dict[str, Any]:
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json", warnings="none")
        if hasattr(value, "to_dict"):
            return value.to_dict()
        if isinstance(value, dict):
            return value
        raise ProviderError(f"Unsupported response payload type: {type(value)!r}")
