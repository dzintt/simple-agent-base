from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence

from pydantic import BaseModel

from simple_agent_base import Agent, AgentConfig
from simple_agent_base.providers.base import (
    ProviderCompletedEvent,
    ProviderEvent,
    ProviderResponse,
    ProviderTextDeltaEvent,
)
from simple_agent_base.types import ConversationItem, JSONObject


class StaticProvider:
    async def create_response(
        self,
        *,
        input_items: Sequence[ConversationItem],
        tools: Sequence[JSONObject],
        response_model: type[BaseModel] | None = None,
    ) -> ProviderResponse:
        return self._response()

    async def stream_response(
        self,
        *,
        input_items: Sequence[ConversationItem],
        tools: Sequence[JSONObject],
        response_model: type[BaseModel] | None = None,
    ) -> AsyncIterator[ProviderEvent]:
        yield ProviderTextDeltaEvent(delta="Hello ")
        yield ProviderTextDeltaEvent(delta="from a custom provider.")
        yield ProviderCompletedEvent(response=self._response())

    async def close(self) -> None:
        return None

    @staticmethod
    def _response() -> ProviderResponse:
        return ProviderResponse(
            response_id="custom_resp_1",
            output_text="Hello from a custom provider.",
            output_items=[
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Hello from a custom provider.",
                        }
                    ],
                }
            ],
            raw_response={"id": "custom_resp_1", "provider": "static"},
        )


async def main() -> None:
    async with Agent(
        config=AgentConfig(model="static-demo"),
        provider=StaticProvider(),
    ) as agent:
        result = await agent.run("Say hello.")
        print(result.output_text)

        async for event in agent.stream("Say hello again."):
            if event.type == "text_delta" and event.delta:
                print(event.delta, end="")
        print()


if __name__ == "__main__":
    asyncio.run(main())
