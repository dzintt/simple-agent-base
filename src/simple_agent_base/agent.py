from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator, Callable, Sequence
from typing import Any

from pydantic import BaseModel

from simple_agent_base.chat import ChatSession
from simple_agent_base.config import AgentConfig
from simple_agent_base.errors import MaxTurnsExceededError, MCPApprovalRequiredError
from simple_agent_base.mcp import (
    ApprovalHandler,
    MCPApprovalRequest,
    MCPCallRecord,
    MCPServer,
    mcp_approval_request_from_item,
    mcp_call_record_from_item,
)
from simple_agent_base.providers.base import ConversationItem, Provider
from simple_agent_base.providers.openai import OpenAIResponsesProvider
from simple_agent_base.sync_utils import SyncRuntime, ensure_sync_allowed, run_sync_awaitable
from simple_agent_base.tools import ToolRegistry
from simple_agent_base.types import (
    AgentEvent,
    AgentRunResult,
    ChatMessage,
    ChatSnapshot,
    FilePart,
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
        system_prompt: str | None = None,
        mcp_servers: Sequence[MCPServer] | None = None,
        approval_handler: ApprovalHandler | None = None,
    ) -> None:
        self.config = config
        self.registry = tools if isinstance(tools, ToolRegistry) else ToolRegistry(tools)
        self.provider = provider or OpenAIResponsesProvider(config)
        self.system_prompt = self._clean_system_prompt(system_prompt)
        self.mcp_servers: list[MCPServer] = list(mcp_servers or [])
        self.approval_handler = approval_handler
        self._sync_runtime: SyncRuntime | None = None

    async def run(
        self,
        input_data: str | Sequence[MessageInput],
        *,
        response_model: type[BaseModel] | None = None,
        system_prompt: str | None = None,
    ) -> AgentRunResult:
        transcript = self._normalize_input(input_data)
        transcript = self._prepend_system_prompt(
            transcript,
            system_prompt=self._resolve_system_prompt(system_prompt),
        )
        return await self._run_transcript(transcript, response_model=response_model)

    async def stream(
        self,
        input_data: str | Sequence[MessageInput],
        *,
        response_model: type[BaseModel] | None = None,
        system_prompt: str | None = None,
    ) -> AsyncIterator[AgentEvent]:
        transcript = self._normalize_input(input_data)
        transcript = self._prepend_system_prompt(
            transcript,
            system_prompt=self._resolve_system_prompt(system_prompt),
        )
        async for event in self._stream_transcript(transcript, response_model=response_model):
            yield event

    def chat(
        self,
        messages: str | Sequence[MessageInput] | None = None,
        *,
        system_prompt: str | None = None,
    ) -> ChatSession:
        initial_items: list[ConversationItem] = []
        if messages is not None:
            initial_items = self._normalize_input(messages)
        return ChatSession(
            self,
            items=initial_items,
            system_prompt=self._resolve_system_prompt(system_prompt),
        )

    def chat_from_snapshot(
        self,
        snapshot: ChatSnapshot | dict[str, Any],
    ) -> ChatSession:
        validated = snapshot if isinstance(snapshot, ChatSnapshot) else ChatSnapshot.model_validate(snapshot)
        return ChatSession(
            self,
            items=validated.items,
            system_prompt=self._clean_system_prompt(validated.system_prompt),
        )

    async def aclose(self) -> None:
        await self.provider.close()

    def run_sync(
        self,
        input_data: str | Sequence[MessageInput],
        *,
        response_model: type[BaseModel] | None = None,
        system_prompt: str | None = None,
    ) -> AgentRunResult:
        return self._run_sync_call(
            lambda: self.run(
                input_data,
                response_model=response_model,
                system_prompt=system_prompt,
            ),
            api_name="run_sync()",
            async_hint="await agent.run(...)",
        )

    def stream_sync(
        self,
        input_data: str | Sequence[MessageInput],
        *,
        response_model: type[BaseModel] | None = None,
        system_prompt: str | None = None,
    ):
        return self._stream_sync_call(
            lambda: self.stream(
                input_data,
                response_model=response_model,
                system_prompt=system_prompt,
            ),
            api_name="stream_sync()",
            async_hint="async for event in agent.stream(...)",
        )

    def close(self) -> None:
        ensure_sync_allowed("close()", "await agent.aclose()")
        if self._sync_runtime is not None:
            try:
                self._sync_runtime.run(lambda: self.aclose())
            finally:
                self._sync_runtime.close()
                self._sync_runtime = None
            return

        run_sync_awaitable(self.aclose())

    async def _run_transcript(
        self,
        transcript: list[ConversationItem],
        *,
        response_model: type[BaseModel] | None = None,
    ) -> AgentRunResult:
        tool_results: list[ToolExecutionResult] = []
        mcp_calls: list[MCPCallRecord] = []
        raw_responses: list[dict[str, Any]] = []

        for _ in range(self.config.max_turns):
            response = await self.provider.create_response(
                input_items=transcript,
                tools=self._build_tool_params(),
                response_model=response_model,
            )
            raw_responses.append(response.raw_response or {})
            transcript.extend(response.output_items)

            mcp_calls.extend(self._collect_mcp_calls(response.output_items))
            approval_requests = self._collect_mcp_approval_requests(response.output_items)

            if not response.tool_calls and not approval_requests:
                return AgentRunResult(
                    output_text=response.output_text,
                    output_data=response.output_data,
                    response_id=response.response_id,
                    tool_results=tool_results,
                    mcp_calls=mcp_calls,
                    raw_responses=raw_responses,
                )

            for result in await self._execute_tool_batch(response.tool_calls):
                tool_results.append(result)
                transcript.append(self._tool_output_item(result))

            for approval_item in await self._resolve_approvals(approval_requests):
                transcript.append(approval_item)

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
        mcp_calls: list[MCPCallRecord] = []
        raw_responses: list[dict[str, Any]] = []

        try:
            for _ in range(self.config.max_turns):
                final_response = None
                streamed_approval_requests: list[MCPApprovalRequest] = []

                async for event in self.provider.stream_response(
                    input_items=transcript,
                    tools=self._build_tool_params(),
                    response_model=response_model,
                ):
                    if event.type == "text_delta":
                        yield AgentEvent(type="text_delta", delta=event.delta)
                    elif event.type == "mcp_call_started":
                        yield AgentEvent(type="mcp_call_started", mcp_call=event.mcp_call)
                    elif event.type == "mcp_call_completed":
                        yield AgentEvent(type="mcp_call_completed", mcp_call=event.mcp_call)
                    elif event.type == "mcp_approval_requested":
                        streamed_approval_requests.append(event.mcp_approval)
                        yield AgentEvent(type="mcp_approval_requested", mcp_approval=event.mcp_approval)
                    elif event.type == "completed":
                        final_response = event.response

                if final_response is None:
                    raise MaxTurnsExceededError("Provider stream completed without a final response.")

                raw_responses.append(final_response.raw_response or {})
                transcript.extend(final_response.output_items)

                mcp_calls.extend(self._collect_mcp_calls(final_response.output_items))
                approval_requests = (
                    streamed_approval_requests
                    if streamed_approval_requests
                    else self._collect_mcp_approval_requests(final_response.output_items)
                )

                if not final_response.tool_calls and not approval_requests:
                    result = AgentRunResult(
                        output_text=final_response.output_text,
                        output_data=final_response.output_data,
                        response_id=final_response.response_id,
                        tool_results=tool_results,
                        mcp_calls=mcp_calls,
                        raw_responses=raw_responses,
                    )
                    yield AgentEvent(type="completed", result=result)
                    return

                for call in final_response.tool_calls:
                    yield AgentEvent(type="tool_call_started", tool_call=call)

                for tool_result in await self._execute_tool_batch(final_response.tool_calls):
                    tool_results.append(tool_result)
                    transcript.append(self._tool_output_item(tool_result))
                    yield AgentEvent(type="tool_call_completed", tool_result=tool_result)

                for approval_item in await self._resolve_approvals(approval_requests):
                    transcript.append(approval_item)

            raise MaxTurnsExceededError(
                f"Agent exceeded max_turns={self.config.max_turns} before reaching a final response."
            )
        except Exception as exc:
            yield AgentEvent(type="error", error=str(exc))

    async def _execute_tool(self, call: ToolCallRequest) -> ToolExecutionResult:
        return await self.registry.execute(call)

    async def _execute_tool_batch(
        self,
        calls: Sequence[ToolCallRequest],
    ) -> list[ToolExecutionResult]:
        if not self.config.parallel_tool_calls:
            results: list[ToolExecutionResult] = []
            for call in calls:
                results.append(await self._execute_tool(call))
            return results

        return list(await asyncio.gather(*(self._execute_tool(call) for call in calls)))

    def _build_tool_params(self) -> list[dict[str, Any]]:
        tools = list(self.registry.to_openai_tools())
        tools.extend(server.to_tool_param() for server in self.mcp_servers)
        return tools

    @staticmethod
    def _collect_mcp_calls(
        output_items: Sequence[ConversationItem],
    ) -> list[MCPCallRecord]:
        records: list[MCPCallRecord] = []
        for item in output_items:
            if not isinstance(item, dict) or item.get("type") != "mcp_call":
                continue
            records.append(mcp_call_record_from_item(item))
        return records

    @staticmethod
    def _collect_mcp_approval_requests(
        output_items: Sequence[ConversationItem],
    ) -> list[MCPApprovalRequest]:
        approvals: list[MCPApprovalRequest] = []
        for item in output_items:
            if not isinstance(item, dict) or item.get("type") != "mcp_approval_request":
                continue
            approvals.append(mcp_approval_request_from_item(item))
        return approvals

    async def _resolve_approvals(
        self,
        approvals: Sequence[MCPApprovalRequest],
    ) -> list[dict[str, Any]]:
        if not approvals:
            return []

        if self.approval_handler is None:
            raise MCPApprovalRequiredError(
                "An MCP server requested approval but no approval_handler was provided. "
                'Set require_approval="never" on the MCPServer or pass approval_handler=... to Agent(...).'
            )

        items: list[dict[str, Any]] = []
        for approval in approvals:
            result = self.approval_handler(approval)
            if inspect.isawaitable(result):
                approved = bool(await result)
            else:
                approved = bool(result)
            items.append(
                {
                    "type": "mcp_approval_response",
                    "approval_request_id": approval.id,
                    "approve": approved,
                }
            )
        return items

    def _run_sync_call(
        self,
        awaitable_factory: Callable[[], Any],
        *,
        api_name: str,
        async_hint: str,
    ) -> Any:
        ensure_sync_allowed(api_name, async_hint)
        return self._get_sync_runtime().run(awaitable_factory)

    def _stream_sync_call(
        self,
        async_iterable_factory: Callable[[], AsyncIterator[AgentEvent]],
        *,
        api_name: str,
        async_hint: str,
    ):
        ensure_sync_allowed(api_name, async_hint)
        return self._get_sync_runtime().iterate(async_iterable_factory)

    def _get_sync_runtime(self) -> SyncRuntime:
        if self._sync_runtime is None:
            self._sync_runtime = SyncRuntime()
        return self._sync_runtime

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

    def _resolve_system_prompt(self, system_prompt: str | None) -> str | None:
        cleaned = self._clean_system_prompt(system_prompt)
        if cleaned is not None:
            return cleaned
        return self.system_prompt

    @staticmethod
    def _clean_system_prompt(system_prompt: str | None) -> str | None:
        if system_prompt is None:
            return None
        cleaned = system_prompt.strip()
        if not cleaned:
            return None
        return cleaned

    @classmethod
    def _prepend_system_prompt(
        cls,
        items: list[ConversationItem],
        *,
        system_prompt: str | None,
    ) -> list[ConversationItem]:
        if system_prompt is None:
            return list(items)

        return [
            cls._message_to_item(ChatMessage(role="developer", content=system_prompt)),
            *items,
        ]

    @classmethod
    def _strip_prepended_system_prompt(
        cls,
        items: Sequence[ConversationItem],
        *,
        system_prompt: str | None,
    ) -> list[ConversationItem]:
        if system_prompt is None:
            return list(items)

        expected_item = cls._message_to_item(ChatMessage(role="developer", content=system_prompt))
        result = list(items)
        if result and result[0] == expected_item:
            return result[1:]
        return result

    @staticmethod
    def _persistable_items(items: Sequence[ConversationItem]) -> list[ConversationItem]:
        return [item for item in items if item.get("type") == "message"]

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
            content_parts: list[TextPart | ImagePart | FilePart] = []
            saw_rich_content = False

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
                            saw_rich_content = True
                    elif block_type == "input_file":
                        file_payload = {
                            "file_url": block.get("file_url"),
                            "file_data": block.get("file_data"),
                            "filename": block.get("filename"),
                        }
                        try:
                            content_parts.append(FilePart.model_validate(file_payload))
                            saw_rich_content = True
                        except Exception:
                            continue

            if saw_rich_content and content_parts:
                messages.append(ChatMessage(role=role, content=content_parts))
            elif text_parts:
                messages.append(ChatMessage(role=role, content="".join(text_parts)))

        return messages

    @staticmethod
    def _content_part_to_item(part: TextPart | ImagePart | FilePart) -> dict[str, Any]:
        if isinstance(part, TextPart):
            return {
                "type": "input_text",
                "text": part.text,
            }

        if isinstance(part, FilePart):
            item: dict[str, Any] = {"type": "input_file"}
            if part.file_url is not None:
                item["file_url"] = part.file_url
            if part.file_data is not None:
                item["file_data"] = part.file_data
            if part.filename is not None:
                item["filename"] = part.filename
            return item

        return {
            "type": "input_image",
            "image_url": part.image_url,
            "detail": part.detail,
        }
