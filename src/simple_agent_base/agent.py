from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import TypeVar

from pydantic import BaseModel

from simple_agent_base.chat import ChatSession
from simple_agent_base.config import AgentConfig
from simple_agent_base.errors import (
    MaxTurnsExceededError,
    MCPApprovalRequiredError,
    ToolExecutionError,
    ToolRegistrationError,
)
from simple_agent_base.mcp import (
    ApprovalHandler,
    MCPApprovalRequest,
    MCPBridgeManager,
    MCPCallRecord,
    MCPToolDefinition,
    MCPServer,
    build_mcp_approval_request,
    mcp_result_payload,
    normalize_mcp_tool_result,
    run_approval_handler,
)
from simple_agent_base.providers.base import Provider, ProviderResponse
from simple_agent_base.providers.openai import OpenAIResponsesProvider
from simple_agent_base.sync_utils import SyncRuntime, ensure_sync_allowed, run_sync_awaitable
from simple_agent_base.tools import ToolRegistry
from simple_agent_base.types import (
    AgentEvent,
    AgentRunResult,
    ChatMessage,
    ChatSnapshot,
    ConversationItem,
    FilePart,
    ImagePart,
    MessageInput,
    JSONObject,
    TextPart,
    ToolCallRequest,
    ToolExecutionResult,
    UsageMetadata,
)

T = TypeVar("T")


@dataclass(slots=True)
class _ExecutedCall:
    tool_result: ToolExecutionResult
    mcp_call: MCPCallRecord | None = None
    emit_mcp_events: bool = True


@dataclass(slots=True)
class _PreparedToolCall:
    events: list[AgentEvent]
    mcp_tool: MCPToolDefinition | None = None


class Agent:
    def __init__(
        self,
        config: AgentConfig,
        tools: list[Callable[..., object]] | ToolRegistry | None = None,
        provider: Provider | None = None,
        system_prompt: str | None = None,
        mcp_servers: Sequence[MCPServer] | None = None,
        hosted_tools: Sequence[JSONObject] | None = None,
        approval_handler: ApprovalHandler | None = None,
    ) -> None:
        self.config = config
        self.registry = tools if isinstance(tools, ToolRegistry) else ToolRegistry(tools)
        self.provider = provider or OpenAIResponsesProvider(config)
        self.system_prompt = self._clean_system_prompt(system_prompt)
        self.mcp_servers: list[MCPServer] = list(mcp_servers or [])
        self.hosted_tools: list[JSONObject] = self._validate_hosted_tools(hosted_tools)
        self.approval_handler = approval_handler
        self._mcp_manager = MCPBridgeManager(self.mcp_servers)
        self._sync_runtime: SyncRuntime | None = None

    async def run(
        self,
        input_data: str | Sequence[MessageInput],
        *,
        response_model: type[BaseModel] | None = None,
        system_prompt: str | None = None,
    ) -> AgentRunResult:
        transcript = self._build_transcript(
            input_data,
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
        transcript = self._build_transcript(
            input_data,
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
        snapshot: ChatSnapshot | JSONObject,
    ) -> ChatSession:
        validated = snapshot if isinstance(snapshot, ChatSnapshot) else ChatSnapshot.model_validate(snapshot)
        return ChatSession(
            self,
            items=validated.items,
            system_prompt=self._clean_system_prompt(validated.system_prompt),
        )

    async def aclose(self) -> None:
        try:
            await self._mcp_manager.close()
        finally:
            await self.provider.close()

    async def __aenter__(self) -> "Agent":
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.aclose()

    def __enter__(self) -> "Agent":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

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
    ) -> Iterator[AgentEvent]:
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
        await self._ensure_mcp_ready()
        tool_results: list[ToolExecutionResult] = []
        mcp_calls: list[MCPCallRecord] = []
        usage_by_response: list[UsageMetadata] = []
        raw_responses: list[JSONObject] = []

        for _ in range(self.config.max_turns):
            response = await self.provider.create_response(
                input_items=transcript,
                tools=self._build_tool_params(),
                response_model=response_model,
            )
            raw_responses.append(response.raw_response or {})
            if response.usage is not None:
                usage_by_response.append(response.usage)
            transcript.extend(response.output_items)

            if not response.tool_calls:
                return self._build_run_result(
                    response=response,
                    tool_results=tool_results,
                    mcp_calls=mcp_calls,
                    usage_by_response=usage_by_response,
                    raw_responses=raw_responses,
                )

            for executed in await self._execute_tool_batch(response.tool_calls):
                tool_results.append(executed.tool_result)
                if executed.mcp_call is not None:
                    mcp_calls.append(executed.mcp_call)
                transcript.append(self._tool_output_item(executed.tool_result))

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
        usage_by_response: list[UsageMetadata] = []
        raw_responses: list[JSONObject] = []

        await self._ensure_mcp_ready()
        for _ in range(self.config.max_turns):
            final_response = None

            async for event in self.provider.stream_response(
                input_items=transcript,
                tools=self._build_tool_params(),
                response_model=response_model,
            ):
                if event.type in {"text_delta", "reasoning_delta"}:
                    yield AgentEvent(type=event.type, delta=event.delta)
                elif event.type == "tool_arguments_delta":
                    yield AgentEvent(
                        type="tool_arguments_delta",
                        delta=event.delta,
                        tool_item_id=event.item_id,
                        tool_name=event.name,
                    )
                elif event.type == "completed":
                    final_response = event.response

            if final_response is None:
                raise MaxTurnsExceededError("Provider stream completed without a final response.")

            raw_responses.append(final_response.raw_response or {})
            if final_response.usage is not None:
                usage_by_response.append(final_response.usage)
            transcript.extend(final_response.output_items)

            if not final_response.tool_calls:
                result = self._build_run_result(
                    response=final_response,
                    tool_results=tool_results,
                    mcp_calls=mcp_calls,
                    usage_by_response=usage_by_response,
                    raw_responses=raw_responses,
                )
                yield AgentEvent(type="completed", result=result)
                return

            for call in final_response.tool_calls:
                yield AgentEvent(type="tool_call_started", tool_call=call)

            async for event in self._execute_tool_batch_stream(
                final_response.tool_calls,
                tool_results=tool_results,
                mcp_calls=mcp_calls,
                transcript=transcript,
            ):
                yield event

        raise MaxTurnsExceededError(
            f"Agent exceeded max_turns={self.config.max_turns} before reaching a final response."
        )

    async def _ensure_mcp_ready(self) -> None:
        await self._mcp_manager.ensure_initialized()
        local_names = {definition.name for definition in self.registry.list_definitions()}
        duplicate_names = sorted(local_names & self._mcp_manager.tool_names())
        if duplicate_names:
            duplicate_list = ", ".join(duplicate_names)
            raise ToolRegistrationError(f"MCP tool names conflict with local tools: {duplicate_list}")

    async def _execute_tool(
        self,
        call: ToolCallRequest,
        *,
        skip_mcp_approval: bool = False,
    ) -> _ExecutedCall:
        if self._mcp_manager.has_tool(call.name):
            return await self._execute_mcp_tool(call, skip_approval=skip_mcp_approval)
        return _ExecutedCall(
            tool_result=await self._with_tool_timeout(
                self.registry.execute(call),
                timeout_message=f"Tool '{call.name}' timed out after {self.config.tool_timeout} seconds.",
            )
        )

    async def _execute_tool_batch(
        self,
        calls: Sequence[ToolCallRequest],
    ) -> list[_ExecutedCall]:
        if not self.config.parallel_tool_calls:
            results: list[_ExecutedCall] = []
            for call in calls:
                results.append(await self._execute_tool(call))
            return results

        return list(await asyncio.gather(*(self._execute_tool(call) for call in calls)))

    async def _execute_tool_batch_stream(
        self,
        calls: Sequence[ToolCallRequest],
        *,
        tool_results: list[ToolExecutionResult],
        mcp_calls: list[MCPCallRecord],
        transcript: list[ConversationItem],
    ) -> AsyncIterator[AgentEvent]:
        if self.config.parallel_tool_calls:
            pending: list[asyncio.Task[_ExecutedCall]] = []
            for call in calls:
                prepared = self._prepare_tool_call_for_stream(call)
                for event in prepared.events:
                    yield event
                skip_mcp_approval = False
                if prepared.mcp_tool is not None:
                    approved = await self._approve_mcp_call(
                        self._build_mcp_approval_request(prepared.mcp_tool, call)
                    )
                    if not approved:
                        for event in self._append_executed_call_events(
                            self._build_denied_mcp_call(prepared.mcp_tool, call),
                            tool_results=tool_results,
                            mcp_calls=mcp_calls,
                            transcript=transcript,
                        ):
                            yield event
                        continue
                    yield self._build_mcp_started_event(call)
                    skip_mcp_approval = True
                pending.append(
                    asyncio.create_task(
                        self._execute_tool(call, skip_mcp_approval=skip_mcp_approval)
                    )
                )

            if not pending:
                return

            executed_calls = await asyncio.gather(*pending)
            for executed in executed_calls:
                for event in self._append_executed_call_events(
                    executed,
                    tool_results=tool_results,
                    mcp_calls=mcp_calls,
                    transcript=transcript,
                ):
                    yield event
            return

        for call in calls:
            prepared = self._prepare_tool_call_for_stream(call)
            for event in prepared.events:
                yield event
            skip_mcp_approval = False
            if prepared.mcp_tool is not None:
                approved = await self._approve_mcp_call(
                    self._build_mcp_approval_request(prepared.mcp_tool, call)
                )
                if not approved:
                    for event in self._append_executed_call_events(
                        self._build_denied_mcp_call(prepared.mcp_tool, call),
                        tool_results=tool_results,
                        mcp_calls=mcp_calls,
                        transcript=transcript,
                    ):
                        yield event
                    continue
                yield self._build_mcp_started_event(call)
                skip_mcp_approval = True
            executed = await self._execute_tool(call, skip_mcp_approval=skip_mcp_approval)
            for event in self._append_executed_call_events(
                executed,
                tool_results=tool_results,
                mcp_calls=mcp_calls,
                transcript=transcript,
            ):
                yield event

    def _build_tool_params(self) -> list[JSONObject]:
        tools = list(self.registry.to_openai_tools())
        tools.extend(self._mcp_manager.to_openai_tools())
        tools.extend(self.hosted_tools)
        return tools

    @staticmethod
    def _validate_hosted_tools(
        hosted_tools: Sequence[JSONObject] | None,
    ) -> list[JSONObject]:
        if not hosted_tools:
            return []

        validated: list[JSONObject] = []
        for index, entry in enumerate(hosted_tools):
            if not isinstance(entry, dict):
                raise ToolRegistrationError(
                    f"hosted_tools[{index}] must be a dict, got {type(entry).__name__}."
                )
            tool_type = entry.get("type")
            if not isinstance(tool_type, str) or not tool_type:
                raise ToolRegistrationError(
                    f"hosted_tools[{index}] must include a non-empty string 'type' field."
                )
            validated.append(dict(entry))
        return validated

    @staticmethod
    def _build_run_result(
        *,
        response: ProviderResponse,
        tool_results: list[ToolExecutionResult],
        mcp_calls: list[MCPCallRecord],
        usage_by_response: list[UsageMetadata],
        raw_responses: list[JSONObject],
    ) -> AgentRunResult:
        return AgentRunResult(
            output_text=response.output_text,
            reasoning_summary=response.reasoning_summary,
            output_data=response.output_data,
            response_id=response.response_id,
            tool_results=tool_results,
            mcp_calls=mcp_calls,
            usage=Agent._aggregate_usage(usage_by_response),
            usage_by_response=list(usage_by_response),
            raw_responses=raw_responses,
        )

    @staticmethod
    def _aggregate_usage(usages: Sequence[UsageMetadata]) -> UsageMetadata | None:
        input_total = Agent._sum_optional_usage_field(usages, "input_tokens")
        output_total = Agent._sum_optional_usage_field(usages, "output_tokens")
        total = Agent._sum_optional_usage_field(usages, "total_tokens")

        if input_total is None and output_total is None and total is None:
            return None

        return UsageMetadata(
            input_tokens=input_total,
            output_tokens=output_total,
            total_tokens=total,
        )

    @staticmethod
    def _sum_optional_usage_field(usages: Sequence[UsageMetadata], field_name: str) -> int | None:
        values = [getattr(usage, field_name) for usage in usages]
        present = [value for value in values if isinstance(value, int)]
        return sum(present) if present else None

    async def _execute_mcp_tool(
        self,
        call: ToolCallRequest,
        *,
        skip_approval: bool = False,
    ) -> _ExecutedCall:
        tool = self._mcp_manager.get_tool(call.name)
        approved = True

        if tool.require_approval and not skip_approval:
            approved = await self._approve_mcp_call(self._build_mcp_approval_request(tool, call))

        if not approved:
            return self._build_denied_mcp_call(tool, call)

        async def call_mcp() -> tuple[MCPToolDefinition, object, str]:
            called_tool, raw_result = await self._mcp_manager.call_tool(
                namespaced_name=call.name,
                arguments=call.arguments,
            )
            output = normalize_mcp_tool_result(raw_result)
            return called_tool, raw_result, output

        try:
            tool, raw_result, output = await self._with_tool_timeout(
                call_mcp(),
                timeout_message=(
                    f"MCP tool '{tool.tool_name}' timed out after "
                    f"{self.config.tool_timeout} seconds."
                ),
            )
        except ToolExecutionError:
            raise
        except Exception as exc:
            raise ToolExecutionError(f"MCP tool '{tool.tool_name}' failed: {exc}") from exc

        if raw_result.isError:
            raise ToolExecutionError(f"MCP tool '{tool.tool_name}' failed: {output}")

        return _ExecutedCall(
            tool_result=ToolExecutionResult(
                call_id=call.call_id,
                name=call.name,
                arguments=call.arguments,
                output=output,
                raw_output=mcp_result_payload(raw_result),
            ),
            mcp_call=MCPCallRecord(
                id=call.call_id,
                server_name=tool.server_name,
                name=tool.tool_name,
                arguments=call.arguments,
                output=output,
            ),
        )

    async def _with_tool_timeout(
        self,
        awaitable: Awaitable[T],
        *,
        timeout_message: str,
    ) -> T:
        if self.config.tool_timeout is None:
            return await awaitable

        try:
            return await asyncio.wait_for(awaitable, timeout=self.config.tool_timeout)
        except TimeoutError as exc:
            raise ToolExecutionError(timeout_message) from exc

    def _prepare_tool_call_for_stream(
        self,
        call: ToolCallRequest,
    ) -> _PreparedToolCall:
        if not self._mcp_manager.has_tool(call.name):
            return _PreparedToolCall(events=[])

        tool = self._mcp_manager.get_tool(call.name)
        events: list[AgentEvent] = []
        if tool.require_approval:
            approval_event = self._build_mcp_approval_event(call)
            if approval_event is not None:
                events.append(approval_event)
            return _PreparedToolCall(events=events, mcp_tool=tool)

        events.append(self._build_mcp_started_event(call))
        return _PreparedToolCall(events=events)

    def _build_denied_mcp_call(self, tool: MCPToolDefinition, call: ToolCallRequest) -> _ExecutedCall:
        message = "MCP tool call denied by approval handler."
        return _ExecutedCall(
            tool_result=ToolExecutionResult(
                call_id=call.call_id,
                name=call.name,
                arguments=call.arguments,
                output=message,
            ),
            mcp_call=MCPCallRecord(
                id=call.call_id,
                server_name=tool.server_name,
                name=tool.tool_name,
                arguments=call.arguments,
                error=message,
            ),
            emit_mcp_events=False,
        )

    def _append_executed_call_events(
        self,
        executed: _ExecutedCall,
        *,
        tool_results: list[ToolExecutionResult],
        mcp_calls: list[MCPCallRecord],
        transcript: list[ConversationItem],
    ) -> list[AgentEvent]:
        events: list[AgentEvent] = []
        if executed.mcp_call is not None:
            mcp_calls.append(executed.mcp_call)
            if executed.emit_mcp_events:
                events.append(AgentEvent(type="mcp_call_completed", mcp_call=executed.mcp_call))

        tool_results.append(executed.tool_result)
        transcript.append(self._tool_output_item(executed.tool_result))
        events.append(AgentEvent(type="tool_call_completed", tool_result=executed.tool_result))
        return events

    def _build_mcp_approval_request(
        self,
        tool: MCPToolDefinition,
        call: ToolCallRequest,
    ) -> MCPApprovalRequest:
        return build_mcp_approval_request(
            request_id=f"mcp-approval-{call.call_id}",
            server_name=tool.server_name,
            tool_name=tool.tool_name,
            arguments=call.arguments,
        )

    def _build_mcp_approval_event(self, call: ToolCallRequest) -> AgentEvent | None:
        tool = self._mcp_manager.get_tool(call.name)
        if not tool.require_approval:
            return None
        return AgentEvent(
            type="mcp_approval_requested",
            mcp_approval=self._build_mcp_approval_request(tool, call),
        )

    def _build_mcp_started_event(self, call: ToolCallRequest) -> AgentEvent:
        tool = self._mcp_manager.get_tool(call.name)
        return AgentEvent(
            type="mcp_call_started",
            mcp_call=MCPCallRecord(
                id=call.call_id,
                server_name=tool.server_name,
                name=tool.tool_name,
                arguments=call.arguments,
            ),
        )

    async def _approve_mcp_call(self, approval: MCPApprovalRequest) -> bool:
        if self.approval_handler is None:
            raise MCPApprovalRequiredError(
                "An MCP tool requires approval but no approval_handler was provided. "
                "Set require_approval=False on the MCPServer or pass approval_handler=... to Agent(...)."
            )

        result = run_approval_handler(self.approval_handler, approval)
        if inspect.isawaitable(result):
            return bool(await result)
        return bool(result)

    def _run_sync_call(
        self,
        awaitable_factory: Callable[[], Awaitable[T]],
        *,
        api_name: str,
        async_hint: str,
    ) -> T:
        ensure_sync_allowed(api_name, async_hint)
        return self._get_sync_runtime().run(awaitable_factory)

    def _stream_sync_call(
        self,
        async_iterable_factory: Callable[[], AsyncIterator[T]],
        *,
        api_name: str,
        async_hint: str,
    ) -> Iterator[T]:
        ensure_sync_allowed(api_name, async_hint)
        return self._get_sync_runtime().iterate(async_iterable_factory)

    def _get_sync_runtime(self) -> SyncRuntime:
        if self._sync_runtime is None:
            self._sync_runtime = SyncRuntime()
        return self._sync_runtime

    @staticmethod
    def _user_message(prompt: str) -> ConversationItem:
        return Agent._message_to_item(ChatMessage(role="user", content=prompt))

    @staticmethod
    def _tool_output_item(result: ToolExecutionResult) -> ConversationItem:
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
        return self._resolve_system_prompt_with_default(system_prompt, self.system_prompt)

    def _resolve_system_prompt_with_default(
        self,
        system_prompt: str | None,
        default: str | None,
    ) -> str | None:
        cleaned = self._clean_system_prompt(system_prompt)
        if cleaned is not None:
            return cleaned
        return default

    def _build_transcript(
        self,
        input_data: str | Sequence[MessageInput],
        *,
        system_prompt: str | None,
        prefix_items: Sequence[ConversationItem] | None = None,
    ) -> list[ConversationItem]:
        transcript = list(prefix_items or [])
        transcript.extend(self._normalize_input(input_data))
        return self._prepend_system_prompt(transcript, system_prompt=system_prompt)

    def _persist_chat_items(
        self,
        transcript: Sequence[ConversationItem],
        *,
        system_prompt: str | None,
    ) -> list[ConversationItem]:
        return self._persistable_items(
            self._strip_prepended_system_prompt(transcript, system_prompt=system_prompt)
        )

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
            content: str | list[JSONObject] = message.content
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
    def _content_part_to_item(part: TextPart | ImagePart | FilePart) -> JSONObject:
        if isinstance(part, TextPart):
            return {
                "type": "input_text",
                "text": part.text,
            }

        if isinstance(part, FilePart):
            item: JSONObject = {"type": "input_file"}
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
