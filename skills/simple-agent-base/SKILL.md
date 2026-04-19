---
name: simple-agent-base
description: Use when working in this repository or when writing code that uses the simple_agent_base package. Covers the package mental model, the correct public APIs, tool registration rules, chat/session behavior, structured output, streaming, multimodal inputs, sync wrappers, cleanup, and the source files to inspect when changing behavior.
---

# Simple Agent Base

Use this skill whenever you need to write code against `simple_agent_base` or modify the package itself.

Do not re-derive the package API from scratch. Use the patterns in this skill first. Only read the deeper docs if the task depends on exact edge-case behavior.

## What This Package Is

`simple_agent_base` is a small async-first harness over the OpenAI Responses API.

It provides:

- one-shot runs with `Agent.run(...)`
- streaming with `Agent.stream(...)`
- local Python tools with automatic JSON schema generation
- structured outputs with `response_model=...`
- in-memory chat sessions with `agent.chat()`
- snapshot and restore with `chat.snapshot()` and `agent.chat_from_snapshot(...)`
- image and file inputs through `ChatMessage` content parts
- sync wrappers for non-async programs

It is not a full agent framework. Do not invent concepts like planning, memory systems, approval flows, hosted tools, retrieval, or workflow orchestration unless the user explicitly wants to build them on top.

## Public API

Use these exports from `simple_agent_base`:

- `Agent`
- `AgentConfig`
- `ChatSession`
- `ChatMessage`
- `TextPart`
- `ImagePart`
- `FilePart`
- `ChatSnapshot`
- `ToolRegistry`
- `tool`

## Choose The Right API

Use this routing:

- one request, final answer: `await agent.run(...)`
- one request, streamed output: `async for event in agent.stream(...)`
- multi-turn conversation state: `chat = agent.chat()`
- exact resume from saved chat: `agent.chat_from_snapshot(...)`
- typed final output: pass `response_model=MyPydanticModel`
- text-only request: pass a plain string
- multimodal request: pass `ChatMessage(..., content=[...parts...])`
- synchronous program: `run_sync(...)` or `stream_sync(...)`

## Core Usage Patterns

### Basic Agent

```python
from simple_agent_base import Agent, AgentConfig

agent = Agent(config=AgentConfig(model="gpt-5.4"))

try:
    result = await agent.run("Say hello in one sentence.")
    print(result.output_text)
finally:
    await agent.aclose()
```

### Agent With Tools

```python
from simple_agent_base import Agent, AgentConfig, tool


@tool
async def ping(message: str) -> str:
    """Echo a message back."""
    return f"pong: {message}"


agent = Agent(
    config=AgentConfig(model="gpt-5.4"),
    tools=[ping],
)
```

### Chat Session

```python
chat = agent.chat(system_prompt="You are concise.")

await chat.run("My name is Anson.")
result = await chat.run("What name did I tell you?")
```

### Structured Output

```python
from pydantic import BaseModel


class Person(BaseModel):
    name: str
    age: int


result = await agent.run(
    "Extract the person from: Sarah is 29 years old.",
    response_model=Person,
)

print(result.output_data)
```

### Streaming

```python
async for event in agent.stream("Explain async IO in one sentence."):
    if event.type == "text_delta" and event.delta:
        print(event.delta, end="")
    elif event.type == "completed" and event.result is not None:
        print(event.result.output_text)
```

### Multimodal Input

```python
from simple_agent_base import ChatMessage, ImagePart, TextPart

result = await agent.run(
    [
        ChatMessage(
            role="user",
            content=[
                TextPart("Describe this image."),
                ImagePart.from_file("cat.png", detail="high"),
            ],
        )
    ]
)
```

## Exact Behavioral Rules

### Input Normalization

- a plain string becomes one user message
- in a message sequence, bare strings also become user messages
- explicit messages validate through `ChatMessage`
- `ChatMessage.content` can be a plain string or a list of content parts

### System Prompt Behavior

The convenience `system_prompt` is implemented as a prepended `developer` message.

Important consequences:

- agent-level `system_prompt` applies by default
- per-call `system_prompt` overrides the agent default for that call
- `chat = agent.chat(system_prompt=...)` sets a default prompt for that chat
- `chat.history` does not include the convenience prompt
- chat snapshots store the session prompt separately instead of persisting a fake developer message into history

### Chat Persistence

- `chat.history` returns reconstructed `ChatMessage` values
- `chat.items` returns stored raw conversation items
- only `message` items are persisted in chat state
- tool output items are not exposed in `chat.history`
- after a successful streamed turn, the completed turn is persisted

### Streaming Semantics

`Agent.stream(...)` emits `AgentEvent` values with these types:

- `text_delta`
- `tool_call_started`
- `tool_call_completed`
- `completed`
- `error`

Important behavior:

- final-result APIs raise runtime errors
- streaming APIs emit one `error` event and end the stream
- structured output appears only on the final `completed` event

### Run Result Fields

`AgentRunResult` contains:

- `output_text`
- `output_data`
- `response_id`
- `tool_results`
- `raw_responses`

## Tool Rules

Use `@tool` on plain Python callables.

Requirements:

- every parameter must have a type annotation
- `*args` and `**kwargs` are not allowed
- the default tool name is the function name
- the default tool description is the first docstring line
- sync tools run in a worker thread
- tool outputs are serialized to strings before being sent back to the model

You can override tool metadata:

```python
@tool(name="lookup_user", description="Fetch a user record.")
async def get_user(user_id: int) -> str:
    return '{"id": 1, "name": "Ada"}'
```

### Parallel Tool Calls

Tool execution is sequential by default.

Enable same-turn parallel execution with:

```python
AgentConfig(
    model="gpt-5.4",
    parallel_tool_calls=True,
)
```

Use this only when tools are independent. Do not enable it for tools with ordering dependencies or shared mutable state.

## Multimodal Rules

Use content parts for multimodal messages:

- `TextPart("...")`
- `ImagePart.from_url(...)`
- `ImagePart.from_file(...)`
- `FilePart.from_url(...)`
- `FilePart.from_file(...)`

Image file support:

- PNG
- JPEG
- WEBP
- GIF

`ImagePart.from_file(...)` converts the file to a Base64 data URL.

File part validation:

- exactly one of `file_url` or `file_data` must be set
- if using `file_data`, `filename` is required

`FilePart.from_file(...)` converts supported document-like files to Base64 data URLs.

## Sync Wrapper Rules

Use sync wrappers only in fully synchronous programs.

```python
agent = Agent(config=AgentConfig(model="gpt-5.4"))

try:
    result = agent.run_sync("Say hello.")
finally:
    agent.close()
```

Restrictions:

- `run_sync()` cannot be used inside a running event loop
- `stream_sync()` cannot be used inside a running event loop
- `chat.run_sync()` and `chat.stream_sync()` have the same restriction

If you used sync wrappers, call `agent.close()` when done.

## Cleanup Rules

For async usage:

```python
try:
    ...
finally:
    await agent.aclose()
```

For sync usage:

```python
try:
    ...
finally:
    agent.close()
```

## Configuration

`AgentConfig` fields:

- `model`
- `api_key`
- `base_url`
- `max_turns`
- `parallel_tool_calls`
- `temperature`
- `timeout`

Environment variables:

- `OPENAI_API_KEY`
- `OPENAI_MODEL`
- `OPENAI_BASE_URL`

## Failure Model

`run()` and `run_sync()` may raise:

- `ProviderError`
- `ToolExecutionError`
- `MaxTurnsExceededError`

`stream()` and `stream_sync()` convert runtime failures into:

- `AgentEvent(type="error", error="...")`

## Source Files To Inspect

When changing behavior, inspect these files first:

- `src/simple_agent_base/agent.py`
  Main run and stream loops.
- `src/simple_agent_base/chat.py`
  Chat sessions, persistence, and snapshots.
- `src/simple_agent_base/types.py`
  Public models and multimodal helpers.
- `src/simple_agent_base/config.py`
  Settings.
- `src/simple_agent_base/providers/openai.py`
  OpenAI Responses provider.
- `src/simple_agent_base/tools/base.py`
  Tool schema helpers.
- `src/simple_agent_base/tools/decorators.py`
  `@tool`.
- `src/simple_agent_base/tools/registry.py`
  Tool registration and execution.
- `src/simple_agent_base/sync_utils.py`
  Sync runtime.

Use these tests as the behavioral contract:

- `tests/test_agent.py`
- `tests/test_streaming.py`
- `tests/test_tools.py`

## Preferred Workflow For Coding Agents

When asked to use this package:

1. Default to `Agent`, not custom wrappers, unless the codebase already has one.
2. Use async APIs unless the surrounding program is clearly synchronous.
3. Use `@tool` for local tools instead of inventing manual schemas.
4. Use `response_model` when the caller needs typed output.
5. Use `chat = agent.chat()` for multi-turn state instead of manually rebuilding history unless the task needs explicit message control.
6. Close agents explicitly with `await agent.aclose()` or `agent.close()`.
7. If behavior is unclear, read only the relevant deep doc:
   - `docs/usage.md`
   - `docs/tools.md`
   - `docs/structured-output.md`
   - `docs/architecture.md`
   - `docs/development.md`

## What Not To Invent

Do not assume the package has:

- planning
- agent memory beyond in-memory chat history
- retrieval
- hosted tools
- approval flows
- retries
- durable workflow state
- provider-agnostic abstractions

If a task needs those features, build them explicitly on top or state that they are outside the package's current scope.
