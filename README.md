# Agent Harness Base

Small async-first Python harness for building OpenAI-powered agents.

This package gives you a thin layer over the OpenAI Responses API with a small public API:

- `Agent`
- `AgentConfig`
- `ChatMessage`
- `TextPart`
- `ImagePart`
- `FilePart`
- `ChatSession`
- `ChatSnapshot`
- `ToolRegistry`
- `tool`

Use it when you want:

- one-shot runs or streaming
- local Python tools with JSON-schema generation
- optional same-turn parallel tool execution
- structured outputs with Pydantic
- in-memory chat history with exact snapshot/restore
- explicit sync wrappers for non-async code
- multimodal inputs for images and documents

Use something else when you want:

- a full agent framework with planning, memory, or orchestration
- hosted tool execution
- retrieval, vector storage, or workflow state machines

This repo is a reusable base layer, not an opinionated product framework.

## How This Compares To Other Python Packages

This repo is for teams that want more structure than the raw OpenAI SDK, but less framework than a full agent platform.

Why people pick this module:

- one small API for `run`, `stream`, tools, structured output, chat state, images, and files
- OpenAI-specific behavior instead of provider abstraction layers you may not need
- code you can read end-to-end in one sitting
- easy to reuse across projects without adopting a bigger runtime model

The tradeoff is deliberate:

| If you want... | Best fit |
|---|---|
| direct access to the raw OpenAI API surface with minimal abstraction | OpenAI Python SDK |
| a small OpenAI-specific harness that removes repeated agent plumbing without becoming a framework | this module |
| OpenAI's broader code-first agent runtime with orchestration, state, approvals, and more advanced patterns | OpenAI Agents SDK |
| a provider-agnostic typed agent framework with more built-in abstractions | PydanticAI |
| structured outputs and validation more than agent/session orchestration | Instructor |
| graph-based workflows, middleware, and a larger agent ecosystem | LangChain agents |

This module is the best fit when you want the middle ground:

- more ergonomic than writing the tool loop and message normalization yourself
- easier to own than a larger framework
- narrow enough that contributors can understand the whole package quickly

## Repository Layout

Use this structure when you add features or examples:

```text
agent-harness-base/
|-- src/
|   `-- agent_harness/
|       |-- __init__.py           # Public package exports
|       |-- agent.py              # Core run/stream loop
|       |-- chat.py               # Stateful chat sessions and snapshots
|       |-- config.py             # AgentConfig and env-backed settings
|       |-- errors.py             # Public exception types
|       |-- sync_utils.py         # Sync wrappers over the async runtime
|       |-- types.py              # Pydantic models and multimodal parts
|       |-- providers/
|       |   |-- base.py           # Provider protocol and response types
|       |   `-- openai.py         # OpenAI Responses API implementation
|       `-- tools/
|           |-- base.py           # Tool schema helpers
|           |-- decorators.py     # @tool decorator
|           `-- registry.py       # Tool registration and execution
|-- examples/                     # Copy-paste usage examples
|-- scripts/                      # Manual and live verification scripts
|-- tests/                        # Local unit tests with fake providers
|-- pyproject.toml                # Package metadata and dependencies
|-- README.md                     # Main developer guide
`-- .env.example                  # Environment variable template
```

Contributor guide:

- put package code under `src/agent_harness/`
- add new public APIs to `src/agent_harness/__init__.py`
- add runnable usage samples under `examples/`
- add verification scripts under `scripts/`
- add local tests under `tests/`

## Install

This repo is already structured as an installable Python package. Right now the public install path is GitHub.

Install from GitHub with `pip`:

```bash
python -m pip install "git+https://github.com/dzintt/agent-harness-base.git"
```

Install from GitHub with `uv`:

```bash
uv add "git+https://github.com/dzintt/agent-harness-base.git"
```

Install from a local checkout with `pip`:

```bash
python -m pip install .
```

Install from a local checkout with `uv`:

```bash
uv sync
```

Install local development dependencies with `pip`:

```bash
python -m pip install -e ".[dev]"
```

Install local development dependencies with `uv`:

```bash
uv sync --dev
```

Import it as:

```python
import agent_harness
```

If you want users to run `pip install agent-harness-base` or `uv add agent-harness-base` without a GitHub URL, publish the built package to PyPI:

```bash
uv build
uv publish
```

Set your API key:

```bash
$env:OPENAI_API_KEY="your_key_here"
```

Run the smallest example:

```bash
uv run python examples/basic_agent.py
```

Minimal usage after install:

`pip` install example:

```bash
python -m pip install "git+https://github.com/dzintt/agent-harness-base.git"
python -c "from agent_harness import Agent, AgentConfig; print(AgentConfig(model='gpt-5.4'))"
```

`uv` install example:

```bash
uv add "git+https://github.com/dzintt/agent-harness-base.git"
uv run python -c "from agent_harness import Agent, AgentConfig; print(AgentConfig(model='gpt-5.4'))"
```

## Quickstart

```python
import asyncio

from agent_harness import Agent, AgentConfig, tool


@tool
async def ping(message: str) -> str:
    """Echo a message back."""
    return f"pong: {message}"


async def main() -> None:
    agent = Agent(
        config=AgentConfig(model="gpt-5.4"),
        tools=[ping],
        system_prompt="You are concise and helpful.",
    )

    try:
        result = await agent.run("Call ping with hello and tell me the result.")
        print(result.output_text)
        print(result.tool_results)
    finally:
        await agent.aclose()


asyncio.run(main())
```

`result` is an `AgentRunResult` with:

- `output_text`: final assistant text
- `output_data`: validated Pydantic object when you pass `response_model=...`
- `tool_results`: local tool execution results
- `response_id`: provider response id when available
- `raw_responses`: provider payloads for debugging

## How The Harness Works

The runtime model is simple:

1. You pass a string or a list of messages to `Agent.run(...)` or `Agent.stream(...)`.
2. The harness converts that input into Responses API items.
3. If you set `system_prompt`, the harness prepends it as a `developer` message.
4. The provider sends the request to the OpenAI Responses API.
5. If the model returns tool calls, the harness runs the local tools and appends their outputs to the transcript.
6. The harness repeats until the model returns a final assistant response or `max_turns` is hit.

Nothing else hides behind the abstraction. `Agent` owns the loop. `ChatSession` stores conversation state. `ToolRegistry` owns tool schemas and execution.

## Pick The Right API

Use this guide:

- One request, final answer: `await agent.run(...)`
- One request, incremental text: `async for event in agent.stream(...)`
- Synchronous script: `agent.run_sync(...)`
- Synchronous streaming script: `agent.stream_sync(...)`
- Multi-turn conversation with in-memory state: `chat = agent.chat()`
- Resume a saved conversation exactly: `agent.chat_from_snapshot(...)`
- Typed final output: pass `response_model=MyPydanticModel`
- Images or files: pass `ChatMessage(..., content=[...parts...])`

## Public API

### `Agent`

Main entrypoint.

```python
from agent_harness import Agent, AgentConfig

agent = Agent(
    config=AgentConfig(model="gpt-5.4"),
    tools=[...],          # optional
    system_prompt="...",  # optional
)
```

Important methods:

- `await agent.run(input_data, response_model=None, system_prompt=None)`
- `async for event in agent.stream(input_data, response_model=None, system_prompt=None)`
- `agent.run_sync(...)`
- `agent.stream_sync(...)`
- `agent.chat(messages=None, system_prompt=None)`
- `agent.chat_from_snapshot(snapshot)`
- `await agent.aclose()`
- `agent.close()`

### `ChatSession`

Stateful wrapper around `Agent`.

```python
chat = agent.chat(system_prompt="You are concise.")

await chat.run("My name is Anson.")
result = await chat.run("What's my name?")

print(result.output_text)
print(chat.history)
```

Useful methods:

- `await chat.run(...)`
- `async for event in chat.stream(...)`
- `chat.run_sync(...)`
- `chat.stream_sync(...)`
- `chat.snapshot()`
- `chat.export()`
- `chat.reset()`

### Message Types

Use a plain string for normal text-only requests:

```python
result = await agent.run("Hello")
```

Use explicit messages when you already have conversation history or need multimodal content:

```python
from agent_harness import ChatMessage

messages = [
    ChatMessage(role="user", content="My name is Anson."),
    ChatMessage(role="assistant", content="Noted."),
    ChatMessage(role="user", content="What's my name?"),
]

result = await agent.run(messages)
```

`ChatMessage.role` can be:

- `user`
- `assistant`
- `developer`
- `system`

### Content Parts

- `TextPart("...")`
- `ImagePart.from_url(...)`
- `ImagePart.from_file(...)`
- `FilePart.from_url(...)`
- `FilePart.from_file(...)`

## Plain Text And Message Input

The input shape stays simple unless you need more.

### Smallest call

```python
result = await agent.run("Say hello.")
print(result.output_text)
```

### Explicit message list

```python
from agent_harness import ChatMessage

result = await agent.run(
    [
        ChatMessage(role="system", content="You are concise."),
        ChatMessage(role="user", content="My name is Anson."),
        ChatMessage(role="assistant", content="Noted."),
        ChatMessage(role="user", content="What's my name?"),
    ]
)
```

Notes:

- `input_data` can be a string or a sequence of messages
- inside a sequence, bare strings are treated as user messages
- the harness accepts `ChatMessage` instances or plain dicts that validate into `ChatMessage`

## System Prompt

You can set a default prompt on the agent:

```python
agent = Agent(
    config=AgentConfig(model="gpt-5.4"),
    system_prompt="You are concise and helpful.",
)
```

Override it for one call:

```python
result = await agent.run(
    "Explain async IO.",
    system_prompt="You are a patient teacher.",
)
```

Set a default prompt for a chat session:

```python
chat = agent.chat(system_prompt="You are a terse coding assistant.")
```

Important behavior:

- the convenience `system_prompt` is sent as a `developer` message
- if you also pass explicit `system` or `developer` messages, the harness sends both
- `chat.history` does not include the convenience prompt
- chat snapshots store the session `system_prompt`, not a fake persisted message for it

## Tools

Define tools with `@tool`:

```python
from agent_harness import tool


@tool
async def lookup_user(user_id: int) -> str:
    """Return a serialized user record."""
    return '{"id": 1, "name": "Ada"}'
```

Register them:

```python
agent = Agent(
    config=AgentConfig(model="gpt-5.4"),
    tools=[lookup_user],
)
```

Tool rules:

- tools can be `async def` or normal `def`
- every parameter needs a type annotation
- `*args` and `**kwargs` are not allowed
- the tool description comes from the first docstring line unless you override it
- sync tools run in a worker thread
- tool outputs are serialized to strings before the harness sends them back to the model

You can customize the metadata:

```python
@tool(name="lookup_user", description="Fetch a user record.")
async def get_user(user_id: int) -> str:
    return '{"id": 1, "name": "Ada"}'
```

### Parallel Tool Calls

Tool execution is sequential by default.

Enable same-turn parallel execution only when your tools are independent:

```python
agent = Agent(
    config=AgentConfig(
        model="gpt-5.4",
        parallel_tool_calls=True,
    ),
    tools=[get_weather, get_news],
)
```

When enabled:

- the harness sends `parallel_tool_calls=True` to the Responses API
- if the model returns multiple tool calls in the same turn, the harness executes them concurrently
- returned tool results still keep the model's original call order

Good fits:

- independent API fetches
- read-only database queries
- weather, news, and market lookups in one turn

Bad fits:

- create invoice then email invoice
- checkout then send receipt
- any tool set with ordering or shared mutable state

## Structured Output

Pass a Pydantic model as `response_model`:

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

Structured outputs work with normal runs, streaming, and tool calls.

Notes:

- the final parsed object is available as `result.output_data`
- the raw assistant text still remains in `result.output_text`
- if the provider cannot satisfy the structured output request, `run()` raises and `stream()` emits an `error` event

## Streaming

Use `stream()` when you want incremental text or tool lifecycle events:

```python
async for event in agent.stream("Explain async IO in one sentence."):
    if event.type == "text_delta" and event.delta:
        print(event.delta, end="")
    elif event.type == "completed" and event.result is not None:
        print()
        print(event.result.output_text)
```

Event types:

- `text_delta`
- `tool_call_started`
- `tool_call_completed`
- `completed`
- `error`

Structured streaming:

```python
from pydantic import BaseModel


class Summary(BaseModel):
    title: str
    bullets: list[str]


async for event in agent.stream("Summarize this text.", response_model=Summary):
    if event.type == "completed" and event.result is not None:
        print(event.result.output_data)
```

Important behavior:

- text arrives through `text_delta`
- structured output appears only on the final `completed` event
- if the stream fails, the harness yields one `error` event and ends the stream

## Chat Sessions

Use a chat session when you want the harness to hold conversation state for you:

```python
chat = agent.chat(system_prompt="You are concise.")

await chat.run("My name is Anson.")
result = await chat.run("What's my name?")

print(result.output_text)
print(chat.history)
```

`chat.history` returns simple `ChatMessage` values for display or storage.

Important behavior:

- the session stores message history in memory
- each `chat.run(...)` sends prior conversation items plus the new turn
- tool output items are not exposed in `chat.history`
- `chat.reset()` clears the stored conversation

### Snapshot And Restore

Use snapshots when you need exact resumable chat state:

```python
chat = agent.chat(system_prompt="You are concise.")

await chat.run("My name is Anson.")
await chat.run("My favorite color is teal.")

snapshot = chat.snapshot()
payload = chat.export()
restored = agent.chat_from_snapshot(payload)
```

Notes:

- `chat.snapshot()` returns a typed `ChatSnapshot`
- `chat.export()` returns a JSON-friendly dict
- `agent.chat_from_snapshot(...)` restores the conversation items and chat `system_prompt`
- snapshots restore conversation state only
- snapshots do not restore model config, tools, or provider settings

## Images

Use an explicit content list only when you need multimodal input:

```python
from agent_harness import ChatMessage, ImagePart, TextPart

result = await agent.run(
    [
        ChatMessage(
            role="user",
            content=[
                TextPart("Describe this image."),
                ImagePart.from_url("https://example.com/cat.png"),
            ],
        )
    ]
)
```

Local image input:

```python
ImagePart.from_file("cat.png")
```

Supported local image formats:

- PNG
- JPEG
- WEBP
- GIF

`ImagePart.from_file(...)` reads the file and converts it to a Base64 data URL.

## Files

File input uses the same multimodal message shape:

```python
from agent_harness import ChatMessage, FilePart, TextPart

result = await agent.run(
    [
        ChatMessage(
            role="user",
            content=[
                TextPart("Summarize this PDF."),
                FilePart.from_file("report.pdf"),
            ],
        )
    ]
)
```

Remote file input:

```python
FilePart.from_url("https://example.com/report.pdf")
```

`FilePart.from_file(...)` reads the file, infers a supported MIME type, and converts it to a Base64 data URL.

Common supported file types include:

- PDF
- TXT
- CSV and TSV
- JSON
- HTML, XML, YAML, Markdown
- DOC and DOCX
- XLS and XLSX
- PPT and PPTX
- RTF
- ODT

This helper does not use the OpenAI Files API. It keeps local file input self-contained and works with OpenAI-compatible base URLs.

## Sync Usage

The library is async-first. Use the sync wrappers only in synchronous programs.

### Sync run

```python
agent = Agent(config=AgentConfig(model="gpt-5.4"))

try:
    result = agent.run_sync("Say hello.")
    print(result.output_text)
finally:
    agent.close()
```

### Sync stream

```python
for event in agent.stream_sync("Explain async IO in one sentence."):
    if event.type == "text_delta" and event.delta:
        print(event.delta, end="")
```

Important behavior:

- `run_sync()` and `stream_sync()` cannot run inside an existing event loop
- use `await agent.run(...)` and `async for event in agent.stream(...)` inside async apps
- call `agent.close()` when you finish using sync wrappers

## Configuration

`AgentConfig` fields:

```python
AgentConfig(
    model="gpt-5.4",
    api_key=None,
    base_url=None,
    max_turns=8,
    parallel_tool_calls=False,
    temperature=None,
    timeout=None,
)
```

Environment variables:

- `OPENAI_API_KEY`
- `OPENAI_MODEL`
- `OPENAI_BASE_URL`

Notes:

- `model` is required unless `OPENAI_MODEL` is set
- `api_key` defaults from `OPENAI_API_KEY`
- `base_url` lets you target an OpenAI-compatible endpoint
- `max_turns` limits tool-call loops before the harness raises `MaxTurnsExceededError`

## Errors And Failure Model

`run()` and `run_sync()` raise exceptions such as:

- `ProviderError`
- `ToolExecutionError`
- `MaxTurnsExceededError`

Streaming behaves differently:

- provider, tool, and loop errors are converted into an `AgentEvent(type="error", ...)`
- after the `error` event, the stream ends

This split is intentional:

- final-result APIs raise
- streaming APIs emit lifecycle events

## Example Index

See [examples/README.md](examples/README.md) for a guided list.

Main examples:

- [basic_agent.py](examples/basic_agent.py)
- [sync_agent.py](examples/sync_agent.py)
- [sync_streaming.py](examples/sync_streaming.py)
- [chat_session.py](examples/chat_session.py)
- [chat_persistence.py](examples/chat_persistence.py)
- [system_prompt.py](examples/system_prompt.py)
- [parallel_tools.py](examples/parallel_tools.py)
- [image_input.py](examples/image_input.py)
- [file_input.py](examples/file_input.py)
- [chat_with_images.py](examples/chat_with_images.py)
- [structured_output.py](examples/structured_output.py)
- [structured_with_tools.py](examples/structured_with_tools.py)
- [streaming.py](examples/streaming.py)

## Development

Run the test suite:

```bash
uv run pytest
```

Run the live end-to-end script when you want to verify the real provider path with an API key:

```bash
uv run python scripts/live_e2e_test.py
```

The unit tests run without a real API key. The live script does not.
