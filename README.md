# Simple Agent Base

Small async-first Python package for building OpenAI-powered agents on top of the Responses API.

This project gives you a thin layer over the OpenAI Python SDK for the parts that repeat in most agent projects:

- request/response loops
- local tool registration and execution
- structured outputs with Pydantic
- streaming text and tool events
- in-memory chat history
- snapshot and restore
- image and file inputs
- sync wrappers for non-async code

It does not try to be a full agent framework. It does not add planning, memory systems, retrieval, workflow orchestration, or hosted tools.

## What This Project Is For

Use this package when you want:

- a small OpenAI-specific harness instead of a large framework
- code you can read and own end-to-end
- local Python tools with automatic JSON schema generation
- one-shot runs, streaming, and chat sessions through one API
- structured outputs without writing the tool loop yourself

Use something else when you want:

- multi-agent orchestration
- long-running workflows or state machines
- retrieval, vector storage, or memory layers
- provider-agnostic abstractions

The main idea is simple: this package sits between the raw OpenAI SDK and a full framework.

## Install

Right now the package installs from GitHub or a local checkout.

Install from GitHub with `pip`:

```bash
python -m pip install "git+https://github.com/dzintt/simple-agent-base.git"
```

Install from GitHub with `uv`:

```bash
uv add "git+https://github.com/dzintt/simple-agent-base.git"
```

Install from a local checkout with `pip`:

```bash
python -m pip install .
```

Install from a local checkout with `uv`:

```bash
uv sync
```

Install development dependencies:

```bash
uv sync --dev
```

Set your API key:

```bash
$env:OPENAI_API_KEY="your_key_here"
```

You can also set the model through the environment:

```bash
$env:OPENAI_MODEL="gpt-5.4"
```

## Quickstart

```python
import asyncio

from simple_agent_base import Agent, AgentConfig, tool


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

`agent.run(...)` returns an `AgentRunResult`:

- `output_text`: final assistant text
- `output_data`: parsed Pydantic object when you pass `response_model=...`
- `tool_results`: local tool results from the run
- `response_id`: provider response id when available
- `raw_responses`: raw provider payloads for debugging

## The API Surface

Most projects only need these exports:

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
- `MCPServer`
- `MCPApprovalRequest`
- `MCPCallRecord`

## How It Works

The runtime model is small:

1. You call `Agent.run(...)` or `Agent.stream(...)` with a string or a list of messages.
2. The package normalizes that input into Responses API items.
3. If you passed `system_prompt`, it prepends that prompt as a `developer` message.
4. The provider sends the request to the OpenAI Responses API.
5. If the model returns tool calls, the package executes the local tools and appends their outputs to the transcript.
6. The loop repeats until the model returns a final answer or `max_turns` is reached.

`Agent` owns the loop. `ChatSession` owns in-memory conversation state. `ToolRegistry` owns tool schemas and execution.

## Common Tasks

### One Request

Use `run(...)` for a normal one-shot request:

```python
result = await agent.run("Say hello in one sentence.")
print(result.output_text)
```

### Streaming

Use `stream(...)` when you want incremental text or tool lifecycle events:

```python
async for event in agent.stream("Explain async IO in one sentence."):
    if event.type == "text_delta" and event.delta:
        print(event.delta, end="")
    elif event.type == "completed" and event.result is not None:
        print()
        print(event.result.output_text)
```

Streaming event types:

- `text_delta`
- `tool_call_started`
- `tool_call_completed`
- `completed`
- `error`

### Tools

Define tools with `@tool`:

```python
from simple_agent_base import tool


@tool
async def lookup_user(user_id: int) -> str:
    """Return a serialized user record."""
    return '{"id": 1, "name": "Ada"}'
```

Register them on the agent:

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
- tool outputs are serialized to strings before the package sends them back to the model

You can override the tool name and description:

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

- the package sends `parallel_tool_calls=True` to the Responses API
- if the model returns multiple tool calls in one turn, the package executes them concurrently
- tool results keep the model's original call order

Good fits:

- independent API calls
- read-only database queries
- weather, news, and market lookups in one turn

Bad fits:

- create invoice, then email invoice
- checkout, then send receipt
- anything that depends on ordering or shared mutable state

### MCP Servers

Give the model access to a client-side MCP server by passing `mcp_servers` to the agent:

```python
import sys
from pathlib import Path

from simple_agent_base import Agent, AgentConfig, MCPServer

fixture_server = Path("tests/fixtures/mcp_demo_server.py").resolve()

agent = Agent(
    config=AgentConfig(model="gpt-5.4"),
    mcp_servers=[
        MCPServer.stdio(
            name="demo",
            command=sys.executable,
            args=[str(fixture_server), "stdio"],
            require_approval=False,
        )
    ],
)

result = await agent.run("Use the demo MCP echo tool with the message 'hello'.")
print(result.output_text)
for call in result.mcp_calls:
    print(call.server_name, call.name, call.arguments)
```

This package supports client-side MCP only. The library owns the MCP connection, discovers tools locally, exposes them to the model as function tools, executes the chosen MCP call locally, and records the activity in `result.mcp_calls`.

`MCPServer` fields:

- `name`: required server identifier used to namespace discovered tools
- `allowed_tools`: optional list of MCP tool names to expose
- `require_approval`: if `True`, the local `approval_handler` is invoked before each MCP call
- `MCPServer.stdio(...)`: configure a subprocess-backed MCP server
- `MCPServer.http(...)`: configure a streamable HTTP MCP server

For trusted public or read-only servers, set `require_approval=False` to skip approval plumbing:

```python
import sys
from pathlib import Path

fixture_server = Path("tests/fixtures/mcp_demo_server.py").resolve()

agent = Agent(
    config=AgentConfig(model="gpt-5.4"),
    mcp_servers=[
        MCPServer.stdio(
            name="demo",
            command=sys.executable,
            args=[str(fixture_server), "stdio"],
            require_approval=False,
        )
    ],
)
```

To gate tools, set `require_approval=True` and pass an `approval_handler`:

```python
from simple_agent_base import MCPApprovalRequest

def approve(request: MCPApprovalRequest) -> bool:
    return input(f"Run {request.server_name}.{request.name}? [y/N] ").lower() == "y"

agent = Agent(
    config=AgentConfig(model="gpt-5.4"),
    mcp_servers=[
        MCPServer.http(
            name="gh",
            url="http://127.0.0.1:8000/mcp",
            require_approval=True,
        )
    ],
    approval_handler=approve,
)
```

The handler can be sync or async. If approvals are requested and no handler is set, the package raises `MCPApprovalRequiredError`.

Streaming surfaces three new event types alongside the existing ones:

- `mcp_call_started`
- `mcp_call_completed`
- `mcp_approval_requested`

### Structured Output

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

Structured output works with normal runs, streaming, and tool calls.

### Chat Sessions

Use a chat session when you want the package to keep history in memory:

```python
chat = agent.chat(system_prompt="You are concise.")

await chat.run("My name is Anson.")
result = await chat.run("What's my name?")

print(result.output_text)
print(chat.history)
```

`chat.history` returns `ChatMessage` values for display or storage.

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

Snapshots restore conversation items plus the chat session `system_prompt`. They do not restore model config, tools, or provider settings.

### Images And Files

Use explicit content parts for multimodal input:

```python
from simple_agent_base import ChatMessage, ImagePart, TextPart

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

Local images:

```python
ImagePart.from_file("cat.png")
```

Supported local image formats:

- PNG
- JPEG
- WEBP
- GIF

File input uses the same shape:

```python
from simple_agent_base import ChatMessage, FilePart, TextPart

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

Common supported file types:

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

`ImagePart.from_file(...)` and `FilePart.from_file(...)` read local files and convert them to Base64 data URLs. This helper does not use the OpenAI Files API.

### Sync Usage

The package is async-first. Use the sync wrappers only in synchronous programs.

```python
agent = Agent(config=AgentConfig(model="gpt-5.4"))

try:
    result = agent.run_sync("Say hello.")
    print(result.output_text)
finally:
    agent.close()
```

For sync streaming:

```python
for event in agent.stream_sync("Explain async IO in one sentence."):
    if event.type == "text_delta" and event.delta:
        print(event.delta, end="")
```

`run_sync()` and `stream_sync()` cannot run inside an existing event loop.

## Message Model

Use a plain string for text-only requests:

```python
result = await agent.run("Hello")
```

Use explicit messages when you already have history or need multimodal content:

```python
from simple_agent_base import ChatMessage

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

Content parts:

- `TextPart("...")`
- `ImagePart.from_url(...)`
- `ImagePart.from_file(...)`
- `FilePart.from_url(...)`
- `FilePart.from_file(...)`

## System Prompts

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
- if you also pass explicit `system` or `developer` messages, the package sends both
- `chat.history` does not include the convenience prompt
- chat snapshots store the session `system_prompt` separately, not as a persisted message

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
- `max_turns` limits tool-call loops before the package raises `MaxTurnsExceededError`

## Error Handling

`run()` and `run_sync()` raise exceptions such as:

- `ProviderError`
- `ToolExecutionError`
- `MaxTurnsExceededError`

`stream()` and `stream_sync()` also raise these exceptions. They do not convert failures into an `error` event.

## Examples

See [examples/README.md](examples/README.md) for a guided list.

Start with these:

1. [basic_agent.py](examples/basic_agent.py)
2. [structured_output.py](examples/structured_output.py)
3. [streaming.py](examples/streaming.py)
4. [chat_session.py](examples/chat_session.py)
5. [parallel_tools.py](examples/parallel_tools.py)

Other examples:

- [sync_agent.py](examples/sync_agent.py)
- [sync_streaming.py](examples/sync_streaming.py)
- [chat_persistence.py](examples/chat_persistence.py)
- [system_prompt.py](examples/system_prompt.py)
- [image_input.py](examples/image_input.py)
- [file_input.py](examples/file_input.py)
- [chat_with_images.py](examples/chat_with_images.py)
- [structured_with_tools.py](examples/structured_with_tools.py)

## Project Layout

```text
simple-agent-base/
|-- src/
|   `-- simple_agent_base/
|       |-- __init__.py
|       |-- agent.py
|       |-- chat.py
|       |-- config.py
|       |-- errors.py
|       |-- sync_utils.py
|       |-- types.py
|       |-- providers/
|       |   |-- base.py
|       |   `-- openai.py
|       `-- tools/
|           |-- base.py
|           |-- decorators.py
|           `-- registry.py
|-- examples/
|-- scripts/
|-- tests/
|-- pyproject.toml
`-- README.md
```

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
