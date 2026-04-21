# Tools Guide

This guide explains how local tools work in `simple-agent-base`:

- defining tools
- schema generation
- registration
- execution
- parallel tool calls
- errors and edge cases

For the surrounding request lifecycle, see [usage.md](./usage.md). For the runtime internals, see [architecture.md](./architecture.md).

## Mental Model

Tools in this package are plain Python callables that the model can request by name.

The package handles four jobs for you:

1. build a JSON schema from the function signature
2. expose that schema to the Responses API as a function tool
3. validate model-supplied arguments with Pydantic
4. execute the tool locally and send the output back into the transcript

There is no hosted execution layer. Tool code runs in your local Python process.

## Defining Tools

Use the `@tool` decorator:

```python
from simple_agent_base import tool


@tool
async def lookup_user(user_id: int) -> str:
    """Return a serialized user record."""
    return '{"id": 1, "name": "Ada"}'
```

You can decorate:

- `async def` functions
- normal `def` functions

You can also skip the decorator and let the registry build the definition directly from the function, but the decorator is the intended interface.

## Tool Signature Rules

Tool signatures are intentionally strict.

Requirements:

- every parameter must have a type annotation
- `*args` is not allowed
- `**kwargs` is not allowed

Defaults work as expected:

- a parameter with no default becomes required in the generated schema
- a parameter with a default becomes optional in the schema

Example:

```python
@tool
async def greet(name: str, count: int = 1) -> str:
    """Return a greeting."""
    return f"{name}:{count}"
```

This produces a schema where:

- `name` is required
- `count` is optional with a default

If a parameter is missing an annotation, tool definition fails with `ToolDefinitionError`.

## Name And Description Resolution

By default:

- the tool name is the Python function name
- the tool description is the first line of the docstring

Example:

```python
@tool
async def get_weather(city: str) -> str:
    """Return the weather for a city."""
    ...
```

This produces:

- name: `get_weather`
- description: `Return the weather for a city.`

You can override either:

```python
@tool(name="lookup_user", description="Fetch a user record.")
async def get_user(user_id: int) -> str:
    return '{"id": 1, "name": "Ada"}'
```

## Registration

The simplest path is to pass tools directly to `Agent`:

```python
agent = Agent(
    config=AgentConfig(model="gpt-5.4"),
    tools=[lookup_user, get_weather],
)
```

Under the hood, `Agent` creates a `ToolRegistry` unless you pass one directly.

You can also use `ToolRegistry` yourself:

```python
from simple_agent_base import ToolRegistry

registry = ToolRegistry([lookup_user])
registry.register(get_weather)
```

Important behavior:

- tool names must be unique
- duplicate registration raises `ToolRegistrationError`

## JSON Schema Generation

The registry exposes tools to the provider as OpenAI function tools.

Each registered tool becomes a dict with:

- `type: "function"`
- `name`
- `description`
- `parameters`
- `strict`

The parameters schema comes from a generated Pydantic model built from the tool signature.

Important details:

- unknown fields are forbidden during validation
- required vs optional fields come from the signature
- schema generation happens once per tool definition

You can inspect the generated tool payload:

```python
registry = ToolRegistry([lookup_user])
openai_tools = registry.to_openai_tools()
print(openai_tools[0])
```

## Execution Model

When the model returns a function call:

1. the provider converts it into a `ToolCallRequest`
2. the registry looks up the tool by name
3. the arguments are validated against the generated Pydantic model
4. the callable is executed
5. the return value is serialized to a string
6. the package appends a `function_call_output` item to the transcript

The model then sees that tool output on the next turn.

## Sync And Async Tool Execution

Async tools:

- executed directly with `await`

Sync tools:

- executed with `asyncio.to_thread(...)`

This means sync tools do not block the event loop directly, but they still run inside the same Python process.

### Example Async Tool

```python
@tool
async def ping(message: str) -> str:
    """Echo a message back."""
    return f"pong: {message}"
```

### Example Sync Tool

```python
@tool
def sync_ping(message: str) -> str:
    """Echo a message back synchronously."""
    return f"sync-pong: {message}"
```

## Tool Output Serialization

Tool outputs must be sent back to the model as strings.

Serialization rules:

- if the tool returns `str`, the package uses it as-is
- if the tool returns a Pydantic model, the package dumps it as JSON
- otherwise, the package JSON-serializes the returned value with `default=str`

That means these all work:

- strings
- dicts
- lists
- numbers
- booleans
- Pydantic models

Example:

```python
@tool
async def get_flags() -> dict[str, bool]:
    """Return feature flags."""
    return {"beta": True, "search": False}
```

The output sent back to the model will be JSON text.

## Tool Results In Final Output

Every executed tool produces a `ToolExecutionResult`.

Fields:

- `call_id`
- `name`
- `arguments`
- `output`
- `raw_output`

These results appear in `AgentRunResult.tool_results` in the order the model requested them.

Important detail:

- even when a same-turn tool batch runs in parallel, the final tool results keep the model's original call order

## Parallel Tool Calls

Tool execution is sequential by default.

Enable parallel same-turn execution with:

```python
config = AgentConfig(
    model="gpt-5.4",
    parallel_tool_calls=True,
)
```

What this changes:

- the provider sends `parallel_tool_calls=True` to the Responses API
- if the model returns multiple tool calls in one turn, the package executes them concurrently with `asyncio.gather(...)`

What it does not change:

- the order of `tool_results`
- the requirement that the tools be safe to run independently

Good fits:

- independent network lookups
- read-only queries
- unrelated fetches that can run in parallel

Bad fits:

- tools with ordering dependencies
- tools that mutate the same shared state
- flows where one tool output must exist before the next tool starts

## Streaming Tool Lifecycle

In streamed requests, tool calls produce lifecycle events.

If the provider returns tool calls in a turn:

1. the stream yields `tool_call_started` once for each tool call
2. the package executes the tools
3. the stream yields `tool_call_completed` as each ordered result is appended
4. the next provider turn begins

This lets UIs display:

- which tool the model chose
- when the tool started
- when it finished
- the final tool output

## Errors

Tool-related errors surface in two ways depending on the API.

### `run()` And `run_sync()`

These raise `ToolExecutionError` if:

- the tool name is not registered
- argument validation fails
- the tool callable raises an exception

### `stream()` And `stream_sync()`

These convert failures into:

- `AgentEvent(type="error", error="...")`

Then the stream ends.

### Important Error Detail

The package does not swallow tool failures and continue. A failing tool ends the current run or stream.

## Argument Validation

Arguments from the model are validated through the generated Pydantic arguments model before the tool runs.

That means:

- missing required fields fail
- wrong types fail
- unknown extra fields fail

This is a useful guardrail when the model returns malformed tool arguments.

## Manual Registry Usage

Most users will not need manual registry usage, but it is available:

```python
registry = ToolRegistry()
registry.register(lookup_user)

definition = registry.get("lookup_user")
print(definition.name)
print(definition.parameters)
```

Methods:

- `register(tool_fn)`
- `get(name)`
- `list_definitions()`
- `to_openai_tools()`
- `execute(call)`

## Behavior The Package Does Not Provide

The tool system is intentionally narrow. It does not provide:

- hosted tool execution
- approvals
- retries
- side-effect safety controls
- dependency-aware tool scheduling
- automatic state management between tools

If you need those features, build them on top of this package or use a larger framework.

## Best Practices

- keep tool names explicit and stable
- write docstrings that tell the model when the tool is useful
- use typed parameters, not loose dictionaries, when possible
- return predictable JSON-shaped data when the model needs to reason over the result
- only enable parallel tool calls for independent operations
- keep side effects isolated and deliberate

## Minimal Example

```python
import asyncio

from simple_agent_base import Agent, AgentConfig, tool


@tool
async def get_weather(city: str) -> str:
    """Return the weather for a city."""
    return f"{city}: 72F and sunny"


async def main() -> None:
    agent = Agent(
        config=AgentConfig(model="gpt-5.4"),
        tools=[get_weather],
    )

    try:
        result = await agent.run("Use the weather tool for San Francisco.")
        print(result.output_text)
        print(result.tool_results)
    finally:
        await agent.aclose()


asyncio.run(main())
```

## MCP Servers

Note: the MCP support in this package is client-side only. The hosted OpenAI `{"type":"mcp"}` model is no longer supported. Prefer `MCPServer.stdio(...)` or `MCPServer.http(...)` as shown in [examples/mcp_server.py](../examples/mcp_server.py).

In addition to local Python tools, you can give the model access to MCP (Model Context Protocol) servers that this library connects to directly. The agent discovers the remote tools locally, exposes them to the model as normal function tools, executes the chosen MCP call locally, and records each invocation in `AgentRunResult.mcp_calls`.

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
```

`mcp_servers` is independent of `tools=`. You can mix local Python tools and bridged MCP tools on the same agent.

### Configuration

`MCPServer` has two constructors:

| Field | Purpose |
| --- | --- |
| `MCPServer.stdio(name=..., command=..., args=..., env=..., cwd=...)` | Start an MCP server as a local subprocess and connect over stdio. |
| `MCPServer.http(name=..., url=..., headers=...)` | Connect to a streamable HTTP MCP server. |
| `name` | Required server identifier used to namespace discovered tools. |
| `allowed_tools` | Optional `list[str]` of MCP tool names to expose to the model. |
| `require_approval` | If `True`, the local `approval_handler` runs before each MCP call. |

### Approvals

For trusted or read-only servers, set `require_approval=False` and nothing else is needed:

```python
agent = Agent(
    config=AgentConfig(model="gpt-5.4"),
    mcp_servers=[
        MCPServer.http(
            name="deepwiki",
            url="https://mcp.deepwiki.com/mcp",
            require_approval=False,
        )
    ],
)
```

If you want human approval, pass an `approval_handler`. The handler can be sync or async:

```python
from simple_agent_base import MCPApprovalRequest

def approve(request: MCPApprovalRequest) -> bool:
    # sync or async; return True to allow, False to deny
    return request.name in {"read_file", "list_files"}

agent = Agent(
    config=AgentConfig(model="gpt-5.4"),
    mcp_servers=[
        MCPServer.http(
            name="gh",
            url="https://gitmcp.io/owner/repo",
            require_approval=True,
        )
    ],
    approval_handler=approve,
)
```

If approvals are requested and no handler is set, the run raises `MCPApprovalRequiredError`.

### Inspecting MCP activity

After a run, `AgentRunResult.mcp_calls` contains an `MCPCallRecord` per invocation with `server_name`, `name`, `arguments`, `output`, and `error`.

When streaming, three additional event types are emitted:

- `mcp_call_started`
- `mcp_call_completed`
- `mcp_approval_requested`

