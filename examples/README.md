# Examples

These examples show the intended ways to use the harness.

Run any example from the repo root:

```bash
uv run python examples/basic_agent.py
```

All examples assume:

- you already ran `uv sync`
- `OPENAI_API_KEY` is set
- you are running from the repo root

`OPENAI_MODEL` is optional if you want `AgentConfig()` to pick the model from the environment.

## Start Here

If you are new to the repo, read these in this order:

1. `basic_agent.py`
2. `structured_output.py`
3. `streaming.py`
4. `chat_session.py`
5. `parallel_tools.py`

## Example Guide

### `basic_agent.py`

Use this first.

Shows:

- the smallest useful `Agent`
- a tool defined with `@tool`
- a normal `await agent.run(...)` call
- how to read `output_text` and `tool_results`

### `sync_agent.py`

Use this when your program is fully synchronous.

Shows:

- `agent.run_sync(...)`
- `agent.close()`

### `sync_streaming.py`

Use this when you want streaming output from sync code.

Shows:

- `agent.stream_sync(...)`
- handling `text_delta` and `completed` events

### `chat_session.py`

Use this when you want the harness to keep conversation history for you.

Shows:

- `chat = agent.chat()`
- follow-up calls with `chat.run(...)`
- `chat.history`

### `chat_persistence.py`

Use this when you want to save and resume chat state.

Shows:

- `chat.snapshot()`
- `chat.export()`
- `agent.chat_from_snapshot(...)`

### `system_prompt.py`

Use this when you want behavior steering without manually building message lists.

Shows:

- agent-level `system_prompt`
- per-run `system_prompt`
- chat-session `system_prompt`

### `parallel_tools.py`

Use this only when your tools are independent and safe to run together.

Shows:

- `AgentConfig(parallel_tool_calls=True)`
- concurrent same-turn tool execution

### `image_input.py`

Use this for a one-turn image prompt.

Shows:

- `ChatMessage`
- `TextPart`
- `ImagePart.from_file(...)`

### `file_input.py`

Use this for a one-turn document prompt.

Shows:

- `ChatMessage`
- `TextPart`
- `FilePart.from_file(...)`

### `chat_with_images.py`

Use this when the first turn contains an image and later turns refer back to it.

Shows:

- multimodal chat history
- follow-up questions after an image turn

### `structured_output.py`

Use this when you want a typed final result.

Shows:

- `response_model=...`
- reading `result.output_data`

### `structured_with_tools.py`

Use this when the model must call a tool and still return typed output.

Shows:

- tools plus structured output in one run

### `mcp_server.py`

Use this when you want to give the model access to a remote MCP server.

Shows:

- `MCPServer(server_label=..., server_url=...)`
- `Agent(mcp_servers=[...])`
- reading `result.mcp_calls`
- an `approval_handler` with `require_approval="always"`

### `streaming.py`

Use this when you want progressive output or tool lifecycle events.

Shows:

- `agent.stream(...)`
- `text_delta`
- `tool_call_started`
- `tool_call_completed`
- `completed`
- `error`
