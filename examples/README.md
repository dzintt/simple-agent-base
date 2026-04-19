# Examples

These examples are meant to be copy-paste friendly reference points for the main ways this harness is intended to be used.

Run any example from the repo root:

```bash
uv run python examples/basic_agent.py
```

## Example List

- `basic_agent.py`
  - Smallest useful agent setup
  - Good starting point when you just want tools plus plain text output

- `sync_agent.py`
  - Shows the smallest sync `run_sync(...)` flow and explicit `close()`
  - Use this when you want to stay entirely in normal synchronous Python

- `sync_streaming.py`
  - Shows how to iterate over streaming events from sync code
  - Use this when you want streaming output without using `asyncio`

- `chat_session.py`
  - Shows how to keep follow-up message history automatically
  - Use this for chat apps and multi-turn conversations

- `chat_persistence.py`
  - Shows exact chat snapshot export and restore
  - Use this when you want to resume a chat later without rebuilding history manually

- `system_prompt.py`
  - Shows the first-class `system_prompt` shortcut on the agent, on a chat session, and as a one-off override
  - Use this when you want behavior steering without manually prepending system or developer messages

- `parallel_tools.py`
  - Shows the single config flag for enabling parallel same-turn tool execution
  - Use this when your tools are independent and safe to run concurrently

- `image_input.py`
  - Shows the simplest multimodal prompt with text plus one image
  - Use this when you want image understanding without a persistent chat session

- `file_input.py`
  - Shows the simplest multimodal prompt with text plus one file such as a PDF
  - Use this when you want document or file understanding without a persistent chat session

- `chat_with_images.py`
  - Shows a chat session where the first turn includes an image and the second turn follows up on it
  - Use this for multimodal chat apps

- `structured_output.py`
  - Shows the simplest possible structured output flow
  - Use this when you want `result.output_data` from a Pydantic schema

- `structured_with_tools.py`
  - Shows tools plus structured output together
  - Use this when the model needs to call local code and still return a typed final result

- `streaming.py`
  - Shows how to stream text deltas and inspect the final completed event
  - Use this when you want terminal-style progressive output

## Shared Requirements

All examples assume:

- you already ran `uv sync`
- `OPENAI_API_KEY` is set
- you are running from the repo root

Optional:

- set `OPENAI_MODEL` if you want to construct `AgentConfig()` from environment defaults
