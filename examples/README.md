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

- `chat_session.py`
  - Shows how to keep follow-up message history automatically
  - Use this for chat apps and multi-turn conversations

- `system_prompt.py`
  - Shows the first-class `system_prompt` shortcut on the agent, on a chat session, and as a one-off override
  - Use this when you want behavior steering without manually prepending system or developer messages

- `image_input.py`
  - Shows the simplest multimodal prompt with text plus one image
  - Use this when you want image understanding without a persistent chat session

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
