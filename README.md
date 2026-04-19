## Agent Harness Base

Small async-first base for building agents with the OpenAI Python SDK.

The package exposes a minimal public API:

- `Agent`
- `AgentConfig`
- `ToolRegistry`
- `tool`

It is designed for:

- async execution with `AsyncOpenAI`
- simple decorator-based tool definitions
- sequential local tool execution
- streaming text deltas through an async iterator
- trivial structured outputs with Pydantic models

## What This Gives You

This project is a small reusable base for future agent projects.

You can:

- create an `Agent`
- give the agent a first-class `system_prompt`
- register async tools with `@tool`
- call `await agent.run(...)` for a normal request
- call `await agent.run([...messages...])` when you already have explicit conversation history
- create `chat = agent.chat()` for persistent follow-up conversations
- call `await agent.run(..., response_model=MySchema)` for structured output
- call `agent.stream(...)` for incremental text and a final structured result

The goal is to keep the public API small while still covering the common paths you will use repeatedly in future projects.

## Install

```bash
uv sync
```

Set your API key:

```bash
$env:OPENAI_API_KEY="your_key_here"
```

Run any example with:

```bash
uv run python examples/basic_agent.py
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
        config=AgentConfig(model="gpt-5"),
        tools=[ping],
        system_prompt="You are concise and helpful.",
    )

    result = await agent.run("Call ping with hello and tell me the result.")
    print(result.output_text)


asyncio.run(main())
```

The result object includes:

- `result.output_text`: the final assistant text
- `result.output_data`: validated structured output when you pass `response_model=...`
- `result.tool_results`: tool execution history
- `result.raw_responses`: raw-ish provider payloads for debugging

## System Prompt

You can set a first-class `system_prompt` without manually constructing a message list:

```python
from agent_harness import Agent, AgentConfig

agent = Agent(
    config=AgentConfig(model="gpt-5"),
    system_prompt="You are concise and helpful.",
)

result = await agent.run("Say hello in five words or fewer.")
print(result.output_text)
```

You can override it for one request:

```python
result = await agent.run(
    "Explain async IO.",
    system_prompt="You are a teacher who explains things simply.",
)
```

And you can give a chat session its own default prompt:

```python
chat = agent.chat(system_prompt="You are a terse coding assistant.")

await chat.run("Help me prepare for a backend interview.")
result = await chat.run("Ask me the next question.")
```

Notes:

- `system_prompt` is the convenience path
- advanced users can still pass explicit `ChatMessage(role="system", ...)` or `ChatMessage(role="developer", ...)`
- if you pass both, the harness sends both and does not deduplicate them

## Images

The API is intentionally two-tier:

- plain text stays trivial
- images use a multimodal message shape only when you need them

### Plain text

```python
result = await agent.run("Hello")
```

### Remote image URL

```python
from agent_harness import Agent, AgentConfig, ChatMessage, ImagePart, TextPart

agent = Agent(config=AgentConfig(model="gpt-5"))

result = await agent.run(
    [
        ChatMessage(
            role="user",
            content=[
                TextPart("What is in this image?"),
                ImagePart.from_url("https://example.com/cat.png"),
            ],
        )
    ]
)
```

### Local image file

```python
result = await agent.run(
    [
        ChatMessage(
            role="user",
            content=[
                TextPart("Describe this image."),
                ImagePart.from_file("cat.png"),
            ],
        )
    ]
)
```

### Chat with images

```python
chat = agent.chat()

await chat.run(
    [
        ChatMessage(
            role="user",
            content=[
                TextPart("Remember this photo."),
                ImagePart.from_file("photo.jpg"),
            ],
        )
    ]
)

result = await chat.run("What was in the photo?")
```

## Conversation History

There are now two trivial ways to work with more than one message.

### 1. Pass a list of messages directly

Use this when you already store conversation history yourself:

```python
from agent_harness import Agent, AgentConfig, ChatMessage

agent = Agent(config=AgentConfig(model="gpt-5"))

result = await agent.run(
    [
        ChatMessage(role="system", content="You are concise."),
        ChatMessage(role="user", content="My name is Anson."),
        ChatMessage(role="assistant", content="Noted."),
        ChatMessage(role="user", content="What's my name?"),
    ]
)
```

### 2. Use a chat session

Use this when you want the harness to keep the conversation history for you:

```python
chat = agent.chat(system_prompt="You are concise.")

await chat.run("My name is Anson.")
result = await chat.run("What's my name?")

print(result.output_text)
print(chat.history)
```

`chat.history` gives you a simple list of `ChatMessage` values for display or storage.
It does not include the convenience `system_prompt` unless you explicitly passed a system or developer message yourself.

## Examples

Available examples:

- [basic_agent.py](/C:/Users/Anson/Desktop/agent-harness-base/examples/basic_agent.py): smallest possible agent with one tool
- [chat_session.py](/C:/Users/Anson/Desktop/agent-harness-base/examples/chat_session.py): persistent conversation history for follow-up chat apps
- [system_prompt.py](/C:/Users/Anson/Desktop/agent-harness-base/examples/system_prompt.py): first-class system prompt defaults and one-off overrides
- [image_input.py](/C:/Users/Anson/Desktop/agent-harness-base/examples/image_input.py): one-turn multimodal input with text plus an image
- [chat_with_images.py](/C:/Users/Anson/Desktop/agent-harness-base/examples/chat_with_images.py): follow-up chat after sending an image
- [structured_output.py](/C:/Users/Anson/Desktop/agent-harness-base/examples/structured_output.py): extract typed data with a Pydantic schema
- [structured_with_tools.py](/C:/Users/Anson/Desktop/agent-harness-base/examples/structured_with_tools.py): combine tools and structured outputs
- [streaming.py](/C:/Users/Anson/Desktop/agent-harness-base/examples/streaming.py): stream text deltas and read the final result
- [examples/README.md](/C:/Users/Anson/Desktop/agent-harness-base/examples/README.md): example index and when to use each one

## Structured Outputs

Pass a Pydantic model directly to `run()` and read the validated result from `output_data`:

```python
from pydantic import BaseModel
from agent_harness import Agent, AgentConfig


class Person(BaseModel):
    name: str
    age: int


agent = Agent(config=AgentConfig(model="gpt-5"))

result = await agent.run(
    "Extract the person from: Sarah is 29 years old.",
    response_model=Person,
)

print(result.output_data)
```

This is the trivial path the harness is optimized for:

```python
result = await agent.run("...", response_model=MySchema)
data = result.output_data
```

Structured outputs also work alongside tools:

```python
from pydantic import BaseModel
from agent_harness import Agent, AgentConfig, tool


class WeatherAnswer(BaseModel):
    city: str
    temperature_f: int
    summary: str


@tool
async def get_weather(city: str) -> str:
    """Return fake weather data."""
    return '{"city":"San Francisco","temperature_f":65,"summary":"Foggy"}'


agent = Agent(config=AgentConfig(model="gpt-5"), tools=[get_weather])

result = await agent.run(
    "Use the weather tool and return a structured answer.",
    response_model=WeatherAnswer,
)

print(result.output_data)
```

When you pass `response_model=...`:

- the harness uses the OpenAI SDK structured output path
- the final parsed value is returned as `result.output_data`
- the normal text result still remains available as `result.output_text`

## Streaming

Use `run()` when you want a single final result object.

Use `stream()` when you want incremental text and tool lifecycle events:

```python
async for event in agent.stream("Say hi"):
    if event.type == "text_delta":
        print(event.delta, end="")
```

Structured streaming exposes the parsed schema on the final completed event:

```python
async for event in agent.stream("Summarize this text.", response_model=Summary):
    if event.type == "text_delta":
        print(event.delta, end="")
    elif event.type == "completed":
        print(event.result.output_data)
```

Streaming event types:

- `text_delta`: incremental assistant text
- `tool_call_started`: a tool call is about to run
- `tool_call_completed`: a tool finished running
- `completed`: final `AgentRunResult`
- `error`: the stream failed

Structured output in streaming mode is exposed only on the final `completed` event:

- you still receive text deltas as normal
- `event.result.output_data` is populated only after the response is complete and validated

Streaming also works with chat sessions:

```python
chat = agent.chat()

async for event in chat.stream("Hello there"):
    ...

async for event in chat.stream("Follow up on that"):
    ...
```

## Tools

Tools are async functions decorated with `@tool`:

```python
from agent_harness import tool


@tool
async def lookup_user(user_id: int) -> str:
    """Return a serialized user record."""
    return '{"id": 1, "name": "Ada"}'
```

Tool rules:

- tools must use `async def`
- every parameter must have a type annotation
- the tool description comes from the first line of the docstring
- tools are executed sequentially

You can pass tools as a list:

```python
agent = Agent(
    config=AgentConfig(model="gpt-5"),
    tools=[lookup_user],
)
```

## Environment

The harness reads `OPENAI_API_KEY` automatically if `api_key` is not passed directly.

An `OPENAI_MODEL` value can also be used as a convenience when creating `AgentConfig()` from environment-backed values.

## API Summary

- `Agent(config, tools=None, provider=None, system_prompt=None)`: main entrypoint
- `await agent.run(input_data, response_model=None, system_prompt=None)`: final result API for one message or many
- `agent.stream(input_data, response_model=None, system_prompt=None)`: streaming API for one message or many
- `agent.chat(messages=None, system_prompt=None)`: create a persistent chat session with in-memory history
- `AgentConfig(...)`: runtime configuration
- `ChatMessage(role, content)`: simple message type for explicit conversation history
- `TextPart(...)`: text content inside a multimodal message
- `ImagePart.from_url(...)` / `ImagePart.from_file(...)`: image content inside a multimodal message
- `@tool`: tool decorator
- `ToolRegistry`: explicit tool registration if you need it

## Development

Run tests:

```bash
uv run pytest
```

This repo keeps tests fully local and does not require a real OpenAI API key for the test suite.
