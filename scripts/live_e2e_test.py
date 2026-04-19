from __future__ import annotations

import asyncio
import tempfile
import zlib
import struct
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from agent_harness import Agent, AgentConfig, ChatMessage, ImagePart, TextPart, tool

class Person(BaseModel):
    name: str
    age: int


class ColorAnswer(BaseModel):
    dominant_color: str


class CheckResult(BaseModel):
    name: str
    ok: bool
    error: str | None = None


@tool
async def ping_tool(message: str) -> str:
    """Echo a message back with a fixed prefix."""
    return f"pong: {message}"


def section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def ensure(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def make_config() -> AgentConfig:
    # Uses OPENAI_* env vars provided at runtime and adds deterministic defaults for the live check.
    return AgentConfig(temperature=0, timeout=120, max_turns=6)


def make_red_png_file() -> Path:
    handle = tempfile.NamedTemporaryFile(prefix="agent-harness-live-", suffix=".png", delete=False)
    path = Path(handle.name)
    handle.write(make_solid_png(64, 64, red=255, green=0, blue=0))
    handle.close()
    return path


def make_solid_png(width: int, height: int, *, red: int, green: int, blue: int) -> bytes:
    png_header = b"\x89PNG\r\n\x1a\n"

    def chunk(chunk_type: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + chunk_type
            + data
            + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    row = b"\x00" + bytes([red, green, blue]) * width
    raw = row * height
    idat = zlib.compress(raw)
    return png_header + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


def print_config_summary() -> None:
    config = make_config()
    print("Runtime configuration:")
    print(f"- model: {config.model}")
    print(f"- base_url: {config.base_url}")
    print(f"- temperature: {config.temperature}")
    print(f"- timeout: {config.timeout}")
    print(f"- max_turns: {config.max_turns}")
    print("- api_key: provided via environment")


def describe_message_content(content: Any) -> list[str]:
    if isinstance(content, str):
        return [f"text: {content}"]

    lines: list[str] = []
    if isinstance(content, list):
        for index, part in enumerate(content, start=1):
            if isinstance(part, TextPart):
                lines.append(f"part {index}: text -> {part.text}")
            elif isinstance(part, ImagePart):
                if part.image_url.startswith("data:"):
                    preview = f"data-url ({len(part.image_url)} chars)"
                else:
                    preview = part.image_url
                lines.append(f"part {index}: image -> {preview} [detail={part.detail}]")
            elif isinstance(part, dict):
                part_type = part.get("type")
                if part_type == "text":
                    lines.append(f"part {index}: text -> {part.get('text')}")
                elif part_type == "image":
                    image_url = part.get("image_url", "")
                    if isinstance(image_url, str) and image_url.startswith("data:"):
                        preview = f"data-url ({len(image_url)} chars)"
                    else:
                        preview = str(image_url)
                    lines.append(f"part {index}: image -> {preview} [detail={part.get('detail', 'auto')}]")
                else:
                    lines.append(f"part {index}: raw -> {part}")
            else:
                lines.append(f"part {index}: raw -> {part}")
    return lines


def print_input(label: str, input_data: str | list[ChatMessage]) -> None:
    print(label)
    if isinstance(input_data, str):
        print(f"- user: {input_data}")
        return

    for index, message in enumerate(input_data, start=1):
        print(f"- message {index}: role={message.role}")
        for line in describe_message_content(message.content):
            print(f"  {line}")


def print_result_details(result: Any) -> None:
    print("Result summary:")
    print(f"- response_id: {result.response_id}")
    print(f"- output_text: {result.output_text}")
    print(f"- structured_output: {result.output_data}")
    print(f"- tool_result_count: {len(result.tool_results)}")
    if result.tool_results:
        for tool_result in result.tool_results:
            print(
                f"  - tool {tool_result.name}("
                f"arguments={tool_result.arguments}) -> {tool_result.output}"
            )


def print_chat_history(chat_history: list[ChatMessage]) -> None:
    print("Chat history:")
    for index, message in enumerate(chat_history, start=1):
        print(f"- history message {index}: role={message.role}")
        for line in describe_message_content(message.content):
            print(f"  {line}")


async def run_check(name: str, fn: Any) -> CheckResult:
    section(name)
    try:
        await fn()
    except Exception as exc:
        print(f"STATUS: FAIL")
        print(f"ERROR: {exc}")
        return CheckResult(name=name, ok=False, error=str(exc))

    print("STATUS: PASS")
    return CheckResult(name=name, ok=True)


async def run_plain_text() -> None:
    agent = Agent(config=make_config())
    prompt = "Reply with exactly the text plain-text-ok."
    try:
        print_input("Sending prompt:", prompt)
        result = await agent.run(prompt)
        print_result_details(result)
        ensure("plain-text-ok" in result.output_text.lower(), "Plain text response did not contain expected text.")
    finally:
        await agent.aclose()


async def run_structured_output() -> None:
    agent = Agent(config=make_config())
    prompt = "Return a person record with the name Sarah and the age 29."
    try:
        print_input("Sending prompt:", prompt)
        print("Requested response_model: Person")
        result = await agent.run(
            prompt,
            response_model=Person,
        )
        print_result_details(result)
        ensure(result.output_data is not None, "Structured output was not returned.")
        ensure(result.output_data.name.lower() == "sarah", "Structured output name was incorrect.")
        ensure(result.output_data.age == 29, "Structured output age was incorrect.")
    finally:
        await agent.aclose()


async def run_system_prompt() -> None:
    agent = Agent(
        config=make_config(),
        system_prompt="For the next reply, answer with exactly the text system-default-ok.",
    )
    default_prompt = "What should you reply with?"
    override_prompt = "What should you reply with now?"
    try:
        print("Agent default system_prompt:")
        print("- For the next reply, answer with exactly the text system-default-ok.")
        print_input("Sending prompt:", default_prompt)
        default_result = await agent.run(default_prompt)
        print_result_details(default_result)
        ensure(
            "system-default-ok" in default_result.output_text.lower(),
            "Agent-level system prompt did not steer the response as expected.",
        )

        print()
        print("Per-run override system_prompt:")
        print("- For the next reply, answer with exactly the text system-override-ok.")
        print_input("Sending prompt:", override_prompt)
        override_result = await agent.run(
            override_prompt,
            system_prompt="For the next reply, answer with exactly the text system-override-ok.",
        )
        print_result_details(override_result)
        ensure(
            "system-override-ok" in override_result.output_text.lower(),
            "Per-run system prompt override did not steer the response as expected.",
        )

        print()
        print("Chat session default system_prompt:")
        print("- For every reply in this chat, answer with exactly the text chat-system-ok.")
        chat = agent.chat(system_prompt="For every reply in this chat, answer with exactly the text chat-system-ok.")
        chat_prompt = "What should you reply with in this chat?"
        print_input("Sending prompt:", chat_prompt)
        chat_result = await chat.run(chat_prompt)
        print_result_details(chat_result)
        print_chat_history(chat.history)
        ensure(
            "chat-system-ok" in chat_result.output_text.lower(),
            "Chat session system prompt did not steer the response as expected.",
        )
        ensure(
            all(message.role != "developer" for message in chat.history),
            "System prompt leaked into chat history.",
        )
    finally:
        await agent.aclose()


async def run_tool_call() -> None:
    agent = Agent(config=make_config(), tools=[ping_tool])
    prompt = "Call the ping_tool with the message 'live-test'. Then tell me the tool result."
    try:
        print_input("Sending prompt:", prompt)
        print("Registered tools:")
        print("- ping_tool(message: str) -> str")
        result = await agent.run(prompt)
        print_result_details(result)
        ensure(len(result.tool_results) >= 1, "Expected at least one tool result.")
        ensure(result.tool_results[0].name == "ping_tool", "Unexpected tool name.")
        ensure(result.tool_results[0].output == "pong: live-test", "Unexpected tool output.")
    finally:
        await agent.aclose()


async def run_chat_history() -> None:
    agent = Agent(config=make_config())
    chat = agent.chat()
    first_prompt = "My favorite color is teal. Remember it for later."
    second_prompt = "What is my favorite color? Reply with the color only."
    try:
        print_input("First turn:", first_prompt)
        first = await chat.run(first_prompt)
        print_result_details(first)
        print_input("Second turn:", second_prompt)
        second = await chat.run(second_prompt)
        print_result_details(second)
        print_chat_history(chat.history)
        ensure("teal" in second.output_text.lower(), "Chat history was not preserved across turns.")
        ensure(len(chat.history) >= 4, "Chat history did not retain both turns.")
    finally:
        await agent.aclose()


async def run_streaming() -> None:
    agent = Agent(config=make_config())
    saw_delta = False
    final_text = None
    prompt = "Explain async agents in one short sentence."
    try:
        print_input("Sending prompt:", prompt)
        print("Streaming deltas:")
        async for event in agent.stream(prompt):
            if event.type == "text_delta" and event.delta:
                saw_delta = True
                print(f"- delta: {event.delta}")
            elif event.type == "completed" and event.result is not None:
                final_text = event.result.output_text
                print("Completed event received.")
                print_result_details(event.result)
        ensure(saw_delta, "Streaming did not emit any text deltas.")
        ensure(final_text is not None and final_text.strip(), "Streaming did not produce a final result.")
    finally:
        await agent.aclose()


async def run_image_structured_output() -> None:
    image_path = make_red_png_file()
    agent = Agent(config=make_config())
    input_data = [
        ChatMessage(
            role="user",
            content=[
                TextPart(
                    "The attached image is a solid color square. "
                    "Choose the dominant color from red, green, blue, black, or white. "
                    "Return the answer in the structured output."
                ),
                ImagePart.from_file(str(image_path), detail="high"),
            ],
        )
    ]
    try:
        print(f"Created temporary image file: {image_path}")
        print_input("Sending multimodal input:", input_data)
        print("Requested response_model: ColorAnswer")
        result = await agent.run(
            input_data,
            response_model=ColorAnswer,
        )
        print_result_details(result)
        ensure(result.output_data is not None, "Image structured output was not returned.")
        ensure("red" in result.output_data.dominant_color.lower(), "Image analysis did not identify red.")
    finally:
        await agent.aclose()
        image_path.unlink(missing_ok=True)


async def run_chat_with_image_follow_up() -> None:
    image_path = make_red_png_file()
    agent = Agent(config=make_config())
    chat = agent.chat()
    first_input = [
        ChatMessage(
            role="user",
            content=[
                TextPart("Remember this image for a follow-up question."),
                ImagePart.from_file(str(image_path), detail="high"),
            ],
        )
    ]
    second_prompt = "What was the dominant color in the image? Choose from red, green, blue, black, or white. Reply with one word."
    try:
        print(f"Created temporary image file: {image_path}")
        print_input("First turn:", first_input)
        first = await chat.run(first_input)
        print_result_details(first)
        print_input("Second turn:", second_prompt)
        follow_up = await chat.run(second_prompt)
        print_result_details(follow_up)
        print_chat_history(chat.history)
        ensure(first.output_text.strip(), "Initial image chat turn returned an empty response.")
        ensure("red" in follow_up.output_text.lower(), "Image context was not preserved in chat history.")
        ensure(len(chat.history) >= 4, "Chat history did not preserve the image turn.")
    finally:
        await agent.aclose()
        image_path.unlink(missing_ok=True)


async def main() -> None:
    print_config_summary()

    results = [
        await run_check("Plain Text", run_plain_text),
        await run_check("Structured Output", run_structured_output),
        await run_check("System Prompt", run_system_prompt),
        await run_check("Tool Call", run_tool_call),
        await run_check("Chat History", run_chat_history),
        await run_check("Streaming", run_streaming),
        await run_check("Image Structured Output", run_image_structured_output),
        await run_check("Chat With Image Follow-Up", run_chat_with_image_follow_up),
    ]

    section("Summary")
    passed = [result for result in results if result.ok]
    failed = [result for result in results if not result.ok]
    print(f"Passed: {len(passed)}")
    print(f"Failed: {len(failed)}")
    for result in results:
        status = "PASS" if result.ok else "FAIL"
        suffix = "" if result.error is None else f" -> {result.error}"
        print(f"- {status}: {result.name}{suffix}")

    if failed:
        raise SystemExit(1)

    print("All live end-to-end checks passed.")


if __name__ == "__main__":
    asyncio.run(main())
