import asyncio

from agent_harness import Agent, AgentConfig, ChatMessage, FilePart, TextPart


async def main() -> None:
    agent = Agent(config=AgentConfig(model="gpt-5"))

    try:
        result = await agent.run(
            [
                ChatMessage(
                    role="user",
                    content=[
                        TextPart("Summarize this PDF in one short paragraph."),
                        FilePart.from_file("report.pdf"),
                    ],
                )
            ]
        )
        print(result.output_text)
    finally:
        await agent.aclose()


if __name__ == "__main__":
    asyncio.run(main())
