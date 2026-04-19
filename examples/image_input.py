import asyncio

from agent_harness import Agent, AgentConfig, ChatMessage, ImagePart, TextPart


async def main() -> None:
    agent = Agent(config=AgentConfig(model="gpt-5"))

    try:
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

        print("Final text:")
        print(result.output_text)
    finally:
        await agent.aclose()


if __name__ == "__main__":
    asyncio.run(main())
