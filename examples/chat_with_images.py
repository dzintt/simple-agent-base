import asyncio

from agent_harness import Agent, AgentConfig, ChatMessage, ImagePart, TextPart


async def main() -> None:
    agent = Agent(config=AgentConfig(model="gpt-5"))
    chat = agent.chat()

    try:
        await chat.run(
            [
                ChatMessage(
                    role="user",
                    content=[
                        TextPart("Remember this image."),
                        ImagePart.from_file("photo.jpg"),
                    ],
                )
            ]
        )
        follow_up = await chat.run("What was in the image?")

        print("Follow-up reply:")
        print(follow_up.output_text)
        print()
        print("Conversation history:")
        for message in chat.history:
            print(f"- {message.role}: {message.content}")
    finally:
        await agent.aclose()


if __name__ == "__main__":
    asyncio.run(main())
