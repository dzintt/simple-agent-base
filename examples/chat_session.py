import asyncio

from simple_agent_base import Agent, AgentConfig


async def main() -> None:
    agent = Agent(config=AgentConfig(model="gpt-5.4"))
    chat = agent.chat()

    try:
        first = await chat.run("My name is Anson.")
        second = await chat.run("What name did I tell you?")

        print("First reply:")
        print(first.output_text)
        print()
        print("Second reply:")
        print(second.output_text)
        print()
        print("Conversation history:")
        for message in chat.history:
            print(f"- {message.role}: {message.content}")
    finally:
        await agent.aclose()


if __name__ == "__main__":
    asyncio.run(main())
