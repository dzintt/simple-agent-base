import asyncio

from simple_agent_base import Agent, AgentConfig, tool


@tool
async def ping(message: str) -> str:
    """Echo a message back."""
    return f"pong: {message}"


async def main() -> None:
    agent = Agent(
        config=AgentConfig(model="gpt-5.4"),
        tools=[ping],
    )

    result = await agent.run("Call ping with hello and tell me the result.")
    try:
        print("Final text:")
        print(result.output_text)
        print()
        print("Tool results:")
        for tool_result in result.tool_results:
            print(f"- {tool_result.name}: {tool_result.output}")
    finally:
        await agent.aclose()


if __name__ == "__main__":
    asyncio.run(main())
