import asyncio

from pydantic import BaseModel

from simple_agent_base import Agent, AgentConfig


class Person(BaseModel):
    name: str
    age: int
    city: str


async def main() -> None:
    agent = Agent(config=AgentConfig(model="gpt-5.4"))

    try:
        result = await agent.run(
            "Extract a person from this sentence: Sarah is 29 years old and lives in Seattle.",
            response_model=Person,
        )

        print("Raw text:")
        print(result.output_text)
        print()
        print("Parsed object:")
        print(result.output_data)
    finally:
        await agent.aclose()


if __name__ == "__main__":
    asyncio.run(main())
