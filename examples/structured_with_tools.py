import asyncio

from pydantic import BaseModel

from simple_agent_base import Agent, AgentConfig, tool


class WeatherAnswer(BaseModel):
    city: str
    temperature_f: int
    summary: str


@tool
async def get_weather(city: str) -> str:
    """Return weather information for a city as JSON text."""
    fake_weather = {
        "san francisco": '{"city":"San Francisco","temperature_f":65,"summary":"Foggy"}',
        "phoenix": '{"city":"Phoenix","temperature_f":92,"summary":"Hot and sunny"}',
    }
    return fake_weather.get(city.lower(), f'{{"city":"{city}","temperature_f":70,"summary":"Mild"}}')


async def main() -> None:
    agent = Agent(
        config=AgentConfig(model="gpt-5.4"),
        tools=[get_weather],
    )

    try:
        result = await agent.run(
            "Use the weather tool for San Francisco and return a structured weather answer.",
            response_model=WeatherAnswer,
        )

        print("Final text:")
        print(result.output_text)
        print()
        print("Structured output:")
        print(result.output_data)
        print()
        print("Tool results:")
        for tool_result in result.tool_results:
            print(f"- {tool_result.name}: {tool_result.output}")
    finally:
        await agent.aclose()


if __name__ == "__main__":
    asyncio.run(main())
