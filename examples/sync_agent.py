from agent_harness import Agent, AgentConfig


def main() -> None:
    agent = Agent(config=AgentConfig(model="gpt-5"))

    try:
        result = agent.run_sync("Say hello in one short sentence.")
        print("Final text:")
        print(result.output_text)
    finally:
        agent.close()


if __name__ == "__main__":
    main()
