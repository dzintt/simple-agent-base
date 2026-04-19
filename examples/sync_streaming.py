from agent_harness import Agent, AgentConfig


def main() -> None:
    agent = Agent(config=AgentConfig(model="gpt-5"))

    try:
        for event in agent.stream_sync("Explain async IO in one short sentence."):
            if event.type == "text_delta" and event.delta:
                print(event.delta, end="")
            elif event.type == "completed":
                print()
                print()
                print("Final text:")
                print(event.result.output_text if event.result is not None else "")
    finally:
        agent.close()


if __name__ == "__main__":
    main()
