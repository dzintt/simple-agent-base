# Docs

This folder contains the detailed documentation for `simple-agent-base`.

Use the top-level [README](../README.md) for the project overview and quickstart. Use these docs when you need exact behavior, API details, or contributor context.

## Reading Order

If you are new to the package:

1. [usage.md](./usage.md)
2. [tools.md](./tools.md)
3. [structured-output.md](./structured-output.md)

If you are extending or modifying the package:

1. [architecture.md](./architecture.md)
2. [development.md](./development.md)

## Guide Map

- [usage.md](./usage.md)
  End-to-end usage of `Agent`, `ChatSession`, multimodal inputs, system prompts, streaming, snapshots, and sync wrappers.
- [tools.md](./tools.md)
  Tool definition, schema generation, registration, execution, parallel tool calls, and failure behavior.
- [structured-output.md](./structured-output.md)
  Typed outputs with Pydantic, behavior during runs and streams, tool interactions, and parse failures.
- [architecture.md](./architecture.md)
  Internal runtime model, transcript flow, provider abstraction, sync runtime, and source file responsibilities.
- [development.md](./development.md)
  Local setup, tests, live verification, package layout, and contribution guidance.
