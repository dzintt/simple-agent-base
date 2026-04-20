"""Use a remote MCP server with an Agent.

Run:
    uv run python examples/mcp_server.py

Requires a valid OPENAI_API_KEY. The deepwiki server is public and requires
no authentication. With the default `require_approval="never"`, MCP tool
invocations run automatically without any approval plumbing.
"""

from __future__ import annotations

import asyncio
import os

from simple_agent_base import (
    Agent,
    AgentConfig,
    MCPApprovalRequest,
    MCPServer,
)


async def basic_mcp_run() -> None:
    agent = Agent(
        config=AgentConfig(model=os.environ.get("OPENAI_MODEL", "gpt-5")),
        mcp_servers=[
            MCPServer(
                server_label="deepwiki",
                server_url="https://mcp.deepwiki.com/mcp",
            )
        ],
    )

    try:
        result = await agent.run(
            "Use the deepwiki MCP server to summarize the README of "
            "modelcontextprotocol/python-sdk in three short bullets."
        )
    finally:
        await agent.aclose()

    print("=== Model output ===")
    print(result.output_text)
    print()
    print("=== MCP tool calls ===")
    for call in result.mcp_calls:
        print(f"- {call.server_label}.{call.name}({call.arguments})")


async def mcp_run_with_approvals() -> None:
    def approve(request: MCPApprovalRequest) -> bool:
        print(f"[approval] {request.server_label}.{request.name}({request.arguments})")
        # In a real app you'd prompt the user, check a policy, etc.
        return True

    agent = Agent(
        config=AgentConfig(model=os.environ.get("OPENAI_MODEL", "gpt-5")),
        mcp_servers=[
            MCPServer(
                server_label="deepwiki",
                server_url="https://mcp.deepwiki.com/mcp",
                require_approval="always",
            )
        ],
        approval_handler=approve,
    )

    try:
        result = await agent.run(
            "Ask deepwiki what the MCP python-sdk is and summarize in one line."
        )
    finally:
        await agent.aclose()

    print("=== With-approval output ===")
    print(result.output_text)


async def main() -> None:
    await basic_mcp_run()
    print()
    await mcp_run_with_approvals()


if __name__ == "__main__":
    asyncio.run(main())
