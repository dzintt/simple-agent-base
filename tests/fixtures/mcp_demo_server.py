from __future__ import annotations

import sys

import uvicorn
from mcp import types
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Demo MCP", json_response=True)


@mcp.tool()
def echo(message: str) -> str:
    """Echo a message."""
    return f"echo:{message}"


@mcp.tool()
def add(a: int, b: int) -> str:
    """Add two integers and return text."""
    return str(a + b)


@mcp.tool()
def structured_value(name: str) -> dict[str, str]:
    """Return a structured value."""
    return {"name": name}


@mcp.tool()
def fail(message: str) -> types.CallToolResult:
    """Return an MCP error result."""
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=message)],
        isError=True,
    )


app = mcp.streamable_http_app()


def main() -> None:
    mode = sys.argv[1]
    if mode == "stdio":
        mcp.run(transport="stdio")
        return

    if mode == "http":
        port = int(sys.argv[2])
        uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
        return

    raise SystemExit(f"Unknown mode: {mode}")


if __name__ == "__main__":
    main()
