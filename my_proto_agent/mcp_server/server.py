# mcp_server/server.py
from fastmcp import FastMCP
import asyncio

mcp = FastMCP("Demo MCP Server")


@mcp.tool()
def add(a: int, b: int) -> int:
    return a + b


if __name__ == "__main__":
    asyncio.run(mcp.run_sse_async(host="0.0.0.0", port=8001))
