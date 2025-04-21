from typing import Any
import httpx
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("weather")

@mcp.tool()
async def fetch_team_capacity() -> str:
    return "Team capacity is 30 story points."

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')