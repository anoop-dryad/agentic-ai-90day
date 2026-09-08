"""Smoke-test the canopy MCP server over stdio."""

import asyncio

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "canopy_agent.mcp_server"],
        env=None,  # inherits your current env — CANOPY_API_KEY etc. must be set in this shell
    )
    async with (
        stdio_client(server_params) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()

        tools = await session.list_tools()
        print("tools:", [t.name for t in tools.tools])

        r = await session.call_tool("get_device_status", {"device_id": "dev-001"})
        print("dev-001:", r.content[0].text)

        r = await session.call_tool("get_device_status", {"device_id": "dev-999"})
        print("dev-999:", r.content[0].text)

        r = await session.call_tool(
            "search_docs", {"question": "what is calibration mode?"}
        )
        print("docs:", r.content[0].text)

        r = await session.call_tool(
            "search_docs", {"question": "airspeed of a swallow"}
        )
        print("offtopic:", r.content[0].text)


asyncio.run(main())
