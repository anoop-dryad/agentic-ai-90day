"""Smoke-test the downlink tools over a real MCP round-trip.
Requires: MCP server running, backend up, env vars set.
Run: python scripts/downlink_smoke.py"""

import asyncio
import json

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def parse(r):
    """Pull the tool's dict out of the CallToolResult."""
    if getattr(r, "structured_content", None):
        return r.structured_content
    if r.content and hasattr(r.content[0], "text"):
        try:
            return json.loads(r.content[0].text)
        except (ValueError, TypeError):
            return r.content[0].text
    return None


async def main():
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "canopy_agent.mcp_server"],
        env=None,
    )
    async with (
        stdio_client(server_params) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        print("tools:", [t.name for t in (await session.list_tools()).tools])

        # 1. propose valid — should describe + issue a token, send NOTHING
        r = await session.call_tool(
            "propose_downlink", {"device_id": "dev-003", "command": "reset"}
        )
        p = parse(r)
        print("\n[propose valid]   ", p)
        assert p["proposable"] is True, "valid propose should be proposable"
        token = p.get("confirmation_token")
        assert token, "propose must return a confirmation_token"

        # 2. propose invalid command — should reject with valid_commands list
        r = await session.call_tool(
            "propose_downlink", {"device_id": "dev-003", "command": "explode"}
        )
        p = parse(r)
        print("[propose bad cmd] ", p)
        assert p["proposable"] is False
        assert "valid_commands" in p

        # 3. propose bad device — should reject not-found
        r = await session.call_tool(
            "propose_downlink", {"device_id": "dev-999", "command": "reset"}
        )
        p = parse(r)
        print("[propose bad dev] ", p)
        assert p["proposable"] is False

        # 4. send WITHOUT token — should be refused (structural gate)
        r = await session.call_tool(
            "send_downlink",
            {"device_id": "dev-003", "command": "reset", "confirmation_token": ""},
        )
        p = parse(r)
        print("[send no token]   ", p)
        assert p["sent"] is False, "send must refuse without a valid token"

        # 5. send WITH the valid token from step 1 — should queue
        r = await session.call_tool(
            "send_downlink",
            {"device_id": "dev-003", "command": "reset", "confirmation_token": token},
        )
        p = parse(r)
        print("[send valid]      ", p)
        assert p["sent"] is True
        assert p["status"] == "queued"  # honest: queued, not delivered
        print("\n✅ downlink smoke passed — downlink_id:", p.get("downlink_id"))


if __name__ == "__main__":
    asyncio.run(main())
