"""MCP server exposing canopy's gated device + docs tools.

The GATES live in backend.py / rag.py — this server is a thin adapter that
exposes them as MCP tools. Safety travels with the tool: any MCP client
calling these gets the verify-or-escalate / grounding behavior."""

from mcp.server import MCPServer

from canopy_agent.backend import get_device
from canopy_agent.health import compute_health
from canopy_agent.observability import log, new_trace
from canopy_agent.rag import search_docs_gated

mcp = MCPServer("canopy")


@mcp.tool()
def get_device_status(device_id: str) -> dict:
    """Get the current health of a Silvanet device by its ID (e.g. 'dev-001').

    Returns verified device data with computed health flags, or an unverified
    signal the caller must relay honestly (never invent a status).
    """

    new_trace()
    log.info(
        "mcp_tool_call",
        extra={"data": {"tool": "get_device_status", "device_id": device_id}},
    )

    result = get_device(device_id)
    if not result.ok:
        if result.data and result.data.get("not_found"):
            return {
                "verified": False,
                "not_found": True,
                "reason": result.reason,
            }
        if result.data and result.data.get("auth_failed"):
            return {
                "verified": False,
                "auth_failed": True,
                "reason": result.reason,
            }
        return {"verified": False, "reason": result.reason}

    device = result.data
    if not device or "id" not in device:
        return {
            "verified": False,
            "reason": "backend returned ok but data incomplete",
        }

    health = compute_health(device)
    log.info(
        "mcp_tool_result",
        extra={
            "data": {
                "tool": "get_device_status",
                "device_id": device_id,
                "verified": True,
                "healthy": health["healthy"],
            }
        },
    )
    return {
        "verified": True,
        "id": device["id"],
        "name": device.get("name", "unknown"),
        "status_field": device.get("status", "unknown"),
        "battery_pct": device.get("battery_pct"),
        "last_seen": device.get("last_seen"),
        "health": health,
    }


@mcp.tool()
def search_docs(question: str) -> dict:
    """Search Dryad Silvanet documentation for how-to, troubleshooting, and
    knowledge questions. Returns grounded doc excerpts, or indicates no
    relevant docs were found (caller must not invent an answer).
    """

    new_trace()
    log.info(
        "mcp_tool_call", extra={"data": {"tool": "search_docs", "question": question}}
    )

    result = search_docs_gated(question)
    if not result.grounded:
        log.info(
            "mcp_tool_result",
            extra={"data": {"tool": "search_docs", "grounded": False}},
        )
        return {"grounded": False, "reason": result.reason}

    log.info(
        "mcp_tool_result",
        extra={
            "data": {"tool": "search_docs", "grounded": True, "n": len(result.chunks)}
        },
    )
    return {"grounded": True, "excerpts": result.chunks}


if __name__ == "__main__":
    mcp.run()  # stdio transport by default
