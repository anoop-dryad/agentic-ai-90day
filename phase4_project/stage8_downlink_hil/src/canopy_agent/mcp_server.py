"""MCP server exposing canopy's gated device + docs tools.

The GATES live in backend.py / rag.py — this server is a thin adapter that
exposes them as MCP tools. Safety travels with the tool: any MCP client
calling these gets the verify-or-escalate / grounding behavior."""

import uvicorn
from mcp.server import MCPServer

from canopy_agent.backend import get_device, send_downlink_to_backend
from canopy_agent.confirmation import make_confirmation_token, valid_confirmation_token
from canopy_agent.health import compute_health
from canopy_agent.observability import log, new_trace
from canopy_agent.rag import search_docs_gated

mcp = MCPServer("canopy")

VALID_COMMANDS = {"reset", "calibrate", "set_interval"}


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


@mcp.tool()
def propose_downlink(device_id: str, command: str) -> dict:
    """Validate a proposed downlink WITHOUT sending it. Use when a user asks
    to send a command to a device. Returns what WOULD be sent so the user can
    confirm. This tool NEVER sends — it only checks the device exists and the
    command is valid, then describes the pending action for confirmation.
    """

    new_trace()
    log.info(
        "propose_downlink",
        extra={
            "data": {
                "device_id": device_id,
                "command": command,
            }
        },
    )

    if command not in VALID_COMMANDS:
        return {
            "proposable": False,
            "reason": f"'{command}' is not a valid command",
            "valid_commands": sorted(VALID_COMMANDS),
        }

    result = get_device(device_id=device_id)
    if not result.ok:
        if result.data and result.data.get("not_found"):
            return {
                "proposable": False,
                "reason": f"device '{device_id}' not found",
            }

        return {
            "proposable": False,
            "reason": result.reason,
        }

    device = result.data
    token = make_confirmation_token(device_id, command)  # signed/hashed, short TTL
    return {
        "proposable": True,
        "device_id": device["id"],
        "device_name": device.get("name", "unknown"),
        "command": command,
        "token": token,
        "warning": "This queues a command to a physical device and requires confirmation.",
    }


@mcp.tool()
def send_downlink(device_id: str, command: str, confirmation_token: str) -> dict:
    """Queue a downlink command to a device. This ACTUALLY queues it and must
    ONLY be called after explicit user confirmation. Reports status honestly
    (queued — recorded for delivery, not confirmed delivered).
    """
    new_trace()
    log.info(
        "send_downlink",
        extra={
            "data": {
                "device_id": device_id,
                "command": command,
            }
        },
    )

    # STRUCTURAL GATE: send is impossible without a valid token from propose
    if not valid_confirmation_token(confirmation_token, device_id, command):
        return {
            "sent": False,
            "reason": "missing or invalid confirmation — must propose first",
        }

    result = send_downlink_to_backend(device_id, command)
    if not result.ok:
        return {
            "sent": False,
            "reason": result.reason,
        }
    return {
        "sent": True,
        "status": result.data.get("status", "queued"),
        "downlink_id": result.data.get("id"),
        "device_id": device_id,
        "command": command,
    }


app = mcp.streamable_http_app()
if __name__ == "__main__":
    # 0.0.0.0 so it's reachable from outside a container (not just localhost)
    uvicorn.run(app, host="0.0.0.0", port=8000)
