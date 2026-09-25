"""
LLM-powered MCP client: Gemini decides which MCP tool to call.

Uses the ClientSession + stdio_client pattern (works in this mcp version),
not the tutorial's high-level Client.
https://modelcontextprotocol.io/docs/2026-07-28/develop/build-client
Gemini replaces Claude as the brain.
"""

import json as _json
import time

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from canopy_agent.config import settings
from canopy_agent.observability import log, new_trace

client = genai.Client(api_key=settings.GEMINI_API_KEY)

SYSTEM_PROMPT = (
    "You are a Dryad Silvanet support assistant. "
    "Use get_device_status for a device's live status/health, search_docs for "
    "how-to/knowledge questions. A question may need both. "
    "If a tool returns verified=false or grounded=false, never invent an answer — "
    "say you couldn't verify / have no docs, and escalate. Report device HEALTH, "
    "not just the raw status."
)


def _mcp_tools_to_gemini(mcp_tools):
    """Convert MCP tool schemas → Gemini function declarations."""
    decls = []
    for t in mcp_tools:
        decls.append(
            types.FunctionDeclaration(
                name=t.name,
                description=t.description or "",
                parameters=t.input_schema,  # MCP's JSON schema → Gemini params
            )
        )
    return [types.Tool(function_declarations=decls)]


async def process_query(session, question: str) -> tuple[str, list[dict], dict | None]:
    """Returns (answer, trace, pending_downlink).
    pending_downlink is set if the LLM proposed one this turn — else None."""
    new_trace()
    trace = []
    pending = None  # ← capture a proposal if it happens

    tool_list = await session.list_tools()
    gemini_tools = _mcp_tools_to_gemini(tool_list.tools)
    contents = [types.Content(role="user", parts=[types.Part(text=question)])]

    MAX_ROUNDS = 5  # the loop cap — your MAX_ITERATIONS instinct
    for _ in range(MAX_ROUNDS):
        resp = _generate_with_retry(contents, gemini_tools)  # your retry wrapper
        parts = resp.candidates[0].content.parts
        fn_calls = [p.function_call for p in parts if getattr(p, "function_call", None)]

        if not fn_calls:
            answer = "".join(p.text for p in parts if getattr(p, "text", None))
            return answer, trace, pending

        contents.append(resp.candidates[0].content)
        for fc in fn_calls:
            result = await session.call_tool(fc.name, dict(fc.args))
            result_text = "".join(b.text for b in result.content if hasattr(b, "text"))
            parsed = _json.loads(result_text) if result_text else {}

            # ── DETECT A PROPOSAL ──
            if fc.name == "propose_downlink" and parsed.get("proposable"):
                pending = {
                    "device_id": parsed["device_id"],
                    "command": parsed["command"],
                    "token": parsed["token"],
                }
                log.info(
                    "downlink_proposed",
                    extra={
                        "data": {
                            "device_id": pending["device_id"],
                            "command": pending["command"],
                        }
                    },
                )

            trace.append(
                {
                    "step": "tool_call",
                    "tool": fc.name,
                    "args": dict(fc.args),
                    "verified": parsed.get("verified"),
                    "grounded": parsed.get("grounded"),
                    "not_found": parsed.get("not_found"),
                    "healthy": (parsed.get("health") or {}).get("healthy"),
                }
            )

            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            function_response=types.FunctionResponse(
                                name=fc.name, response={"result": result_text}
                            )
                        )
                    ],
                )
            )

    return "Couldn't complete.", trace, pending


async def ask_once(question: str) -> tuple[str, list[dict], dict | None]:
    """One full query with its own connection. Streamlit-friendly."""
    async with (
        streamable_http_client(settings.MCP_SERVER_URL) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        return await process_query(session, question)


def _generate_with_retry(contents, gemini_tools, max_retries=3):
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(
                model=settings.MODEL_NAME,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT, tools=gemini_tools
                ),
            )
        except genai_errors.ServerError:
            if attempt < max_retries - 1:
                time.sleep(5**attempt)  # backoff: 5s, 10s, 15s
                continue
            raise


async def confirm_and_send(pending: dict) -> dict:
    """Send the downlink directly with the stored token. No LLM involved —
    this is the human's confirmation turned into a gated tool call."""
    new_trace()
    log.info(
        "downlink_confirmed",
        extra={
            "data": {"device_id": pending["device_id"], "command": pending["command"]}
        },
    )

    async with (
        streamable_http_client(settings.MCP_SERVER_URL) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        r = await session.call_tool(
            "send_downlink",
            {
                "device_id": pending["device_id"],
                "command": pending["command"],
                "confirmation_token": pending["token"],
            },
        )
        import json as _json

        text = "".join(b.text for b in r.content if hasattr(b, "text"))
        return _json.loads(text) if text else {}
