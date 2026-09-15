"""
LLM-powered MCP client: Gemini decides which MCP tool to call.

Uses the ClientSession + stdio_client pattern (works in this mcp version),
not the tutorial's high-level Client.
https://modelcontextprotocol.io/docs/2026-07-28/develop/build-client
Gemini replaces Claude as the brain.
"""

import asyncio

from google import genai
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


async def process_query(session, question: str) -> str:
    new_trace()
    log.info("query_start", extra={"data": {"question": question}})

    tool_list = await session.list_tools()
    gemini_tools = _mcp_tools_to_gemini(tool_list.tools)
    contents = [types.Content(role="user", parts=[types.Part(text=question)])]

    MAX_ROUNDS = 5  # the loop cap — your MAX_ITERATIONS instinct
    for _ in range(MAX_ROUNDS):
        resp = client.models.generate_content(
            model=settings.MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT, tools=gemini_tools
            ),
        )
        parts = resp.candidates[0].content.parts
        fn_calls = [p.function_call for p in parts if getattr(p, "function_call", None)]

        # no tool calls → Gemini gave the final answer, we're done
        if not fn_calls:
            answer = "".join(p.text for p in parts if getattr(p, "text", None))
            log.info("query_end", extra={"data": {"answer_preview": answer[:200]}})
            return answer

        # otherwise run the tools and loop again
        contents.append(resp.candidates[0].content)
        for fc in fn_calls:
            log.info(
                "mcp_tool_call",
                extra={"data": {"tool": fc.name, "args": dict(fc.args)}},
            )
            result = await session.call_tool(fc.name, dict(fc.args))
            result_text = "".join(b.text for b in result.content if hasattr(b, "text"))
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

    # hit the cap without a final answer
    log.warning("query_max_rounds", extra={"data": {"question": question}})
    return (
        "I gathered some information but couldn't complete a full answer — escalating."
    )


async def main():
    async with (
        streamable_http_client("http://localhost:8000/mcp") as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        print("connected. tools:", [t.name for t in (await session.list_tools()).tools])

        for q in [
            "Is dev-001 online?",
            "What does inactive status mean?",
            "Why is dev-003 inactive?",
            "What's the airspeed of a swallow?",
        ]:
            print(f"\n🧑 {q}\n🤖 {await process_query(session, q)}")


if __name__ == "__main__":
    asyncio.run(main())
