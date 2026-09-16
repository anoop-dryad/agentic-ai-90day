"""Streamlit chat UI for the canopy agent, with a collapsible per-message trace."""

import asyncio

import streamlit as st

from canopy_agent.mcp_client_agent import ask_once

st.set_page_config(page_title="Canopy Support Agent", page_icon="🌲")
st.title("🌲 Canopy — Silvanet Support Agent")

# chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {role, content, trace}


def _render_trace(trace: list[dict]):
    """Render the per-message execution trace as a tidy structured summary."""
    for event in trace:
        if event["step"] == "query":
            st.caption(f"❓ Query: {event['detail']}")
        elif event["step"] == "tool_call":
            tool = event["tool"]
            st.markdown(f"**🔧 {tool}** `{event['args']}`")
            # gate outcome — readable, color-coded via emoji
            if event.get("not_found"):
                st.markdown("   ↳ 🟡 gate: **not found** (device doesn't exist)")
            elif event.get("verified") is True:
                healthy = event.get("healthy")
                badge = "🟢 healthy" if healthy else "🔴 unhealthy"
                st.markdown(f"   ↳ ✅ gate: **verified** — {badge}")
            elif event.get("verified") is False:
                st.markdown("   ↳ 🔴 gate: **could not verify** (escalating)")
            elif event.get("grounded") is True:
                st.markdown("   ↳ ✅ docs: **grounded**")
            elif event.get("grounded") is False:
                st.markdown("   ↳ 🟡 docs: **not grounded** (no relevant docs)")


# --- render the conversation ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # the debug box — only on assistant messages, collapsed by default
        if msg["role"] == "assistant" and msg.get("trace"):
            with st.expander("🔍 Execution trace", expanded=False):
                _render_trace(msg["trace"])

# --- handle new input ---
if question := st.chat_input("Ask about a device or Silvanet docs..."):
    # show the user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # run the agent
    with st.chat_message("assistant"), st.spinner("Thinking..."):
        try:
            answer, trace = asyncio.run(ask_once(question))
        except Exception as e:  # noqa: BLE001
            answer = f"⚠️ Sorry, I hit an error: {type(e).__name__}. Please try again."
            trace = []
            st.error(answer)
        else:
            st.markdown(answer)
            with st.expander("🔍 Execution trace", expanded=False):
                _render_trace(trace)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "trace": trace}
    )
