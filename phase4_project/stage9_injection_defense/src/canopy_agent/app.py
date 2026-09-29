"""Streamlit chat UI for the canopy agent, with a collapsible per-message trace."""

import asyncio

import streamlit as st

from canopy_agent.mcp_client_agent import ask_once, confirm_and_send

st.set_page_config(page_title="Canopy Support Agent", page_icon="🌲")
st.title("🌲 Canopy — Silvanet Support Agent")

# chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {role, content, trace}

# pending downlink awaiting human confirmation (None when nothing pending)
if "pending_downlink" not in st.session_state:
    st.session_state.pending_downlink = None


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

# --- the downlink confirmation gate ---
# Renders when the agent proposed a downlink. The human confirms via BUTTON,
# not by typing — the client holds the token in session_state and, on Confirm,
# calls send_downlink directly. The LLM is out of the loop for the actual send.
if st.session_state.pending_downlink:
    pd = st.session_state.pending_downlink
    st.divider()
    st.warning(
        f"⚠️ **Confirm downlink**\n\n"
        f"Send **`{pd['command']}`** to **{pd['device_id']}**?\n\n"
        f"This queues a command to a physical device and cannot be undone."
    )
    col1, col2 = st.columns(2)

    if col1.button("✅ Confirm & Send", type="primary", use_container_width=True):
        try:
            result = asyncio.run(confirm_and_send(pd))
        except Exception as e:  # noqa: BLE001
            result = {"sent": False, "reason": f"{type(e).__name__}"}

        if result.get("sent"):
            note = (
                f"✅ Downlink **queued** — id `{result.get('downlink_id')}`, "
                f"status: **{result.get('status')}**"
            )
            st.session_state.messages.append({"role": "assistant", "content": note})
        else:
            note = f"❌ Downlink failed: {result.get('reason')}"
            st.session_state.messages.append({"role": "assistant", "content": note})

        st.session_state.pending_downlink = None
        st.rerun()

    if col2.button("❌ Cancel", use_container_width=True):
        st.session_state.messages.append(
            {"role": "assistant", "content": "❌ Cancelled — nothing was sent."}
        )
        st.session_state.pending_downlink = None
        st.rerun()


# --- handle new input ---
# Input is disabled while a downlink is pending — the human must resolve the
# confirmation (Confirm/Cancel) before continuing. Keeps the gate unambiguous.
disabled = st.session_state.pending_downlink is not None
if question := st.chat_input(
    "Ask about a device or Silvanet docs...",
    disabled=disabled,
):
    # show the user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # run the agent
    with st.chat_message("assistant"), st.spinner("Thinking..."):
        try:
            answer, trace, pending = asyncio.run(ask_once(question))
        except Exception as e:  # noqa: BLE001
            answer = f"⚠️ Sorry, I hit an error: {type(e).__name__}. Please try again."
            # st.code(traceback.format_exc()) to get the full error
            trace, pending = [], None
            st.error(answer)
        else:
            st.markdown(answer)
            with st.expander("🔍 Execution trace", expanded=False):
                _render_trace(trace)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "trace": trace}
    )

    if pending:
        st.session_state.pending_downlink = pending
        st.rerun()
