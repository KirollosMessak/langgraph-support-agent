"""Streamlit chat UI for the support agent: live agent steps and a supervisor approval panel.

Run: streamlit run app.py
"""

import sys
import uuid
from pathlib import Path

import streamlit as st
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

sys.path.insert(0, str(Path(__file__).parent / "src"))

from support_agent import store  # noqa: E402
from support_agent.graph import build_graph  # noqa: E402
from support_agent.llm import get_llm, has_api_key  # noqa: E402

st.set_page_config(page_title="AI Support Agent", page_icon="🎧", layout="wide")

STEP_LABELS = {
    "classify": "🧭 Understood the request",
    "retrieve": "📚 Searched store policies",
    "lookup_order": "📦 Looked up the order",
    "check_eligibility": "⚖️ Checked the rules",
    "human_approval": "🧑‍💼 Human approval",
    "execute_action": "✅ Executed the action",
    "escalate": "🙋 Escalated to a human agent",
    "respond": "✍️ Wrote the reply",
}


@st.cache_resource
def get_conn():
    if not store.DEFAULT_DB.exists():
        store.init_db()
    return store.connect()


@st.cache_resource
def get_agent(api_key: str | None):
    return build_graph(get_llm(api_key), get_conn(), checkpointer=InMemorySaver())


def describe(node: str, update: dict) -> str:
    """One readable line per graph step, shown under each reply."""
    update = update or {}
    if node == "classify":
        c = update["classification"]
        extra = f", order {c['order_id']}" if c.get("order_id") else ""
        return f"intent **{c['intent']}**, language {c['language']}, sentiment {c['sentiment']}{extra}"
    if node == "retrieve":
        return "found: " + "; ".join(d["title"] for d in update.get("docs", []))
    if node == "lookup_order":
        o = update.get("order")
        return f"{o['id']}: {o['items']} ({o['status']})" if o else "no matching order on this account"
    if node == "check_eligibility":
        e = update["eligibility"]
        return f"{'eligible' if e['eligible'] else 'not eligible'}: {e['reason']}"
    if node == "execute_action":
        r = update["action_result"]
        return f"{r['action']} #{r['id']} → {r['status']}"
    if node == "escalate":
        return f"ticket #{update['ticket_id']} created"
    if node == "human_approval":
        a = update.get("approval") or {}
        return f"{'approved' if a.get('approved') else 'rejected'} by {a.get('reviewer', '?')}"
    return ""


def run(agent, payload):
    """Stream the graph, collecting steps; returns (reply or None, steps, pending interrupt or None)."""
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    steps, pending, reply = [], None, None
    with st.status("Agent working...", expanded=True) as status:
        for chunk in agent.stream(payload, config, stream_mode="updates"):
            for node, update in chunk.items():
                if node == "__interrupt__":
                    pending = update[0].value
                    continue
                line = f"{STEP_LABELS.get(node, node)}: {describe(node, update)}".rstrip(": ")
                steps.append(line)
                st.write(line)
                if node == "respond":
                    reply = update["messages"][-1].content
        status.update(label="Waiting for supervisor approval" if pending else "Done", state="complete", expanded=False)
    return reply, steps, pending


# ---------- Sidebar: who is logged in, their orders, controls ----------
conn = get_conn()
with st.sidebar:
    st.header("🎧 Nile Gadgets support")
    st.caption("Demo store. All data is fictional.")
    key = None
    if not has_api_key():
        key = st.text_input("Groq API key", type="password", help="Free at console.groq.com/keys")
    customers = {c[0]: f"{c[1]} ({c[0]})" for c in store.CUSTOMERS}
    customer_id = st.selectbox("Logged in as", customers, format_func=customers.get)
    st.subheader("Their orders")
    st.dataframe(
        [
            {"order": o["id"], "items": o["items"], "status": o["status"], "EGP": int(o["total"])}
            for o in store.list_orders(conn, customer_id)
        ],
        hide_index=True,
    )
    col1, col2 = st.columns(2)
    if col1.button("New chat") or st.session_state.get("customer") != customer_id:
        st.session_state.update(thread_id=uuid.uuid4().hex, chat=[], pending=None, customer=customer_id)
    if col2.button("Reset data"):
        store.init_db()
        st.session_state.update(thread_id=uuid.uuid4().hex, chat=[], pending=None)
        st.rerun()

st.title("AI Customer Support Agent")
st.caption(
    "LangGraph agent: answers policy questions, checks orders, and handles refunds and cancellations "
    "with human approval. Try English or Arabic."
)

if not (has_api_key() or key):
    st.info("Add a Groq API key in the sidebar to start (free at console.groq.com/keys).")
    st.stop()
agent = get_agent(key)

for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("steps"):
            with st.expander("Agent steps"):
                st.markdown("\n".join(f"- {s}" for s in msg["steps"]))

# ---------- Supervisor approval panel (human-in-the-loop) ----------
if pending := st.session_state.pending:
    with st.container(border=True):
        st.subheader("🧑‍💼 Supervisor approval needed")
        st.write(
            f"**Refund {pending['amount']:,.0f} EGP** for order **{pending['order_id']}** "
            f"({pending['items']}), customer {pending['customer_id']}"
        )
        st.caption(f"Rule check: {pending['reason']}  ·  Customer said: “{pending['customer_message']}”")
        note = st.text_input("Note to customer (optional)")
        approve, reject = st.columns(2)
        decision = None
        if approve.button("✅ Approve refund", type="primary"):
            decision = {"approved": True, "reviewer": "supervisor", "note": note}
        if reject.button("❌ Reject"):
            decision = {"approved": False, "reviewer": "supervisor", "note": note}
        if decision:
            reply, steps, _ = run(agent, Command(resume=decision))
            st.session_state.pending = None
            st.session_state.chat.append({"role": "assistant", "content": reply, "steps": steps})
            st.rerun()
    st.stop()

if prompt := st.chat_input("Ask about an order, returns, shipping... / اسأل عن طلبك"):
    st.session_state.chat.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        reply, steps, pending = run(agent, {"messages": [HumanMessage(prompt)], "customer_id": customer_id})
    if pending:
        st.session_state.pending = pending
        st.session_state.chat.append(
            {
                "role": "assistant",
                "content": "_One moment, I've sent your refund request to our team for approval..._",
                "steps": steps,
            }
        )
    else:
        st.session_state.chat.append({"role": "assistant", "content": reply, "steps": steps})
    st.rerun()

with st.expander("Activity log (actions and escalations)"):
    st.write("**Actions**", store.recent(conn, "actions") or "none yet")
    st.write("**Tickets**", store.recent(conn, "tickets") or "none yet")
