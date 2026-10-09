"""Test fixtures: a rule-based fake LLM, so the graph is tested without an API key or network."""

import re
from datetime import date

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from support_agent import store
from support_agent.graph import Classification, build_graph

TODAY = date(2026, 10, 1)


class FakeLLM:
    """Classifies with keywords and 'replies' by echoing the context it was given."""

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        if isinstance(messages[0].content, str) and messages[0].content.startswith("You route"):
            return self._classify(messages)
        return AIMessage(messages[0].content)  # the system prompt, which contains the CONTEXT

    def _classify(self, messages):
        humans = [m.content for m in messages if isinstance(m, HumanMessage)]
        text = humans[-1].lower()
        ids = re.findall(r"ORD-\d+", " ".join(humans), re.I)
        intent = (
            "talk_to_human"
            if "human" in text
            else "refund_request"
            if "refund" in text
            else "cancel_order"
            if "cancel" in text
            else "order_status"
            if "where" in text or "status" in text
            else "greeting"
            if text.strip() in {"hi", "hello"}
            else "policy_question"
        )
        return Classification(
            intent=intent,
            order_id=ids[-1].upper() if ids else None,
            language="ar" if re.search(r"[؀-ۿ]", text) else "en",
            sentiment="negative" if "angry" in text else "neutral",
            wants_human=False,
            search_query=text,
        )


@pytest.fixture
def conn(tmp_path):
    db = tmp_path / "store.db"
    store.init_db(db, today=TODAY)
    return store.connect(db)


@pytest.fixture
def agent(conn):
    return build_graph(FakeLLM(), conn, checkpointer=InMemorySaver(), today=TODAY)


def chat(agent, text, customer="C001", thread="t1"):
    config = {"configurable": {"thread_id": thread}}
    return agent.invoke({"messages": [HumanMessage(text)], "customer_id": customer}, config), config
