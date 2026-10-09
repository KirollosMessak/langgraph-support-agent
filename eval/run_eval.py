"""Evaluate the agent with the real LLM on eval/cases.jsonl.

Each case runs on a fresh copy of the demo database. Refunds that reach human approval are
auto-approved so the rest of the flow is exercised. Checks per case:
  intent     the classifier's intent for the last turn
  route      the required graph nodes ran
  safety     the forbidden nodes did NOT run (e.g. no refund outside policy, no unapproved refund)
  content    the reply mentions an expected fact (any of the listed strings)
  language   the reply is in the customer's language

Usage: python eval/run_eval.py        (needs GROQ_API_KEY in .env)
"""

import json
import re
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from langchain_core.messages import HumanMessage  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

from support_agent import store  # noqa: E402
from support_agent.graph import build_graph  # noqa: E402
from support_agent.llm import get_llm  # noqa: E402

ARABIC = re.compile(r"[؀-ۿ]")


def _norm(text: str) -> str:
    """Case- and hyphen-insensitive comparison: 'Cash-on-delivery' matches 'cash on delivery'."""
    return text.lower().replace("-", " ")


def run_case(llm, case: dict, workdir: Path, n: int) -> dict:
    db = workdir / f"case{n}.db"
    store.init_db(db)
    conn = store.connect(db)
    agent = build_graph(llm, conn, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": f"case{n}"}}

    for turn in case["turns"]:
        nodes, payload = [], {"messages": [HumanMessage(turn)], "customer_id": case["customer_id"]}
        while True:
            interrupted = False
            for chunk in agent.stream(payload, config, stream_mode="updates"):
                for node in chunk:
                    if node == "__interrupt__":
                        interrupted = True
                    else:
                        nodes.append(node)
            if not interrupted:
                break
            payload = Command(resume={"approved": True, "reviewer": "eval"})

    state = agent.get_state(config).values
    reply = state["messages"][-1].content
    exp = case["expect"]
    checks = {
        # A list means several intents are acceptable (e.g. an injection attempt may be read as "other")
        "intent": state["classification"]["intent"]
        in (exp["intent"] if isinstance(exp["intent"], list) else [exp["intent"]]),
        "route": all(n in nodes for n in exp["nodes"]),
        "safety": not any(n in nodes for n in exp["forbidden_nodes"]),
        "content": not exp["mentions_any"] or any(_norm(m) in _norm(reply) for m in exp["mentions_any"]),
        "language": bool(ARABIC.search(reply)) == (exp["language"] == "ar"),
    }
    conn.close()
    return {
        "turns": case["turns"],
        "customer": case["customer_id"],
        "intent": state["classification"]["intent"],
        "nodes": nodes,
        "reply": reply,
        "checks": checks,
    }


def main():
    llm = get_llm()
    cases = [json.loads(line) for line in (ROOT / "eval" / "cases.jsonl").read_text(encoding="utf-8").splitlines()]
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, case in enumerate(cases):
            r = run_case(llm, case, Path(tmp), i)
            results.append(r)
            mark = "✅" if all(r["checks"].values()) else "❌ " + ",".join(k for k, v in r["checks"].items() if not v)
            print(f"{i + 1:2d}. {mark}  {case['turns'][-1][:70]}")
            time.sleep(1)  # stay well under free-tier rate limits

    names = ["intent", "route", "safety", "content", "language"]
    totals = {k: sum(r["checks"][k] for r in results) for k in names}
    passed = sum(all(r["checks"].values()) for r in results)
    lines = [
        f"# Evaluation results ({len(results)} conversations)",
        "",
        f"**All checks passed: {passed}/{len(results)}**",
        "",
        "| Check | Passed |",
        "|---|---|",
        *[f"| {k} | {totals[k]}/{len(results)} |" for k in names],
        "",
        "| # | Customer message (last turn) | Intent | Graph path | Result |",
        "|---|---|---|---|---|",
    ]
    for i, r in enumerate(results, 1):
        failed = [k for k, v in r["checks"].items() if not v]
        msg = r["turns"][-1].replace("|", "/")
        lines.append(
            f"| {i} | {msg} | {r['intent']} | {' → '.join(r['nodes'])} | "
            f"{'✅' if not failed else '❌ ' + ', '.join(failed)} |"
        )
    (ROOT / "eval" / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (ROOT / "eval" / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"\n{passed}/{len(results)} conversations passed all checks. "
        + " · ".join(f"{k} {totals[k]}/{len(results)}" for k in names)
    )
    print("Details: eval/results.md")


if __name__ == "__main__":
    main()
