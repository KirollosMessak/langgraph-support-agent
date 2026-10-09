"""Policy knowledge base: markdown documents split by section and searched with BM25.

BM25 keyword search needs no embedding model download and works well for short policy texts.
Arabic questions are handled by the classifier, which writes an English search query.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Plus

POLICY_DIR = Path(__file__).resolve().parents[2] / "data" / "policies"


@dataclass
class Section:
    source: str  # file name
    title: str  # "Returns and refunds > Return window"
    text: str


def load_sections(policy_dir: Path = POLICY_DIR) -> list[Section]:
    sections = []
    for path in sorted(policy_dir.glob("*.md")):
        doc_title, current, lines = path.stem, None, []
        for line in path.read_text(encoding="utf-8").splitlines() + ["## "]:
            if line.startswith("# "):
                doc_title = line[2:].strip()
            elif line.startswith("## "):
                if current and lines:
                    sections.append(Section(path.name, f"{doc_title} > {current}", " ".join(lines).strip()))
                current, lines = line[3:].strip(), []
            elif line.strip():
                lines.append(line.strip())
    return sections


# Filler words that match every document and drown out the meaningful ones
STOPWORDS = set(
    "a an the and or but if of to in on at for from by with about as is are was were be been being do does did "
    "have has had i you your we our us it its this that these those what which who whom how when where why can "
    "could would should will shall may might must my me any some something there their they them than then so "
    "not no yes please hi hello many much".split()
)


def _stem(word: str) -> str:
    """Very light English stemming so "time"/"times" and "refund"/"refunds" match."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _tokens(text: str) -> list[str]:
    return [_stem(w) for w in re.findall(r"\w+", text.lower()) if w not in STOPWORDS]


class PolicySearch:
    def __init__(self, sections: list[Section] | None = None):
        self.sections = sections or load_sections()
        self.index = BM25Plus([_tokens(f"{s.title} {s.text}") for s in self.sections])

    def search(self, query: str, k: int = 3) -> list[Section]:
        scores = self.index.get_scores(_tokens(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.sections[i] for i in ranked[:k]]
