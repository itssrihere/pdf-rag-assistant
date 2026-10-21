"""Answer-level checks: citation verification (deterministic) and LLM-judged faithfulness and relevancy."""

import re
from dataclasses import dataclass

from pydantic import BaseModel, Field

# "... (2021-Some paper title.pdf, p. 3; Other paper, pp. 5, 7)": split a parenthetical on ";" and read "name, p. N"
PAREN = re.compile(r"\(([^()]+)\)")
PAGED = re.compile(r"^(?P<name>.+?),\s*pp?\.\s*(?P<pages>\d[\d,\s\-–]*)$", re.I)
# author-year references to outside works: "Hiller et al., 2021", "Parzych & Klionsky, 2014", "Smith, 2019"
AUTHOR_YEAR = re.compile(r"^[A-Z][\w'’\-]+(?:\s+(?:et al\.?|&\s+[A-Z][\w'’\-]+|and\s+[A-Z][\w'’\-]+))?,?\s*(?:19|20)\d{2}\b")


SOURCE_PREFIX = re.compile(r"^(?:sources?|see|from)\s*:?\s+", re.I)   # "(source: file.pdf, p. 3)"
PAGE_ONLY = re.compile(r"^pp?\.?\s*\d", re.I)                          # "(p. 1, p. 2)": no file named


@dataclass
class Citation:
    name: str
    pages: list[int]
    kind: str = "paged"   # "paged" (file/title + page), "author_year" (outside work) or "unnamed" (page only)


def _pages(text: str) -> list[int]:
    pages = []
    for part in re.split(r"[,\s]+", text.strip()):
        span = re.fullmatch(r"(\d+)[\-–](\d+)", part)
        if span:
            pages += range(int(span[1]), int(span[2]) + 1)
        elif part.isdigit():
            pages.append(int(part))
    return pages


def parse_citations(answer: str) -> list[Citation]:
    """Citations written as "(name, p. N)" plus bare author-year references such as "(Smith et al., 2019)"."""
    found = []
    for group in PAREN.findall(answer):
        for segment in (s.strip() for s in group.split(";")):
            paged = PAGED.match(segment)
            if paged:
                name = SOURCE_PREFIX.sub("", paged["name"].strip())
                kind = "unnamed" if PAGE_ONLY.match(name) else "author_year" if AUTHOR_YEAR.match(name) else "paged"
                found.append(Citation(name, _pages(paged["pages"]), kind))
            elif AUTHOR_YEAR.match(segment):
                found.append(Citation(segment, [], "author_year"))
    return found


def _norm(name: str) -> str:
    return re.sub(r"\s+", " ", name.lower().removesuffix(".pdf")).strip()


def check_citations(citations: list[Citation], retrieved: list[tuple[str, int | None]]) -> dict:
    """Counts, per answer: *grounded* citations (name matches, as a prefix since the model often truncates titles, a
    paper that was actually retrieved), *page_correct* ones (every cited page was among that paper's retrieved pages),
    *outside* (author-year references to works that are not in the collection), *unmatched* (a name that matches no
    retrieved paper) and *unnamed* (a page number with no file)."""
    stems = {_norm(name): {p for n, p in retrieved if n == name and p} for name in {n for n, _ in retrieved}}
    counts = {"total": len(citations), "grounded": 0, "page_correct": 0, "outside": 0, "unmatched": 0, "unnamed": 0}
    for citation in citations:
        if citation.kind == "author_year":
            counts["outside"] += 1
        elif citation.kind == "unnamed":
            counts["unnamed"] += 1
        else:
            cited = _norm(citation.name)
            match = next((s for s in stems if len(cited) >= 8 and (s.startswith(cited) or cited.startswith(s))), None)
            if match is None:
                counts["unmatched"] += 1
                continue
            counts["grounded"] += 1
            if citation.pages and all(p in stems[match] for p in citation.pages):
                counts["page_correct"] += 1
    return counts


class Claim(BaseModel):
    text: str = Field(description="One factual claim made by the answer")
    supported: bool = Field(description="True only if the context directly supports the claim")


class FaithfulnessVerdict(BaseModel):
    claims: list[Claim]


class RelevancyVerdict(BaseModel):
    score: int = Field(ge=1, le=5, description="1 = off-topic, 5 = directly and completely answers the question")
    reason: str


FAITHFULNESS_PROMPT = (
    "You check whether an answer is supported by the context it was written from. Split the answer into its "
    "individual factual claims (ignore citations, hedges and filler). For each claim, mark it supported only if the "
    "context directly states or clearly implies it. Judge against the context only; ignore your own knowledge. "
    "If the answer says the information is not available, it has no claims.\n\nCONTEXT:\n{context}\n\nANSWER:\n{answer}"
)
RELEVANCY_PROMPT = (
    "Rate how directly and completely the answer addresses the question, from 1 (off-topic or evasive) to 5 "
    "(directly and completely answers it). Ignore whether the facts are correct. An answer that correctly says the "
    "information is unavailable should score 5 only if the question truly cannot be answered from the documents."
    "\n\nQUESTION: {question}\n\nANSWER:\n{answer}"
)


def judge_faithfulness(llm, context: str, answer: str) -> dict:
    """`llm` is a chat model; returns the supported fraction and the unsupported claims."""
    verdict = llm.with_structured_output(FaithfulnessVerdict).invoke(
        FAITHFULNESS_PROMPT.format(context=context, answer=answer))
    claims = verdict.claims
    supported = sum(c.supported for c in claims)
    return {"score": supported / len(claims) if claims else 1.0, "claims": len(claims),
            "unsupported": [c.text for c in claims if not c.supported]}


def judge_relevancy(llm, question: str, answer: str) -> dict:
    verdict = llm.with_structured_output(RelevancyVerdict).invoke(
        RELEVANCY_PROMPT.format(question=question, answer=answer))
    return {"score": verdict.score, "reason": verdict.reason}
