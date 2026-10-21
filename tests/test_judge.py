from types import SimpleNamespace

from eval import judge

RETRIEVED = [
    ("2021-β-Cell autophagy in the pathogenesis of type 1 diabetes.pdf", 1),
    ("2021-β-Cell autophagy in the pathogenesis of type 1 diabetes.pdf", 5),
    ("2022-Inside the β cell- molecular stress response pathways in diabetes pathogenesis.pdf", 6),
]


def check(answer):
    return judge.check_citations(judge.parse_citations(answer), RETRIEVED)


# --- parsing

def test_parses_file_citations_with_pages_and_truncated_names():
    cites = judge.parse_citations("Autophagy matters (2021-β-Cell autophagy in the pathogenesis of type 1 diabetes.pdf, p. 1; "
                                  "2022-Inside the β cell, pp. 6, 7).")
    assert [(c.name, c.pages) for c in cites] == [
        ("2021-β-Cell autophagy in the pathogenesis of type 1 diabetes.pdf", [1]),
        ("2022-Inside the β cell", [6, 7]),
    ]


def test_parses_page_ranges():
    assert judge.parse_citations("(some paper title, pp. 3-5)")[0].pages == [3, 4, 5]


def test_parses_author_year_references_without_pages():
    names = [c.name for c in judge.parse_citations("Known (Smith et al., 2019) and (Parzych & Klionsky, 2014).")]
    assert names == ["Smith et al., 2019", "Parzych & Klionsky, 2014"]


def test_ignores_ordinary_parentheses():
    assert judge.parse_citations("Beta cells (the insulin-producing cells) matter (see below).") == []


def test_file_names_starting_with_a_year_are_not_mistaken_for_author_year():
    cites = judge.parse_citations("(2021-β-Cell autophagy in the pathogenesis of type 1 diabetes, p. 1)")
    assert len(cites) == 1 and cites[0].pages == [1]


# --- checking against what was retrieved

def counts(**kw):
    base = {"total": 0, "grounded": 0, "page_correct": 0, "outside": 0, "unmatched": 0, "unnamed": 0}
    return {**base, **kw}


def test_grounded_and_page_correct_citations():
    result = check("A (2021-β-Cell autophagy in the pathogenesis of type 1 diabetes, p. 5). B (2022-Inside the β cell, p. 6).")
    assert result == counts(total=2, grounded=2, page_correct=2)


def test_wrong_page_is_grounded_but_not_page_correct():
    assert check("A (2022-Inside the β cell, p. 99).") == counts(total=1, grounded=1)


def test_outside_author_year_and_unretrieved_papers_are_told_apart():
    result = check("A (2025-Some other paper that was never retrieved, p. 2). B (Hiller et al., 2021, p. 16).")
    assert result == counts(total=2, outside=1, unmatched=1)


def test_source_prefix_is_ignored():
    assert check("A (source: 2022-Inside the β cell- molecular stress response pathways in diabetes pathogenesis.pdf, p. 6)") == counts(total=1, grounded=1, page_correct=1)


def test_page_only_citation_is_unnamed_not_outside():
    assert check("A (p. 1, p. 2).") == counts(total=1, unnamed=1)


def test_very_short_names_do_not_match_by_accident():
    assert check("A (2021, p. 1).") == counts(total=1, unmatched=1)


# --- judges, with a fake chat model

class FakeJudgeLLM:
    def __init__(self, verdict):
        self.verdict = verdict

    def with_structured_output(self, schema):
        return SimpleNamespace(invoke=lambda prompt: self.verdict)


def test_faithfulness_score_is_the_supported_fraction_and_lists_unsupported_claims():
    verdict = judge.FaithfulnessVerdict(claims=[
        judge.Claim(text="X is a protein", supported=True),
        judge.Claim(text="X cures diabetes", supported=False),
        judge.Claim(text="X is found in beta cells", supported=True),
        judge.Claim(text="X was discovered in 1990", supported=False),
    ])
    result = judge.judge_faithfulness(FakeJudgeLLM(verdict), "context", "answer")
    assert result == {"score": 0.5, "claims": 4, "unsupported": ["X cures diabetes", "X was discovered in 1990"]}


def test_answer_with_no_claims_counts_as_faithful():
    assert judge.judge_faithfulness(FakeJudgeLLM(judge.FaithfulnessVerdict(claims=[])), "c", "a")["score"] == 1.0


def test_relevancy_returns_score_and_reason():
    verdict = judge.RelevancyVerdict(score=4, reason="mostly on point")
    assert judge.judge_relevancy(FakeJudgeLLM(verdict), "q", "a") == {"score": 4, "reason": "mostly on point"}
