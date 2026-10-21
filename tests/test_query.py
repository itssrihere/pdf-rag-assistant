from types import SimpleNamespace

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

import query


def doc(text="text", file="a.pdf", page=None):
    metadata = {"source_file": file}
    if page is not None:
        metadata["page"] = page
    return Document(page_content=text, metadata=metadata)


# --- source / context formatting

def test_format_sources_groups_pages_per_file_and_sorts():
    docs = [doc(file="b.pdf", page=0), doc(file="a.pdf", page=4), doc(file="a.pdf", page=2), doc(file="a.pdf", page=2)]
    assert query.format_sources(docs) == "a.pdf (pp. 3, 5); b.pdf (p. 1)"


def test_format_sources_without_page_metadata():
    assert query.format_sources([doc(file="c.pdf")]) == "c.pdf"


def test_format_sources_empty():
    assert query.format_sources([]) == ""


def test_page_numbers_are_one_based():
    assert query.format_sources([doc(page=0)]) == "a.pdf (p. 1)"


def test_format_context_labels_each_chunk_with_file_and_page():
    context = query.format_context([doc("alpha", "a.pdf", page=2), doc("beta", "b.pdf")])
    assert context == "[a.pdf, p. 3]\nalpha\n\n[b.pdf]\nbeta"


def test_unknown_source_file_is_labelled_unknown():
    assert query.format_sources([Document(page_content="x", metadata={})]) == "unknown"


# --- history trimming

def messages(n_pairs):
    out = []
    for i in range(n_pairs):
        out += [HumanMessage(f"q{i}"), AIMessage(f"a{i}")]
    return out


def test_trim_history_keeps_last_n_pairs():
    trimmed = query.trim_history(messages(4), 2)
    assert [m.content for m in trimmed] == ["q2", "a2", "q3", "a3"]


def test_trim_history_short_history_unchanged():
    assert query.trim_history(messages(1), 5) == messages(1)


def test_trim_history_zero_turns_disables_memory():
    assert query.trim_history(messages(3), 0) == []


# --- ask(): follow-up rewriting

class FakeRetriever:
    def __init__(self, docs):
        self.docs, self.queries = docs, []

    def invoke(self, q):
        self.queries.append(q)
        return self.docs


class FakeLLM:
    """Returns canned replies in order and records the messages it was sent."""

    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    def invoke(self, msgs):
        self.calls.append(msgs)
        return SimpleNamespace(content=self.replies[len(self.calls) - 1])


def test_first_question_skips_rewrite_and_searches_verbatim(capsys):
    retriever, llm = FakeRetriever([doc("ctx", page=1)]), FakeLLM("the answer")

    answer = query.ask(retriever, llm, "What is X?")

    assert answer == "the answer"
    assert retriever.queries == ["What is X?"]
    assert len(llm.calls) == 1
    out = capsys.readouterr().out
    assert "the answer" in out and "Sources: a.pdf (p. 2)" in out


def test_follow_up_is_rewritten_before_retrieval_but_answered_as_asked():
    history = [HumanMessage("What is X?"), AIMessage("X is a thing.")]
    retriever, llm = FakeRetriever([doc()]), FakeLLM("What is X's size?", "It is big.")

    answer = query.ask(retriever, llm, "and its size?", history)

    assert answer == "It is big."
    assert retriever.queries == ["What is X's size?"]  # rewritten question drives retrieval
    assert len(llm.calls) == 2
    answer_prompt = " ".join(str(m.content) for m in llm.calls[1])
    assert "and its size?" in answer_prompt and "X is a thing." in answer_prompt


def test_no_sources_line_when_nothing_retrieved(capsys):
    query.ask(FakeRetriever([]), FakeLLM("I don't know."), "Q?")
    assert "Sources:" not in capsys.readouterr().out
