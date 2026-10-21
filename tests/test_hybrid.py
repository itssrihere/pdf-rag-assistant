from langchain_core.documents import Document

import hybrid


def doc(id_, text):
    return Document(id=id_, page_content=text, metadata={"source_file": id_ + ".pdf"})


def test_tokenize_lowercases_splits_on_punctuation_and_drops_stopwords():
    assert hybrid.tokenize("What is the JSM-7800F used for?") == ["jsm", "7800f", "used"]
    assert hybrid.tokenize("rs1936208, PA5-26020") == ["rs1936208", "pa5", "26020"]


def test_rrf_rewards_agreement_between_lists():
    assert hybrid.rrf_fuse([["a", "b", "c"], ["b", "c", "a"]], top=3)[0] == "b"


def test_rrf_includes_items_found_by_only_one_list():
    fused = hybrid.rrf_fuse([["a", "b"], ["x"]])
    assert set(fused) == {"a", "b", "x"}


def test_rrf_respects_top_and_breaks_ties_by_first_list_order():
    assert hybrid.rrf_fuse([["a", "b"], ["c"]], top=2) == ["a", "c"]  # a and c both rank 1; a was seen first
    assert hybrid.rrf_fuse([[]]) == []


def test_keyword_index_finds_an_exact_rare_token():
    index = hybrid.KeywordIndex([
        ("p1", doc("p1", "autophagy in beta cells and diabetes")),
        ("p2", doc("p2", "the SNP rs1936208 lies between two genes")),
        ("p3", doc("p3", "macrophages and inflammation")),
    ])
    assert index.search("Which region contains rs1936208?", 3) == ["p2"]


def test_keyword_index_returns_nothing_when_no_word_matches():
    index = hybrid.KeywordIndex([("p1", doc("p1", "beta cells")), ("p2", doc("p2", "macrophages"))])
    assert index.search("quantum chromodynamics", 5) == []
    assert index.search("the of and", 5) == []  # only stopwords


class FakeStore:
    def __init__(self, docs):
        self.docs = docs

    def similarity_search(self, question, k):
        return self.docs[:k]


def test_hybrid_surfaces_a_keyword_match_that_vector_search_ranks_low():
    docs = [doc(f"d{i}", f"generic text about cells number {i}") for i in range(10)]
    target = doc("target", "the PA5-26020 antibody was used at 1:200")
    store = FakeStore(docs + [target])  # vector search ranks the target last
    index = hybrid.KeywordIndex([(d.id, d) for d in docs + [target]])
    retriever = hybrid.HybridRetriever(store, index, fetch_k=11)

    assert "target" not in [d.id for d in retriever.search("PA5-26020 antibody", k=3, mode="dense")]
    assert "target" in [d.id for d in retriever.search("PA5-26020 antibody", k=3, mode="hybrid")]
    assert [d.id for d in retriever.search("PA5-26020 antibody", k=1, mode="bm25")] == ["target"]


def test_hybrid_includes_chunks_only_the_keyword_index_found():
    store = FakeStore([doc("v1", "something else")])
    keyword_only = doc("k1", "unique-token-xyz appears here")
    filler = [(f"f{i}", doc(f"f{i}", f"filler text {i} about cells")) for i in range(5)]  # BM25 weights need N > 2
    index = hybrid.KeywordIndex([("v1", doc("v1", "something else")), ("k1", keyword_only)] + filler)
    retriever = hybrid.HybridRetriever(store, index, fetch_k=5)
    assert {d.id for d in retriever.search("unique-token-xyz", k=5)} == {"v1", "k1"}  # dense found v1, BM25 found k1


def test_rrf_weights_let_one_list_dominate():
    lists = [["a", "b"], ["x"]]
    assert hybrid.rrf_fuse(lists, top=1)[0] == "a"                       # equal weight: rank-1 tie goes to the first list
    assert hybrid.rrf_fuse(lists, top=1, weights=[1.0, 5.0])[0] == "x"   # a heavier second list wins
