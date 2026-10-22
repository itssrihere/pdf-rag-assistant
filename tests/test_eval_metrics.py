from eval import metrics

RETRIEVED = [("a.pdf", 3), ("b.pdf", 1), ("a.pdf", 7), ("c.pdf", None)]


def test_first_file_rank():
    assert metrics.first_file_rank(RETRIEVED, "a.pdf") == 1
    assert metrics.first_file_rank(RETRIEVED, "c.pdf") == 4
    assert metrics.first_file_rank(RETRIEVED, "z.pdf") is None


def test_first_page_rank_requires_both_file_and_page():
    assert metrics.first_page_rank(RETRIEVED, "a.pdf", [7]) == 3
    assert metrics.first_page_rank(RETRIEVED, "a.pdf", [3, 7]) == 1
    assert metrics.first_page_rank(RETRIEVED, "b.pdf", [3]) is None  # page 3 exists, but in another file
    assert metrics.first_page_rank(RETRIEVED, "a.pdf", [99]) is None


def test_hit_at_k_boundaries():
    assert metrics.hit_at_k(3, 3) and not metrics.hit_at_k(3, 2)
    assert not metrics.hit_at_k(None, 100)


def test_reciprocal_rank():
    assert metrics.reciprocal_rank(1) == 1.0
    assert metrics.reciprocal_rank(4) == 0.25
    assert metrics.reciprocal_rank(None) == 0.0


def test_file_recall_at_k_counts_distinct_expected_files_only():
    assert metrics.file_recall_at_k(RETRIEVED, ["a.pdf", "b.pdf", "z.pdf"], 4) == 2 / 3
    assert metrics.file_recall_at_k(RETRIEVED, ["a.pdf", "b.pdf", "z.pdf"], 1) == 1 / 3
    assert metrics.file_recall_at_k(RETRIEVED, [], 4) == 0.0


def test_file_precision_at_k():
    assert metrics.file_precision_at_k(RETRIEVED, ["a.pdf"], 4) == 0.5    # a.pdf holds 2 of the 4 chunks
    assert metrics.file_precision_at_k(RETRIEVED, ["a.pdf", "b.pdf"], 2) == 1.0
    assert metrics.file_precision_at_k(RETRIEVED, ["z.pdf"], 4) == 0.0
    assert metrics.file_precision_at_k([], ["a.pdf"], 4) == 0.0


def test_page_precision_at_k_is_strict_about_pages():
    assert metrics.page_precision_at_k(RETRIEVED, "a.pdf", [3], 4) == 0.25   # only the page-3 chunk counts
    assert metrics.page_precision_at_k(RETRIEVED, "a.pdf", [3, 7], 4) == 0.5


def test_any_location_metrics_accept_a_fact_documented_in_two_places():
    retrieved = [("guide.pdf", 9), ("appendix.pdf", 33), ("other.pdf", 1)]
    locations = {"guide.pdf": [4], "appendix.pdf": [33, 34]}
    assert metrics.first_file_rank_any(retrieved, set(locations)) == 1
    assert metrics.first_page_rank_any(retrieved, locations) == 2          # guide.pdf p9 is the wrong page
    assert metrics.first_page_rank_any([("other.pdf", 4)], locations) is None
    assert metrics.page_precision_at_k_any(retrieved, locations, 3) == 1 / 3
    assert metrics.first_file_rank_any([("other.pdf", 1)], set(locations)) is None
