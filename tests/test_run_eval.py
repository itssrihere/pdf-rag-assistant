from eval import run_eval


def test_single_location_questions_still_work():
    q = {"expected_file": "a.pdf", "expected_pages": [3]}
    assert run_eval.question_locations(q) == {"a.pdf": [3]}


def test_expected_locations_override_the_single_file_fields():
    q = {"expected_locations": [{"file": "guide.pdf", "pages": [2]}, {"file": "appendix.pdf", "pages": [9, 10]}]}
    assert run_eval.question_locations(q) == {"guide.pdf": [2], "appendix.pdf": [9, 10]}


def test_a_hit_in_either_documented_location_counts():
    q = {"category": "exact_term", "expected_locations": [{"file": "guide.pdf", "pages": [2]}, {"file": "appendix.pdf", "pages": [9]}]}
    result = run_eval.score_question(q, {4: [("noise.pdf", 1), ("appendix.pdf", 9)], 8: [("noise.pdf", 1), ("appendix.pdf", 9)]}, [4, 8])
    assert result["file_hit@4"] and result["page_hit@4"] and result["page_rank"] == 2


def test_load_questions_reads_any_file(tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_text('{"id": "x", "category": "specific"}\n\n')
    assert run_eval.load_questions(path) == [{"id": "x", "category": "specific"}]
