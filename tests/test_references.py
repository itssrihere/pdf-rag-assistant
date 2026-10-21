from langchain_core.documents import Document

import config
import ingest

BODY = "Cells were treated and measured. Results are shown in Figure 2. "
ENTRY = "{n}. Smith AB, Jones C, et al. A study of cells. J Cell Biol. 2019;70(8):1885-1897.\n"


def page(text, index):
    return Document(page_content=text, metadata={"page": index, "source_file": "p.pdf"})


def ref_list(first, count=12):
    return "".join(ENTRY.format(n=first + i) for i in range(count))


def body_pages(n):
    return [page(BODY * 5, i) for i in range(n)]


def texts(pages):
    return [p.page_content for p in pages]


def test_paper_without_references_is_returned_untouched():
    pages = body_pages(5)
    assert ingest.remove_references(pages) == pages


def test_text_after_heading_is_cut_and_later_reference_pages_dropped():
    pages = body_pages(6) + [page("Discussion ends here.\nReferences\n" + ref_list(1), 6), page(ref_list(13), 7)]
    kept = ingest.remove_references(pages)
    assert len(kept) == 7
    assert kept[-1].page_content.strip() == "Discussion ends here."
    assert not any("J Cell Biol" in t for t in texts(kept))


def test_page_numbers_survive_for_kept_pages():
    pages = body_pages(6) + [page("References\n" + ref_list(1), 6)]
    kept = ingest.remove_references(pages)
    assert [p.metadata["page"] for p in kept] == [0, 1, 2, 3, 4, 5]


def test_heading_that_starts_a_page_drops_the_whole_page():
    pages = body_pages(6) + [page("REFERENCES\n" + ref_list(1), 6)]
    assert len(ingest.remove_references(pages)) == 6


def test_science_style_heading_variant():
    pages = body_pages(6) + [page("REFERENCES AND NOTES\n" + ref_list(1), 6)]
    assert len(ingest.remove_references(pages)) == 6


def test_content_after_the_reference_list_is_kept():
    legends = page("Figure 1. Overview of the study design. Figure 2. Survival curves.", 8)
    pages = body_pages(6) + [page("References\n" + ref_list(1), 6), page(ref_list(13), 7), legends]
    kept = ingest.remove_references(pages)
    assert legends in kept and len(kept) == 6 + 1


def test_references_mentioned_early_in_the_paper_are_ignored():
    pages = [page("Intro\nReferences\nsee below", 0)] + body_pages(9)
    assert ingest.remove_references(pages) == pages


def test_headingless_numbered_list_at_the_end_is_dropped():
    pages = body_pages(6) + [page(ref_list(1, 10), 6), page(ref_list(11, 25), 7)]
    kept = ingest.remove_references(pages)
    assert [p.metadata["page"] for p in kept] == [0, 1, 2, 3, 4, 5]


def test_body_text_with_a_few_citations_is_not_mistaken_for_references():
    cited = page("Prior work (Smith et al., 2019) and (Lee et al., 2020) support this. " * 3, 7)
    pages = body_pages(6) + [page("References\n" + ref_list(1), 6), cited]
    assert cited in ingest.remove_references(pages)


# --- wired into ingest()

def run_ingest(tmp_path, monkeypatch, **kwargs):
    stored = []
    pages = body_pages(4) + [page("References\n" + ref_list(1), 4)]

    class FakeStore:
        def get_by_ids(self, ids):
            return []

        def add_documents(self, docs, ids):
            stored.extend(docs)

    (tmp_path / "paper.pdf").write_bytes(b"%PDF")
    monkeypatch.setattr(ingest, "build_store", lambda reset=False: FakeStore())
    monkeypatch.setattr(ingest, "load_pages", lambda path: pages)
    monkeypatch.setattr(config, "CHUNK_SIZE", 200)
    ingest.ingest(str(tmp_path), **kwargs)
    return stored


def test_ingest_drops_bibliography_by_default(tmp_path, monkeypatch):
    stored = run_ingest(tmp_path, monkeypatch)
    assert stored and not any("J Cell Biol" in d.page_content for d in stored)


def test_keep_references_flag_embeds_them(tmp_path, monkeypatch):
    stored = run_ingest(tmp_path, monkeypatch, keep_references=True)
    assert any("J Cell Biol" in d.page_content for d in stored)
