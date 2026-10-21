import pytest
from langchain_core.documents import Document

import config
import ingest


class FakeTable:
    """Stands in for the pgvector table: `id` is a global primary key, lookups are per collection
    (matching langchain-postgres, where add_documents upserts by id and get_by_ids filters by collection)."""

    def __init__(self):
        self.rows = {}  # id -> (collection, Document)
        self.add_calls = 0

    def store_for(self, collection):
        return FakeStore(self, collection)


class FakeStore:
    def __init__(self, table, collection):
        self.table, self.collection = table, collection

    def get_by_ids(self, ids):
        return [self.table.rows[i][1] for i in ids if i in self.table.rows and self.table.rows[i][0] == self.collection]

    def add_documents(self, docs, ids):
        self.table.add_calls += 1
        for doc, id_ in zip(docs, ids):
            self.table.rows[id_] = (self.collection, doc)


@pytest.fixture
def table(monkeypatch):
    table = FakeTable()
    monkeypatch.setattr(ingest, "build_store", lambda reset=False: table.store_for(config.COLLECTION_NAME))
    monkeypatch.setattr(config, "CHUNK_SIZE", 100)
    monkeypatch.setattr(config, "CHUNK_OVERLAP", 10)
    monkeypatch.setattr(config, "COLLECTION_NAME", "col_a")
    return table


@pytest.fixture
def fake_pdf_text(monkeypatch):
    """PyPDFLoader replacement: every PDF 'contains' two pages of text derived from its file name."""

    def fake_load_pages(path):
        text = ("lorem ipsum dolor sit amet " * 8)
        return [Document(page_content=text, metadata={"page": p, "source": str(path)}) for p in range(2)]

    monkeypatch.setattr(ingest, "load_pages", fake_load_pages)


def make_pdf(folder, name, content=b"%PDF-fake-1"):
    path = folder / name
    path.write_bytes(content)
    return path


# --- file_hash / find_pdfs

def test_file_hash_depends_on_content_not_name(tmp_path):
    a = make_pdf(tmp_path, "a.pdf", b"same")
    b = make_pdf(tmp_path, "b.pdf", b"same")
    c = make_pdf(tmp_path, "c.pdf", b"different")
    assert ingest.file_hash(a) == ingest.file_hash(b)
    assert ingest.file_hash(a) != ingest.file_hash(c)


def test_find_pdfs_searches_subfolders_and_ignores_other_file_types(tmp_path):
    make_pdf(tmp_path, "b.pdf")
    make_pdf(tmp_path, "a.pdf")
    make_pdf(tmp_path, "notes.txt")
    (tmp_path / "sub" / "deeper").mkdir(parents=True)
    make_pdf(tmp_path / "sub", "nested.pdf")
    make_pdf(tmp_path / "sub" / "deeper", "deep.pdf")
    found = [p.relative_to(tmp_path).as_posix() for p in ingest.find_pdfs(str(tmp_path))]
    assert found == ["a.pdf", "b.pdf", "sub/deeper/deep.pdf", "sub/nested.pdf"]


def test_find_pdfs_matches_extension_case_insensitively(tmp_path):
    make_pdf(tmp_path, "UPPER.PDF")
    make_pdf(tmp_path, "Mixed.Pdf")
    assert [p.name for p in ingest.find_pdfs(str(tmp_path))] == ["Mixed.Pdf", "UPPER.PDF"]


def test_find_pdfs_skips_hidden_folders_and_macos_resource_files(tmp_path):
    make_pdf(tmp_path, "real.pdf")
    make_pdf(tmp_path, "._real.pdf")
    (tmp_path / ".git").mkdir()
    make_pdf(tmp_path / ".git", "hidden.pdf")
    assert [p.name for p in ingest.find_pdfs(str(tmp_path))] == ["real.pdf"]


def test_find_pdfs_exits_when_folder_has_no_pdfs(tmp_path):
    with pytest.raises(SystemExit):
        ingest.find_pdfs(str(tmp_path))


# --- ingest(): storing and skipping

def test_ingest_stores_chunks_with_collection_and_hash_ids(tmp_path, table, fake_pdf_text):
    pdf = make_pdf(tmp_path, "paper.pdf")
    ingest.ingest(str(tmp_path))

    prefix = f"col_a:{ingest.file_hash(pdf)}"
    ids = sorted(table.rows, key=lambda i: int(i.rsplit(":", 1)[1]))
    assert len(ids) > 1
    assert ids == [f"{prefix}:{n}" for n in range(len(ids))]
    assert all(doc.metadata["source_file"] == "paper.pdf" for _, doc in table.rows.values())


def test_rerun_skips_already_ingested_file(tmp_path, table, fake_pdf_text, capsys):
    make_pdf(tmp_path, "paper.pdf")
    ingest.ingest(str(tmp_path))
    rows_after_first = dict(table.rows)

    ingest.ingest(str(tmp_path))

    assert table.add_calls == 1
    assert table.rows == rows_after_first
    assert "Skipping (already ingested): paper.pdf" in capsys.readouterr().out


def test_renamed_file_is_still_skipped(tmp_path, table, fake_pdf_text):
    pdf = make_pdf(tmp_path, "old-name.pdf")
    ingest.ingest(str(tmp_path))
    pdf.rename(tmp_path / "new-name.pdf")

    ingest.ingest(str(tmp_path))

    assert table.add_calls == 1


def test_edited_file_is_ingested_again(tmp_path, table, fake_pdf_text):
    pdf = make_pdf(tmp_path, "paper.pdf", b"version 1")
    ingest.ingest(str(tmp_path))
    pdf.write_bytes(b"version 2")

    ingest.ingest(str(tmp_path))

    assert table.add_calls == 2


def test_same_pdf_in_two_collections_does_not_collide(tmp_path, table, fake_pdf_text, monkeypatch):
    make_pdf(tmp_path, "paper.pdf")
    ingest.ingest(str(tmp_path))
    first = {i for i, (col, _) in table.rows.items() if col == "col_a"}

    monkeypatch.setattr(config, "COLLECTION_NAME", "col_b")
    ingest.ingest(str(tmp_path))

    assert {i for i, (col, _) in table.rows.items() if col == "col_a"} == first  # untouched
    assert {i for i, (col, _) in table.rows.items() if col == "col_b"}
    assert table.add_calls == 2  # col_b was not mistaken for "already ingested"


def test_file_without_extractable_text_is_not_stored(tmp_path, table, monkeypatch, capsys):
    make_pdf(tmp_path, "scan.pdf")
    monkeypatch.setattr(ingest, "load_pages", lambda path: [])

    ingest.ingest(str(tmp_path))

    assert table.rows == {}
    assert "No extractable text in scan.pdf" in capsys.readouterr().out


def test_reset_flag_is_passed_to_build_store(tmp_path, fake_pdf_text, monkeypatch):
    make_pdf(tmp_path, "paper.pdf")
    seen = []
    table = FakeTable()
    monkeypatch.setattr(ingest, "build_store", lambda reset=False: seen.append(reset) or table.store_for("col_a"))

    ingest.ingest(str(tmp_path), reset=True)

    assert seen == [True]


def test_sources_are_cited_by_relative_path_so_same_named_files_stay_distinct(tmp_path, table, fake_pdf_text):
    (tmp_path / "2023").mkdir()
    (tmp_path / "2024").mkdir()
    make_pdf(tmp_path / "2023", "report.pdf", b"first")
    make_pdf(tmp_path / "2024", "report.pdf", b"second")
    ingest.ingest(str(tmp_path))
    assert {doc.metadata["source_file"] for _, doc in table.rows.values()} == {"2023/report.pdf", "2024/report.pdf"}


def test_flat_folder_citations_are_unchanged(tmp_path, table, fake_pdf_text):
    make_pdf(tmp_path, "paper.pdf")
    ingest.ingest(str(tmp_path))
    assert {doc.metadata["source_file"] for _, doc in table.rows.values()} == {"paper.pdf"}
