"""
Scans DOCS_DIR for PDFs, chunks them, embeds, and stores in pgvector.
PDFs already stored (matched by content hash) are skipped.
Usage: python ingest.py [--reset]
"""

import argparse
import hashlib
import re
import sys
from pathlib import Path

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from pdf_text import load_pages


REFERENCE_HEADING = re.compile(
    r"^\s*(?:\d+\.?\s*)?(?:references?(?: and notes| cited)?|bibliography|literature cited|works cited)\s*:?\s*$",
    re.I | re.M,
)
CITATION_CUE = re.compile(  # reference-list fragments: "2021;70(8):", "et al." (not "et al., 2019)"), "doi:", "Smith AB,"
    r"\b(?:19|20)\d{2}\s*[;,.]?\s*\d{1,4}\s*[(:]|\b(?:19|20)\d{2}\b\s*;\s*\d|et al\.(?!,?\s*\(?(?:19|20)\d{2})|doi\.org|doi:|\bPMID\b"
    r"|\b[A-Z][a-z]+\s+[A-Z]{1,3}[,.]\s",
    re.I,
)
NUMBERED_ENTRY = re.compile(r"(?m)^\s*\d{1,3}\.\s+[A-Z][\w'’-]+\s+[A-Z]{1,3}[,.]")  # "12. Smith AB,"
MIN_CUES_FOR_REFERENCE_PAGE = 6  # figure legends/tables score 0-2; reference pages score 13+
MIN_ENTRIES_FOR_HEADINGLESS_LIST = 8
HEADING_SEARCH_FROM = 0.3  # ignore "References" lines in the first 30% of a document


def find_reference_start(pages: list[Document]) -> tuple[int, int] | None:
    """(page index, character offset) where the bibliography begins, or None.

    Uses a "References"-style heading in the latter 70% of the document; failing that, a run of
    numbered-entry pages at the very end (some journals print no heading)."""
    for i in range(int(HEADING_SEARCH_FROM * len(pages)), len(pages)):
        match = REFERENCE_HEADING.search(pages[i].page_content)
        if match:
            return i, match.start()
    start = None
    for i in range(len(pages) - 1, -1, -1):
        if len(NUMBERED_ENTRY.findall(pages[i].page_content)) < MIN_ENTRIES_FOR_HEADINGLESS_LIST:
            break
        start = i
    return (start, 0) if start is not None else None


def remove_references(pages: list[Document]) -> list[Document]:
    """Drops the bibliography so reference-list entries are never embedded or retrieved.

    Text after the heading is cut on its page, and later pages are dropped only if they are
    densely bibliographic; figure legends, tables and methods that follow the list are kept.
    Kept pages retain their original page-number metadata."""
    start = find_reference_start(pages)
    if start is None:
        return pages
    first, offset = start
    kept = list(pages[:first])
    head_text = pages[first].page_content[:offset]
    if head_text.strip():
        kept.append(Document(page_content=head_text, metadata=pages[first].metadata))
    kept += [p for p in pages[first + 1:] if len(CITATION_CUE.findall(p.page_content)) < MIN_CUES_FOR_REFERENCE_PAGE]
    return kept


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_pdfs(docs_dir: str) -> list[Path]:
    root = Path(docs_dir)
    pdf_files = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() == ".pdf"
        and not any(part.startswith(".") for part in path.relative_to(root).parts)  # hidden dirs, "._x.pdf" junk
    )
    if not pdf_files:
        print(f"No PDF files found in {docs_dir}")
        sys.exit(1)
    return pdf_files


def chunk_documents(docs: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    return splitter.split_documents(docs)


def build_store(reset: bool = False) -> PGVector:
    embeddings = OpenAIEmbeddings(
        openai_api_key=config.OPENAI_API_KEY,
        max_retries=6,  # ride out brief network drops during long ingests
        # Small requests: on some networks, uploads over ~20 KB randomly fail with SSL "bad record mac".
        chunk_size=16,  # chunks per embeddings request (library default: 1000)
    )
    store = PGVector(
        embeddings=embeddings,
        collection_name=config.COLLECTION_NAME,
        connection=config.DB_URL,
        use_jsonb=True,
    )
    if reset:
        print("Resetting existing collection...")
        store.delete_collection()
        store.create_collection()
    return store


def ingest(docs_dir: str, reset: bool = False, keep_references: bool = False) -> None:
    """Embeds and stores each PDF, skipping files whose content is already stored.

    Chunk IDs are "<collection>:<sha256 of file>:<chunk index>", so re-running is idempotent,
    a file that changed on disk is ingested again under its new hash, and the same PDF can
    live in several collections (IDs are a global primary key, hence the collection prefix).
    Bibliographies are dropped unless `keep_references` is set. Files already stored are skipped
    as-is, so changing this setting needs `reset=True` to take effect on existing data."""
    pdf_files = find_pdfs(docs_dir)
    print(f"Found {len(pdf_files)} PDF(s), including subfolders")
    store = build_store(reset=reset)

    added = skipped = 0
    for pdf in pdf_files:
        name = pdf.relative_to(docs_dir).as_posix()  # cited as e.g. "reports/q3.pdf"; a top-level file is just its name
        id_prefix = f"{config.COLLECTION_NAME}:{file_hash(pdf)}"
        if store.get_by_ids([f"{id_prefix}:0"]):
            print(f"  Skipping (already ingested): {name}")
            skipped += 1
            continue

        print(f"  Loading: {name}")
        pages = load_pages(pdf)
        for page in pages:
            page.metadata["source_file"] = name
        if not keep_references:
            body = remove_references(pages)
            if len(body) != len(pages):
                print(f"  Dropped {len(pages) - len(body)} reference page(s)")
            pages = body
        chunks = chunk_documents(pages)
        if not chunks:
            print(f"  No extractable text in {name}, skipping")
            continue

        store.add_documents(chunks, ids=[f"{id_prefix}:{i}" for i in range(len(chunks))])
        print(f"  Stored {len(chunks)} chunks from {len(pages)} pages")
        added += 1

    print(f"Done. {added} PDF(s) ingested, {skipped} skipped.")


def main():
    parser = argparse.ArgumentParser(description="Ingest PDFs into pgvector")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing collection before ingesting",
    )
    parser.add_argument(
        "--keep-references",
        action="store_true",
        help="Embed bibliographies too (dropped by default)",
    )
    parser.add_argument(
        "--docs-dir",
        default=config.DOCS_DIR,
        required=config.DOCS_DIR is None,
        help="Path to the folder containing PDF files",
    )
    args = parser.parse_args()

    print(f"Scanning {args.docs_dir} for PDFs...")
    ingest(args.docs_dir, reset=args.reset, keep_references=args.keep_references)


if __name__ == "__main__":
    main()
