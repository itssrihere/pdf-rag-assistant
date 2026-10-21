"""Page-by-page PDF text extraction with a selectable backend (PDF_EXTRACTOR).

Every backend returns one Document per page with `page` (0-based) and `total_pages` metadata, so page-level
citations work the same whichever extractor is used."""

from pathlib import Path

from langchain_core.documents import Document

import config

EXTRACTORS = ("pypdf", "pypdfium2")


def _page_documents(path: Path, texts: list[str]) -> list[Document]:
    return [
        Document(page_content=text, metadata={"source": str(path), "page": i, "total_pages": len(texts)})
        for i, text in enumerate(texts)
    ]


def _pypdf(path: Path) -> list[Document]:
    from langchain_community.document_loaders import PyPDFLoader

    return PyPDFLoader(str(path)).load()


def _pypdfium2(path: Path) -> list[Document]:
    import pypdfium2  # optional: pip install pypdfium2 (BSD/Apache)

    pdf = pypdfium2.PdfDocument(str(path))
    try:
        return _page_documents(path, [pdf[i].get_textpage().get_text_range() for i in range(len(pdf))])
    finally:
        pdf.close()


def load_pages(path: Path, extractor: str | None = None) -> list[Document]:
    extractor = (extractor or config.PDF_EXTRACTOR).lower()
    backends = {"pypdf": _pypdf, "pypdfium2": _pypdfium2}
    if extractor not in backends:
        raise ValueError(f"PDF_EXTRACTOR must be one of {list(EXTRACTORS)}, got {extractor!r}")
    return backends[extractor](Path(path))
