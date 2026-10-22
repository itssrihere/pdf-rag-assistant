"""
Builds the second evaluation corpus: the Spring Boot 3.1.12 reference documentation, split by chapter and section
into a folder tree of PDFs (a realistic "docs folder", unlike the single-file original).

    python -m eval.spring_docs ./spring-docs            # downloads the official PDF, then splits it
    python -m eval.spring_docs ./spring-docs --pdf x.pdf  # split a PDF you already have

The PDF is published by the Spring project (https://docs.spring.io/spring-boot/docs/3.1.12/reference/pdf/);
it is downloaded on demand and is not part of this repository. Questions: eval/spring_questions.jsonl.
"""

import argparse
import re
import urllib.request
from pathlib import Path

from pypdf import PdfReader, PdfWriter

URL = "https://docs.spring.io/spring-boot/docs/3.1.12/reference/pdf/spring-boot-reference.pdf"


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60]


def outline_entries(reader: PdfReader) -> list[tuple[int, str, int]]:
    """(depth, title, 1-based start page) for every bookmark."""
    entries = []

    def walk(items, depth=0):
        for item in items:
            if isinstance(item, list):
                walk(item, depth + 1)
            else:
                entries.append((depth, item.title.strip(), reader.get_destination_page_number(item) + 1))

    walk(reader.outline)
    return entries


def split(pdf_path: Path, out_dir: Path) -> int:
    """Writes the front matter, then one folder per chapter holding one PDF per section (sections that start on the
    same page are kept together). Returns the number of files written."""
    reader = PdfReader(str(pdf_path))
    total = len(reader.pages)
    outline = outline_entries(reader)
    chapters = [(title, page) for depth, title, page in outline if depth == 0]

    def write(path: Path, first: int, last: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = PdfWriter()
        for i in range(first - 1, last):
            writer.add_page(reader.pages[i])
        with open(path, "wb") as fh:
            writer.write(fh)

    written = 1
    write(out_dir / "00-front-matter" / "table-of-contents.pdf", 1, chapters[2][1] - 1)  # title page + contents
    for index in range(2, len(chapters)):
        title, start = chapters[index]
        end = chapters[index + 1][1] - 1 if index + 1 < len(chapters) else total
        name = slug(re.sub(r"^Chapter \d+\. ", "", title))
        folder = out_dir / f"{index - 1:02d}-{name}"
        sections = []  # sections sharing a start page are merged under the first one's title
        for depth, section_title, page in outline:
            if depth == 1 and start <= page <= end and (not sections or sections[-1][1] != page):
                sections.append((section_title, page))
        if len(sections) < 2:
            write(folder / f"{name}.pdf", start, end)
            written += 1
            continue
        if sections[0][1] > start:
            write(folder / "introduction.pdf", start, sections[0][1] - 1)
            written += 1
        for k, (section_title, page) in enumerate(sections):
            last = sections[k + 1][1] - 1 if k + 1 < len(sections) else end
            write(folder / f"{slug(section_title)}.pdf", page, max(page, last))
            written += 1
    return written


def main():
    parser = argparse.ArgumentParser(description="Prepare the Spring Boot docs corpus for the evaluation")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--pdf", type=Path, help="use this PDF instead of downloading the official one")
    args = parser.parse_args()
    pdf = args.pdf
    if pdf is None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        pdf = args.out_dir / "spring-boot-reference-3.1.12.pdf"
        if not pdf.exists():
            print(f"Downloading {URL} ...")
            urllib.request.urlretrieve(URL, pdf)
    count = split(pdf, args.out_dir / "docs")
    print(f"Wrote {count} PDFs to {args.out_dir / 'docs'}")
    print(f"Next: COLLECTION_NAME=spring_docs DOCS_DIR={args.out_dir / 'docs'} python ingest.py\n"
          f"      COLLECTION_NAME=spring_docs python -m eval.run_eval --questions eval/spring_questions.jsonl")


if __name__ == "__main__":
    main()
