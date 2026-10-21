import pytest

import pdf_text

def build_pdf(text: str) -> bytes:
    """A valid one-page PDF containing `text`, with a correct cross-reference table."""
    stream = f"BT /F1 12 Tf 20 50 Td ({text}) Tj ET".encode()
    objects = [
        b"<</Type/Catalog/Pages 2 0 R>>",
        b"<</Type/Pages/Kids[3 0 R]/Count 1>>",
        b"<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 100]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>",
        b"<</Length %d>>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>",
    ]
    out, offsets = b"%PDF-1.4\n", []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref_at = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    return out + b"trailer\n<</Root 1 0 R/Size %d>>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref_at)


@pytest.fixture
def pdf(tmp_path):
    path = tmp_path / "hello.pdf"
    path.write_bytes(build_pdf("Hello world"))
    return path


@pytest.mark.parametrize("extractor", ["pypdf", "pypdfium2"])
def test_backends_return_one_document_per_page_with_page_metadata(pdf, extractor):
    if extractor == "pypdfium2":
        pytest.importorskip("pypdfium2")
    pages = pdf_text.load_pages(pdf, extractor)
    assert len(pages) == 1
    assert "Hello world" in pages[0].page_content
    assert pages[0].metadata["page"] == 0


def test_unknown_extractor_is_rejected(pdf):
    with pytest.raises(ValueError, match="PDF_EXTRACTOR"):
        pdf_text.load_pages(pdf, "tesseract")
