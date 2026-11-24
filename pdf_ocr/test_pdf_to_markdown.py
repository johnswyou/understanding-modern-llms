"""Offline tests: a fake Mistral client stands in for the network."""

import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

import pdf_to_markdown as m


def make_page(index, markdown, images=()):
    return NS(index=index, markdown=markdown, images=list(images))


def make_response(pages):
    return NS(pages=pages, model="mistral-ocr-test")


class FakeClient:
    def __init__(self, response=None, ocr_error=None):
        self.calls = []
        self.deleted = []
        self.response, self.ocr_error = response, ocr_error
        self.files = NS(
            upload=lambda file, purpose: NS(id="file-1"),
            get_signed_url=lambda file_id: NS(url="https://signed"),
            delete=lambda file_id: self.deleted.append(file_id),
        )
        self.ocr = NS(process=self._process)

    def _process(self, **kwargs):
        self.calls.append(kwargs)
        if self.ocr_error:
            raise self.ocr_error
        return self.response


def test_figure_placeholder_replaced_by_description():
    ann = json.dumps({"figure_type": "chart", "description": "Loss falls\nwith compute."})
    page = make_page(
        0, "Text\n\n![img-0.jpeg](img-0.jpeg)\n\nMore",
        [NS(id="img-0.jpeg", image_annotation=ann)],
    )
    out = m.describe_figures_in_page(page)
    assert "![img-0.jpeg]" not in out
    assert "> **Figure `img-0.jpeg` (chart):** Loss falls\n> with compute." in out


def test_image_without_annotation_keeps_placeholder():
    page = make_page(0, "![img-0.jpeg](img-0.jpeg)", [NS(id="img-0.jpeg", image_annotation=None)])
    assert m.describe_figures_in_page(page) == "![img-0.jpeg](img-0.jpeg)"


def test_non_json_annotation_and_missing_placeholder_are_kept():
    page = make_page(0, "Body", [NS(id="img-3.jpeg", image_annotation="plain text")])
    assert m.describe_figures_in_page(page).endswith("(figure):** plain text")


def test_convert_writes_markdown_and_requests_annotations(tmp_path):
    pdf = tmp_path / "in" / "sub" / "paper.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"%PDF-1.4")
    client = FakeClient(make_response([make_page(0, "Hello"), make_page(1, "World")]))

    m.convert_single_pdf(client, pdf, tmp_path / "in", tmp_path / "out", describe_figures=True)

    out = (tmp_path / "out" / "sub" / "paper.md").read_text()
    assert 'source_file: "paper.pdf"' in out and "page_count: 2" in out
    assert "<!-- page: 1 -->\n\nHello" in out and "<!-- page: 2 -->\n\nWorld" in out
    call = client.calls[0]
    assert call["bbox_annotation_format"]["type"] == "json_schema"
    assert call["extract_header"] and call["extract_footer"] and call["include_image_base64"] is False
    assert client.deleted == ["file-1"]


def test_annotations_requested_by_default_and_not_when_disabled(tmp_path):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    client = FakeClient(make_response([make_page(0, "Hi")]))
    m.convert_single_pdf(client, pdf, tmp_path, tmp_path / "out")
    m.convert_single_pdf(client, pdf, tmp_path, tmp_path / "out", describe_figures=False, skip_existing=False)
    assert "bbox_annotation_format" in client.calls[0]
    assert "bbox_annotation_format" not in client.calls[1]


def test_remote_file_deleted_when_ocr_fails(tmp_path):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    client = FakeClient(ocr_error=RuntimeError("boom"))
    with pytest.raises(RuntimeError):
        m.convert_single_pdf(client, pdf, tmp_path, tmp_path / "out")
    assert client.deleted == ["file-1"]


def test_existing_output_skipped_unless_overwrite(tmp_path):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "paper.md").write_text("old")
    client = FakeClient(make_response([make_page(0, "new")]))

    assert m.convert_single_pdf(client, pdf, tmp_path, tmp_path / "out")[1] == "skipped"
    assert client.calls == []
    m.convert_single_pdf(client, pdf, tmp_path, tmp_path / "out", skip_existing=False)
    assert "new" in (tmp_path / "out" / "paper.md").read_text()
