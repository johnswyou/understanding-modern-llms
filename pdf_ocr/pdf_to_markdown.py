#!/usr/bin/env python3
"""Batch-convert a directory of PDFs into agent-readable Markdown via Mistral OCR."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
import sys
import time

from pydantic import BaseModel, Field

try:
    from mistralai.client import Mistral  # SDK v2+
except ImportError:
    from mistralai import Mistral  # Fallback for SDK v1
from mistralai.extra import response_format_from_pydantic_model


class FigureAnnotation(BaseModel):
    """Structured text description Mistral generates for each figure region."""

    figure_type: str = Field(
        description="One of: chart, diagram, table, photo, equation, other."
    )
    description: str = Field(
        description=(
            "Plain-text description for a reader who cannot see the figure. "
            "Charts: axes with units, each series, trends and key values. "
            "Diagrams: components and how they connect. "
            "Transcribe any visible text and labels."
        )
    )


def describe_figures_in_page(page) -> str:
    """Replace each image placeholder in the page Markdown with its text description.

    A no-op for images without an annotation (i.e. when --describe-figures is off).
    """
    body = (page.markdown or "").strip()
    for image in page.images or []:
        if not image.image_annotation:
            continue
        try:
            annotation = json.loads(image.image_annotation)
            kind = annotation.get("figure_type", "figure")
            text = annotation.get("description", image.image_annotation)
        except (ValueError, AttributeError):
            kind, text = "figure", image.image_annotation
        quoted = str(text).strip().replace("\n", "\n> ")
        block = f"> **Figure `{image.id}` ({kind}):** {quoted}"
        body, replaced = re.subn(
            rf"!\[[^\]]*\]\({re.escape(image.id)}\)", lambda _: block, body
        )
        if not replaced:  # placeholder missing from the Markdown: keep the description anyway
            body += f"\n\n{block}"
    return body


def build_agent_markdown(pdf_path: Path, ocr_response) -> str:
    """Assemble OCR pages into clean Markdown with YAML frontmatter and page markers."""
    pages = ocr_response.pages or []
    frontmatter = "\n".join(
        [
            "---",
            f'source_file: "{pdf_path.name}"',
            f"page_count: {len(pages)}",
            f'ocr_model: "{ocr_response.model}"',
            "---",
            "",
        ]
    )

    sections: list[str] = []
    for page in pages:
        page_num = page.index + 1
        sections.append(f"<!-- page: {page_num} -->\n\n{describe_figures_in_page(page)}")

    return frontmatter + "\n\n---\n\n".join(sections) + "\n"


def convert_single_pdf(
    client: Mistral,
    pdf_path: Path,
    input_dir: Path,
    output_dir: Path,
    *,
    strip_headers_footers: bool = True,
    describe_figures: bool = True,
    skip_existing: bool = True,
) -> tuple[Path, str]:
    """Upload one PDF, run Mistral OCR, write Markdown, and delete the remote file."""
    relative_path = pdf_path.relative_to(input_dir)
    out_path = (output_dir / relative_path).with_suffix(".md")

    if skip_existing and out_path.exists():
        return pdf_path, "skipped"

    uploaded_file = None
    try:
        with pdf_path.open("rb") as f:
            uploaded_file = client.files.upload(
                file={"file_name": pdf_path.name, "content": f},
                purpose="ocr",
            )

        signed_url = client.files.get_signed_url(file_id=uploaded_file.id)

        extra = {}
        if describe_figures:
            extra["bbox_annotation_format"] = response_format_from_pydantic_model(
                FigureAnnotation
            )

        ocr_response = client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "document_url",
                "document_url": signed_url.url,
            },
            table_format=None,  # Inline Markdown tables
            include_image_base64=False,  # Keep output lightweight for agents
            extract_header=strip_headers_footers,
            extract_footer=strip_headers_footers,
            **extra,
        )

        markdown_text = build_agent_markdown(pdf_path, ocr_response)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(markdown_text, encoding="utf-8")
        return pdf_path, f"ok ({len(ocr_response.pages)} pages -> {out_path})"

    finally:
        if uploaded_file is not None:
            try:
                client.files.delete(file_id=uploaded_file.id)
            except Exception as cleanup_err:
                print(
                    f"[warn] Failed to delete remote file {uploaded_file.id}: {cleanup_err}",
                    file=sys.stderr,
                )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert all PDFs in a directory to agent-readable Markdown using Mistral OCR."
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing PDF files.")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Destination directory for .md files (default: <input_dir>/markdown_output).",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Search subdirectories recursively for PDFs.",
    )
    parser.add_argument(
        "--keep-headers-footers",
        action="store_true",
        help="Keep repeating page headers/footers inside the main Markdown body.",
    )
    parser.add_argument(
        "--describe-figures",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Replace chart/diagram image placeholders with generated text descriptions "
        "(default: on; uses Mistral's bbox annotations, which cost more than plain OCR). "
        "Use --no-describe-figures to turn it off.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-process PDFs even if the output .md file already exists.",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=3,
        help="Number of concurrent OCR requests (default: 3).",
    )
    args = parser.parse_args()

    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        sys.exit(
            "Error: MISTRAL_API_KEY is not set in the current environment.\n"
            "Ensure it is exported in ~/.bashrc or ~/.zshrc and run `source ~/.bashrc` (or `source ~/.zshrc`)."
        )

    input_dir = args.input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        sys.exit(f"Error: Input directory not found: {input_dir}")

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else input_dir / "markdown_output"
    )

    pattern = "**/*.pdf" if args.recursive else "*.pdf"
    pdf_files = sorted(p for p in input_dir.glob(pattern) if p.is_file())

    if not pdf_files:
        sys.exit(f"No PDF files found in {input_dir}")

    print(f"Found {len(pdf_files)} PDF(s). Writing Markdown to: {output_dir}")
    start_time = time.perf_counter()
    failures = 0

    with Mistral(api_key=api_key) as client:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {
                pool.submit(
                    convert_single_pdf,
                    client,
                    pdf,
                    input_dir,
                    output_dir,
                    strip_headers_footers=not args.keep_headers_footers,
                    describe_figures=args.describe_figures,
                    skip_existing=not args.overwrite,
                ): pdf
                for pdf in pdf_files
            }

            for future in as_completed(futures):
                pdf = futures[future]
                try:
                    _, status = future.result()
                    print(f"[done] {pdf.name}: {status}")
                except Exception as exc:
                    failures += 1
                    print(f"[fail] {pdf.name}: {exc}", file=sys.stderr)

    elapsed = time.perf_counter() - start_time
    print(f"Finished in {elapsed:.1f}s ({len(pdf_files) - failures}/{len(pdf_files)} succeeded).")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
