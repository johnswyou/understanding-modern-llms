Converting local PDFs into agent-ready Markdown with Mistral OCR uses a four-step pipeline: upload the PDF with `purpose="ocr"`, generate a temporary signed URL, run `client.ocr.process` with `model="mistral-ocr-latest"`, and delete the remote file afterward.

## Why These OCR Settings Work Best for Agents

* **Inline Markdown tables (`table_format=None`):** Embeds tables directly in the page's Markdown flow rather than pulling them into a separate `tables` array behind placeholders like `[tbl-3.html](tbl-3.html)`. Agents parse full context without needing to stitch separate structures back together.
* **Noise reduction (`extract_header=True`, `extract_footer=True`):** Separates repeating page headers and footers out of `page.markdown` into dedicated fields, preventing repetitive chrome from polluting chunking and retrieval.
* **Token efficiency (`include_image_base64=False`):** Omits raw base64 image payloads from the response so files stay lightweight and text-focused.
* **Structured provenance:** Adding YAML frontmatter and `<!-- page: N -->` markers lets downstream agents track the source filename and cite exact page numbers.

## 1. Setup

Install the current Mistral Python SDK (`v2`, which imports from `mistralai.client`):

```bash
pip install "mistralai>=2"

```

Verify that your shell profile (`~/.bashrc` or `~/.zshrc`) has exported your key into the active session:

```bash
# Reload if you recently edited your profile
source ~/.bashrc   # or: source ~/.zshrc

# Confirm the variable is available to Python
echo "${MISTRAL_API_KEY:+Key is set}"

```

## 2. Python Conversion Script

Save the following script as `pdf_to_markdown.py`. It reads `MISTRAL_API_KEY` from your shell environment, converts every PDF in a target directory (preserving subfolder structure), cleans up uploaded files on Mistral's servers, and supports parallel processing.

```python
#!/usr/bin/env python3
"""Batch-convert a directory of PDFs into agent-readable Markdown via Mistral OCR."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import sys
import time

try:
    from mistralai.client import Mistral  # SDK v2+
except ImportError:
    from mistralai import Mistral  # Fallback for SDK v1


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
        body = (page.markdown or "").strip()
        sections.append(f"<!-- page: {page_num} -->\n\n{body}")

    return frontmatter + "\n\n---\n\n".join(sections) + "\n"


def convert_single_pdf(
    client: Mistral,
    pdf_path: Path,
    input_dir: Path,
    output_dir: Path,
    *,
    strip_headers_footers: bool = True,
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
    pdf_files = sorted(
        p for p in input_dir.glob(pattern) if p.is_file()
    )

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


if __name__ == "__main__":
    main()

```

## 3. Running the Script

```bash
# Convert all PDFs in ./pdfs into ./pdfs/markdown_output
python pdf_to_markdown.py ./pdfs

# Recursively convert PDFs into a specific folder with 5 parallel workers
python pdf_to_markdown.py ./docs -o ./agent_corpus -r -w 5

# Force re-conversion of existing .md files
python pdf_to_markdown.py ./docs -o ./agent_corpus --overwrite

```

## Scaling & Caveats

* **Large archives (hundreds or thousands of PDFs):** Switch from synchronous calls to Mistral's Batch API (`endpoint="/v1/ocr"`), which processes workloads asynchronously at a 50% cost reduction.
* **Charts and diagrams:** Because `include_image_base64=False` leaves image placeholders like `![img-0.jpeg](img-0.jpeg)` without text descriptions, if your PDFs rely heavily on charts or figures, you can add Mistral's `bbox_annotation_format` parameter with a Pydantic schema to automatically generate structured text descriptions for each visual region.
* **Cron jobs or non-interactive runners:** Non-interactive processes do not source `~/.bashrc` or `~/.zshrc` automatically. If you schedule this script via `cron` or an agent runner, pass the environment variable explicitly or wrap the command in `bash -lc 'python pdf_to_markdown.py ...'`.