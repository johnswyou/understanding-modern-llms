# pdf_ocr

Converts a directory of PDFs into agent-readable Markdown with Mistral OCR. Each output file has YAML frontmatter (source file, page count, model) and a `<!-- page: N -->` marker before each page, so an agent can cite page numbers. See [solution_sketch.md](solution_sketch.md) for why the OCR settings were chosen.

## Setup

```bash
pip install "mistralai>=2"
echo "${MISTRAL_API_KEY:+Key is set}"   # must print "Key is set"
```

`MISTRAL_API_KEY` must be exported in your shell. Cron jobs and agent runners don't read `~/.bashrc`, so wrap the command in `bash -lc '...'` there.

## Convert `papers/twelve-essential`

Run from the repository root:

```bash
python pdf_ocr/pdf_to_markdown.py papers/twelve-essential
```

This writes one `.md` file per PDF to `papers/twelve-essential/markdown_output/`. Every PDF is uploaded to Mistral and deleted from Mistral's servers after its OCR run finishes.

The folder also holds `on-the-biology-of-a-large-language-model.html`. The converter reads only PDFs, so it skips that file.

## Options

| Option | Effect |
| --- | --- |
| `-o, --output-dir DIR` | Write Markdown to `DIR`. The default is `<input_dir>/markdown_output`. |
| `-r, --recursive` | Also convert PDFs in subfolders, keeping the folder structure. |
| `--no-describe-figures` | Turn off figure descriptions, which are on by default (see below). |
| `--keep-headers-footers` | Leave repeating page headers and footers in the page text. By default they are removed. |
| `--overwrite` | Re-convert PDFs whose `.md` file already exists. By default they are skipped. |
| `-w, --workers N` | Number of parallel OCR requests. The default is 3. |

The exit code is 1 if any PDF failed. Failed PDFs are listed on stderr, and rerunning the same command retries only those.

## Charts and diagrams

The images themselves are not saved, so a bare placeholder such as `![img-0.jpeg](img-0.jpeg)` would tell an agent nothing. By default the script asks Mistral for a structured annotation of each figure (`figure_type` and a plain-text `description`) and replaces the placeholder with:

```markdown
> **Figure `img-0.jpeg` (chart):** Training loss on the y-axis falls as compute on the x-axis grows...
```

Annotations are billed at a higher rate than plain OCR, so pass `--no-describe-figures` for text-only documents. Existing `.md` files are skipped on later runs, so add `--overwrite` when you want to redo a file after changing this setting.

## Tests

```bash
pip install pytest
cd pdf_ocr && python -m pytest
```

The tests use a fake client, need no API key, and make no network calls. The script itself has not been run against the live Mistral API.

## Large archives

For hundreds or thousands of PDFs, Mistral's Batch API (`endpoint="/v1/ocr"`) is cheaper. This script does not use it.
