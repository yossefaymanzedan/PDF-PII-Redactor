# PDF PII Detector

> **Language scope: English only (current release)**  
> This build processes documents in **English**. Multi-language OCR and NER are on the roadmap and will be added in a future release.

---

## Non-technical quick start (no setup)

If you do not have Python or do not want to install anything:

1. Download the folder repository.
2. Inside that folder, double-click **`run.bat`**.
   - First run only: sets up a local environment and downloads required libraries and optionally the AI model.
   - Later runs: skips downloads and starts the app immediately.
   - You will be guided through simple prompts.
   - When finished, your default browser opens the app UI.

> **Required components - Tesseract + `pytesseract` (English only)**  
> - **Tesseract OCR** reads text from images such as scanned PDFs.  
>   • Windows: <https://github.com/UB-Mannheim/tesseract/wiki>  
>   • macOS (Homebrew): `brew install tesseract`  
>   • Linux (Debian/Ubuntu): `sudo apt-get install tesseract-ocr`  
> - Ensure the **English language data (`eng`)** is installed. Other languages will be supported in a future update.

---

## What this app does

- Mixed PDF OCR: combines native PDF text with per-image OCR, with an optional masked full-page fallback.
- PII detection: fast regex rules plus an optional NER model (`dslim/bert-base-NER`) for richer entities such as names, organizations, and locations.
- Subject grouping: pairs NAME with EMAIL and associates nearby PII using layout affinity to form subjects.
- Minimal exports: creates `detections_min.csv` with a deterministic `subject_key`, duplicate values with score and provenance, plus `detections_min.jsonl` and a `quality_report.json`.
- Charts and quick analytics: counts, hits per page, and top subjects by risk.
- Artifacts for teams: redaction worklist, HR contacts pivot, annotated previews (optional).
- Batch mode and caching: process folders of PDFs. Re-runs are faster.

---

## Quickstart for developers

### 1) Requirements

- **Python 3.9+**
- **Tesseract OCR** installed and on PATH (English language data `eng`)  
  - Windows: <https://github.com/UB-Mannheim/tesseract/wiki>  
  - macOS (Homebrew): `brew install tesseract`  
  - Linux (Debian/Ubuntu): `sudo apt-get install tesseract-ocr`
- **`pytesseract`** (Python wrapper for Tesseract) - install from PyPI: <https://pypi.org/project/pytesseract/>

### 2) Install Python dependencies

```bash
# from repo root
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 3) Run the UI

```bash
python app.py
```
Your browser should open to the Gradio interface. If not, copy the local URL printed in the terminal and paste it into the browser.

> **Note:** The app currently assumes **English** OCR. Multi-language selection will arrive in a future update.

---

## Using the app

### Tab 1 - Extract detections
1. Upload a PDF.
2. Choose OCR **DPI** (300 is a good default).  
   Language is fixed to English in this build.
3. Optional: enable the NER model `dslim/bert-base-NER`.
4. Click **Process PDF**. Outputs include:
   - `words_split.csv` (all tokens)
   - `detections.csv` (per-hit rows with labels, scores, page coordinates)
   - Page preview images with green or blue token boxes

### Tab 2 - Post-process CSVs and charts or export
1. Provide the `detections.csv` from Tab 1.
2. Optional: provide the previews folder to draw annotations.
3. Set **Min confidence** for charts and annotations (defaults from YAML).
4. Optional: provide a **YAML config** to override risk weights, duplicate policy, and other settings.
5. Optional: **Scrub raw_text and line_text** to drop those columns in artifacts.
6. Click **Run post-processor**. Outputs include:
   - `detections_min.csv` (subject level, with `subject_key`, duplicate values and provenance)
   - `detections_min.jsonl`, `quality_report.json`
   - `counts_by_label.csv`, `risk_by_subject.csv`, `risk_by_page.csv`
   - `labels_bar.png`, `top_subjects_risk.png`, `hits_per_page.png`
   - `redaction_worklist.csv`, `contacts_hr.csv`
   - `previews_annotated/` (if previews folder provided)

---

## CLI usage (batch or single file)

```bash
# Batch process a folder recursively, English OCR
python app.py --batch "C:\\path\\to\\pdfs" --dpi 300 --conf 60 --scrub-raw-text

# Single file, then auto-run post-processor, English OCR
python app.py --file "C:\\path\\to\\file.pdf" --dpi 300 --conf 60 --ner_model "dslim/bert-base-NER"

# Run without launching the UI (useful for servers)
python app.py --file "C:\\path\\to\\file.pdf" --no-gui
```
**Common flags**

- `--dpi` Rasterization DPI for OCR (100 to 600, default 300)  
- `--conf` Minimum OCR confidence to keep tokens (0 to 100, default 60)  
- `--ner_model` Use `dslim/bert-base-NER` if you want NER (optional)  
- `--scrub-raw-text` Drop `raw_text` and `line_text` in post-processed artifacts  
- `--yaml` Path to a config file to override defaults

> **Deprecated for now:** `--lang` may not return the expected results. It is planned for future enhancements.

---

## How it works

### Pipeline stages

1. **PDF parsing and tokenization**  
   - Extract native PDF words via PyMuPDF.  
   - Locate embedded images and OCR only those regions. If needed, perform a masked full-page OCR pass where native words are blanked out to avoid duplicates.

2. **OCR (Tesseract via `pytesseract`)**  
   - Rasterize pages at a configurable **DPI** and call Tesseract through `pytesseract`.  
   - Ingest word-level boxes, confidences, and normalized text for downstream detection.  
   - Current build: OCR language is **English** (`eng`).

3. **PII detection**  
   - Regex for strong entities: EMAIL, PHONE, SSN, EIN, CREDIT_CARD with Luhn check, IBAN, DATE with normalization, and others.  
   - Optional NER (`dslim/bert-base-NER`) for NAME, ORG, LOCATION. Light post-filters remove short junk tokens and overlaps are resolved to prefer regex on conflicts.

4. **Subject grouping**  
   - Identify NAME and EMAIL seeds and merge likely pairs using layout proximity and a name to email local-part affinity score.  
   - Attach nearby detections to the best seed using a layout score such as column match, y-distance, x-overlap, and block continuity.

5. **Minimal subject view**  
   - Deterministic `subject_key = sha1(normalized_best_name|best_email)` for cross-page and cross-file de-duplication.  
   - Duplicate values per field are joined or top-k (configurable) with optional `|score:` and `|page:` or `|block:` provenance tags.

6. **Exports and QA**  
   - `detections.csv`, `detections_min.csv`, `detections_min.jsonl`, `quality_report.json`.  
   - CSVs for label counts, per-page hits, per-subject risk, plus simple charts via matplotlib.  
   - Optional annotated previews to visually review hits.

### Key modules

- **`process_pdf_worker`**: end to end per-PDF extraction, OCR, detection, grouping.  
- **`detect_pii`**: combines regex and optional NER, resolves span conflicts.  
- **`build_detections_min`**: merges per-hit rows into a subject-level table with `subject_key`, duplicate handling, and provenance.  
- **`postprocess_detections`**: writes team-friendly artifacts, charts, and JSONL.  
- **OCR stack**: PyMuPDF for rasterize and parse then Tesseract OCR engine then `pytesseract` Python bridge.  
- **Model**: `dslim/bert-base-NER` (optional).

---

## Roadmap - multi-language support

- Add language selector in the UI, starting with English and more languages later.
- Extend the CLI with `--lang` accepting `eng`, `spa`, `deu`, and others.
- Bundle language specific post-filters and locale aware patterns for phones, IBAN formats, and dates.

---

## Tips and troubleshooting

- OCR quality: raise DPI to 300 to 400. Ensure scans are clear and English text is legible.
- NER is optional. You will still get robust detections from regex alone.
- Role emails such as `hr@` or `info@` are de-emphasized unless adjacent to a name. This is configurable in YAML.
- Scrub content: enable Scrub raw_text and line_text from artifacts to minimize raw strings in outputs.

---

## License and attribution

**Copyright © 2025 Yossef Ayman Zedan**

- You may use, modify, and distribute this software for **non-commercial** purposes (individuals, companies, organizations, and others).
- **You may NOT sell it or profit commercially from it** without permission.
- For **commercial licensing** (any selling or profit-making use), contact: **yossefaymanzedan@gmail.com**.

Please retain this notice in derivative works.
