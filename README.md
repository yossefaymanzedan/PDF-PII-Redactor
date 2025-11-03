# PDF PII Detector & Post-Processor (OCR + NER/regex + Grouping)

> **Language scope: English-only (current release)**  
> This build processes documents in **English**. Multi‑language OCR/NER is on the roadmap and will be added in a future release.

---

**💡 Non‑technical? Start here — zero setup required**

If you don’t have Python or you don’t want to install anything:

1. Download the folder **`PDF-PPI-Redactor-Non-Technical`** from this project’s release/package.
2. Inside that folder, double‑click **`run.bat`** (already included).
   - First run only: it sets up a local environment and downloads required libraries and (optionally) the AI model.
   - Next runs: it skips downloads and starts the app immediately.
   - You’ll be guided through simple prompts (press Enter a few times).
   - When it finishes, your default browser opens the app UI.

> **Required components — Tesseract + `pytesseract` (English only)**  
> - **Tesseract OCR** is the engine that reads text from images (e.g., scanned PDFs).  
>   • Windows: <https://github.com/UB-Mannheim/tesseract/wiki>  
>   • macOS (Homebrew): `brew install tesseract`  
>   • Linux (Debian/Ubuntu): `sudo apt-get install tesseract-ocr`  
> - Make sure the **English language data (`eng`)** is installed. Other languages will be supported in a future update.

---

## What this app does

- **Mixed PDF OCR**: combines native PDF text + per‑image OCR, with an optional masked full‑page fallback.
- **PII detection**: fast **regex** rules plus an optional **NER model** (`dslim/bert-base-NER`) for richer entities (Names/Orgs/Locations).
- **Subject grouping**: pairs NAME↔EMAIL and associates nearby PII using layout affinity to form “subjects.”
- **Minimal exports**: creates `detections_min.csv` with deterministic `subject_key`, duplicate values (with score & provenance), plus `detections_min.jsonl` and a `quality_report.json`.
- **Charts & quick analytics**: counts, hits per page, and top subjects by risk.
- **Artifacts for teams**: redaction worklist, HR contacts pivot, annotated previews (optional).
- **Batch mode & caching**: process folders of PDFs; re‑runs are faster.

---

## Quickstart (Developers)

### 1) Requirements

- **Python 3.9+**
- **Tesseract OCR** installed and on PATH (English language data `eng`)  
  - Windows: <https://github.com/UB-Mannheim/tesseract/wiki>  
  - macOS (Homebrew): `brew install tesseract`  
  - Linux (Debian/Ubuntu): `sudo apt-get install tesseract-ocr`
- **`pytesseract`** (Python wrapper for Tesseract) — install from PyPI: <https://pypi.org/project/pytesseract/>

### 2) Install Python deps

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

> **Note:** The app currently assumes **English** OCR. Multi‑language selection will arrive in a future update.

---

## Using the App

### Tab 1 — “Extract detections”
1. Upload a PDF.
2. Choose OCR **DPI** (300 is a good default).  
   *Language is fixed to English in this build.*
3. (Optional) enable the NER model: `dslim/bert-base-NER`.
4. Click **Process PDF**. You’ll get:
   - `words_split.csv` (all tokens)
   - `detections.csv` (per‑hit rows with labels, scores, page coords)
   - Page preview images with green/blue token boxes

### Tab 2 — “Post‑process CSVs & charts / Export”
1. Provide the `detections.csv` from Tab 1.
2. (Optional) Provide the previews folder to draw annotations.
3. Set **Min confidence** for charts/annotations (defaults from YAML).
4. (Optional) Provide a **YAML config** to override risk weights, dup policy, etc.
5. (Optional) **Scrub raw_text/line_text** to drop those columns in artifacts.
6. Click **Run post‑processor**. You’ll get:
   - `detections_min.csv` (subject‑level, with `subject_key`, dup values + provenance)
   - `detections_min.jsonl`, `quality_report.json`
   - `counts_by_label.csv`, `risk_by_subject.csv`, `risk_by_page.csv`
   - `labels_bar.png`, `top_subjects_risk.png`, `hits_per_page.png`
   - `redaction_worklist.csv`, `contacts_hr.csv`
   - `previews_annotated/` (if previews folder provided)

---

## CLI (batch or single file)

```bash
# Batch process a folder (recursively finds PDFs) — English OCR
python app.py --batch "C:\\path\\to\\pdfs" --dpi 300 --conf 60 --scrub-raw-text

# Single file, then auto-run post-processor — English OCR
python app.py --file "C:\\path\\to\\file.pdf" --dpi 300 --conf 60 --ner_model "dslim/bert-base-NER"

# Run without launching the UI (useful for servers)
python app.py --file "C:\\path\\to\\file.pdf" --no-gui
```
**Common flags**

- `--dpi` Rasterization DPI for OCR (100–600, default 300)  
- `--conf` Minimum OCR confidence to keep tokens (0–100, default 60)  
- `--ner_model` Use **only** `dslim/bert-base-NER` if you want NER (optional)  
- `--scrub-raw-text` Drop `raw_text`/`line_text` in post‑processed artifacts  
- `--yaml` Path to a config file to override defaults

> **Deprecated for now:** `--lang` may not return the expected results. It is planned for future enhancements.

---

## How it works (for technical readers)

### Pipeline stages

1. **PDF parsing + tokenization**  
   - Extract **native PDF words** via PyMuPDF.  
   - Locate **embedded images** and OCR only those regions; if needed, do a **masked full‑page OCR** pass where native words are blanked out to avoid duplicates.

2. **OCR (Tesseract via `pytesseract`)**  
   - We rasterize pages at configurable **DPI** and call Tesseract through `pytesseract`.  
   - We ingest word‑level boxes, confidences, and normalized text for downstream detection.  
   - **Current build:** OCR language is **English** (`eng`).

3. **PII detection**  
   - **Regex** for strong entities: EMAIL, PHONE, SSN, EIN, CREDIT_CARD (Luhn‑checked), IBAN, DATE (normalized), etc.  
   - **Optional NER** (`dslim/bert-base-NER`) for NAME/ORG/LOCATION. Light post‑filters remove short junk tokens and overlaps are resolved to **prefer regex** on conflicts.

4. **Subject grouping**  
   - Identify **NAME/EMAIL seeds** and merge likely pairs using layout proximity + a **name↔email local‑part affinity** score.  
   - Attach nearby detections to the best seed using a **layout score** (column match, y‑distance, x‑overlap, block continuity, etc.).

5. **Minimal subject view**  
   - Deterministic **`subject_key = sha1(normalized_best_name|best_email)`** for cross‑page/file de‑duplication.  
   - Duplicate values per field are joined or **top‑k** (configurable) with optional **`|score:`** and **`|page:`/`|block:`** provenance tags.

6. **Exports & QA**  
   - `detections.csv`, `detections_min.csv`, `detections_min.jsonl`, `quality_report.json`.  
   - CSVs for label counts, per‑page hits, per‑subject risk; simple charts (matplotlib).  
   - Optional **annotated previews** to visually review hits.

### Key functions & modules

- **`process_pdf_worker`**: end‑to‑end per‑PDF extraction, OCR, detection, grouping.  
- **`detect_pii`**: combines regex and optional NER, resolves span conflicts.  
- **`build_detections_min`**: merges per‑hit rows into a subject‑level table with `subject_key`, dup handling, and provenance.  
- **`postprocess_detections`**: writes team‑friendly artifacts, charts, and JSONL.  
- **OCR stack**: PyMuPDF (rasterize & parse) → Tesseract (OCR engine) → `pytesseract` (Python bridge).  
- **Model**: `dslim/bert-base-NER` (only, optional).

---

## Roadmap — Multi‑language Support

- Add language selector in UI (e.g., English + future languages).
- Extend CLI with `--lang` accepting `eng`, `spa`, `deu`, etc.
- Bundle language‑specific post‑filters and locale‑aware patterns (phones, IBAN formats, dates).

---

## Tips & Troubleshooting

- **OCR quality**: raise **DPI** (300–400); ensure scans are clear and English text is legible.
- **NER is optional**: you’ll still get robust detections from regex alone.
- **Role emails** (e.g., `hr@`, `info@`) are de‑emphasized unless adjacent to a **NAME** (configurable in YAML).
- **Scrub content**: enable *Scrub raw_text/line_text from artifacts* to minimize raw strings in outputs.

---

## License & Attribution

**Copyright © 2025 Yossef Ayman Zedan**

- You may use, modify, and distribute this software for **non‑commercial** purposes (individuals, companies, organizations, etc.).
- **You may NOT sell it or profit commercially from it** without permission.
- For **commercial licensing** (any selling/profit‑making use), contact: **yossefaymanzedan@gmail.com**.

Please retain this notice in derivative works.
