# =============================================================================
# app.py
#
# PDF PII Detector & Post-Processor
# - Mixed PDF OCR (native text + per-image OCR + optional masked full-page OCR)
# - PII detection via NER (optional HF model) + robust regex rules
# - Subject grouping with deterministic subject keys across pages/files
# - Duplicate/conflict handling with provenance & confidence tagging
# - Exports: detections.csv → detections_min.csv + detections_min.jsonl + quality_report.json
# - Charts & quick analytics; optional annotated previews
# - Caching & batch mode; simple Gradio UI + CLI
#
# Copyright (c) 2025 Yossef Ayman Zedan
# Permission is granted to individuals, companies, organizations, etc. to use,
# modify, and distribute this software for non-commercial purposes.
# You may NOT sell it or profit commercially from it.
# For commercial licensing (selling or profit-making use), contact:
#   yossefaymanzedan@gmail.com
# =============================================================================

import argparse
import hashlib
import io
import json
import math
import os
import re
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import fitz  # PyMuPDF
import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytesseract
import yaml
from PIL import Image, ImageDraw

# -----------------------------------------------------------------------------
# Runtime environment tweaks
# -----------------------------------------------------------------------------

os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
DEFAULT_NER_DEVICE = int(os.environ.get("PII_NER_DEVICE", "-1"))
SENTRY_DSN = os.environ.get("SENTRY_DSN", "")

# Optionally set Tesseract path (overridable by UI/CLI)
pytesseract.pytesseract.tesseract_cmd = os.environ.get(
    "TESSERACT_EXE", r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# =============================================================================
# CONFIG (override via YAML)
# =============================================================================

DEFAULT_CONFIG: Dict[str, Any] = {
    "DEFAULT_MIN_CONF": 0.85,
    "RISK_WEIGHTS": {
        "SSN": 10,
        "CREDIT_CARD": 8,
        "IBAN": 6,
        "EIN": 5,
        "DATE": 3,
        "PHONE": 2,
        "EMAIL": 1,
        "LOCATION": 1,
        "NAME": 0,
        "ORG": 0,
        "MISC": 0,
    },
    # Still used for redaction worklist & metrics (no actual redacted PDF output)
    "TEAM_SENSITIVE": ["SSN", "CREDIT_CARD", "IBAN", "EIN"],
    "HR_CONTACT_LABELS": ["NAME", "EMAIL", "PHONE"],
    "LABEL_TO_MINCOL": {
        "NAME": "name",
        "EMAIL": "email",
        "PHONE": "phone",
        "SSN": "ssn",
        "CREDIT_CARD": "credit",
        "IBAN": "iban",
        "DATE": "date",
        "LOCATION": "location",
        "ORG": "org",
    },
    # How to store duplicates in detections_min.csv
    "MIN_DUP_POLICY": {
        "mode": "list",  # "list" or "topk"
        "k": 2,  # used if mode == "topk"
        "include_confidence": True,  # append |score:0.99
        "include_provenance": True,  # append |page:1|block:3
        "separator": "; ",
    },
    "ROLE_EMAIL_POLICY": {"keep_if_adjacent_to_name": True},
}


def load_config(yaml_path: Optional[str]) -> Dict[str, Any]:
    """Load optional YAML config and deep-merge selective dicts."""
    cfg = DEFAULT_CONFIG.copy()
    if yaml_path and Path(yaml_path).exists():
        with open(yaml_path, "r", encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
        for k, v in user.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    return cfg


# =============================================================================
# OCR / PDF UTILITIES
# =============================================================================

def coerce_path(file_input) -> Path:
    """
    Accepts:
      - str / Path (path to file)
      - dict with {"path": "..."} or {"data": bytes/base64, "orig_name": "..."}
    Returns a local Path (writes temp file if needed).
    """
    if file_input is None:
        raise ValueError("No file provided.")

    if isinstance(file_input, (str, Path)):
        return Path(file_input)

    if isinstance(file_input, dict):
        p = file_input.get("path")
        if p:
            return Path(p)

        data = file_input.get("data")
        orig = file_input.get("orig_name") or "upload.pdf"
        if data is None:
            raise ValueError("Uploaded file has no path or data.")

        tmpdir = Path(tempfile.mkdtemp(prefix="pdf_mixed_"))
        outp = tmpdir / orig
        with open(outp, "wb") as f:
            if isinstance(data, bytes):
                f.write(data)
            else:
                if isinstance(data, str) and data.startswith("data:"):
                    import base64

                    b64 = data.split(",", 1)[1]
                    f.write(base64.b64decode(b64))
                else:
                    f.write(bytes(data))
        return outp

    return Path(str(file_input))


def words_from_pdf(page: fitz.Page) -> pd.DataFrame:
    """Extract native PDF words (no OCR)."""
    words = page.get_text("words")
    cols = ["x", "y", "w", "h", "text", "conf", "source", "block_id", "line_no"]
    if not words:
        return pd.DataFrame(columns=cols)

    rows = []
    for x0, y0, x1, y1, txt, block_no, line_no, *_ in words:
        rows.append(
            dict(
                x=float(x0),
                y=float(y0),
                w=float(x1 - x0),
                h=float(y1 - y0),
                text=str(txt),
                conf=100.0,
                source="native",
                block_id=int(block_no),
                line_no=int(line_no),
            )
        )
    return pd.DataFrame(rows, columns=cols)


def rasterize_page(page: fitz.Page, dpi: int = 300) -> Tuple[Image.Image, float, float]:
    """Rasterize PDF page to PIL image, return image and x/y scale factors to map PDF→pixels."""
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    mode = "RGB" if pix.alpha == 0 else "RGBA"
    img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
    if mode == "RGBA":
        img = img.convert("RGB")
    sx = pix.width / page.rect.width
    sy = pix.height / page.rect.height
    return img, sx, sy


def ocr_image(pil_im: Image.Image, lang: str = "eng", psm: int = 6, oem: int = 1) -> pd.DataFrame:
    """Run Tesseract on a PIL image and return word-level dataframe."""
    cfg = f"--psm {psm} --oem {oem}"
    df = pytesseract.image_to_data(
        pil_im, lang=lang, config=cfg, output_type=pytesseract.Output.DATAFRAME
    )
    if df is None or df.empty:
        return pd.DataFrame(columns=["x", "y", "w", "h", "conf", "text"])

    df = df[df.level == 5].copy()
    df.rename(columns={"left": "x", "top": "y", "width": "w", "height": "h"}, inplace=True)
    df["text"] = df["text"].fillna("").astype(str)
    df = df[df["text"].str.strip().ne("")]

    # Remove very large blocks (e.g., full-page images from noise)
    area = df["w"] * df["h"]
    big = (pil_im.width * pil_im.height) * 0.05
    df = df[area < big]

    # Coerce types
    df["conf"] = pd.to_numeric(df["conf"], errors="coerce").fillna(-1).astype(float)
    for c in ["x", "y", "w", "h"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(float)

    return df[["x", "y", "w", "h", "conf", "text"]]


def map_pixels_to_pdf(df: pd.DataFrame, sx: float, sy: float, x0: float = 0.0, y0: float = 0.0) -> pd.DataFrame:
    """Convert OCR pixel coordinates to PDF coordinate space."""
    out = df.copy()
    out["x"] = (out["x"] / sx) + float(x0)
    out["y"] = (out["y"] / sy) + float(y0)
    out["w"] = out["w"] / sx
    out["h"] = out["h"] / sy
    return out


def enumerate_image_placements(page: fitz.Page) -> List[Dict[str, Any]]:
    """Find embedded images and their placements on a page."""
    placements = []
    seen = {}
    for img in page.get_images(full=True):
        xref = img[0]
        rects = page.get_image_rects(xref)
        if not rects:
            continue
        for rect in rects:
            inst = seen.get(xref, 0)
            placements.append({"xref": xref, "instance": inst, "bbox": rect})
            seen[xref] = inst + 1
    return placements


def page_mid_x(page: fitz.Page) -> float:
    """Middle x coordinate of the page (used for rough column split)."""
    r = page.rect
    return float((r.x0 + r.x1) / 2.0)


def column_id_for_bbox(x: float, w: float, mid_x: float) -> int:
    """Crude 2-column detector: left=1, right=2."""
    cx = x + w / 2.0
    return 1 if cx < mid_x else 2


# =============================================================================
# PII DETECTOR (NER + RULES)
# =============================================================================

NER = None
NER_MODEL_USED = None


def init_ner(model_name: Optional[str]) -> None:
    """Optionally initialize a HF NER pipeline."""
    global NER, NER_MODEL_USED
    NER = None
    NER_MODEL_USED = None

    if not model_name:
        return

    try:
        from transformers import pipeline

        NER = pipeline(
            "ner",
            model=model_name,
            aggregation_strategy="simple",
            device=DEFAULT_NER_DEVICE,
        )
        NER_MODEL_USED = model_name
        print(f"[i] NER enabled: {model_name}")
    except Exception as e:
        print("[i] NER disabled:", e)
        NER = None


NER_MAP = {"PER": "NAME", "ORG": "ORG", "LOC": "LOCATION", "MISC": "MISC"}

RX = {
    "EMAIL": re.compile(
        r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,24}(?![A-Za-z0-9._%+-])"
    ),
    # permissive; accepts 7–15 digits total
    "PHONE": re.compile(
        r"(?:(?<!\d)(?:\+?\d{1,3}[\s\-\.)]?)?(?:\(?\d{1,4}\)?[\s\-.]?)?\d{2,4}[\s\-.]?\d{3,4}(?:[\s\-]?\d{2,4})?(?!\d))"
    ),
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "EIN": re.compile(r"\b\d{2}-\d{7}\b"),
    "CREDIT_CARD": re.compile(
        r"(?<!\d)(?:4\d{3}|5[1-5]\d{2}|3[47]\d{2}|6(?:011|5\d{2}))(?:[ -]?\d{4}){2,3}(?!\d)"
    ),
    "IBAN": re.compile(r"\b([A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){3,7})\b"),
    "DATE": re.compile(r"\b(\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b"),
}

ROLE_EMAILS = {"hr", "info", "support", "contact", "sales", "billing", "noreply", "no-reply", "hello", "admin"}
JUNK_UPPER = {"OCR", "IBAN", "IBA", "ID", "IN"}


def is_role_email(addr: str) -> bool:
    """Detect role/local-part emails like hr@, info@, etc., or 1–2 char locals."""
    try:
        local = addr.split("@", 1)[0].lower()
        return local in ROLE_EMAILS or len(local) <= 2
    except Exception:
        return False


def _digits(s: str) -> int:
    return sum(ch.isdigit() for ch in s)


def _luhn_ok(s: str) -> bool:
    """Basic Luhn check for credit cards."""
    digs = [int(c) for c in s if c.isdigit()]
    if len(digs) < 12:
        return False
    checksum, dbl = 0, False
    for d in reversed(digs):
        v = d * 2 if dbl else d
        if v > 9:
            v -= 9
        checksum += v
        dbl = not dbl
    return checksum % 10 == 0


def _canon_phone(s: str) -> str:
    s = s.strip()
    plus = "+" if s.lstrip().startswith("+") else ""
    return plus + "".join(ch for ch in s if ch.isdigit())


def _canon_cc(s: str) -> str:
    return "".join(ch for ch in s if ch.isdigit())


def _canon_date(s: str) -> str:
    """Return YYYY-MM-DD if possible, otherwise the original string."""
    try:
        import datetime as _dt

        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
            return s
        parts = re.split(r"[/-]", s)
        if len(parts) == 3:
            a, b, c = parts
            if len(c) == 2:
                c = "20" + c
            for (y, m, d) in ((c, a, b), (c, b, a)):
                try:
                    dt = _dt.date(int(y), int(m), int(d))
                    return dt.isoformat()
                except Exception:
                    pass
    except Exception:
        pass
    return s


def _is_ner_crumb(lbl: str, txt: str) -> bool:
    """Filter tiny/upper junk from NER for ORG/LOC/MISC near strong regex hits."""
    t = (txt or "").strip()
    if not t:
        return True
    if lbl in {"ORG", "LOCATION", "MISC"} and len(t) <= 2:
        return True
    if lbl in {"ORG", "LOCATION", "MISC"} and t.isupper() and len(t) <= 4:
        return True
    if t in JUNK_UPPER:
        return True
    return False


def _rule_hits(text: str) -> List[Dict[str, Any]]:
    """Regex-based hits with canonicalization & validation (Luhn, length, etc.)."""
    hits: List[Dict[str, Any]] = []
    for label, pattern in RX.items():
        for m in pattern.finditer(text):
            s, e = m.span()
            val = text[s:e]
            canonical = val
            if label == "IBAN":
                c = re.sub(r"\s+", "", val)
                if not (15 <= len(c) <= 34):
                    continue
                canonical = c
            elif label == "CREDIT_CARD":
                if not _luhn_ok(val):
                    continue
                canonical = _canon_cc(val)
            elif label == "PHONE":
                if not (7 <= _digits(val) <= 15):
                    continue
                canonical = _canon_phone(val)
            elif label == "DATE":
                canonical = _canon_date(val)

            hits.append(
                {
                    "label": label,
                    "start": s,
                    "end": e,
                    "text": val,
                    "score": 0.99,
                    "source": "regex",
                    "canonical": canonical,
                }
            )
    return hits


def _ner_hits(text: str) -> List[Dict[str, Any]]:
    """NER-based hits (optional)."""
    if NER is None:
        return []
    out: List[Dict[str, Any]] = []
    for ent in NER(text):
        group = ent.get("entity_group") or ent.get("entity")
        mapped = NER_MAP.get(group)
        if not mapped:
            continue

        s, e = int(ent["start"]), int(ent["end"])
        piece = text[s:e]
        if _is_ner_crumb(mapped, piece):
            continue

        out.append(
            {
                "label": mapped,
                "start": s,
                "end": e,
                "text": piece,
                "score": float(ent["score"]),
                "source": "ner",
                "canonical": piece,
            }
        )
    return out


def _overlap(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return not (a[1] <= b[0] or b[1] <= a[0])


def _merge_preferring_regex(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Resolve overlaps between regex and NER spans.
    Prefer regex for strong labels; drop tiny uppercase junk hugging strong spans.
    """
    strong_labels = {"IBAN", "CREDIT_CARD", "SSN", "EMAIL"}
    strong_spans = [
        (r["start"], r["end"], r["label"])
        for r in results
        if r["source"] == "regex" and r["label"] in strong_labels
    ]

    def _near_strong(r: Dict[str, Any]) -> bool:
        if r["source"] != "ner":
            return False
        txt = r.get("text", "")
        if not (1 <= len(txt) <= 4 and txt.isupper()):
            return False
        s, e = r["start"], r["end"]
        for (as_, ae, _) in strong_spans:
            if abs(s - ae) <= 2 or abs(as_ - e) <= 2:
                return True
        return False

    # Remove NER overlaps against strong regex spans
    filtered: List[Dict[str, Any]] = []
    for r in results:
        if r["source"] == "ner":
            span = (r["start"], r["end"])
            if any(_overlap(span, (a, b)) and r["label"] != lab for (a, b, lab) in strong_spans):
                continue
            if _near_strong(r):
                continue
        filtered.append(r)

    # Sort and de-duplicate by span preference
    filtered.sort(key=lambda r: (r["start"], -(r["end"] - r["start"])))
    final: List[Dict[str, Any]] = []
    for r in filtered:
        keep = True
        for f in list(final):
            if _overlap((r["start"], r["end"]), (f["start"], f["end"])):
                pri = {"regex": 2, "ner": 1}
                if pri.get(r["source"], 0) < pri.get(f["source"], 0):
                    keep = False
                    break
                elif pri.get(r["source"], 0) > pri.get(f["source"], 0):
                    final.remove(f)
                    break
                else:
                    if (r.get("score", 0), r["end"] - r["start"]) <= (
                        f.get("score", 0),
                        f["end"] - f["start"],
                    ):
                        keep = False
                        break
                    else:
                        final.remove(f)
                        break
        if keep:
            final.append(r)

    # Exact dedupe
    seen, uniq = set(), []
    for h in final:
        key = (h["label"], h["start"], h["end"], h["canonical"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(h)
    return uniq


def detect_pii(text: str) -> List[Dict[str, Any]]:
    """Run both NER and regex, then merge with conflict resolution."""
    ner = _ner_hits(text)
    rx = _rule_hits(text)
    return _merge_preferring_regex(ner + rx)


# =============================================================================
# TEXT NORMALIZATION / GROUPING
# =============================================================================

_SPLIT_FIX = re.compile(r"\b([A-Z])\s+([a-z]{2,})\b")  # fixes "J ane" → "Jane"


def normalize_line_text(t: str) -> str:
    if not t:
        return t
    t = _SPLIT_FIX.sub(lambda m: m.group(1) + m.group(2), t)
    t = re.sub(r"\s{2,}", " ", t)
    return t.strip()


def tidy_name(n: Optional[str]) -> Optional[str]:
    if not n:
        return n
    n = normalize_line_text(n)
    n = re.sub(r"\s{2,}", " ", n).strip()
    return n


def build_lines_from_words(words_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Group words (native + OCR) into line-level rows with bounding boxes."""
    lines: List[Dict[str, Any]] = []
    if words_df.empty:
        return lines

    for (blk, lno), g in words_df.groupby(["block_id", "line_no"], dropna=False):
        g = g.sort_values(["y", "x"])
        text = " ".join(g["text"].astype(str))
        text = normalize_line_text(text)

        x0 = g["x"].min()
        y0 = g["y"].min()
        x1 = (g["x"] + g["w"]).max()
        y1 = (g["y"] + g["h"]).max()

        lines.append(
            {
                "block_id": int(blk) if pd.notna(blk) else -1,
                "line_no": int(lno) if pd.notna(lno) else -1,
                "x": float(x0),
                "y": float(y0),
                "w": float(x1 - x0),
                "h": float(y1 - y0),
                "y_center": float((y0 + y1) / 2.0),
                "text": text,
            }
        )
    return lines


def _x_overlap_ratio(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    ax0, ax1 = a["x"], a["x"] + a["w"]
    bx0, bx1 = b["x"], b["x"] + b["w"]
    inter = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    denom = max(1e-6, min(a["w"], b["w"]))
    return float(inter / denom)


def layout_score(item: Dict[str, Any], seed: Dict[str, Any]) -> float:
    """Heuristic to attach an item to a NAME/EMAIL seed based on layout proximity."""
    if item["page"] != seed["page"]:
        return -1e9

    sc = 0.0
    if item.get("column_id") == seed.get("column_id"):
        sc += 2.0

    # Vertical proximity
    dy = abs(item.get("y_center", 0) - seed.get("y_center", 0))
    sc += max(0.0, 2.5 - (dy / 40.0))

    # Horizontal overlap
    r = _x_overlap_ratio(item, seed)
    sc += (1.2 * r - 0.6)

    # Distance of centers
    cx_item = item["x"] + item["w"] / 2.0
    cx_seed = seed["x"] + seed["w"] / 2.0
    dx = abs(cx_item - cx_seed)
    if dx > 160:
        sc -= 0.8
    if dx > 280:
        sc -= 0.6

    # Block/paragraph signals
    same_block = (item.get("block_id") == seed.get("block_id") and seed.get("block_id", -999) != -1)
    if same_block:
        sc += 0.4
    else:
        sc -= 0.5
        if item.get("block_id") == -1:
            sc -= 0.6

    if item.get("block_id") == -1:
        sc -= 0.4

    return sc


def _name_email_affinity(name: str, email: str) -> float:
    """Compare name tokens vs email local-part tokens."""
    try:
        local = email.split("@", 1)[0].lower()
        parts = [p for p in re.split(r"[._+\-]+", local) if p]
        n_tokens = [t.lower() for t in re.split(r"\s+", name.strip()) if t]
        if not parts or not n_tokens:
            return 0.0
        hits = sum(1 for t in n_tokens if t in parts)
        return hits / max(1, len(n_tokens))
    except Exception:
        return 0.0


def _fix_split_name_spacing(s: str) -> str:
    """Merge split initials-tokens like 'J ane' -> 'Jane'."""
    toks = s.split()
    out = []
    i = 0
    while i < len(toks):
        if (
            i + 1 < len(toks)
            and len(toks[i]) == 1
            and toks[i].isalpha()
            and toks[i].isupper()
            and toks[i + 1].islower()
            and len(toks[i + 1]) >= 2
        ):
            out.append(toks[i] + toks[i + 1])
            i += 2
        else:
            out.append(toks[i])
            i += 1
    return " ".join(out)


def _prune_misc_near_strong(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop MISC hits when strong labels exist on the same line."""
    strong = {"EMAIL", "SSN", "CREDIT_CARD", "IBAN", "DATE", "PHONE", "NAME", "ORG", "LOCATION", "EIN"}
    by_line: Dict[Tuple[int, int, str], List[Dict[str, Any]]] = {}
    for h in hits:
        key = (h["page"], h.get("block_id"), h.get("line_text", ""))
        by_line.setdefault(key, []).append(h)

    out: List[Dict[str, Any]] = []
    for _, arr in by_line.items():
        has_strong = any(h["label"] in strong and h["label"] != "MISC" for h in arr)
        for h in arr:
            if h["label"] == "MISC" and has_strong:
                continue
            out.append(h)
    return out


def merge_adjacent_names_on_line(hits: List[Dict[str, Any]], max_gap_px: float = 10.0) -> List[Dict[str, Any]]:
    """Join adjacent NAME tokens on same line into a single NAME span."""
    names = [h for h in hits if h["label"] == "NAME"]
    others = [h for h in hits if h["label"] != "NAME"]

    by_line: Dict[Tuple[int, int, int, str], List[Dict[str, Any]]] = {}
    for h in names:
        key = (h["page"], h.get("column_id"), h.get("block_id"), h.get("line_text", ""))
        by_line.setdefault(key, []).append(h)

    merged_name_rows: List[Dict[str, Any]] = []
    for _, arr in by_line.items():
        arr.sort(key=lambda r: (r["x"], r["y"]))
        buf: List[Dict[str, Any]] = []
        for h in arr:
            if not buf:
                buf.append(dict(h))
                continue
            prev = buf[-1]
            gap = h["x"] - (prev["x"] + prev["w"])
            same_line = abs(h["y"] - prev["y"]) < 4.0
            if gap <= max_gap_px and same_line and prev["page"] == h["page"]:
                prev["end"] = max(prev.get("end", 0), h.get("end", 0))
                prev["w"] = (h["x"] + h["w"]) - prev["x"]
                prev["raw_text"] = (prev["raw_text"] + " " + h["raw_text"]).strip()
                prev["text"] = prev["raw_text"]
                prev["score"] = max(prev.get("score", 0.0), h.get("score", 0.0))
            else:
                buf.append(dict(h))
        merged_name_rows.extend(buf)

    cleaned: List[Dict[str, Any]] = []
    for h in merged_name_rows:
        if h["label"] == "NAME":
            txt = (h.get("raw_text") or "").strip()
            fixed = _fix_split_name_spacing(txt)
            if fixed:
                h["raw_text"] = fixed
                h["text"] = fixed
            txt = (h.get("raw_text") or "").strip()
            if len(txt) <= 2 and h.get("score", 0) < 0.98:
                continue
        cleaned.append(h)
    return others + cleaned


def _merge_seed_pairs(seeds: List[Dict[str, Any]], y_gap_px: float = 18.0, min_affinity: float = 0.60):
    """
    Merge NAME/EMAIL seeds into subject seeds using UFDS (disjoint-set),
    based on vertical distance and name-email affinity.
    """
    parent = list(range(len(seeds)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(len(seeds)):
        si = seeds[i]
        for j in range(i + 1, len(seeds)):
            sj = seeds[j]
            pair = {si["label"], sj["label"]}
            if pair != {"NAME", "EMAIL"}:
                continue

            layout_close = (
                si.get("block_id", -1) == sj.get("block_id", -2)
                and abs(si["y_center"] - sj["y_center"]) <= y_gap_px
            )
            name = si["raw_text"] if si["label"] == "NAME" else sj["raw_text"]
            email = sj["raw_text"] if si["label"] == "NAME" else si["raw_text"]
            aff = _name_email_affinity(name, email)
            same_line_hint = ("@" in (si.get("line_text", "")) or "@" in (sj.get("line_text", "")))

            if (aff >= min_affinity) or (layout_close and same_line_hint):
                union(i, j)

    groups: Dict[int, List[Dict[str, Any]] ] = {}
    for idx in range(len(seeds)):
        r = find(idx)
        groups.setdefault(r, []).append(seeds[idx])

    merged = []
    for _, group in groups.items():
        subj_name = None
        for s in group:
            if s["label"] == "NAME":
                subj_name = s["raw_text"]
                break
        merged.append({"seeds": group, "subject_name": subj_name})
    return merged


def _pretty_name_from_email(email: str) -> Optional[str]:
    """Guess 'First Last' from email local-part tokens."""
    try:
        local = email.split("@", 1)[0]
        parts = [p for p in re.split(r"[._+\-]+", local) if p]
        if not parts:
            return None
        return " ".join(p.capitalize() for p in parts[:3])
    except Exception:
        return None


def group_locations_on_line(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Roll up multiple LOCATION crumbs on same line into a single representative
    (keep the longest piece from that line).
    """
    by_key: Dict[Tuple[int, int, int, str], List[Dict[str, Any]]] = {}
    for h in hits:
        key = (h["page"], h.get("column_id"), h.get("block_id"), h.get("line_text", ""))
        by_key.setdefault(key, []).append(h)

    out: List[Dict[str, Any]] = []
    for _, arr in by_key.items():
        locs = [a for a in arr if a["label"] == "LOCATION"]
        if len(locs) <= 1:
            out.extend(arr)
            continue

        # keep the longest LOCATION as representative
        locs_sorted = sorted(locs, key=lambda r: len(r.get("raw_text", "")), reverse=True)
        rep = locs_sorted[0]
        out.extend([a for a in arr if a["label"] != "LOCATION"])
        out.append(rep)

    return out


# =============================================================================
# CACHING (page-level)
# =============================================================================

def _page_cache_key(doc_path: Path, page_no: int, dpi: int, lang: str, conf: float) -> str:
    """Create a key that reflects file, page, OCR params, and native text length."""
    h = hashlib.sha1()
    h.update(str(doc_path.resolve()).encode())
    h.update(f":p{page_no}:dpi{dpi}:lang{lang}:conf{conf}".encode())
    try:
        with fitz.open(doc_path) as d:
            t = d[page_no - 1].get_text("text") or ""
        h.update(str(len(t)).encode())
    except Exception:
        pass
    return h.hexdigest()


def _cache_dir() -> Path:
    p = Path(tempfile.gettempdir()) / "pii_pdf_cache"
    p.mkdir(parents=True, exist_ok=True)
    return p


# =============================================================================
# MAIN WORKER (PROCESS PDF)
# =============================================================================

def process_pdf_worker(
    pdf_path: Path,
    dpi: int = 300,
    lang: str = "eng",
    conf: float = 60.0,
    overlap_iou: float = 0.2,  # reserved, currently not used
    start_page: int = 1,
    end_page: Optional[int] = None,
    fallback_fullpage_ocr: bool = True,
    role_email_keep_if_adjacent: bool = True,
    ner_model: Optional[str] = None,
    tesseract_path: Optional[str] = None,
    log_jsonl: Optional[Path] = None,
):
    """
    Process a single PDF:
      - Extract native text & OCR image regions
      - Build lines, detect PII
      - Group by subjects (NAME/EMAIL seeds)
      - Emit words_split.csv, detections.csv and preview PNGs
    """

    if tesseract_path and str(tesseract_path).strip():
        pytesseract.pytesseract.tesseract_cmd = str(tesseract_path).strip()

    init_ner(ner_model)

    doc = fitz.open(pdf_path)
    n_pages = len(doc)
    if end_page is None or end_page > n_pages:
        end_page = n_pages
    if start_page < 1:
        start_page = 1
    if start_page > end_page:
        start_page, end_page = 1, n_pages

    run_id = uuid.uuid4().hex[:10]
    out_dir = Path(tempfile.gettempdir()) / "gradio" / run_id / (pdf_path.stem + "_split")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_tokens: List[pd.DataFrame] = []
    gallery_items: List[Tuple[str, str]] = []
    page_logs: List[str] = []
    all_detections_rows: List[Dict[str, Any]] = []

    sid = 1
    t0 = time.time()

    for pno in range(start_page - 1, end_page):
        t_page = time.time()
        page = doc[pno]
        mid_x = page_mid_x(page)

        # ---------- caching ----------
        cache_key = _page_cache_key(pdf_path, pno + 1, dpi, lang, conf)
        cache_file = _cache_dir() / f"{cache_key}_tokens.csv"

        if cache_file.exists():
            tokens = pd.read_csv(cache_file)
            native = tokens[tokens["source"] == "native"].copy()
            ocr_pdf = tokens[tokens["source"] != "native"].copy()
        else:
            # Native text
            native = words_from_pdf(page)
            native.insert(0, "page", pno + 1)

            # OCR (per-image)
            raster, sx, sy = rasterize_page(page, dpi=dpi)
            placements = enumerate_image_placements(page)

            ocr_rows = []
            for p in placements:
                x0, y0, x1, y1 = tuple(p["bbox"])
                crop = raster.crop((int(x0 * sx), int(y0 * sy), int(x1 * sx), int(y1 * sy)))
                df = ocr_image(crop, lang=lang, psm=6, oem=1)
                if df.empty:
                    continue
                df = df[df["conf"] >= float(conf)].copy()
                df_pdf = map_pixels_to_pdf(df, sx, sy, x0=x0, y0=y0)
                df_pdf = df_pdf.assign(block_id=-1, line_no=-1)
                df_pdf.insert(0, "page", pno + 1)
                df_pdf["source"] = "ocr_img"
                ocr_rows.append(
                    df_pdf[
                        ["page", "x", "y", "w", "h", "conf", "text", "source", "block_id", "line_no"]
                    ]
                )

            ocr_pdf = (
                pd.concat(ocr_rows, ignore_index=True)
                if ocr_rows
                else pd.DataFrame(
                    columns=["page", "x", "y", "w", "h", "conf", "text", "source", "block_id", "line_no"]
                )
            )

            # Optional masked full-page OCR (fall back when needed)
            if fallback_fullpage_ocr and (len(placements) == 0 or len(ocr_pdf) < 5):
                masked = raster.copy()
                if not native.empty:
                    draw_mask = ImageDraw.Draw(masked)
                    for _, r in native.iterrows():
                        draw_mask.rectangle(
                            [int(r.x * sx), int(r.y * sy), int((r.x + r.w) * sx), int((r.y + r.h) * sy)],
                            fill="white",
                        )
                full_df = ocr_image(masked, lang=lang, psm=6, oem=1)
                full_df = full_df[full_df["conf"] >= float(conf)].copy()
                if not full_df.empty:
                    full_pdf = map_pixels_to_pdf(full_df, sx, sy)
                    full_pdf = full_pdf.assign(block_id=-1, line_no=-1)
                    full_pdf.insert(0, "page", pno + 1)
                    full_pdf["source"] = "ocr_page"
                    ocr_pdf = pd.concat([ocr_pdf, full_pdf], ignore_index=True)

            # Merge native + OCR tokens and cache
            cols = ["page", "x", "y", "w", "h", "conf", "text", "source", "block_id", "line_no"]
            native = native.reindex(columns=cols, fill_value=-1)
            tokens = pd.concat([native, ocr_pdf], ignore_index=True)
            tokens.to_csv(cache_file, index=False)

        # ---------- build page preview image ----------
        try:
            raster, sx, sy = rasterize_page(page, dpi=dpi)
            prev = raster.copy()
            draw = ImageDraw.Draw(prev)

            for _, r in native.iterrows():
                draw.rectangle(
                    [int(r.x * sx), int(r.y * sy), int((r.x + r.w) * sx), int((r.y + r.h) * sy)],
                    outline=(0, 255, 0),
                    width=2,
                )
            for _, r in ocr_pdf.iterrows():
                draw.rectangle(
                    [int(r.x * sx), int(r.y * sy), int((r.x + r.w) * sx), int((r.y + r.h) * sy)],
                    outline=(0, 0, 255),
                    width=2,
                )

            prev_path = out_dir / f"page_{pno + 1:03d}_preview.png"
            prev.save(prev_path)
            gallery_items.append((str(prev_path), f"Page {pno + 1}"))
        except Exception:
            pass

        # ---------- group tokens into lines ----------
        lines = build_lines_from_words(tokens)
        for ln in lines:
            ln["page"] = pno + 1
            ln["column_id"] = column_id_for_bbox(ln["x"], ln["w"], mid_x)

        # ---------- detect PII per line ----------
        raw_hits_this_page: List[Dict[str, Any]] = []
        for ln in lines:
            detections = detect_pii(ln["text"])
            for d in detections:
                raw_hits_this_page.append(
                    {
                        "page": pno + 1,
                        "label": d["label"],
                        "text": d["canonical"],
                        "raw_text": d["text"],
                        "score": float(d.get("score", 0.0)),
                        "source": d.get("source", "regex"),
                        "start": int(d["start"]),
                        "end": int(d["end"]),
                        "x": ln["x"],
                        "y": ln["y"],
                        "w": ln["w"],
                        "h": ln["h"],
                        "y_center": ln["y_center"],
                        "block_id": ln["block_id"],
                        "column_id": ln["column_id"],
                        "line_text": ln["text"],
                    }
                )

        # Cleanups & enrichment
        raw_hits_this_page = merge_adjacent_names_on_line(raw_hits_this_page, max_gap_px=10.0)
        raw_hits_this_page = _prune_misc_near_strong(raw_hits_this_page)
        raw_hits_this_page = group_locations_on_line(raw_hits_this_page)

        # ---------- subject grouping (seeded by NAME/EMAIL) ----------
        seed_candidates = [
            h for h in raw_hits_this_page if h["label"] in {"NAME", "EMAIL"} and h.get("block_id", -1) != -1
        ]

        # paragraph lines (block_id == -1) with NAME/EMAIL pairs
        by_line_para: Dict[str, List[Dict[str, Any]]] = {}
        for h in raw_hits_this_page:
            if h.get("block_id") == -1 and h["label"] in {"NAME", "EMAIL"}:
                by_line_para.setdefault(h.get("line_text", ""), []).append(h)

        para_pairs: List[Dict[str, Any]] = []
        for line_text, arr in by_line_para.items():
            names = [a for a in arr if a["label"] == "NAME"]
            emails = [a for a in arr if a["label"] == "EMAIL" and not is_role_email(a["raw_text"])]
            for n in names:
                for e in emails:
                    aff = _name_email_affinity(n["raw_text"], e["raw_text"])
                    if aff >= 0.55 or "@" in line_text:
                        para_pairs.extend([n, e])

        def _key(h):
            return (h["label"], h["page"], h["x"], h["y"], h["w"], h["h"])

        seen_keys = {_key(s) for s in seed_candidates}
        for h in para_pairs:
            if _key(h) not in seen_keys:
                seed_candidates.append(h)
                seen_keys.add(_key(h))

        # role-email filtering by policy
        seeds: List[Dict[str, Any]] = []
        for h in seed_candidates:
            if h["label"] == "EMAIL":
                if is_role_email(h["raw_text"]) and not role_email_keep_if_adjacent:
                    continue
                if is_role_email(h["raw_text"]) and role_email_keep_if_adjacent:
                    same_line_name = any(
                        (x["label"] == "NAME" and x.get("block_id") == h.get("block_id") and x.get("line_text") == h.get("line_text"))
                        for x in seed_candidates
                    )
                    if not same_line_name:
                        continue
            seeds.append(h)

        merged_seed_groups = _merge_seed_pairs(seeds, y_gap_px=28.0, min_affinity=0.50)

        # build subjects
        subjects: List[Dict[str, Any]] = []
        for g in merged_seed_groups:
            subj_name = g["subject_name"]
            if not subj_name:
                emails = [s["raw_text"] for s in g["seeds"] if s["label"] == "EMAIL" and s["raw_text"]]
                if emails:
                    guess = _pretty_name_from_email(emails[0])
                    if guess:
                        subj_name = guess

            subj_name = tidy_name(subj_name) or ""
            subjects.append({"subject_id": sid, "name": subj_name, "seeds": g["seeds"], "items": []})
            sid += 1

        leftovers: List[Dict[str, Any]] = []
        for h in raw_hits_this_page:
            if h in seeds:
                continue

            best, best_sc, best_seed = None, -1e9, None
            for subj in subjects:
                for s in subj["seeds"]:
                    sc = layout_score(h, s)

                    if h["label"] == "EMAIL":
                        name_seeds = [t for t in subj["seeds"] if t["label"] == "NAME" and t.get("raw_text")]
                        for ns in name_seeds:
                            sc += 0.8 * _name_email_affinity(ns["raw_text"], h["raw_text"])
                        if is_role_email(h["raw_text"]):
                            sc -= 1.0

                    if sc > best_sc:
                        best, best_sc, best_seed = subj, sc, s

            def _required_score(item: Dict[str, Any], seed: Optional[Dict[str, Any]]) -> float:
                base = 2.6
                if not seed:
                    return base
                if item.get("block_id") != seed.get("block_id"):
                    base += 0.7
                if item.get("block_id") == -1:
                    base += 0.6
                if seed:
                    r = _x_overlap_ratio(item, seed)
                    if r < 0.20:
                        base += 0.5
                return base

            need = _required_score(h, best_seed)
            if best and best_sc >= need:
                best["items"].append(h)
            else:
                leftovers.append(h)

        # Attach leftovers by nearest y-center subject on same page
        for h in leftovers:
            candidates: List[Tuple[float, Dict[str, Any]]] = []
            for subj in subjects:
                for s in subj["seeds"]:
                    if s["page"] == h["page"]:
                        candidates.append((abs(h["y_center"] - s["y_center"]), subj))
            if candidates:
                _, subj = min(candidates, key=lambda t: t[0])
                subj["items"].append(h)
            else:
                subjects.append({"subject_id": sid, "name": None, "seeds": [], "items": [h]})
                sid += 1

        # Emit per-item rows
        for subj in subjects:
            subject_id = subj["subject_id"]
            subject_name = subj["name"] or ""
            pack = subj["seeds"] + subj["items"]

            for h in pack:
                if h["label"] == "NAME":
                    fixed = _fix_split_name_spacing(h.get("raw_text", ""))
                    if fixed:
                        h["raw_text"] = fixed
                        h["text"] = fixed

                all_detections_rows.append(
                    {
                        "page": h["page"],
                        "subject_id": subject_id,
                        "subject_name": subject_name,
                        "label": h["label"],
                        "text": h["text"],
                        "raw_text": h["raw_text"],
                        "score": h["score"],
                        "source": h["source"],
                        "x": h["x"],
                        "y": h["y"],
                        "w": h["w"],
                        "h": h["h"],
                        "column_id": h["column_id"],
                        "block_id": h["block_id"],
                        "y_center": h["y_center"],
                        "line_text": h["line_text"],
                    }
                )

        all_tokens.append(tokens)

        dt = time.time() - t_page
        page_logs.append(f"p{pno + 1}: tokens={len(tokens)} time={dt:.3f}s")
        if log_jsonl:
            with open(log_jsonl, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {"event": "page_done", "page": pno + 1, "tokens": len(tokens), "time_s": dt}
                    )
                    + "\n"
                )

    doc.close()

    # ---------- persist outputs ----------
    words_df = (
        pd.concat(all_tokens, ignore_index=True)
        if all_tokens
        else pd.DataFrame(columns=["page", "x", "y", "w", "h", "conf", "text", "source", "block_id", "line_no"])
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    words_csv = out_dir / "words_split.csv"
    words_df.to_csv(words_csv, index=False, encoding="utf-8")

    det_df = (
        pd.DataFrame(all_detections_rows)
        if all_detections_rows
        else pd.DataFrame(
            columns=[
                "page",
                "subject_id",
                "subject_name",
                "label",
                "text",
                "raw_text",
                "score",
                "source",
                "x",
                "y",
                "w",
                "h",
                "column_id",
                "block_id",
                "y_center",
                "line_text",
            ]
        )
    )
    det_csv = out_dir / "detections.csv"
    det_df.to_csv(det_csv, index=False, encoding="utf-8")

    total_t = time.time() - t0
    header = f"✅ Processed {pdf_path.name} | Pages {start_page}–{end_page} of {n_pages} | DPI={dpi}, Lang={lang}, conf≥{conf}"
    footer = f"📁 Outputs: {out_dir.as_posix()} | ⏱ {total_t:.2f}s"
    log = header + "\n" + footer + "\n" + " | ".join(page_logs)

    if log_jsonl:
        with open(log_jsonl, "a", encoding="utf-8") as f:
            f.write(
                json.dumps({"event": "run_done", "pages": end_page - start_page + 1, "time_s": total_t})
                + "\n"
            )

    return log, str(words_csv), str(det_csv), gallery_items, str(out_dir)


# =============================================================================
# POST-PROCESSOR
# =============================================================================

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def coerce_numeric(df: pd.DataFrame, cols: List[str]):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def add_risk(df: pd.DataFrame, risk_weights: Dict[str, int]) -> pd.DataFrame:
    """Compute risk = weight(label) * confidence."""
    df = df.copy()
    df["score"] = pd.to_numeric(df.get("score", 1.0), errors="coerce").fillna(0.0).clip(0, 1)
    df["risk_unit"] = df["label"].map(risk_weights).fillna(0)
    df["risk"] = df["risk_unit"] * df["score"]
    return df


def chart_bar(series: pd.Series, title: str, outpath: Path, xlabel: str = "", ylabel: str = "Count"):
    """Simple bar chart helper."""
    if series is None or series.empty:
        return
    plt.figure(figsize=(8, 5))
    series.sort_values(ascending=False).plot(kind="bar")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


# ---- Deterministic subject key helpers ------------------------------------------------

def _norm(s: str) -> str:
    return (s or "").strip().lower()


def subject_stable_key(best_name: str, best_email: str) -> str:
    """sha1(lower(name)|lower(email)) to match across pages/files."""
    base = f"{_norm(best_name)}|{_norm(best_email)}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()


def _fmt_dup_value(val: str, score: float, page: Any, block: Any, cfg: Dict[str, Any]) -> str:
    """Format a value with |score: and |page:|block: tags per policy."""
    if val is None:
        val = ""
    out = str(val).strip()
    if cfg["include_confidence"]:
        out += f"|score:{score:.2f}"
    if cfg["include_provenance"]:
        out += f"|page:{page}|block:{block}"
    return out


def _emit_dup_values(values: List[Tuple[str, float, int, int]], min_cfg: Dict[str, Any]) -> str:
    """Join or top-k the values, ordered by (score desc, length desc)."""
    if not values:
        return ""
    mode = min_cfg["mode"]
    sep = min_cfg["separator"]

    if mode == "topk":
        k = int(min_cfg.get("k", 1))
        values = sorted(values, key=lambda t: (t[1], len(str(t[0]))), reverse=True)[:k]
    else:
        values = sorted(values, key=lambda t: (t[1], len(str(t[0]))), reverse=True)

    return sep.join(_fmt_dup_value(v, s, p, b, min_cfg) for (v, s, p, b) in values)


def _build_quality_report(min_df: pd.DataFrame, out_dir: Path) -> Path:
    """Emit quality_report.json with basic completeness and suspicious patterns."""
    total = len(min_df)
    pct = lambda n: (0 if total == 0 else round(100.0 * n / total, 2))

    suspicious_phones: List[str] = []
    for _, r in min_df.iterrows():
        ph = str(r.get("phone", "") or "")
        first = ph.split(";")[0] if ";" in ph else ph
        digs = sum(ch.isdigit() for ch in first)
        if 0 < digs < 7:
            suspicious_phones.append(r.get("name", ""))

    report = {
        "subjects_total": total,
        "pct_with_email": pct(min_df["email"].astype(str).str.strip().ne("").sum()),
        "pct_with_phone": pct(min_df["phone"].astype(str).str.strip().ne("").sum()),
        "pct_with_date": pct(min_df["date"].astype(str).str.strip().ne("").sum()),
        "suspicious_phones_count": len(suspicious_phones),
        "suspicious_phones_names": suspicious_phones[:50],
    }

    p = out_dir / "quality_report.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return p


def _jsonl_export(min_df: pd.DataFrame, out_dir: Path) -> Path:
    """Line-delimited JSON mirroring detections_min.csv."""
    p = out_dir / "detections_min.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for _, r in min_df.iterrows():
            f.write(json.dumps({k: (None if pd.isna(v) else v) for k, v in r.items()}, ensure_ascii=False) + "\n")
    return p


def _apply_scrub(df: pd.DataFrame, scrub_raw_text: bool) -> pd.DataFrame:
    """Optionally remove raw_text/line_text fields from downstream artifacts."""
    if not scrub_raw_text:
        return df
    keep = [c for c in df.columns if c not in {"raw_text", "line_text"}]
    return df[keep].copy()


# ---- subject-first merge → detections_min.csv ----------------------------------------

def build_detections_min(df: pd.DataFrame, outdir: Path, cfg: Dict[str, Any]) -> Path:
    """
    Build subject-level table with deterministic subject_key and duplicate handling.
    Keeps provenance columns for quick auditing of best entry per field.
    """
    desired_cols = ["subject_key", "name", "email", "phone", "ssn", "credit", "iban", "date", "location", "org"]
    prov_cols = [
        "email_source_page",
        "email_block_id",
        "email_score",
        "phone_source_page",
        "phone_block_id",
        "phone_score",
        "ssn_source_page",
        "ssn_block_id",
        "ssn_score",
        "credit_source_page",
        "credit_block_id",
        "credit_score",
        "iban_source_page",
        "iban_block_id",
        "iban_score",
        "date_source_page",
        "date_block_id",
        "date_score",
        "location_source_page",
        "location_block_id",
        "location_score",
        "org_source_page",
        "org_block_id",
        "org_score",
    ]

    label_to_col = cfg["LABEL_TO_MINCOL"]
    dup_cfg = cfg["MIN_DUP_POLICY"]

    out_path = outdir / "detections_min.csv"
    if df is None or df.empty:
        pd.DataFrame(columns=desired_cols + prov_cols).to_csv(out_path, index=False)
        return out_path

    tmp = df.copy()
    tmp["label_col"] = tmp["label"].map(label_to_col)
    tmp = tmp[tmp["label_col"].notna()].copy()
    if "score" in tmp.columns:
        tmp["score"] = pd.to_numeric(tmp["score"], errors="coerce").fillna(0.0)
    else:
        tmp["score"] = 0.0

    # subjkey is the local "page::subject_id" (used internally for grouping before stable key)
    tmp["subjkey"] = tmp["page"].astype(str) + "::" + tmp["subject_id"].astype(str)

    # Best per (subjkey, label_col) but also keep duplicates for list/topk emission
    tmp_sorted = tmp.sort_values(["subjkey", "label_col", "score"], ascending=[True, True, False])
    best_per_subj = tmp_sorted.drop_duplicates(subset=["subjkey", "label_col"], keep="first")
    dup_groups = tmp_sorted.groupby(["subjkey", "label_col"])

    # Best name/email per subjkey for deterministic subject key + display
    names = (
        best_per_subj[best_per_subj["label_col"] == "name"][["subjkey", "text", "score"]]
        .rename(columns={"text": "best_name", "score": "best_name_score"})
        .copy()
    )
    emails = (
        best_per_subj[best_per_subj["label_col"] == "email"][["subjkey", "text", "score", "page", "block_id"]]
        .rename(columns={"text": "best_email", "score": "best_email_score", "page": "best_email_page", "block_id": "best_email_block"})
        .copy()
    )

    # Prefer non-role emails if available (but keep role if none)
    if not emails.empty:
        emails["email_is_role"] = emails["best_email"].apply(lambda e: is_role_email(str(e)) if pd.notna(e) else False)
        emails = (
            emails.sort_values(["subjkey", "email_is_role", "best_email_score"], ascending=[True, True, False])
            .drop_duplicates("subjkey", keep="first")
        )

    keyed = pd.merge(best_per_subj, names, on="subjkey", how="left")
    keyed = pd.merge(keyed, emails, on="subjkey", how="left")

    per_subj = keyed.groupby("subjkey")

    buckets: Dict[str, Dict[str, Any]] = {}
    bucket_scores: Dict[str, Dict[str, float]] = {}

    def _display_name(nm: str, em: str) -> str:
        nm = (nm or "").strip()
        if nm:
            return nm
        pretty = _pretty_name_from_email(em or "") or (em or "")
        return pretty

    for subj, g in per_subj:
        row = g.iloc[0]
        best_nm = str(row.get("best_name") or "")
        best_em = str(row.get("best_email") or "")

        subj_key = subject_stable_key(best_nm, best_em)
        disp_name = _display_name(best_nm, best_em)

        if subj_key not in buckets:
            buckets[subj_key] = {c: "" for c in desired_cols + prov_cols}
            buckets[subj_key]["subject_key"] = subj_key
            buckets[subj_key]["name"] = disp_name
            bucket_scores[subj_key] = {c: -1.0 for c in desired_cols}
            bucket_scores[subj_key]["name"] = float(row.get("best_name_score", 0.0))
        else:
            sc = float(row.get("best_name_score", 0.0))
            if sc > bucket_scores[subj_key]["name"]:
                buckets[subj_key]["name"] = disp_name
                bucket_scores[subj_key]["name"] = sc

        # Collect duplicates/provenance for each label_col within this subj
        for label_col in ["email", "phone", "ssn", "credit", "iban", "date", "location", "org"]:
            group = dup_groups.get_group((subj, label_col)) if (subj, label_col) in dup_groups.groups else None
            if group is None or group.empty:
                continue

            values: List[Tuple[str, float, int, int]] = []
            for _, r in group.iterrows():
                values.append(
                    (
                        str(r.get("text", "")).strip(),
                        float(r.get("score", 0.0)),
                        int(r.get("page", 0)),
                        int(r.get("block_id", -1)),
                    )
                )

            joined = _emit_dup_values(values, dup_cfg)
            if joined and joined != buckets[subj_key].get(label_col, ""):
                best = max(values, key=lambda t: t[1])
                buckets[subj_key][label_col] = joined
                buckets[subj_key][f"{label_col}_source_page"] = best[2]
                buckets[subj_key][f"{label_col}_block_id"] = best[3]
                buckets[subj_key][f"{label_col}_score"] = best[1]
                bucket_scores[subj_key][label_col] = best[1]

    out_df = pd.DataFrame(list(buckets.values()), columns=desired_cols + prov_cols)
    out_df.to_csv(out_path, index=False)
    return out_path


def _quality_and_exports(min_csv_path: Path, out_dir: Path) -> Tuple[Path, Path]:
    min_df = pd.read_csv(min_csv_path, dtype=str, keep_default_na=False)
    quality = _build_quality_report(min_df, out_dir)
    jsonl = _jsonl_export(min_df, out_dir)
    return quality, jsonl


def postprocess_detections(
    detections_csv: Union[str, Path],
    out_dir: Optional[Union[str, Path]] = None,
    previews_dir: Optional[Union[str, Path]] = None,
    min_conf: float = 0.85,
    cfg: Dict[str, Any] = DEFAULT_CONFIG,
    scrub_raw_text: bool = False,
):
    """
    Post-process detections.csv to:
      - detections_min.csv (+subject_key, duplicates, provenance)
      - detections_min.jsonl
      - quality_report.json
      - summary charts (counts, top subjects, hits per page)
      - optional annotated previews (draw boxes on preview PNGs)
      - redaction_worklist.csv (for review)
    """

    inp = Path(detections_csv)
    if not inp.exists():
        raise FileNotFoundError(f"detections.csv not found: {inp}")

    outdir = Path(out_dir) if out_dir else (inp.parent / "out")
    safe_mkdir(outdir)

    df = pd.read_csv(inp, dtype=str, keep_default_na=False)
    if scrub_raw_text:
        df = _apply_scrub(df, scrub_raw_text)

    coerce_numeric(df, ["page", "x", "y", "w", "h", "score", "y_center", "column_id", "block_id"])
    df["label"] = df.get("label", "").astype(str)
    df["subject_id"] = df.get("subject_id", "").astype(str)
    df["subject_name"] = df.get("subject_name", "").astype(str)

    df = add_risk(df, cfg["RISK_WEIGHTS"])

    # detections_min (with subject_key + dup handling + provenance)
    detections_min_path = build_detections_min(df, outdir, cfg)

    # Counts
    counts = df["label"].value_counts()
    (outdir / "counts_by_label.csv").write_text(counts.to_csv(header=["count"]), encoding="utf-8")

    # Risk by subject
    by_subject = (
        df.groupby(["subject_id", "subject_name"], dropna=False)
        .agg(hits=("label", "count"), risk=("risk", "sum"), max_conf=("score", "max"))
        .reset_index()
        .sort_values(["risk", "hits"], ascending=False)
    )
    risk_by_subject_csv = outdir / "risk_by_subject.csv"
    by_subject.to_csv(risk_by_subject_csv, index=False)

    # Risk by page
    by_page = (
        df.groupby("page", dropna=False)
        .agg(hits=("label", "count"), risk=("risk", "sum"))
        .reset_index()
        .sort_values("page")
    )
    risk_by_page_csv = outdir / "risk_by_page.csv"
    by_page.to_csv(risk_by_page_csv, index=False)

    # Redaction worklist (for reviewers)
    redaction = df[df["label"].isin(cfg["TEAM_SENSITIVE"]) & (df["score"] >= float(min_conf))].copy()
    redact_cols = [
        c
        for c in [
            "page",
            "x",
            "y",
            "w",
            "h",
            "label",
            "subject_id",
            "subject_name",
            "raw_text",
            "line_text",
            "score",
            "source",
        ]
        if c in redaction.columns
    ]
    redaction = redaction[redact_cols].sort_values(["page", "y", "x"])
    redaction_csv = outdir / "redaction_worklist.csv"
    redaction.to_csv(redaction_csv, index=False)

    # HR contacts quick view
    hr = df[df["label"].isin(cfg["HR_CONTACT_LABELS"])].copy()
    hr = hr.sort_values(["subject_id", "label", "score"], ascending=[True, True, False])
    hr = hr.drop_duplicates(subset=["subject_id", "label"], keep="first")
    pivot_hr = (
        hr.pivot_table(index=["subject_id", "subject_name"], columns="label", values="text", aggfunc="first")
        .reset_index()
        .copy()
    )
    hr_cols = [c for c in ["subject_id", "subject_name", "NAME", "EMAIL", "PHONE"] if c in pivot_hr.columns]
    contacts_hr_csv = outdir / "contacts_hr.csv"
    pivot_hr[hr_cols].to_csv(contacts_hr_csv, index=False)

    # Charts
    labels_png = outdir / "labels_bar.png"
    top_subjects_png = outdir / "top_subjects_risk.png"
    hits_page_png = outdir / "hits_per_page.png"

    chart_bar(counts, "PII labels (count)", labels_png, xlabel="Label")

    top_subj = by_subject.set_index("subject_name")["risk"].head(10)
    chart_bar(top_subj, "Top subjects by risk", top_subjects_png, xlabel="Subject", ylabel="Risk")

    page_hits = by_page.set_index("page")["hits"]
    chart_bar(page_hits, "Hits per page", hits_page_png, xlabel="Page")

    # Annotate page previews
    annotated_dir = None
    if previews_dir and all(col in df.columns for col in ["page", "x", "y", "w", "h"]):
        try:
            prev_dir = Path(previews_dir)
            annotated_dir = outdir / "previews_annotated"
            safe_mkdir(annotated_dir)

            g = df[(df["score"] >= float(min_conf))].groupby("page")
            for page_num, rows in g:
                candidates = [
                    f"page_{int(page_num):03d}_preview.png",
                    f"page_{int(page_num):02d}_preview.png",
                    f"page_{int(page_num)}_preview.png",
                ]
                img_path = None
                for name in candidates:
                    p = prev_dir / name
                    if p.exists():
                        img_path = p
                        break
                if not img_path:
                    continue

                im = Image.open(img_path).convert("RGB")
                draw = ImageDraw.Draw(im)
                for _, r in rows.iterrows():
                    x0, y0 = float(r["x"]), float(r["y"])
                    x1, y1 = x0 + float(r["w"]), y0 + float(r["h"])
                    draw.rectangle([x0, y0, x1, y1], outline=0, width=2)
                im.save(annotated_dir / f"page_{int(page_num):03d}_annotated.png")
        except Exception as e:
            print("[warn] preview annotation skipped:", e)

    # Quality + JSONL
    quality_json, jsonl_path = _quality_and_exports(Path(detections_min_path), outdir)

    # Collect artifact paths
    artifacts = [
        detections_min_path,
        str(outdir / "counts_by_label.csv"),
        str(risk_by_subject_csv),
        str(risk_by_page_csv),
        str(redaction_csv),
        str(contacts_hr_csv),
        str(labels_png),
        str(top_subjects_png),
        str(hits_page_png),
        str(quality_json),
        str(jsonl_path),
    ]
    if annotated_dir and annotated_dir.exists():
        artifacts.extend(sorted(str(p) for p in annotated_dir.glob("*.png")))

    artifacts = [str(p) for p in artifacts]
    labels_png = str(labels_png)
    top_subjects_png = str(top_subjects_png)
    hits_page_png = str(hits_page_png)

    summary = (
        f"✅ Wrote artifacts to: {outdir.resolve()}\n"
        f" - detections_min.csv (+ subject_key, duplicates, provenance)\n"
        f" - detections_min.jsonl, quality_report.json\n"
        f" - counts_by_label.csv, risk_by_subject.csv, risk_by_page.csv\n"
        f" - redaction_worklist.csv, contacts_hr.csv\n"
        f" - labels_bar.png, top_subjects_risk.png, hits_per_page.png\n"
        + (
            f" - previews_annotated/ ({len(list((Path(annotated_dir) if annotated_dir else Path()).glob('*.png')))} pages)\n"
            if annotated_dir
            else ""
        )
    )

    return summary, artifacts, str(labels_png), str(top_subjects_png), str(hits_page_png)


# =============================================================================
# BATCH / CLI
# =============================================================================

def run_batch(
    input_path: Path,
    lang: str,
    dpi: int,
    conf: float,
    ner_model: Optional[str],
    scrub: bool,
    cfg: Dict[str, Any],
) -> None:
    """Batch-process all PDFs under a folder or a single PDF path."""
    pdfs: List[Path] = []
    if input_path.is_dir():
        pdfs = sorted([p for p in input_path.rglob("*.pdf")])
    elif input_path.suffix.lower() == ".pdf":
        pdfs = [input_path]
    else:
        print("[batch] No PDFs found")
        return

    for pdf in pdfs:
        print(f"[batch] Processing {pdf}")
        log_jsonl = _cache_dir() / f"{pdf.stem}_runlog.jsonl"
        log, words_csv, det_csv, gallery, out_dir = process_pdf_worker(
            pdf_path=pdf,
            dpi=dpi,
            lang=lang,
            conf=conf,
            overlap_iou=0.2,
            start_page=1,
            end_page=None,
            fallback_fullpage_ocr=True,
            role_email_keep_if_adjacent=cfg["ROLE_EMAIL_POLICY"]["keep_if_adjacent_to_name"],
            ner_model=ner_model,
            tesseract_path=None,
            log_jsonl=log_jsonl,
        )
        print(log)

        summary, _, *_ = postprocess_detections(
            detections_csv=det_csv,
            out_dir=None,
            previews_dir=out_dir,
            min_conf=cfg["DEFAULT_MIN_CONF"],
            cfg=cfg,
            scrub_raw_text=scrub,
        )
        print(summary)


# =============================================================================
# GRADIO APP (v4)
# =============================================================================

_LANG_OPTIONS = [
    "eng",

]

# Some common NER models
_NER_MODELS = [
    "",  # none
    "dslim/bert-base-NER",  # English

]


def _coerce_lang_value(sel: Union[str, List[str]]) -> str:
    if sel is None:
        return "eng"
    if isinstance(sel, list):
        return "+".join([s for s in sel if s]) or "eng"
    return str(sel) or "eng"


def _open_folder_html(path_str: str) -> str:
    p = Path(path_str)
    return f'<div>Outputs: <a href="file:///{p.as_posix()}" target="_blank">{p}</a></div>'


def gr_process(
    file_obj,
    dpi,
    lang_sel,
    conf,
    role_keep,
    ner_model_sel,
    tesseract_path,
    do_fallback_fullpage_ocr,
):
    pdf_path = coerce_path(file_obj)
    lang = _coerce_lang_value(lang_sel)

    log, words_csv, det_csv, gallery, out_dir = process_pdf_worker(
        pdf_path=pdf_path,
        dpi=int(dpi),
        lang=lang,
        conf=float(conf),
        overlap_iou=0.2,
        start_page=1,
        end_page=None,
        fallback_fullpage_ocr=bool(do_fallback_fullpage_ocr),
        role_email_keep_if_adjacent=bool(role_keep),
        ner_model=(ner_model_sel or None),
        tesseract_path=str(tesseract_path or "").strip() or None,
        log_jsonl=None,
    )
    return log + "\n" + _open_folder_html(out_dir), words_csv, det_csv, gallery, out_dir


def gr_postprocess(
    detections_csv_file,
    out_dir_text,
    previews_dir_text,
    min_conf,
    yaml_config_file,
    scrub_raw_text,
):
    cfg = load_config(
        yaml_config_file["path"] if isinstance(yaml_config_file, dict) else (yaml_config_file or None)
    )
    det_path = coerce_path(detections_csv_file)
    out_dir = (out_dir_text or "").strip() or None
    previews_dir = (previews_dir_text or "").strip() or None

    summary, files, labels_png, top_subj_png, hits_png = postprocess_detections(
        detections_csv=det_path,
        out_dir=out_dir,
        previews_dir=previews_dir,
        min_conf=float(min_conf),
        cfg=cfg,
        scrub_raw_text=bool(scrub_raw_text),
    )
    files = [str(p) for p in files]
    charts = [str(labels_png), str(top_subj_png), str(hits_png)]
    return summary, files, charts


with gr.Blocks(title="PDF PII Redactor") as demo:
    gr.Markdown(
        "## PDF PII Detector"
    )

    with gr.Tabs():
        with gr.TabItem("1) Extract detections"):
            with gr.Row():
                file_in = gr.File(label="Upload PDF", file_types=[".pdf"], interactive=True)
            with gr.Row():
                dpi = gr.Slider(label="DPI", minimum=100, maximum=600, step=25, value=300)
                lang_dd = gr.Dropdown(
                    label="Tesseract languages", choices=_LANG_OPTIONS, value=["eng"], multiselect=True
                )
                conf = gr.Slider(label="Min OCR confidence", minimum=0, maximum=100, step=1, value=60)
            with gr.Row():
                role_keep = gr.Checkbox(label="Keep role emails when adjacent to a NAME", value=True)
                ner_model_sel = gr.Dropdown(label="NER model (optional)", choices=_NER_MODELS, value="")
            with gr.Row():
                tesseract_path = gr.Textbox(
                    label="Tesseract path (optional)",
                    value="",
                    placeholder=r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                )
                do_fallback = gr.Checkbox(
                    label="Fallback masked full-page OCR when needed", value=True
                )

            run_btn = gr.Button("Process PDF", variant="primary")

            log_out = gr.HTML(label="Log")
            words_csv_out = gr.File(label="words_split.csv")
            det_csv_out = gr.File(label="detections.csv")
            gallery = gr.Gallery(label="Page previews", columns=2, preview=True)
            out_dir_hidden = gr.Textbox(visible=False)

            run_btn.click(
                gr_process,
                inputs=[file_in, dpi, lang_dd, conf, role_keep, ner_model_sel, tesseract_path, do_fallback],
                outputs=[log_out, words_csv_out, det_csv_out, gallery, out_dir_hidden],
            )

        with gr.TabItem("2) Post-process CSVs & charts / Export"):
            gr.Markdown("Create CSVs, JSONL, quality report, and annotated previews (optional).")
            with gr.Row():
                detections_csv_in = gr.File(label="detections.csv", file_types=[".csv"], interactive=True)
                out_dir_text = gr.Textbox(label="Output folder (optional)", value="")
            with gr.Row():
                previews_dir_text = gr.Textbox(
                    label="Previews folder (optional, to annotate)",
                    placeholder=".../page_*_preview.png",
                )
                min_conf = gr.Slider(
                    label="Min confidence for actions/annotations",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    value=DEFAULT_CONFIG["DEFAULT_MIN_CONF"],
                )
            with gr.Row():
                yaml_config_file = gr.File(label="Config YAML (optional)", file_types=[".yml", ".yaml"])
                scrub_raw_text = gr.Checkbox(
                    label="Scrub raw_text/line_text from artifacts", value=False
                )

            post_btn = gr.Button("Run post-processor", variant="primary")

            post_log = gr.HTML(label="Post-process log")
            artifacts_out = gr.Files(label="Generated artifacts")
            chart_gallery = gr.Gallery(label="Charts", columns=3, preview=True)

            post_btn.click(
                gr_postprocess,
                inputs=[detections_csv_in, out_dir_text, previews_dir_text, min_conf, yaml_config_file, scrub_raw_text],
                outputs=[post_log, artifacts_out, chart_gallery],
            )


# =============================================================================
# CLI ENTRY
# =============================================================================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="PDF PII Detector & Post-Processor (OCR + NER/regex + grouping)"
    )
    ap.add_argument("--file", type=str, help="Single PDF file to process")
    ap.add_argument("--batch", type=str, help="Folder or a single PDF to batch process")
    ap.add_argument("--lang", type=str, default="eng", help="Tesseract languages like 'eng+deu'")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--conf", type=float, default=60.0)
    ap.add_argument("--ner_model", type=str, default="", help="HF model name for NER (optional)")
    ap.add_argument("--scrub-raw-text", action="store_true", help="Drop raw_text/line_text in artifacts")
    ap.add_argument("--yaml", type=str, default="", help="Config YAML to override risks/thresholds/mappings")
    ap.add_argument("--no-gui", action="store_true", help="Run without Gradio UI")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.yaml or None)

    if args.batch:
        run_batch(
            Path(args.batch),
            lang=args.lang,
            dpi=args.dpi,
            conf=args.conf,
            ner_model=(args.ner_model or None),
            scrub=args.scrub_raw_text,
            cfg=cfg,
        )
    elif args.file:
        pdf = Path(args.file)
        log, words_csv, det_csv, gallery, out_dir = process_pdf_worker(
            pdf_path=pdf,
            dpi=args.dpi,
            lang=args.lang,
            conf=args.conf,
            overlap_iou=0.2,
            start_page=1,
            end_page=None,
            fallback_fullpage_ocr=True,
            role_email_keep_if_adjacent=cfg["ROLE_EMAIL_POLICY"]["keep_if_adjacent_to_name"],
            ner_model=(args.ner_model or None),
            tesseract_path=None,
            log_jsonl=None,
        )
        print(log)

        summary, files, *_ = postprocess_detections(
            detections_csv=det_csv,
            out_dir=None,
            previews_dir=out_dir,
            min_conf=cfg["DEFAULT_MIN_CONF"],
            cfg=cfg,
            scrub_raw_text=args.scrub_raw_text,
        )
        print(summary)

        if not args.no_gui:
            demo.launch()
    else:
        demo.launch()
