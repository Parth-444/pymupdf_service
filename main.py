import pymupdf
import tempfile
import os
from collections import Counter
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="PDF Span Extractor + Classifier", version="2.1.0")

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

SIZE_TOLERANCE = 0.5       # ±0.5pt for size comparisons (handles PDF rounding: 9.08→9, 10.74→11, 3.53→3.5)
HEADER_FONT_SIZE = 11.0    # Header text is always ~11pt across all Rhydburg artworks
HEADER_Y_THRESHOLD = 200   # Spans above this y-coordinate are header candidates

# Component type values found in header's "Component Type" field
COMPONENT_TYPES = {
    "outer carton": "carton",
    "inner carton": "carton",
    "carton": "carton",
    "sticker label": "label",
    "label": "label",
    "insert": "insert",
    "foil": "foil",
    "tube": "tube",
    "packing slip": "packing_slip",
    "sachet": "sachet",
}


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def flags_to_properties(flags: int) -> dict:
    return {
        "superscript": bool(flags & 2**0),
        "italic": bool(flags & 2**1),
        "serif": bool(flags & 2**2),
        "monospace": bool(flags & 2**3),
        "bold": bool(flags & 2**4),
    }


def color_to_hex(color_int: int) -> str:
    r, g, b = pymupdf.sRGB_to_rgb(color_int)
    return f"#{r:02x}{g:02x}{b:02x}"


def is_calibri(font_name: str) -> bool:
    """Check if font is any Calibri variant (Calibri, Calibri-Bold, Calibri-Italic)."""
    return font_name.lower().replace("-", "").replace(" ", "").startswith("calibri")


def size_approx_eq(a: float, b: float, tol: float = SIZE_TOLERANCE) -> bool:
    """Compare two sizes with tolerance for PDF rounding artifacts."""
    return abs(a - b) <= tol


# ─────────────────────────────────────────────────────────────
# STEP 0: FILTER PRODUCTION ANNOTATIONS
#
# Artwork PDFs contain non-artwork spans added by the design tool:
#   - Dimension labels: "150.00 mm", "100.00 mm" (ArialMT, blue)
#   - Layout markers: "Pasting Side" (ArialMT)
#
# These are never Calibri. Filter them into a separate array.
# ─────────────────────────────────────────────────────────────

def is_production_annotation(span: dict) -> bool:
    return not is_calibri(span["font"])


# ─────────────────────────────────────────────────────────────
# STEP 1: DETECT HEADER vs BODY
#
# Every artwork page has a tabular header at the top with fields
# like "Product Name", "AC Reference", "Pantone No." etc.
#
# Header spans are: size ≈ 11pt + positioned in top ~200pt of page.
# A vertical gap separates the last header row from body content.
# ─────────────────────────────────────────────────────────────

def detect_header_boundary(spans: list) -> float:
    """Find the y-coordinate where header ends and body begins."""
    if not spans:
        return 0

    header_candidates = [
        s for s in spans
        if size_approx_eq(s["size"], HEADER_FONT_SIZE, tol=0.5)
        and s["y"] < HEADER_Y_THRESHOLD
    ]

    if not header_candidates:
        return 0

    max_header_y = max(s["y"] + s["height"] for s in header_candidates)

    # Confirm by finding the large y-gap after the header
    all_ys = sorted(set(round(s["y"], 0) for s in spans))
    for i in range(1, len(all_ys)):
        if all_ys[i - 1] >= max_header_y - 10 and all_ys[i] - all_ys[i - 1] > 30:
            return all_ys[i - 1] + 5

    return max_header_y + 5


def is_header_span(span: dict, header_boundary: float) -> bool:
    return span["y"] <= header_boundary and size_approx_eq(span["size"], HEADER_FONT_SIZE, tol=0.5)


# ─────────────────────────────────────────────────────────────
# STEP 2: DETECT COMPONENT TYPE
#
# Reads "Outer Carton" / "Sticker Label" / "Insert" etc. from
# header spans to determine which classification branch to use.
# ─────────────────────────────────────────────────────────────

def detect_component_type(header_spans: list) -> str:
    for span in header_spans:
        text_lower = span["text"].lower().strip()
        for keyword, comp_type in COMPONENT_TYPES.items():
            if keyword in text_lower:
                return comp_type
    return "unknown"


# ─────────────────────────────────────────────────────────────
# STEP 3: FIND X VALUE
#
# X = the trade name font size (largest bold body text).
# This is a measurement, not a rule — X varies per component:
#   Carton → typically 18pt
#   Label  → typically 7pt
#   Insert → typically 5pt
# ─────────────────────────────────────────────────────────────

def find_x_value(body_spans: list) -> float | None:
    if not body_spans:
        return None

    bold_body = [s for s in body_spans if s["bold"]]
    if bold_body:
        return max(s["size"] for s in bold_body)

    return max(s["size"] for s in body_spans)


# ─────────────────────────────────────────────────────────────
# STEP 4: CLASSIFY BODY SPANS
#
# For Carton/Label/Foil/Tube — X / X÷2 size tiers:
#   Size ≈ X + bold     → TRADE_NAME
#   Size ≈ X + unbold   → CLAIM_PHARMA_FORM
#   Size ≈ X/2          → OTHER_CONTENT
#   None of the above   → ANOMALY
#
# For Insert — flat classification (all same size):
#   Bold   → INSERT_HEADING
#   Unbold → INSERT_BODY
# ─────────────────────────────────────────────────────────────

def classify_carton_label_span(span: dict, x_value: float) -> dict:
    """Classify using X / X÷2 tiers. Returns category + tier only."""
    half_x = x_value / 2

    if size_approx_eq(span["size"], x_value):
        if span["bold"]:
            return {"category": "TRADE_NAME", "tier": 1}
        else:
            return {"category": "CLAIM_PHARMA_FORM", "tier": 2}

    if size_approx_eq(span["size"], half_x):
        return {"category": "OTHER_CONTENT", "tier": 3}

    return {"category": "ANOMALY", "tier": None}


def classify_insert_span(span: dict) -> dict:
    """Classify insert spans by bold flag only."""
    if span["bold"]:
        return {"category": "INSERT_HEADING", "tier": None}
    else:
        return {"category": "INSERT_BODY", "tier": None}


# ─────────────────────────────────────────────────────────────
# PIPELINE: CLASSIFY ALL SPANS ON A PAGE
# ─────────────────────────────────────────────────────────────

def classify_page(spans: list) -> dict:
    """
    Full classification pipeline for one page:
      Step 0 → Filter annotations (non-Calibri fonts)
      Step 1 → Split header vs body (by y-position + size)
      Step 2 → Detect component type from header
      Step 3 → Find X from body spans
      Step 4 → Assign category to every span
    """
    if not spans:
        return {
            "classified_spans": [],
            "annotation_spans": [],
            "page_summary": {"component_type": "unknown", "x_value": None},
        }

    # ── Step 0: Filter ──
    artwork_spans = []
    annotation_spans = []
    for s in spans:
        if is_production_annotation(s):
            annotation_spans.append({**s, "category": "PRODUCTION_ANNOTATION"})
        else:
            artwork_spans.append(s)

    # ── Step 1: Header boundary ──
    header_boundary = detect_header_boundary(artwork_spans)
    header_spans = [s for s in artwork_spans if is_header_span(s, header_boundary)]
    body_spans = [s for s in artwork_spans if not is_header_span(s, header_boundary)]

    # ── Step 2: Component type ──
    component_type = detect_component_type(header_spans)

    # ── Step 3: Find X ──
    x_value = find_x_value(body_spans)

    # ── Step 4: Classify ──
    classified = []

    for s in header_spans:
        classified.append({**s, "category": "HEADER", "tier": None, "x_value": None})

    for s in body_spans:
        if component_type == "insert":
            classification = classify_insert_span(s)
        elif x_value is not None:
            classification = classify_carton_label_span(s, x_value)
        else:
            classification = {"category": "UNCLASSIFIED", "tier": None}

        classified.append({**s, **classification, "x_value": x_value})

    # ── Summary ──
    category_counts = dict(Counter(s["category"] for s in classified))

    page_summary = {
        "component_type": component_type,
        "x_value": x_value,
        "x_half_value": round(x_value / 2, 2) if x_value else None,
        "header_boundary_y": round(header_boundary, 2),
        "total_artwork_spans": len(classified),
        "total_annotation_spans": len(annotation_spans),
        "category_breakdown": category_counts,
    }

    return {
        "classified_spans": classified,
        "annotation_spans": annotation_spans,
        "page_summary": page_summary,
    }


# ─────────────────────────────────────────────────────────────
# PDF PROCESSING
# ─────────────────────────────────────────────────────────────

def process_pdf(contents: bytes, filename: str) -> dict:
    tmp_path = os.path.join(tempfile.gettempdir(), f"upload_{filename}")
    try:
        with open(tmp_path, "wb") as f:
            f.write(contents)

        doc = pymupdf.open(tmp_path)
        all_pages = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            page_width = page.rect.width
            page_height = page.rect.height

            blocks = page.get_text("dict", sort=True, flags=pymupdf.TEXTFLAGS_TEXT)["blocks"]

            # ── Extract raw spans ──
            spans_list = []
            for block in blocks:
                if "lines" not in block:
                    continue
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if not text:
                            continue
                        props = flags_to_properties(span["flags"])
                        bbox = span["bbox"]
                        spans_list.append({
                            "text": text,
                            "font": span["font"],
                            "size": round(span["size"], 2),
                            "color": color_to_hex(span["color"]),
                            "bold": props["bold"],
                            "italic": props["italic"],
                            "serif": props["serif"],
                            "monospace": props["monospace"],
                            "superscript": props["superscript"],
                            "x": round(bbox[0], 2),
                            "y": round(bbox[1], 2),
                            "width": round(bbox[2] - bbox[0], 2),
                            "height": round(bbox[3] - bbox[1], 2),
                        })

            # ── Classify ──
            result = classify_page(spans_list)

            all_pages.append({
                "page": page_num + 1,
                "page_width": round(page_width, 2),
                "page_height": round(page_height, 2),
                "span_count": len(spans_list),
                "page_summary": result["page_summary"],
                "classified_spans": result["classified_spans"],
                "annotation_spans": result["annotation_spans"],
            })

        doc.close()

        return {
            "filename": filename,
            "total_pages": len(all_pages),
            "pages": all_pages,
        }

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ─────────────────────────────────────────────────────────────
# API ENDPOINTS
# ─────────────────────────────────────────────────────────────

@app.post("/extract-spans")
async def extract_spans(file: UploadFile = File(...)):
    """Upload PDF as multipart form-data. Key = 'file'."""
    contents = await file.read()
    result = process_pdf(contents, file.filename)
    return JSONResponse(content=result)


@app.post("/extract-spans-binary")
async def extract_spans_binary(request: Request):
    """
    Upload PDF as raw binary body.
    In n8n HTTP Request node:
    - Method: POST
    - URL: .../extract-spans-binary
    - Body Content Type: Binary File
    - Content Type: application/pdf
    """
    contents = await request.body()
    result = process_pdf(contents, "upload.pdf")
    return JSONResponse(content=result)


@app.get("/health")
async def health():
    return {"status": "ok", "pymupdf_version": pymupdf.__version__}