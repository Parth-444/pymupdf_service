import pymupdf
import tempfile
import os
from collections import Counter
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="PDF Span Extractor + Classifier", version="2.0.0")

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

SIZE_TOLERANCE = 0.5          # ±0.5pt for size comparisons (handles PDF rounding: 9.08→9, 10.74→11, 3.53→3.5)
HEADER_FONT_SIZE = 11.0       # SOP: header text must be 11pt
MIN_SIZE_CARTON_LABEL = 3.0   # SOP: minimum for carton/label/foil/sachet
MIN_SIZE_INSERT = 5.0         # SOP: minimum for insert
HEADER_Y_THRESHOLD = 200      # spans above this y-coordinate are header candidates

# Known component type values found in the header's "Component Type" field
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
    """Compare two font sizes with tolerance for PDF rounding artifacts.
    
    Why this exists: PyMuPDF often reports sizes like 9.08 instead of 9,
    10.74 instead of 11, 3.53 instead of 3.5. These are rendering artifacts,
    not actual formatting differences. A ±0.5pt window catches all of these.
    """
    return abs(a - b) <= tol


# ─────────────────────────────────────────────────────────────
# STEP 0: FILTER NON-ARTWORK SPANS
#
# Why: The artwork PDF contains production annotations that are
# NOT part of the actual artwork content. These are dimension
# measurements ("150.00 mm") and layout markers ("Pasting Side")
# added by the design tool. They use ArialMT font in blue (#1c6fa8).
#
# Rule: anything NOT in a Calibri font variant → annotation, skip it.
# ─────────────────────────────────────────────────────────────

def is_production_annotation(span: dict) -> bool:
    return not is_calibri(span["font"])


# ─────────────────────────────────────────────────────────────
# STEP 1: DETECT HEADER vs BODY
#
# Every artwork page has a tabular header at the top containing
# fields like "Product Name", "AC Reference", "Pantone No." etc.
# 
# Detection strategy:
#   1. Header spans are at the top of the page (low y-coordinates)
#   2. They are size ≈ 11pt (SOP rule for header font size)
#   3. There's a visible vertical gap between the last header row
#      and the first body content
#
# The header boundary y-coordinate separates header from body.
# ─────────────────────────────────────────────────────────────

def detect_header_boundary(spans: list) -> float:
    """Find the y-coordinate that separates header table from body content."""
    if not spans:
        return 0

    # Find spans that look like header: size ≈ 11pt AND in the top portion of page
    header_candidates = [
        s for s in spans
        if size_approx_eq(s["size"], HEADER_FONT_SIZE, tol=0.5)
        and s["y"] < HEADER_Y_THRESHOLD
    ]

    if not header_candidates:
        return 0  # no header found on this page

    # Header boundary = bottom edge of the lowest header span + buffer
    max_header_y = max(s["y"] + s["height"] for s in header_candidates)

    # Confirm by looking for a large y-gap after the header
    all_ys = sorted(set(round(s["y"], 0) for s in spans))
    for i in range(1, len(all_ys)):
        if all_ys[i - 1] >= max_header_y - 10 and all_ys[i] - all_ys[i - 1] > 30:
            return all_ys[i - 1] + 5  # boundary sits inside the gap

    return max_header_y + 5


def is_header_span(span: dict, header_boundary: float) -> bool:
    """A span is header if it's above the boundary AND at header font size."""
    return span["y"] <= header_boundary and size_approx_eq(span["size"], HEADER_FONT_SIZE, tol=0.5)


# ─────────────────────────────────────────────────────────────
# STEP 2: DETECT COMPONENT TYPE
#
# The header table has a "Component Type" field with values like
# "Outer Carton", "Sticker Label", "Insert", etc.
# This determines which classification branch to use:
#   - Carton/Label/Foil/Tube → X / X÷2 size tier classification
#   - Insert → flat 5pt classification (bold = heading, unbold = body)
# ─────────────────────────────────────────────────────────────

def detect_component_type(header_spans: list) -> str:
    """Read component type from the header table spans."""
    for span in header_spans:
        text_lower = span["text"].lower().strip()
        for keyword, comp_type in COMPONENT_TYPES.items():
            if keyword in text_lower:
                return comp_type
    return "unknown"


# ─────────────────────────────────────────────────────────────
# STEP 3: FIND X VALUE
#
# X = the font size used for the trade name (salt name + Rhydburg).
# Per the SOP: "Salt Name and Rhydburg → X Bold"
#
# X is the largest bold font size among body (non-header) spans.
# Everything else in the artwork is measured relative to X.
#
# X varies per component:
#   Carton → X=18pt typically
#   Label  → X=7pt typically
#   Insert → X=5pt (but X/2 rule doesn't apply — see classify_insert_span)
# ─────────────────────────────────────────────────────────────

def find_x_value(body_spans: list) -> float | None:
    """Find X = the trade name font size (largest bold body text)."""
    if not body_spans:
        return None

    bold_body = [s for s in body_spans if s["bold"]]
    if bold_body:
        return max(s["size"] for s in bold_body)

    # Fallback: largest body size overall
    return max(s["size"] for s in body_spans)


# ─────────────────────────────────────────────────────────────
# STEP 4: CLASSIFY BODY SPANS
#
# For Carton/Label/Foil/Tube — the X / X÷2 tier system:
#
#   Question 1: Is size ≈ X?
#     YES + bold   → TRADE_NAME      (Tier 1: "Salt Name and Rhydburg → X Bold")
#     YES + unbold → CLAIM_PHARMA_FORM (Tier 2: "Claim and Pharmaceutical form → X Unbold")
#   
#   Question 2: Is size ≈ X/2?
#     YES          → OTHER_CONTENT    (Tier 3: "Other content → X/2 Bold/Unbold")
#   
#   Neither        → ANOMALY          (size doesn't fit any SOP-defined tier)
#
# For Insert — flat classification:
#   Everything is the same size (5pt).
#   Bold = heading, Unbold = body text.
#   The X/2 rule does NOT apply to inserts.
# ─────────────────────────────────────────────────────────────

def classify_carton_label_span(span: dict, x_value: float, component_type: str) -> dict:
    """Classify a body span using X / X÷2 size tiers."""
    half_x = x_value / 2
    min_size = MIN_SIZE_INSERT if component_type == "insert" else MIN_SIZE_CARTON_LABEL

    # ── Tier 1 or 2: size ≈ X ──
    if size_approx_eq(span["size"], x_value):
        if span["bold"]:
            category = "TRADE_NAME"
            size_rule = f"X={x_value}pt Bold"
        else:
            category = "CLAIM_PHARMA_FORM"
            size_rule = f"X={x_value}pt Unbold"
        return {
            "category": category,
            "tier": 1 if span["bold"] else 2,
            "size_rule": size_rule,
            "size_compliant": True,
            "font_compliant": is_calibri(span["font"]),
            "bold_compliant": True,
            "min_size_compliant": span["size"] >= min_size,
            "x_value": x_value,
        }

    # ── Tier 3: size ≈ X/2 ──
    if size_approx_eq(span["size"], half_x):
        return {
            "category": "OTHER_CONTENT",
            "tier": 3,
            "size_rule": f"X/2={half_x}pt",
            "size_compliant": True,
            "font_compliant": is_calibri(span["font"]),
            "bold_compliant": True,  # SOP allows both bold and unbold at X/2
            "min_size_compliant": span["size"] >= min_size,
            "x_value": x_value,
        }

    # ── ANOMALY: doesn't fit any tier ──
    dist_x = abs(span["size"] - x_value)
    dist_half = abs(span["size"] - half_x)
    closest = f"X={x_value}" if dist_x < dist_half else f"X/2={half_x}"

    return {
        "category": "ANOMALY",
        "tier": None,
        "size_rule": f"Expected {closest}pt, got {span['size']}pt",
        "size_compliant": False,
        "font_compliant": is_calibri(span["font"]),
        "bold_compliant": True,
        "min_size_compliant": span["size"] >= min_size,
        "x_value": x_value,
    }


def classify_insert_span(span: dict, x_value: float) -> dict:
    """
    Classify a body span for insert components.
    
    Inserts do NOT follow the X/X÷2 pattern — all body text is the same size (5pt).
    Classification uses bold flag only:
      - Bold text → INSERT_HEADING (section headings, sub-headings)
      - Unbold text → INSERT_BODY (regular content)
    
    Additional SOP rules for inserts:
      - Color: only black
      - Font: Calibri
      - Min size: 5pt
    """
    category = "INSERT_HEADING" if span["bold"] else "INSERT_BODY"

    # Color compliance: SOP says "only black colour" for inserts
    color = span["color"].lower()
    r = int(color[1:3], 16)
    g = int(color[3:5], 16)
    b = int(color[5:7], 16)
    # Accept near-black (#1a1a18, #333333, #000000, #302b25) and white (#ffffff for text on colored bg)
    is_acceptable_color = (r < 80 and g < 80 and b < 80) or (r > 200 and g > 200 and b > 200)

    return {
        "category": category,
        "tier": None,
        "size_rule": f"≥{MIN_SIZE_INSERT}pt",
        "size_compliant": span["size"] >= MIN_SIZE_INSERT - SIZE_TOLERANCE,
        "font_compliant": is_calibri(span["font"]),
        "bold_compliant": True,
        "min_size_compliant": span["size"] >= MIN_SIZE_INSERT - SIZE_TOLERANCE,
        "color_compliant": is_acceptable_color,
        "x_value": x_value,
    }


def classify_header_span(span: dict) -> dict:
    """Classify and check a header table span. SOP: Calibri, 11pt."""
    return {
        "category": "HEADER",
        "tier": None,
        "size_rule": f"={HEADER_FONT_SIZE}pt",
        "size_compliant": size_approx_eq(span["size"], HEADER_FONT_SIZE),
        "font_compliant": is_calibri(span["font"]),
        "bold_compliant": True,
        "min_size_compliant": True,
        "x_value": None,
    }


# ─────────────────────────────────────────────────────────────
# PIPELINE: CLASSIFY ALL SPANS ON A PAGE
# ─────────────────────────────────────────────────────────────

def classify_page(spans: list) -> dict:
    """
    Full classification pipeline for one page:
      Step 0 → Filter annotations (ArialMT)
      Step 1 → Split header vs body (by y-position + size)
      Step 2 → Detect component type from header
      Step 3 → Find X from body spans
      Step 4 → Classify every span into its category
      Step 5 → Compute compliance summary
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
            annotation_spans.append({**s, "category": "PRODUCTION_ANNOTATION", "filtered": True})
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
        classified.append({**s, **classify_header_span(s)})

    for s in body_spans:
        if component_type == "insert":
            classified.append({**s, **classify_insert_span(s, x_value)})
        elif x_value is not None:
            classified.append({**s, **classify_carton_label_span(s, x_value, component_type)})
        else:
            classified.append({**s, "category": "UNCLASSIFIED", "tier": None,
                               "size_rule": "No X value found",
                               "size_compliant": None, "font_compliant": is_calibri(s["font"]),
                               "bold_compliant": None, "min_size_compliant": None, "x_value": None})

    # ── Step 5: Summary ──
    total = len(classified)
    compliant = sum(1 for s in classified
                    if s.get("size_compliant") is True
                    and s.get("font_compliant") is True
                    and s.get("min_size_compliant") is True)
    non_compliant = sum(1 for s in classified
                        if s.get("size_compliant") is False
                        or s.get("font_compliant") is False
                        or s.get("min_size_compliant") is False)
    anomalies = sum(1 for s in classified if s.get("category") == "ANOMALY")
    category_counts = dict(Counter(s["category"] for s in classified))

    if non_compliant > 0:
        verdict = "NON_COMPLIANT"
    elif anomalies > 0:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "COMPLIANT"

    page_summary = {
        "component_type": component_type,
        "x_value": x_value,
        "x_half_value": round(x_value / 2, 2) if x_value else None,
        "header_boundary_y": round(header_boundary, 2),
        "total_artwork_spans": total,
        "total_annotation_spans": len(annotation_spans),
        "compliant_spans": compliant,
        "non_compliant_spans": non_compliant,
        "anomaly_spans": anomalies,
        "category_breakdown": category_counts,
        "verdict": verdict,
    }

    return {
        "classified_spans": classified,
        "annotation_spans": annotation_spans,
        "page_summary": page_summary,
    }


# ─────────────────────────────────────────────────────────────
# PDF PROCESSING (extraction + classification)
# ─────────────────────────────────────────────────────────────

def process_pdf(contents: bytes, filename: str) -> dict:
    tmp_path = os.path.join(tempfile.gettempdir(), f"upload_{filename}")
    try:
        with open(tmp_path, "wb") as f:
            f.write(contents)

        doc = pymupdf.open(tmp_path)
        all_pages = []
        overall_non_compliant = 0
        overall_anomalies = 0

        for page_num in range(len(doc)):
            page = doc[page_num]
            page_width = page.rect.width
            page_height = page.rect.height

            blocks = page.get_text("dict", sort=True, flags=pymupdf.TEXTFLAGS_TEXT)["blocks"]

            # ── Extract raw spans (original logic) ──
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

            # ── Classify spans ──
            result = classify_page(spans_list)

            overall_non_compliant += result["page_summary"].get("non_compliant_spans", 0)
            overall_anomalies += result["page_summary"].get("anomaly_spans", 0)

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

        # ── Overall document verdict ──
        if overall_non_compliant > 0:
            verdict = "NON_COMPLIANT"
        elif overall_anomalies > 0:
            verdict = "INCONCLUSIVE"
        else:
            verdict = "COMPLIANT"

        return {
            "filename": filename,
            "total_pages": len(all_pages),
            "overall_verdict": verdict,
            "overall_non_compliant_spans": overall_non_compliant,
            "overall_anomaly_spans": overall_anomalies,
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