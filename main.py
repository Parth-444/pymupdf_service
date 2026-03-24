import pymupdf
import tempfile
import os
import math
from collections import Counter
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="PDF Span Extractor + Classifier + Structural Analyzer", version="3.0.0")

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

SIZE_TOLERANCE = 0.5       # ±0.5pt for size comparisons (handles PDF rounding: 9.08→9, 10.74→11, 3.53→3.5)
HEADER_FONT_SIZE = 11.0    # Header text is always ~11pt across all Rhydburg artworks
HEADER_Y_THRESHOLD = 200   # Spans above this y-coordinate are header candidates

PT_TO_MM = 0.3528          # 1 PDF point = 0.3528 mm
ROTATION_TOLERANCE = 0.1   # Degrees — angles below this are treated as 0°

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

# SOP Section 4 — Required header fields per component type (in order)
REQUIRED_HEADER_FIELDS = {
    "carton": [
        "Product Name", "AC Reference", "Product Code", "Version#",
        "Component Type", "Style", "Substrate", "GSM", "Size",
        "Pantone No.", "Printing Overlay", "Barcode",
    ],
    "label": [
        "Product Name", "AC Reference", "Product Code", "Version#",
        "Component Type", "Style", "Substrate", "GSM", "Size",
        "Pantone No.", "Printing Overlay", "Barcode",
    ],
    "insert": [
        "Product Name", "AC Reference", "Product Code", "Version#",
        "Component Type", "Style", "Substrate", "GSM", "Size",
        "Pantone No.", "Printing Overlay",
    ],
    # Add more component types as SOP defines them
}

# SOP Section 4 — Expected header table dimensions per component type (width x height in mm)
EXPECTED_HEADER_DIMENSIONS = {
    "carton": {"width_mm": 180, "height_mm": 48},
    "label":  {"width_mm": 180, "height_mm": 36},
    "insert": {"width_mm": 180, "height_mm": 30},
}

# SOP — Expected layout border width
EXPECTED_BORDER_WIDTH_PT = 0.5

# SOP — Expected logo dimensions (width x height in mm)
EXPECTED_LOGO_DIMENSIONS = {"width_mm": 15, "height_mm": 3.5}


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


def rgb_tuple_to_hex(rgb: tuple | None) -> str | None:
    """Convert PyMuPDF RGB float tuple (0-1 range) to hex string."""
    if rgb is None:
        return None
    r, g, b = rgb
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


def is_calibri(font_name: str) -> bool:
    """Check if font is any Calibri variant (Calibri, Calibri-Bold, Calibri-Italic)."""
    return font_name.lower().replace("-", "").replace(" ", "").startswith("calibri")


def size_approx_eq(a: float, b: float, tol: float = SIZE_TOLERANCE) -> bool:
    """Compare two sizes with tolerance for PDF rounding artifacts."""
    return abs(a - b) <= tol


def pt_to_mm(val: float) -> float:
    """Convert PDF points to millimeters."""
    return round(val * PT_TO_MM, 2)


def rect_to_mm(rect) -> dict:
    """Convert a PyMuPDF Rect or 4-tuple from points to mm dict."""
    if hasattr(rect, "x0"):
        x0, y0, x1, y1 = rect.x0, rect.y0, rect.x1, rect.y1
    else:
        x0, y0, x1, y1 = rect
    return {
        "x0_mm": pt_to_mm(x0),
        "y0_mm": pt_to_mm(y0),
        "x1_mm": pt_to_mm(x1),
        "y1_mm": pt_to_mm(y1),
        "width_mm": pt_to_mm(x1 - x0),
        "height_mm": pt_to_mm(y1 - y0),
    }


def compute_rotation_angle(direction: tuple) -> float:
    """Compute rotation angle in degrees from a span's direction vector.
    
    Unrotated text has dir = (1, 0) → angle = 0°.
    PyMuPDF 'dir' field is a unit vector (cos θ, sin θ) of the writing direction.
    """
    dx, dy = direction
    angle_deg = math.degrees(math.atan2(dy, dx))
    return round(angle_deg, 2)


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
# STRUCTURAL EXTRACTION: VECTOR DRAWINGS
#
# Uses page.get_drawings() to extract all vector paths:
#   - Table borders, layout borders, lines, rectangles
#   - Each path has: width (pt), stroke color, fill color, rect
#   - Path items: 'l' (line), 're' (rect), 'qu' (quad), 'c' (curve)
# ─────────────────────────────────────────────────────────────

def extract_drawings(page) -> list:
    """Extract vector graphics/paths from the page.
    
    Returns a list of drawing dicts with:
      - width_pt: stroke width in points
      - stroke_color: hex color of stroke (or None)
      - fill_color: hex color of fill (or None)
      - type: 's' (stroke), 'f' (fill), 'fs' (fill+stroke)
      - rect_pt: bounding box in points [x0, y0, x1, y1]
      - rect_mm: bounding box in mm
      - width_mm, height_mm: dimensions in mm
      - is_closed: whether path is closed
      - item_types: list of path operation types ('l', 're', 'qu', 'c')
      - item_count: number of path operations
    """
    drawings = []
    for path in page.get_drawings():
        rect = path["rect"]
        if rect.is_empty or rect.is_infinite:
            continue
        item_types = [item[0] for item in path["items"]]

        width_val = path.get("width")
        stroke_opacity = path.get("stroke_opacity")
        fill_opacity = path.get("fill_opacity")

        drawings.append({
            "width_pt": round(width_val, 4) if width_val is not None else 0.0,
            "stroke_color": rgb_tuple_to_hex(path.get("color")),
            "fill_color": rgb_tuple_to_hex(path.get("fill")),
            "stroke_opacity": stroke_opacity if stroke_opacity is not None else 1.0,
            "fill_opacity": fill_opacity if fill_opacity is not None else 1.0,
            "type": path.get("type", ""),       # 's', 'f', 'fs'
            "is_closed": path.get("closePath", False),
            "dashes": path.get("dashes", "[] 0"),
            "rect_pt": [round(rect.x0, 2), round(rect.y0, 2),
                        round(rect.x1, 2), round(rect.y1, 2)],
            "rect_mm": rect_to_mm(rect),
            "item_types": item_types,
            "item_count": len(path["items"]),
        })

    return drawings


# ─────────────────────────────────────────────────────────────
# STRUCTURAL EXTRACTION: IMAGES
#
# Uses page.get_images() + page.get_image_bbox() for:
#   - Logo dimension check (15 x 3.5 mm per SOP)
#   - Any embedded image with its on-page bbox
# ─────────────────────────────────────────────────────────────

def extract_images(page, doc) -> list:
    """Extract image metadata and on-page bounding boxes.
    
    Returns a list of image dicts with:
      - xref: PDF cross-reference ID
      - original_width_px, original_height_px: native image size
      - colorspace: e.g. 'DeviceRGB', 'DeviceCMYK'
      - bpc: bits per component
      - bbox_pt: on-page bounding box [x0, y0, x1, y1]
      - bbox_mm: on-page bounding box in mm
      - display_width_mm, display_height_mm: rendered size in mm
      - aspect_ratio: width / height as displayed
    """
    images = []
    for img_item in page.get_images(full=True):
        xref = img_item[0]
        orig_w = img_item[2]    # native width in pixels
        orig_h = img_item[3]    # native height in pixels
        bpc = img_item[4]       # bits per component
        colorspace = img_item[5]  # e.g. 'DeviceRGB'

        # Get on-page bounding box
        try:
            bbox = page.get_image_bbox(img_item)
            if bbox.is_empty or bbox.is_infinite:
                continue
        except Exception:
            continue

        display_w_mm = pt_to_mm(bbox.width)
        display_h_mm = pt_to_mm(bbox.height)

        images.append({
            "xref": xref,
            "original_width_px": orig_w,
            "original_height_px": orig_h,
            "colorspace": colorspace,
            "bpc": bpc,
            "bbox_pt": [round(bbox.x0, 2), round(bbox.y0, 2),
                        round(bbox.x1, 2), round(bbox.y1, 2)],
            "bbox_mm": rect_to_mm(bbox),
            "display_width_mm": display_w_mm,
            "display_height_mm": display_h_mm,
            "aspect_ratio": round(display_w_mm / display_h_mm, 4) if display_h_mm > 0 else None,
        })

    return images


# ─────────────────────────────────────────────────────────────
# STRUCTURAL EXTRACTION: TEXT ROTATIONS
#
# Uses the 'dir' field from get_text("rawdict") spans.
# dir = (cos θ, sin θ) — the writing direction unit vector.
# Unrotated horizontal text → dir = (1, 0) → angle = 0°
# SOP says rotation should be 0.0 unless specified.
# ─────────────────────────────────────────────────────────────

def extract_rotations(dict_blocks: list) -> list:
    """Find all text spans with non-zero rotation.
    
    NOTE: In PyMuPDF, 'dir' (writing direction vector) is on the LINE level,
    not on individual spans. We read dir from lines and associate it with spans.
    
    Returns a list of dicts with:
      - text: first 50 chars of the span
      - angle_deg: rotation angle in degrees
      - direction: raw (dx, dy) direction vector
      - origin: (x, y) starting point of the span
      - font: font name
      - size: font size
    """
    rotated_spans = []
    for block in dict_blocks:
        if "lines" not in block:
            continue
        for line in block["lines"]:
            direction = line.get("dir", (1, 0))
            angle = compute_rotation_angle(direction)
            if abs(angle) <= ROTATION_TOLERANCE:
                continue
            # This entire line is rotated — report each span
            for span in line["spans"]:
                text = span.get("text", "").strip()
                if not text:
                    continue
                rotated_spans.append({
                    "text": text[:50],
                    "angle_deg": angle,
                    "direction": [round(direction[0], 6), round(direction[1], 6)],
                    "origin": [round(span["origin"][0], 2), round(span["origin"][1], 2)],
                    "font": span["font"],
                    "size": round(span["size"], 2),
                })

    return rotated_spans


# ─────────────────────────────────────────────────────────────
# STRUCTURAL ANALYSIS: HEADER FIELD VALIDATION
#
# SOP Section 4 defines required fields per component type.
# We extract field labels from header spans and check presence.
# ─────────────────────────────────────────────────────────────

def validate_header_fields(header_spans: list, component_type: str) -> dict:
    """Check which required header fields are present/missing.
    
    Returns:
      - required_fields: list of fields SOP requires
      - found_fields: list of fields detected in header
      - missing_fields: required but not found
      - extra_fields: found but not in required list
    """
    required = REQUIRED_HEADER_FIELDS.get(component_type, [])
    if not required:
        return {
            "required_fields": [],
            "found_fields": [],
            "missing_fields": [],
            "extra_fields": [],
            "note": f"No required field list defined for component type '{component_type}'",
        }

    # Combine all header text to search for field labels
    header_texts = [s["text"].strip() for s in header_spans]
    header_combined = " ".join(header_texts).lower()

    found = []
    missing = []
    for field in required:
        if field.lower() in header_combined:
            found.append(field)
        else:
            missing.append(field)

    # Identify any extra field-like text (colon-terminated or known field patterns)
    known_lower = {f.lower() for f in required}
    extra = []
    for text in header_texts:
        # Heuristic: field labels often end with ':' or are standalone short texts
        cleaned = text.strip().rstrip(":").strip()
        if cleaned and cleaned.lower() not in known_lower and len(cleaned) < 30:
            # Only flag short-ish text as potential extra fields
            if cleaned.lower() not in " ".join(found).lower():
                extra.append(cleaned)

    return {
        "required_fields": required,
        "found_fields": found,
        "missing_fields": missing,
        "extra_fields": list(set(extra)),
    }


# ─────────────────────────────────────────────────────────────
# STRUCTURAL ANALYSIS: HEADER DIMENSIONS
#
# Compute the bounding box of the header region and compare
# against SOP-specified dimensions per component type.
# Uses drawings (vector rectangles) first, falls back to spans.
# ─────────────────────────────────────────────────────────────

def analyze_header_dimensions(header_spans: list, drawings: list,
                               component_type: str, header_boundary: float) -> dict:
    """Measure the header table region dimensions.
    
    Strategy:
      1. Look for a prominent rectangle in drawings that encloses
         the header spans — this is the actual table border.
      2. Fallback: compute bounding box from header span positions.
    
    Returns measured dimensions and comparison against SOP expected values.
    """
    expected = EXPECTED_HEADER_DIMENSIONS.get(component_type)

    # --- Method 1: Find enclosing rectangle from drawings ---
    # Look for rectangles in the header region (top portion of page)
    header_rects = []
    for d in drawings:
        # Only consider rectangles (path items with 're' or closed 'l' paths)
        has_rect = "re" in d["item_types"]
        # Must be in the header area
        rect_y0 = d["rect_pt"][1]
        rect_y1 = d["rect_pt"][3]
        rect_w = d["rect_pt"][2] - d["rect_pt"][0]
        rect_h = rect_y1 - rect_y0

        if has_rect and rect_y0 < header_boundary + 20 and rect_w > 100 and rect_h > 20:
            header_rects.append({
                "rect_pt": d["rect_pt"],
                "rect_mm": d["rect_mm"],
                "width_mm": d["rect_mm"]["width_mm"],
                "height_mm": d["rect_mm"]["height_mm"],
                "border_width_pt": d["width_pt"],
                "stroke_color": d["stroke_color"],
            })

    # Sort by area (largest first) — the header table is usually the biggest rect
    header_rects.sort(key=lambda r: r["width_mm"] * r["height_mm"], reverse=True)

    # --- Method 2: Fallback to span bounding box ---
    span_bbox = None
    if header_spans:
        min_x = min(s["x"] for s in header_spans)
        min_y = min(s["y"] for s in header_spans)
        max_x = max(s["x"] + s["width"] for s in header_spans)
        max_y = max(s["y"] + s["height"] for s in header_spans)
        span_bbox = {
            "rect_pt": [round(min_x, 2), round(min_y, 2), round(max_x, 2), round(max_y, 2)],
            "rect_mm": rect_to_mm((min_x, min_y, max_x, max_y)),
            "width_mm": pt_to_mm(max_x - min_x),
            "height_mm": pt_to_mm(max_y - min_y),
        }

    # Pick best measurement
    measured = None
    measurement_source = None

    if header_rects:
        measured = header_rects[0]
        measurement_source = "drawing_rectangle"
    elif span_bbox:
        measured = span_bbox
        measurement_source = "span_bounding_box"

    result = {
        "measurement_source": measurement_source,
        "measured": measured,
        "all_header_rectangles": header_rects,      # for debugging
        "span_bounding_box": span_bbox,
        "expected": {
            "width_mm": expected["width_mm"] if expected else None,
            "height_mm": expected["height_mm"] if expected else None,
        } if expected else None,
    }

    # Comparison
    if measured and expected:
        result["width_delta_mm"] = round(measured["width_mm"] - expected["width_mm"], 2)
        result["height_delta_mm"] = round(measured["height_mm"] - expected["height_mm"], 2)
        result["width_match"] = abs(result["width_delta_mm"]) <= 2.0   # 2mm tolerance
        result["height_match"] = abs(result["height_delta_mm"]) <= 2.0

    return result


# ─────────────────────────────────────────────────────────────
# STRUCTURAL ANALYSIS: LAYOUT BORDER
#
# SOP says 0.5pt single line border around the layout.
# We look for the largest rectangle on the page and check
# its stroke width.
# ─────────────────────────────────────────────────────────────

def analyze_layout_border(drawings: list, page_width_pt: float, page_height_pt: float) -> dict:
    """Find the layout border and verify its width.
    
    The layout border is typically the largest stroke-type rectangle
    on the page, close to page edges.
    """
    # Filter to stroked rectangles (not fills)
    stroke_rects = []
    for d in drawings:
        has_rect = "re" in d["item_types"]
        is_stroked = d["type"] in ("s", "fs")
        if has_rect and is_stroked and d["width_pt"] > 0:
            rect_w = d["rect_pt"][2] - d["rect_pt"][0]
            rect_h = d["rect_pt"][3] - d["rect_pt"][1]
            area = rect_w * rect_h
            stroke_rects.append({
                **d,
                "area_pt2": round(area, 2),
                "width_dim_mm": pt_to_mm(rect_w),
                "height_dim_mm": pt_to_mm(rect_h),
            })

    # Sort by area descending — layout border should be the biggest
    stroke_rects.sort(key=lambda r: r["area_pt2"], reverse=True)

    # Heuristic: layout border covers > 50% of page area
    page_area = page_width_pt * page_height_pt
    candidates = [r for r in stroke_rects if r["area_pt2"] > page_area * 0.3]

    border = candidates[0] if candidates else (stroke_rects[0] if stroke_rects else None)

    result = {
        "found": border is not None,
        "expected_width_pt": EXPECTED_BORDER_WIDTH_PT,
    }

    if border:
        result["measured_width_pt"] = border["width_pt"]
        result["width_match"] = abs(border["width_pt"] - EXPECTED_BORDER_WIDTH_PT) < 0.1
        result["stroke_color"] = border["stroke_color"]
        result["rect_pt"] = border["rect_pt"]
        result["rect_mm"] = border["rect_mm"]
        result["dashes"] = border["dashes"]
        result["is_solid"] = border["dashes"] in ("[] 0", "[] 0.0", "")
    else:
        result["note"] = "No layout border rectangle detected"

    return result


# ─────────────────────────────────────────────────────────────
# STRUCTURAL ANALYSIS: LOGO CHECK
#
# SOP says logo should be 15 x 3.5 mm.
# We look for images in the header region that could be the logo.
# ─────────────────────────────────────────────────────────────

def analyze_logo(images: list, header_boundary: float) -> dict:
    """Find the logo image and check its dimensions.
    
    The logo is typically in the header region, and smaller than
    content images. We look for images above header_boundary.
    """
    # Images in header region
    header_images = [
        img for img in images
        if img["bbox_pt"][1] < header_boundary + 20
    ]

    if not header_images:
        return {
            "found": False,
            "note": "No images found in header region",
            "expected_width_mm": EXPECTED_LOGO_DIMENSIONS["width_mm"],
            "expected_height_mm": EXPECTED_LOGO_DIMENSIONS["height_mm"],
        }

    # Pick the one closest to expected logo size
    expected_w = EXPECTED_LOGO_DIMENSIONS["width_mm"]
    expected_h = EXPECTED_LOGO_DIMENSIONS["height_mm"]

    best = None
    best_delta = float("inf")
    for img in header_images:
        delta = (abs(img["display_width_mm"] - expected_w)
                 + abs(img["display_height_mm"] - expected_h))
        if delta < best_delta:
            best_delta = delta
            best = img

    result = {
        "found": True,
        "expected_width_mm": expected_w,
        "expected_height_mm": expected_h,
        "measured_width_mm": best["display_width_mm"],
        "measured_height_mm": best["display_height_mm"],
        "width_delta_mm": round(best["display_width_mm"] - expected_w, 2),
        "height_delta_mm": round(best["display_height_mm"] - expected_h, 2),
        "width_match": abs(best["display_width_mm"] - expected_w) <= 1.0,
        "height_match": abs(best["display_height_mm"] - expected_h) <= 1.0,
        "bbox_pt": best["bbox_pt"],
        "bbox_mm": best["bbox_mm"],
        "aspect_ratio": best["aspect_ratio"],
        "all_header_images_count": len(header_images),
    }

    return result


# ─────────────────────────────────────────────────────────────
# STRUCTURAL ANALYSIS: ROTATION CHECK
#
# SOP says rotation should be 0.0 unless specified.
# Flag any spans with non-zero rotation.
# ─────────────────────────────────────────────────────────────

def analyze_rotations(rotated_spans: list) -> dict:
    """Summarize text rotation findings."""
    if not rotated_spans:
        return {
            "all_zero": True,
            "rotated_span_count": 0,
            "unique_angles": [],
            "details": [],
        }

    angles = list(set(s["angle_deg"] for s in rotated_spans))
    angles.sort()

    return {
        "all_zero": False,
        "rotated_span_count": len(rotated_spans),
        "unique_angles": angles,
        "details": rotated_spans,  # full list of rotated spans with text/position
    }


# ─────────────────────────────────────────────────────────────
# STRUCTURAL ANALYSIS: COLOR SUMMARY
#
# Summarize all colors found in the artwork for comparison
# against SOP's Pantone specifications.
# ─────────────────────────────────────────────────────────────

def analyze_colors(classified_spans: list, drawings: list) -> dict:
    """Collect all unique colors from text and vector graphics.
    
    Returns:
      - text_colors: unique hex colors used in text, with frequency
      - stroke_colors: unique hex colors used in drawing strokes
      - fill_colors: unique hex colors used in drawing fills
    """
    # Text colors by category
    text_color_map = {}
    for s in classified_spans:
        color = s.get("color", "#000000")
        cat = s.get("category", "UNKNOWN")
        key = f"{color}|{cat}"
        if key not in text_color_map:
            text_color_map[key] = {"color": color, "category": cat, "count": 0, "sample_text": s["text"][:30]}
        text_color_map[key]["count"] += 1

    # Drawing colors
    stroke_colors = Counter()
    fill_colors = Counter()
    for d in drawings:
        if d["stroke_color"]:
            stroke_colors[d["stroke_color"]] += 1
        if d["fill_color"]:
            fill_colors[d["fill_color"]] += 1

    return {
        "text_colors": sorted(text_color_map.values(), key=lambda x: -x["count"]),
        "stroke_colors": [{"color": c, "count": n} for c, n in stroke_colors.most_common()],
        "fill_colors": [{"color": c, "count": n} for c, n in fill_colors.most_common()],
        "unique_text_color_count": len(set(s.get("color") for s in classified_spans)),
        "unique_stroke_color_count": len(stroke_colors),
        "unique_fill_color_count": len(fill_colors),
    }


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
            "header_spans": [],
            "body_spans": [],
            "page_summary": {
                "component_type": "unknown",
                "x_value": None,
                "x_half_value": None,
                "header_boundary_y": 0,
                "total_artwork_spans": 0,
                "total_annotation_spans": 0,
                "category_breakdown": {},
            },
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
        "header_spans": header_spans,
        "body_spans": body_spans,
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

            # ── Extract text with "dict" for reliable span text ──
            # NOTE: 'dir' (writing direction) lives on the LINE level in both dict and rawdict.
            # We use dict for everything and extract dir from lines directly.
            dict_data = page.get_text("dict", sort=True, flags=pymupdf.TEXTFLAGS_TEXT)
            dict_blocks = dict_data["blocks"]

            # ── Build a lookup: line origin → direction vector ──
            # In PyMuPDF, 'dir' (writing direction) is on the LINE, not the span.
            # We key by the line's bbox y-center + first span origin for matching.
            direction_lookup = {}
            for block in dict_blocks:
                if "lines" not in block:
                    continue
                for line in block["lines"]:
                    line_dir = line.get("dir", (1, 0))
                    for span in line["spans"]:
                        origin = span.get("origin", (0, 0))
                        key = (round(origin[0], 1), round(origin[1], 1))
                        direction_lookup[key] = line_dir

            # ── Extract raw spans from dict blocks (has 'text' field) ──
            spans_list = []
            for block in dict_blocks:
                if "lines" not in block:
                    continue
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if not text:
                            continue
                        props = flags_to_properties(span["flags"])
                        bbox = span["bbox"]

                        # Look up direction vector from rawdict
                        origin = span.get("origin", (0, 0))
                        origin_key = (round(origin[0], 1), round(origin[1], 1))
                        direction = direction_lookup.get(origin_key, (1, 0))
                        rotation_angle = compute_rotation_angle(direction)

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
                            "rotation_deg": rotation_angle,
                        })

            # ── Classify spans ──
            result = classify_page(spans_list)

            # ── Extract structural elements ──
            drawings = extract_drawings(page)
            images = extract_images(page, doc)
            rotated_spans = extract_rotations(dict_blocks)

            header_boundary = result["page_summary"]["header_boundary_y"]
            component_type = result["page_summary"]["component_type"]

            # ── Structural analysis ──
            header_field_validation = validate_header_fields(
                result["header_spans"], component_type
            )
            header_dimensions = analyze_header_dimensions(
                result["header_spans"], drawings, component_type, header_boundary
            )
            layout_border = analyze_layout_border(drawings, page_width, page_height)
            logo_check = analyze_logo(images, header_boundary)
            rotation_check = analyze_rotations(rotated_spans)
            color_summary = analyze_colors(result["classified_spans"], drawings)

            all_pages.append({
                "page": page_num + 1,
                "page_width_pt": round(page_width, 2),
                "page_height_pt": round(page_height, 2),
                "page_width_mm": pt_to_mm(page_width),
                "page_height_mm": pt_to_mm(page_height),
                "span_count": len(spans_list),

                # ── Existing output ──
                "page_summary": result["page_summary"],
                "classified_spans": result["classified_spans"],
                "annotation_spans": result["annotation_spans"],

                # ── NEW: Structural data (raw extractions) ──
                "structural": {
                    "drawings": drawings,
                    "images": images,
                    "rotated_spans": rotated_spans,
                },

                # ── NEW: Structural analysis (SOP compliance checks) ──
                "compliance_checks": {
                    "header_fields": header_field_validation,
                    "header_dimensions": header_dimensions,
                    "layout_border": layout_border,
                    "logo": logo_check,
                    "rotation": rotation_check,
                    "colors": color_summary,
                },
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

@app.post("/extract-text")
async def extract_text(file: UploadFile):
    doc = pymupdf.open(stream=await file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n\n"
    return {"text": text.strip()}