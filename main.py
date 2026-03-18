import pymupdf
import json
import tempfile
import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

app = FastAPI(title="PDF Span Extractor", version="1.0.0")


def flags_to_properties(flags: int) -> dict:
    """Decompose PyMuPDF font flags into readable properties."""
    return {
        "superscript": bool(flags & 2**0),
        "italic": bool(flags & 2**1),
        "serif": bool(flags & 2**2),
        "monospace": bool(flags & 2**3),
        "bold": bool(flags & 2**4),
    }


def color_to_hex(color_int: int) -> str:
    """Convert PyMuPDF sRGB integer to hex string."""
    r, g, b = pymupdf.sRGB_to_rgb(color_int)
    return f"#{r:02x}{g:02x}{b:02x}"


@app.post("/extract-spans")
async def extract_spans(file: UploadFile = File(...)):
    """
    Upload a PDF file and get back all text spans with font metadata.

    Returns JSON with per-page span data including:
    - text content
    - font name, size, color (hex)
    - bold/italic flags
    - position (x, y) and dimensions (width, height)
    """
    # Save uploaded file to temp location
    tmp_path = os.path.join(tempfile.gettempdir(), f"upload_{file.filename}")
    try:
        contents = await file.read()
        with open(tmp_path, "wb") as f:
            f.write(contents)

        doc = pymupdf.open(tmp_path)
        all_pages = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            page_width = page.rect.width
            page_height = page.rect.height

            blocks = page.get_text("dict", sort=True, flags=pymupdf.TEXTFLAGS_TEXT)["blocks"]
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

            all_pages.append({
                "page": page_num + 1,
                "page_width": round(page_width, 2),
                "page_height": round(page_height, 2),
                "span_count": len(spans_list),
                "spans": spans_list,
            })

        doc.close()

        return JSONResponse(content={
            "filename": file.filename,
            "total_pages": len(all_pages),
            "pages": all_pages,
        })

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.get("/health")
async def health():
    return {"status": "ok", "pymupdf_version": pymupdf.__version__}