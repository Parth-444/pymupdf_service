import pymupdf
import tempfile
import os
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="PDF Span Extractor", version="1.1.0")


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

        return {
            "filename": filename,
            "total_pages": len(all_pages),
            "pages": all_pages,
        }

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


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