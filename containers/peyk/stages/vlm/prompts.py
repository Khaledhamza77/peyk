"""One shared prompt per role, used identically by every provider adapter. Centralized so
tuning wording later means editing this file, not four backend files — same reasoning as
peyk-paddleocr-vl's discovery of the exact "OCR:" prompt PaddleX's own pipeline uses."""

PROMPTS = {
    "ocr": (
        "Transcribe all text visible in this image exactly as it appears, preserving line "
        "breaks and reading order. Output only the transcribed text — no commentary, no "
        "translation, no markdown formatting."
    ),
    "figure": (
        "Describe this image concisely for inclusion in an extracted document transcript. "
        "It is a figure, chart, or stamp cropped from a document page. If it is a chart or "
        "graph, summarize the key data or trend it shows. If it is a stamp or seal, describe "
        "what it depicts and transcribe any visible text on it. If it is a plain photo or "
        "figure, describe its content plainly. Output only the description, one short "
        "paragraph, no preamble."
    ),
    "table": (
        "This image is a table cropped from a document page. Reproduce the full table as "
        "clean HTML using <table>/<tr>/<td> markup, preserving the exact row and column "
        "structure and every cell's text content, in the correct reading order and script "
        "direction (right-to-left if the table's text is Arabic). Output only the HTML table, "
        "no commentary, no markdown code fences."
    ),
    "fullpage": (
        "This image is a full page of a document. Transcribe the entire page to Markdown, "
        "preserving correct reading order (right-to-left if the page's text is Arabic). "
        "Render tables as Markdown tables and figures/stamps as a short bracketed "
        "description, e.g. [Figure: ...]. Output only the Markdown, no commentary, no code "
        "fences around the whole output."
    ),
}
