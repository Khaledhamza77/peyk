"""Published, vendor-reported accuracy figures for every model peyk can dispatch.

WHY THIS FILE EXISTS, AND WHAT IT IS NOT
----------------------------------------
peyk orchestrates third-party models; it does not train any. There is no ground-truth corpus and
no annotated test set in this project, so peyk cannot honestly report a *measured* accuracy number
for anything. What it can do is report what each model's own authors published, with a citation,
and let the demo's side-by-side output examples carry the qualitative argument.

Every number below is therefore:
  - reported by the model's authors (or the benchmark's maintainers), never measured here;
  - carried with the exact benchmark it came from -- a TEDS on PubTabNet and an mAP on DocLayNet
    are not comparable quantities and must never be put in the same column;
  - carried with a source URL, so any figure on a slide can be traced back in one click.

Numbers are NOT comparable across benchmarks, and mostly not comparable across models either
unless both were evaluated on the same benchmark and version. OmniDocBench in particular has
several versions in circulation (v1.5, v1.6, Real5) whose scores are not interchangeable.

THE CAVEAT THAT MATTERS MOST FOR THIS PROJECT
---------------------------------------------
Almost every benchmark below is dominated by English and Chinese, born-digital pages. peyk's
actual corpus is predominantly Arabic, right-to-left, and substantially scanned. A high headline
score is therefore weak evidence for this use case, and the few Arabic-specific and scanned-
specific figures recorded below deserve far more weight than the headline ones. Where a model
publishes such a breakdown it is captured in `subscores`.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Claim:
    """One published figure. `benchmark` must name the exact benchmark AND version where the
    source gives one -- "OmniDocBench v1.5" and "OmniDocBench v1.6" are different scales."""

    benchmark: str
    metric: str
    value: str
    source_url: str
    # Higher is better unless stated -- edit-distance metrics are the common exception, and get
    # an explicit note rather than a silently inverted number.
    note: str = ""


@dataclass(frozen=True)
class ModelCard:
    key: str  # the name used in peyk's own config YAML
    display: str
    hosting: str  # "self-hosted" | "managed"
    roles: tuple[str, ...]  # which peyk job slots this model can fill
    params: str | None = None
    claims: tuple[Claim, ...] = ()
    subscores: tuple[Claim, ...] = ()
    # Set when the vendor publishes nothing comparable. An explicit statement is far better on a
    # slide than a blank cell, which reads as an oversight rather than a finding.
    no_published_parsing_benchmark: str = ""


# --------------------------------------------------------------------------- layout

LAYOUT_CARDS = (
    ModelCard(
        key="heron",
        display="Heron (Docling layout)",
        hosting="self-hosted",
        roles=("layout",),
        params="42.9M (RT-DETRv2)",
        claims=(
            Claim(
                "DocLayNet", "mAP", "0.699",
                "https://arxiv.org/abs/2509.11720",
                "heron, evaluated on all predictions without post-processing",
            ),
            Claim(
                "DocLayNet-v2", "mAP", "0.758",
                "https://arxiv.org/abs/2509.11720",
                "heron-101 -- a LARGER variant than the `heron` peyk dispatches; do not quote "
                "this figure for peyk's default",
            ),
            Claim(
                "DocLayNet-v2", "mAP / latency", "78% mAP, 28 ms/image (A100)",
                "https://arxiv.org/abs/2509.11720",
                "heron-101 again, on a datacentre A100 -- not this project's 12 GB laptop GPU",
            ),
        ),
        no_published_parsing_benchmark=(
            "The HuggingFace card itself publishes no scores (architecture and param count only); "
            "every figure above comes from the linked technical report."
        ),
    ),
    ModelCard(
        key="doclayout-yolo",
        display="DocLayout-YOLO",
        hosting="self-hosted",
        roles=("layout",),
        claims=(
            Claim("DocStructBench", "mAP", "78.8%", "https://arxiv.org/abs/2410.12628"),
            Claim("DocLayNet", "mAP", "79.7%", "https://arxiv.org/abs/2410.12628"),
            Claim("D4LA", "mAP", "70.3%", "https://arxiv.org/abs/2410.12628"),
            Claim("DocStructBench", "throughput", "85.5 FPS", "https://arxiv.org/abs/2410.12628",
                  "authors' hardware, not measured here"),
        ),
    ),
    ModelCard(
        key="pp-doclayout-v2",
        display="PP-DocLayoutV2 (PP-StructureV3)",
        hosting="self-hosted",
        roles=("layout",),
        claims=(
            Claim(
                "OmniDocBench", "edit distance (lower is better)", "0.145 EN / 0.206 ZH",
                "https://paddlepaddle.github.io/PaddleOCR/main/en/version3.x/algorithm/PP-StructureV3/PP-StructureV3.html",
                "PP-StructureV3 end-to-end pipeline score, NOT the layout module in isolation",
            ),
        ),
    ),
    # Surya can also fill the layout slot (config key `surya`), but it is one model serving every
    # role rather than a separate layout model, so it has a single card under OCR_CARDS below
    # instead of a duplicate entry here.
)

# --------------------------------------------------------------------------- table structure

TSR_CARDS = (
    ModelCard(
        key="tableformer",
        display="TableFormer",
        hosting="self-hosted",
        roles=("tsr",),
        claims=(
            Claim("PubTabNet", "TEDS", "96.75%", "https://arxiv.org/abs/2203.01017"),
        ),
        subscores=(
            Claim("PubTabNet (simple tables)", "TEDS", "98.5%", "https://arxiv.org/abs/2203.01017"),
            Claim("PubTabNet (complex tables)", "TEDS", "95.0%", "https://arxiv.org/abs/2203.01017"),
        ),
    ),
    ModelCard(
        key="tatr",
        display="Table Transformer (TATR)",
        hosting="self-hosted",
        roles=("tsr",),
        claims=(
            Claim("PubTables-1M", "GriTS", "0.8433",
                  "https://github.com/microsoft/table-transformer",
                  "after the maintainers' code improvements; 0.8326 before"),
            Claim("ICDAR-2013", "exact match", "75%",
                  "https://github.com/microsoft/table-transformer",
                  "trained on PubTables-1M; 81% and DAR 0.965 when trained on PubTables-1M + FinTabNet"),
        ),
    ),
    ModelCard(
        key="pp-structure-general",
        display="PP-Structure (SLANet_plus)",
        hosting="self-hosted",
        roles=("tsr",),
        claims=(
            Claim(
                "OmniDocBench", "edit distance (lower is better)", "0.145 EN / 0.206 ZH",
                "https://paddlepaddle.github.io/PaddleOCR/main/en/version3.x/algorithm/PP-StructureV3/PP-StructureV3.html",
                "PP-StructureV3 pipeline score, not the table module alone",
            ),
        ),
    ),
    ModelCard(
        key="rapidtable",
        display="RapidTable",
        hosting="self-hosted",
        roles=("tsr",),
        no_published_parsing_benchmark=(
            "Not an independent model: RapidAI re-exports the same PP-Structure/SLANet family "
            "weights to other runtimes (ONNX, OpenVINO, Paddle, TensorRT). Accuracy tracks "
            "whichever upstream checkpoint is loaded; the project publishes no separate score."
        ),
    ),
)

# --------------------------------------------------------------------------- text recognition

OCR_CARDS = (
    ModelCard(
        key="paddleocr-vl",
        display="PaddleOCR-VL-0.9B",
        hosting="self-hosted",
        roles=("ocr", "cell_ocr"),
        params="0.9B",
        claims=(
            Claim("OmniDocBench v1.5", "overall", "92.86", "https://arxiv.org/abs/2510.14528"),
            Claim("OmniDocBench v1.5", "text edit distance (lower is better)", "0.035",
                  "https://arxiv.org/abs/2510.14528"),
            Claim("OmniDocBench v1.5", "table TEDS", "90.89", "https://arxiv.org/abs/2510.14528"),
            Claim("OmniDocBench v1.5", "table TEDS-S", "94.76", "https://arxiv.org/abs/2510.14528"),
        ),
        subscores=(
            # The single most relevant published figure in this entire file for peyk's corpus.
            Claim("OmniDocBench (Arabic)", "accuracy", "80.45", "https://arxiv.org/abs/2510.14528",
                  "highest Arabic score in the paper's comparison, above larger VLMs"),
            Claim("OmniDocBench (Arabic)", "edit distance (lower is better)", "0.122",
                  "https://arxiv.org/abs/2510.14528"),
        ),
    ),
    ModelCard(
        key="surya",
        display="Surya-OCR-2",
        hosting="self-hosted",
        roles=("ocr", "cell_ocr", "tsr", "layout", "fullpage"),
        params="0.65B",
        claims=(
            Claim("olmOCR-bench", "overall", "83.3%", "https://huggingface.co/datalab-to/surya-ocr-2"),
            Claim("llamaindex/ParseBench", "mean", "64.83",
                  "https://huggingface.co/datalab-to/surya-ocr-2"),
        ),
        subscores=(
            # This breakdown independently corroborates what this project found empirically:
            # Surya's full-table path is strong on clean born-digital pages and weak on genuine
            # scans (docs-personal/surya/improvement.md). Worth showing side by side.
            Claim("olmOCR-bench: Base", "score", "99.7%", "https://huggingface.co/datalab-to/surya-ocr-2"),
            Claim("olmOCR-bench: Tiny Text", "score", "93.7%", "https://huggingface.co/datalab-to/surya-ocr-2"),
            Claim("olmOCR-bench: Headers/Footers", "score", "92.5%", "https://huggingface.co/datalab-to/surya-ocr-2"),
            Claim("olmOCR-bench: ArXiv", "score", "88.3%", "https://huggingface.co/datalab-to/surya-ocr-2"),
            Claim("olmOCR-bench: Tables", "score", "86.6%", "https://huggingface.co/datalab-to/surya-ocr-2"),
            Claim("olmOCR-bench: Multi-Column", "score", "82.4%", "https://huggingface.co/datalab-to/surya-ocr-2"),
            Claim("olmOCR-bench: Old Math", "score", "81.4%", "https://huggingface.co/datalab-to/surya-ocr-2"),
            Claim("olmOCR-bench: Old Scans", "score", "41.8%", "https://huggingface.co/datalab-to/surya-ocr-2",
                  "the vendor's own weakest category by a wide margin -- directly relevant, since "
                  "part of peyk's corpus is genuine scans"),
        ),
    ),
    ModelCard(
        key="paddleocr",
        display="PaddleOCR (PP-OCRv5)",
        hosting="self-hosted",
        roles=("ocr", "cell_ocr"),
        claims=(
            Claim("multilingual recognition", "relative improvement", ">30% over PP-OCRv3",
                  "https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/algorithm/PP-OCRv5/PP-OCRv5_multi_languages.en.md",
                  "vendor-reported relative gain, not an absolute accuracy on any named benchmark"),
            Claim("language coverage", "languages", "106",
                  "https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/algorithm/PP-OCRv5/PP-OCRv5_multi_languages.en.md"),
        ),
        subscores=(
            Claim("Arabic recognition training set", "images", "2,676",
                  "https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/algorithm/PP-OCRv5/PP-OCRv5_multi_languages.en.md",
                  "dataset size, not an accuracy -- included because it is the clearest available "
                  "signal of how thin Arabic coverage is for this model family"),
        ),
    ),
    ModelCard(
        key="easyocr",
        display="EasyOCR",
        hosting="self-hosted",
        roles=("ocr", "cell_ocr"),
        no_published_parsing_benchmark=(
            "No vendor-published accuracy on any document-parsing benchmark. Community "
            "comparisons exist but are not authored by the maintainers and are not cited here."
        ),
    ),
    ModelCard(
        key="rapidocr",
        display="RapidOCR",
        hosting="self-hosted",
        roles=("ocr", "cell_ocr"),
        no_published_parsing_benchmark=(
            "Re-export of the PP-OCR family to alternative runtimes rather than an independent "
            "model -- accuracy tracks the upstream PP-OCR checkpoint that is loaded."
        ),
    ),
    ModelCard(
        key="tesseract",
        display="Tesseract",
        hosting="self-hosted",
        roles=("ocr", "cell_ocr"),
        no_published_parsing_benchmark=(
            "No maintainer-published benchmark of this kind. Notable regardless: it was the "
            "strongest backend on this project's own Arabic crop test (n=2 crops), and it is "
            "CPU-only -- zero VRAM cost, which the latency benchmark measures directly."
        ),
    ),
)

# --------------------------------------------------------------------------- managed VLMs

MANAGED_CARDS = (
    ModelCard(
        key="deepseek-ocr",
        display="DeepSeek-OCR (Vertex MaaS)",
        hosting="managed",
        roles=("ocr", "tsr", "figures", "fullpage"),
        claims=(
            Claim("Fox", "precision", "97-98%", "https://arxiv.org/abs/2510.18234",
                  "when text tokens are within 10x vision tokens; >60% at 20x compression"),
        ),
        subscores=(),
        no_published_parsing_benchmark=(
            "Published figures are for the open DeepSeek-OCR weights. Vertex serves "
            "`deepseek-ocr-maas@001`, and Google publishes no separate accuracy for that "
            "endpoint -- so the paper's numbers are indicative, not a guarantee about the "
            "managed variant this pipeline actually calls."
        ),
    ),
)


def _generic_managed_note(display: str) -> ModelCard:
    """Claude / Gemini / Nova / Pixtral entries all share the same honest position, so they are
    generated rather than hand-written: these are general-purpose multimodal models, and their
    vendors publish general multimodal and document-VQA results (DocVQA, ChartQA, MMMU and
    similar) rather than document-*parsing* scores on an OmniDocBench-style benchmark. Third-party
    leaderboards do evaluate some Gemini and GPT versions on OmniDocBench, but coverage is
    partial, version-specific, and not vendor-authored.

    Presenting one of those third-party numbers next to a vendor-published OmniDocBench score
    would be a category error, so this file deliberately records the absence instead."""
    return ModelCard(
        key="",
        display=display,
        hosting="managed",
        roles=("ocr", "cell_ocr", "tsr", "figures", "fullpage"),
        no_published_parsing_benchmark=(
            "General-purpose multimodal model. The vendor publishes document-VQA-style results "
            "(DocVQA/ChartQA/MMMU class), not a document-parsing score comparable to OmniDocBench "
            "or olmOCR-bench. Evidence for this pipeline's use of it is the demo's own output "
            "examples, not a published parsing figure."
        ),
    )


MANAGED_FAMILIES = {
    "claude-*": _generic_managed_note("Anthropic Claude (Haiku / Sonnet / Opus tiers)"),
    "gemini-*": _generic_managed_note("Google Gemini (2.5 / 3.x tiers)"),
    "nova-*": _generic_managed_note("Amazon Nova (Lite / Pro / 2 Lite)"),
    "pixtral-large": _generic_managed_note("Mistral Pixtral Large"),
    "kimi-k2-5": _generic_managed_note("Moonshot Kimi K2.5"),
}

ALL_CARDS = LAYOUT_CARDS + TSR_CARDS + OCR_CARDS + MANAGED_CARDS

BY_KEY = {card.key: card for card in ALL_CARDS if card.key}


def card_for(model_key: str) -> ModelCard | None:
    """Exact match first, then the managed-family wildcards -- so `claude-sonnet-4-5` resolves to
    the shared Claude entry without needing one hand-written card per registry tier."""
    if model_key in BY_KEY:
        return BY_KEY[model_key]
    for pattern, card in MANAGED_FAMILIES.items():
        prefix = pattern.rstrip("*")
        if pattern.endswith("*") and model_key.startswith(prefix):
            return card
        if pattern == model_key:
            return card
    return None


# Every distinct source cited above, for a slide's own reference list.
SOURCES = sorted(
    {claim.source_url for card in ALL_CARDS for claim in (*card.claims, *card.subscores)}
)
