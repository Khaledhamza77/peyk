# Pipeline
Two-step spine: **Layout Understanding** (detect + classify regions) → **Digitalization** (per-element: text, tables, figures) → **Markdown fragments**. Assembly/reading-order stays out of scope. Covers document Families A (structured regulatory/financial), B (legal/contractual), D (correspondence) in one pipeline; Family C (transactional) deferred to its own pipeline later. Privacy constraint: self-hostable or private-cloud (**AWS Bedrock**) only — no public vendor APIs.
# Ranked candidates
## Layout:
PP-DocLayoutV2, Surya-Layout, DocLayout-YOLO, Heron, Detectron2
## OCR (text recognition):
AIN-7B, Claude Opus 4.5, Claude Sonnet 5, Claude Haiku 4.5, Amazon Nova Premier, Amazon Nova Lite, PaddleOCR-VL-0.9B, Granite-Docling-258M, PaddleOCR (PP-OCRv5/v6), EasyOCR, Surya-recognition, Tesseract, RapidOCR, DeepSeek-OCR (unverified — Bedrock access/request format not yet re-confirmed post GCP→AWS move), Mistral OCR (unverified — access not yet confirmed on Bedrock)
— excluded: **Gemini family** (Google-first-party, not available on Bedrock — this was the GCP-era pick, excluded here purely by the platform move, not a quality judgment; see [build_notes.md](build_notes.md) for the GCP-era history), Qwen2.5-VL-32B (no managed API on Bedrock, self-deploy only — needs an expensive dedicated GPU-backed endpoint, disproportionate to PoC scale)
## Table structure recognition:
TableFormer, Table Transformer (TATR), PP-StructureV3 (table module), RapidTable
— born-digital: TSR alone, no OCR pairing. Scanned: TSR + a model from the OCR list.
## Figure/chart/stamp description (ranked by cost):
PaddleOCR-VL-0.9B, Granite-Docling-258M, Amazon Nova Lite, Claude Haiku 4.5, Amazon Nova Premier, Claude Sonnet 5, Claude Opus 4.5
— excluded: **Gemini family** (Google-first-party, not available on Bedrock — same platform-move exclusion as above), Qwen2.5-VL-7B (no managed API on Bedrock, self-deploy only)
— access notes: on GCP, Claude tiers were entitled via Agent Platform's global endpoint but blocked by a zero-default quota, and the Marketplace-purchase step required upgrading off trial billing — see [build_notes.md](build_notes.md) Task 0.4 for the full GCP-era account-friction history. On Bedrock, Claude and Amazon Nova are both first-party-supported model families with self-service model access requests (no purchase step) — expected to be materially less friction, but **not yet re-verified on this AWS account**; confirming Claude Haiku 4.5 access is the first concrete step of the AWS-era Task 0.4, not an assumption to build on top of yet.
