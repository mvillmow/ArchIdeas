# ArchIdeas
Ideas for improving AI model architecture — inference efficiency (TPOT/TTFT), memory bandwidth, adaptive compute, and generation paradigms.

Moved here from `Random/ArchIdeas/` as a standalone repo.

## Contents

- `arch_research_ideas.md` — Index of all 39 ideas in 5 sections (adaptive sparsity, compute reduction, new architectures, state machine/shared weights, memory reduction), each with a one-line description and literature search terms.
- `ArchNotes.pdf` — Supporting architecture notes.
- `research/` — Per-idea research docs (`research_<id>_<slug>.md`, 39 files) plus synthesis docs:
  - `architecture_synthesis.md` — Cross-cutting analysis across all 39 ideas and 6 groups.
  - `priority_ranking.md` — PURSUE / INVESTIGATE / DEPRIORITIZE tiers and implementation sequence.
  - `cross_reference_matrix.md` — Synergy/conflict map between ideas.
  - `implementation_spec_phase1.md` — Inference-only Phase 1 stack (INT4 + per-layer k + KV quant + compressed LM head + pre-attention router A/B).
  - `SHARED_PRELUDE.md` — Canonical baselines (A1 Qwen3.5-27B hybrid, A2 Qwen3-32B dense, B Qwen3.5-397B-A17B MoE, C K2 72.55B dense) and KV-cache formulas. Include verbatim in research prompts.
  - `report/` — LaTeX paper (`paper.tex`, `references.bib`, `paper.pdf`). Build with `./report/build.sh` (needs `tectonic` or `pdflatex`+`bibtex`).

## Top priorities (see `research/priority_ranking.md`)

PURSUE now: per-layer adaptive expert count (1.3), learnable top-k (1.1), INT4 block weights (5.7), compressed dense layers (2.2), prefill/decode split (6.2) + combined AR/split/diffusion (6.4), compressed dictionary (4.7).

## License

Apache 2.0 — see `LICENSE`.
