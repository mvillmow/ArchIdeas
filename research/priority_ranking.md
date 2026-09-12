# Priority Ranking — 39 Architecture Ideas

## Ranking Methodology

Verdicts are drawn from the "Verdict" / "Status" / "Executive Summary" field in each merged research document. Within each tier, ideas are ranked by:

1. **Standalone feasibility** — can the idea be implemented and validated in isolation without prerequisites?
2. **Synergy count** — number of confirmed synergies with other ideas (from cross_reference_matrix.md)
3. **Risk level** — overall risk rating from each document's Risk Assessment section

Tier assignments:
- **Tier 1 — PURSUE**: Verdict is explicitly PURSUE or NOVEL-PURSUE; multiple published implementations confirm the mechanism; implementation risk is LOW or LOW-MEDIUM.
- **Tier 2 — INVESTIGATE**: Verdict is INVESTIGATE FURTHER or NOVEL-INVESTIGATE; mechanism is sound but key empirical questions remain unanswered at target scale.
- **Tier 3 — DEPRIORITIZE**: Verdict is DEPRIORITIZE or the strong form is fundamentally flawed; standalone deployment is not recommended without substantial rework.

---

## Tier 1: PURSUE

| Rank | ID | Name | Verdict | Rationale | Key Dependencies |
|------|----|------|---------|-----------|-----------------|
| 1 | 1.3 | Per-Layer Adaptive Expert Count | PURSUE | Static post-training variant (LExI/Alloc-L style) is production-ready today: 1–2 engineer-days on Baseline B (vLLM confirmed on H100). Delivers ~10–36% MoE FFN FLOPs reduction; ~11–26% net TPOT improvement at k̄_l=7. Multiple independent published validations at scale (GRAPE, DiEP, Alloc-MoE, AdapMoE). Lowest risk of any idea in the corpus. | Baseline B (MoE only); trivially combined with 1.1, 2.2, 5.1 |
| 2 | 1.1 | Learnable Top-k | PURSUE | Multiple published implementations (AdaMoE EMNLP 2024, DynMoE ICLR 2025, ReMoE ICLR 2025) confirm feasibility. Average k drop from 11 to 7 yields ~1.2–1.4× net TPOT on Baseline B. High synergy count (1.3, 2.2, 5.1, 6.5). Primary open question: routing collapse at 512-expert scale. | Baseline B (MoE); AdaMoE/ReMoE reference implementations exist |
| 3 | 5.7 | Block Compressed Weights | PURSUE | Tier 1 INT4 quantization (GPTQ/AWQ/QServe/MARLIN) is production-ready today; delivers ~2.4–3.5× TPOT for all baselines. vLLM/TRT-LLM native support. Fully orthogonal to all other ideas (stacks with 5.1, 1.1, 1.3). Tier 2 Block-LR adds 2–3 month prototype path. | Any baseline; no prerequisite ideas |
| 4 | 2.2 | Compressed Dense Layers | PURSUE | Post-hoc SVD track (SVD-LLM/CALDERA style) applicable immediately to A1/A2 with <1 day effort. Training-native low-rank delivers ~3.4–4.0× TPOT on A2, ~3.5–5.0× on Baseline C. Highest per-idea TPOT potential in the corpus for dense baselines. Especially attractive at Baseline C (d_ff=28672, ~96% MLP compression at r=256). | A1/A2/C (dense/hybrid MLP); multiplicative synergy with 1.1, 1.3, 3.4 |
| 5 | 6.4 | Combined AR Loop + Split + Diffusion | PURSUE (NOVEL) | No published work combines in-weights prefill/decode split + block-diffusion decoder + learned stopping. Component prior art is strong (BD3-LM ICLR 2025 Oral, DistServe OSDI 2024). Multiplicative 4× decode FLOP reduction achievable in bandwidth-bound regime. Highest-novelty publication target in Group 6. | Full custom training; 4–5 stage curriculum; integrates 6.1+6.2+6.3 |
| 6 | 6.2 | Prefill/Decode Split | PURSUE (PARTIAL NOVELTY) | Architectural-level split with regime-specific optimization is genuinely novel. YOCO is closest prior art but focuses on KV cache, not regime optimization. Enables multiplicative combination with 6.3. DeltaNet recurrent state provides compact prefill interface for A1/B. | Custom training from scratch; depends on 3.4 for optimal decode recurrent mode |
| 7 | 4.7 | Compressed Dictionary | PURSUE (static-frequency variant) | Static-frequency reduced LM head: IMPLEMENT IMMEDIATELY — 1–2 days, no quality risk. INT4 LM head: IMPLEMENT IMMEDIATELY. Context-conditioned on draft models (30–50% TPOT): HIGH PRIORITY. 4.7+2.1 hybrid is the recommended production architecture. | No prerequisites for static variant; 1.7 for dynamic masking synergy |

---

## Tier 2: INVESTIGATE

| Rank | ID | Name | Verdict | Rationale | Key Dependencies |
|------|----|------|---------|-----------|-----------------|
| 8 | 5.4 | Linked Attention / Compressed Retrieval | INVESTIGATE | DeepSeek-V4 turns training-native compressed retrieval into the new reference point: CSA compresses KV blocks and sparse-selects top-k compressed entries with a lightning indexer, while retaining a local sliding-window branch. Post-hoc H2O/SnapKV/Quest remains useful for existing checkpoints, but new architectures should benchmark against CSA-style trained compression. | Full-attention or hybrid-attention baselines; multiplicative synergy with 5.1, 5.5 |
| 9 | 5.5 | Ragged Window / Local Branch Attention | INVESTIGATE | DeepSeek-V4 validates the "compressed global + uncompressed local" pattern at 1M context via CSA/HCA plus a 128-token sliding-window branch. Standalone per-token ragged windows remain an engineering gap, but local-window retention is now mandatory in any aggressive KV-compression design. | Full-attention or hybrid-attention baselines; synergy with 5.1 (MULT), 5.4 |
| 10 | 5.1 | TurboQuant / Low-Precision KV | INVESTIGATE | Post-hoc TurboQuant remains immediately deployable for A2/C at long context. DeepSeek-V4 adds separate evidence for low-precision attention state: FP8-dominant KV storage with BF16 RoPE dimensions and FP4 CSA indexer QK computation. KV-QAT extension remains research-grade, but the low-precision path is now stronger. | Any baseline; zero conflicts with weight compression; must account for CSA/HCA-specific KV layouts |
| 11 | 1.2 | Per-Token Adaptive Depth | INVESTIGATE | CALM/LayerSkip analogues confirm up to 2–3× speedup; prefix/postfix structure is genuine architectural novelty. Critical risk: "Diminishing Returns" (arXiv:2603.23701) shows reduced suitability in modern LLMs, especially MoE/SSM. Requires wall-clock TPOT validation on 7–8B model before committing to 27B+ scale. 13–22 engineer-weeks for production quality. | Dense baselines (A2, C) primarily; synergy with 1.1, 3.4, 4.2, 5.1, 5.4 |
| 12 | 4.5 | Learned Residual Flow | INVESTIGATE | ~25% TPOT and TTFT reduction with zero inference overhead if 25% layers stably pruneable. ShortGPT validates ~24–27% redundancy empirically. Train-time gating vs. post-hoc pruning ablation is the critical experiment. Recommended at 1B–3B scale first. | Any baseline; synergy with 4.2, 1.2; partial conflict with 3.4/3.6 |
| 13 | 6.5 | Pre-Attention Expert Router | INVESTIGATE (NOVEL) | No published MoE system routes FFN experts on Q/K projections. V1 (pre-attention residual) A/B test costs <1 day and is zero-risk to prototype. All 6 variants are NOVEL. Claims on expert specialization and load balance are empirically unvalidated hypotheses requiring ablation. High synergy with 1.1, 3.1, 3.3. | Baseline B (MoE primary); confirmed synergies with 3.1, 3.3, 1.1 |
| 14 | 3.3 | Dynamic Expert Router | INVESTIGATE | Cold-start routing quality gap is the key open engineering problem. Router KD tooling is lightweight. Operational value is real (post-training model updates, continual learning). Synergy with 3.2 creates complete MoE lifecycle system. Main impact is operational, not raw TTFT/TPOT. | Baseline B (MoE); 3.2 as operational partner; 6.5 for combined pre-attention routing |
| 15 | 3.2 | Swappable Experts | INVESTIGATE | Proof-of-concept feasible in weeks using vLLM weight update API on Mixtral 8×7B. PHATGOOSE zero-shot routing addresses cold-start. Primary value: deployment agility and knowledge lifecycle management. No raw TPOT/TTFT improvement. | Baseline B (MoE); 3.3 as routing partner; 4.2 for LoRA-delta swap |
| 16 | 4.2 | Shared Core + LoRA | INVESTIGATE | 7–33× MLP-only weight reduction at r=64. Critical unresolved risk: quality recovery when training from scratch at A2 scale (64 layers, d=5120). TPOT does not improve without early-exit synergy (1.2). Best pursued as bundled proposal with 1.2. | A1/A2/C (dense); must be bundled with 1.2 for TPOT benefit; synergy with 3.4, 3.1 |
| 17 | 4.3 | LoRA Everywhere (4.3-C) | INVESTIGATE | 4.3-C (pure A·B inference) delivers ~2× weight memory reduction and ~1.64× TPOT improvement at batch=1. TTFT is ~2× FASTER at r=d/4, not slower (critical correction from original doc). Non-uniform rank allocation required at 27B+. Validate at 1B–7B before committing to scale. | Any dense baseline; synergy with 2.2, 4.2, 3.2 |
| 18 | 3.1 | Layer-Level MoE | INVESTIGATE | Per-token attention specialization is novel and potentially high-impact. KV cache multiplication (k× per full-block layer) is a fundamental architectural constraint. Sparse expert KV history semantics must be characterized at small scale first. High technical risk. HIGH stage-1 experiment priority: 1–2 weeks, <1B params. | New architecture (not retrofit); synergy with 1.1, 5.1, 1.6, 4.2, 6.5 |
| 19 | 1.4 | Learned Dense/Sparse Layer | INVESTIGATE | Gradient-based layer-type assignment is novel at LLM pretraining scale. DeepSeek-V4 weakens the old dense-first baseline by using MoE in every Transformer block and Hash routing for the first 3 MoE layers; the target comparison is now learned assignment vs all-MoE plus deterministic early routing, not just vs DeepSeek-V3 3+58. Training complexity still requires 1–3B validation first. | Any baseline; requires training from scratch; synergy with 1.1, 1.3, 2.2 |
| 20 | 1.6 | Learned Layer Type | INVESTIGATE | Static 4-primitive NAS variant is low-risk, high-impact extension of validated 2-primitive hybrid paradigm. Proxy-based NAS at 1–3B: ~60 GPU-days. Use corrected TTFT/TPOT estimates (context-length-specific). Dynamic routing variant (1C) is secondary, higher-risk. | New architecture; MAD/Composer NAS framework; synergy with 1.1, 1.2, 5.1–5.5 |
| 21 | 1.7 | Dynamic Vocabulary | INVESTIGATE | Standalone TPOT impact 2.4–6.0% for large models — meaningful but not transformative. System-level impact up to 2.95× (CSV-Decode bundle). Highest-value deployment: speculative decoding draft models (1.12×–2.23× speedup demonstrated). Hard missing-token failure mode requires fallback mechanism. | Any baseline; synergy with 4.7 (MULT — highest impact), 2.1 |
| 22 | 5.8 | Block Sparse Weights | INVESTIGATE | Hardware support (cuSPARSELt 2:4, BLaST BSR) is production-quality. Quality recovery at 32B is the primary unknown; ~1.3–1.6× TPOT at 50% BSR at batch=1. Post-hoc 2:4 on 70B+ models: +1.86–2.04 PPL (Thanos/Wanda). Recommended path: validate on Qwen3-7B or 14B first. | Any baseline; synergy with 5.7, 5.9, 1.5, 3.5 |
| 23 | 5.9 | Dynamic Numeric Type | INVESTIGATE | Per-scalar formulation converges to SpQR/OWQ (already published). Reframed as prompt-conditioned runtime group-type assignment — this is the novel direction. Proof-of-concept required. Near-lossless quality at ~3.9-bit effective with SpQR baseline. | Any baseline; synergy with 5.8, 1.5 |
| 24 | 5.10 | Block Dynamic Compression | INVESTIGATE | Pivot to SpQR replication on Qwen3-32B (Phase 1) then MX+-style unified layout (Phase 2: ~4.50 bits/weight, high Blackwell compatibility) before investing in generalized k-level flags (b_eff=6.37 bits — not competitive). Novel contribution is the interleaved block layout for sequential memory access. | Any baseline; synergy with 5.7, 5.8 |
| 25 | 5.6 | Double Attention | INVESTIGATE | Vision precedent (DaViT) confirms quality gains. Efficiency NEGATIVE without depth reduction; POTENTIALLY POSITIVE (~0.74× TPOT) with ~34% fewer layers. Implementation is trivial; 1 GPU-day pilot is cheap. Without depth reduction, limited deployment value. | Full-attention baselines (A2, C); strongest synergy with 5.4 |
| 26 | 5.3 | Grammar Attention | INVESTIGATE | Practical variant reduces to NSA (ACL 2025) with incremental novelty. Key synergies: 5.1 (quantize retained KV), 5.2 (gated layers need no KV). Deployable as post-hoc NSA replication. | Full-attention baselines; synergy with 5.1, 5.2 |
| 27 | 3.7 | Learnable State Machine | INVESTIGATE | Inference-speedup case is real (1.33× TPOT at p=0.5, r=0.5). No published experiment validates FSM-governed block routing in an LLM. Joint 3.7+4.1 implementation strongly recommended. Priority path: soft FSM first, then hard routing. | Any baseline but risky on A1 (competing DeltaNet state); synergy with 4.1, 3.6, 3.5 |
| 28 | 4.1 | State Machine Core | INVESTIGATE | Merged 4.1+3.7 implementation at 1–3B scale; FSM infrastructure is identical and both must be co-developed. ~2% TPOT overhead in pure conditioning form; ~28% TPOT improvement when combined with 3.7 block-skipping. | Dense baselines preferred; joint 3.7+4.1 required; synergy with 3.1, 1.2, 3.5 |
| 29 | 3.4 | Recursive Internal State | INVESTIGATE | Validated at LLM scale (Huginn 3.5B, Ouro 2.6B, AdaPonderLM). Huginn at r=32 matches 50B-class models. RLTT +14.4% MATH-500 (unreviewed preprint). Benefits narrow to reasoning-dedicated endpoints. Implement 3.4 before 3.6. | Any baseline; synergy with 3.6, 3.7, 4.2, 2.2 (MULT), 6.1 |
| 30 | 6.1 | InArch AR Loop | INVESTIGATE | Technically sound; prior art density is high. Incremental novelty narrows to hybrid model application with trained halting at large scale. Engineering contribution framing recommended. LoopFormer (arXiv:2602.11451) is a direct novelty threat. | Any baseline, especially A1/B (DeltaNet hybrid for recurrent state reuse) |
| 31 | 2.1 | Hierarchical Frequency Dictionary | INVESTIGATE | Essentially adaptive softmax with known reference implementation (facebook/adaptive-softmax). 1–2 engineer-weeks to integrate. Low-to-medium impact at 32B+ (LM head is 2.4–4.7% of total weight); HIGH impact for small draft models. 4.7+2.1 hybrid is recommended production architecture. | Any baseline; incompatible with standard weight-tied embeddings |
| 32 | 3.5 | Gated Internal DAG | INVESTIGATE | High technical risk — no published results at LLM scale; sequential gate critical path is fundamental latency obstacle. 2-level binary DAG experiment required before larger investment. 1.2–1.7× TPOT vs dense if achievable. 6–12 months for 7B research prototype. | Dense baselines; synergy with 3.4, 3.6, 5.8, 1.6 |
| 33 | 4.6 | Mixture of Models | INVESTIGATE | Conditional on 5-step ablation sequence. Full MoMoRA is 6–18 month engineering effort; start with token-level routing signal validation (1–2 months). Component evidence: +7.6% AlpacaEval 2.0 (MoA), +4% accuracy at equal cost (FrugalGPT). HIGH technical risk from routing collapse with M=3 models. | Multiple base models required; custom CUDA kernels; synergy with 3.4, 1.1, 3.2 |
| 34 | 5.2 | LSTM-Gated Attention | INVESTIGATE | Validated at scale (RWKV 14B, GLA ICML 2024, xLSTM NeurIPS 2024, Griffin-7B). Replacing attention with gated linear attention in some layers reduces KV cache and improves TPOT at long context. Requires training from scratch; retrofit (LoLCATs-style) is the lower-risk path for existing models. | New training; synergy with 3.1, 6.3, 5.1 |
| 35 | 1.5 | Learned Sparsity Type | INVESTIGATE | Hardware-grounded distinction between N:M and MoE sparsity is real. Gate training complexity is HIGH. Mode collapse to single type is likely without careful regularization. No published demonstration of training-time type selection between N:M and MoE routing. Run N:M vs MoE ablation at equal FLOPs first, without the gate. | New architecture; 7–13 months research effort; synergy with 1.1, 1.3, 5.8 |
| 36 | 3.6 | Recursive Internal DAG | INVESTIGATE | Three interacting instability sources. Loop+DAG+persistent state has not been trained at any scale. Implement 3.4 then 3.5 as prerequisites before attempting 3.6. 6–12 months for production system. | Requires 3.4 and 3.5 as prerequisites; synergy with 3.7, 4.2, 2.2 |

---

## Tier 3: DEPRIORITIZE

| Rank | ID | Name | Verdict | Rationale | Key Dependencies |
|------|----|------|---------|-----------|-----------------|
| 37 | 3.8 | Trainable Activation | DEPRIORITIZE | No inference speedup without sparsity emerging empirically. Quality improvement real but poorly extrapolated from ≤1B scale to 27B+. Windowed variant confirmed infeasible (permutation invariance). Only revisit if: (a) sparsity emerges empirically at 27B+, or (b) reframe as PACT-style per-neuron quantization-aware clipping (low priority). | Any baseline; minimal synergies; recommended reframe: bundled with quantization-aware training |
| 38 | 4.4 | Skip List Layers (strong form) | DEPRIORITIZE | Strong form: gradient attenuation ~10⁻¹⁰ between skip points — identical to plain network failure. No inference benefit from residual connections (O(d) vs O(d·d_ff) MLP). DeepSeek-V4's mHC result strengthens the opposite path: preserve residual highways and enrich them with constrained learned mixing. Treat hybrid Hyper-Connections/mHC-style residual enrichment as a separate serious baseline, not as validation of skip-list replacement. | Strong form deprioritized; hybrid mHC-style residual enrichment belongs with 4.5-style stability/capacity work |
| 39 | 6.3 | Block Diffusion AR Decoder | DEPRIORITIZE (standalone) | Core mechanism is fully published as BD3-LM (ICLR 2025 Oral). Proposing 6.3 alone has no novelty. PURSUE only as a component of 6.4 (Combined AR+Split+Diffusion), where the integration with 6.2 and 6.1 creates the genuine research contribution. | Only valuable as part of 6.4; standalone = DEPRIORITIZE |

---

## Recommended Implementation Sequence

The following sequence maximizes early wins while building toward longer-horizon research contributions:

**Immediate (0–2 weeks, no training required):**
1. **1.3** (static post-training variant) on Baseline B — run LExI sensitivity profiling, deploy via vLLM; 1–2 engineer-days
2. **5.7** (INT4 quantization, Tier 1) on all baselines — deploy GPTQ/AWQ via vLLM; ~1 day
3. **4.7** (static-frequency LM head) on all baselines — 1–2 days; INT4 LM head additive
4. **5.1** (TurboQuant post-hoc) on A2 at 262K context — immediate for long-context serving

**Short-term (1–8 weeks, light training):**
5. **1.1** (learnable top-k) on Baseline B — requires training loop modification; use AdaMoE/ReMoE as reference
6. **5.4** + **5.5** + **5.1** combination on A2/C — prototype against a DeepSeek-V4-style CSA/HCA reference design before investing in post-hoc-only eviction
7. **6.5** (V1 pre-attention router) A/B test on Baseline B — <1 GPU-day

**Medium-term (2–6 months, training at 1–7B scale):**
8. **2.2** (training-native low-rank) on A2/C — highest TPOT potential for dense baselines
9. **4.5** (learned residual flow) ablation at 1B–3B scale — compare vs ShortGPT post-hoc pruning
10. **3.7 + 4.1** (joint FSM prototype) at 1–3B scale — soft FSM first
11. **3.4** (recursive internal state) prototype at 3.5B scale following Huginn training recipe

**Long-term (6–18 months, full research projects):**
12. **6.4** (combined AR+split+diffusion) — highest-novelty training project
13. **1.6** (learned layer type) — NAS at 1B, validate at 7B, scale to 27B+
14. **3.1** (layer-level MoE) — semantics validation experiment first (1–2 weeks, <1B params)
15. **1.4** (learned dense/sparse assignment) — only after 1.3 post-training variant is validated

---

## Baseline C Impact on Priority

Baseline C (K2 family, 72.55B dense, d=8192) changes the following priority assessments:

- **2.2 and 5.7 move UP** in practical priority for Baseline C: at d_ff=28672, MLP bandwidth is ~112 GB, and 96% compression at r=256 yields ~3.5–5.0× TPOT improvement. This makes 2.2 arguably the single highest-impact idea for Baseline C.
- **1.1, 1.3, 6.5 (MoE routing) become inapplicable** to Baseline C directly; they require sparse upcycling first (which adds 50% of pretraining compute per Komatsuzaki et al. 2023).
- **4.3-C (pure A·B inference) becomes more attractive** for Baseline C: d=8192 means ~2× weight memory at BF16; pure A·B at r=d/4=2048 delivers ~2× memory reduction and ~1.64× TPOT.
- **5.1 remains valuable** at long context for Baseline C: ~1.33× TPOT at 262K context (lower than A2's ~1.63× because weight bandwidth dominates at short context for the larger model).
- **6.4 remains applicable** to Baseline C as a dense model; prefill state compression via DeltaNet is the only component that requires architectural rework (since C may not have DeltaNet layers).

## DeepSeek-V4 Impact on Priority

DeepSeek-V4 adds a new large-scale MoE evidence baseline: V4-Pro has 1.6T total parameters, 49B active parameters, 61 layers, 384 routed experts plus one shared expert per MoE layer, fixed 6 routed experts per token, and 1M context. It changes priority in three ways:

- **5.3/5.4/5.5 move up for long-context architecture work** because CSA/HCA demonstrates that trained compressed attention can make 1M context operational. Post-hoc KV eviction is still useful, but new-model research should treat CSA/HCA as the baseline to beat.
- **4.4 hybrid residual work should be reframed around mHC**, not sparse skip-list replacement. The strong form remains Tier 3, but mHC-style constrained residual mixing should be considered a scale-tested ingredient for stability.
- **6.4 stays Tier 1 only as an integration claim.** Efficient 1M-token AR attention is now published in an open model family; novelty remains in combining architectural prefill/decode split, block diffusion, and learned stopping.
