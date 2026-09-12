# Implementation Specification — Phase 1 Prototype

## Objective

This specification defines a concrete, achievable Phase 1 experimental model that combines the top PURSUE and high-confidence INVESTIGATE ideas into a single model demonstrating measurable TPOT and TTFT improvements over the base model. The goal is to ship a working experimental model within 2–4 engineer-months that yields quantified inference speedup results suitable for a technical report and that provides the empirical foundation for the Phase 2 (training-time) research program.

Phase 1 is strictly inference-only on existing checkpoints. No retraining required for the selected ideas.

---

## Selected Ideas for Phase 1

The following 5 ideas are selected for Phase 1 based on: PURSUE verdict, post-hoc applicability (no retraining), production-framework compatibility, and orthogonality (each addresses a different bandwidth bottleneck).

| Priority | ID | Name | Verdict | Target Baseline | Expected TPOT |
|----------|----|------|---------|-----------------|---------------|
| 1 | 5.7 | Block Compressed Weights (INT4) | PURSUE | A2 (Qwen3-32B) | 2.4–3.5× |
| 2 | 1.3 | Per-Layer Adaptive Expert Count | PURSUE | B (Qwen3.5-397B-A17B) | ~1.1–1.3× net |
| 3 | 5.1 | TurboQuant KV Cache | INVESTIGATE (high confidence) | A2 (Qwen3-32B) at 262K context | ~1.63× |
| 4 | 4.7 | Compressed Dictionary (static LM head) | PURSUE | All baselines | ~2–6% |
| 5 | 6.5 | Pre-Attention Expert Router (V1) | INVESTIGATE (NOVEL) | B (Qwen3.5-397B-A17B) | ~0% TPOT delta (quality measurement) |

Ideas 5.7, 1.3, 5.1, and 4.7 are pure inference-time modifications applied post-hoc. Idea 6.5 V1 is an A/B training experiment to characterize routing quality change from pre-attention placement.

---

## Base Model Choice

**A2: Qwen3-32B Dense** is the recommended base for Phase 1 ablations on ideas 5.7, 5.1, and 4.7. Rationale:
- Dense architecture eliminates MoE routing complexity from the ablation loop
- 32B scale is large enough for results to generalize to 72B+ but small enough to run experiments in hours
- Single-GPU (A100-80G with INT4) or dual-GPU inference is feasible
- Ideas 5.7+5.1+4.7 compose fully independently on this model

**B: Qwen3.5-397B-A17B MoE** is used for ideas 1.3 and 6.5. Rationale:
- 1.3 and 6.5 require an existing MoE architecture
- vLLM production support confirmed for 1.3 (LExI authors tested on H100 via vLLM)
- Baseline B's 512-expert scale provides the highest-novelty application context for both ideas

---

## Architecture Modifications

### Modification 1: INT4 Block Quantization (Idea 5.7, Tier 1)

**What to change:**
Apply GPTQ or AWQ INT4 weight quantization (group size = 128) to all MLP projection weights and attention QKV/output projection weights in A2 (Qwen3-32B). Use MARLIN or QServe kernel for decode-time inference; retain W4A16 inference mode (weights INT4, activations BF16).

Toolchain: `autoawq` or `gptq-py` for quantization; `vllm` with MARLIN backend for serving. Quantization calibration: 512 samples from Wikitext-2 or PilE, 2048 token context.

**Parameter count impact:** Zero (INT4 is a weight storage format, not an architectural change). Total weight storage: ~64 GB BF16 → ~16.5 GB INT4 + ~0.5 GB scales/zeros = ~17 GB.

**Expected TPOT/TTFT change vs. A2 BF16 baseline:**
- TPOT (batch=1, 32K context): ~2.4–3.5× improvement. Canonical formula: (16.5+8.59)/(64+8.59) = 25.09/72.59 = 0.346× → **2.89× theoretical**; practical range 2.4–3.5× accounting for dequantization overhead.
- TTFT (8K prompt, W4A16): Approximately unchanged. W4A16 does not reduce prefill FLOPs; compute-bound prefill is unaffected. W4A8 variant would deliver ~2× TTFT improvement but is a separate evaluation.

**Risk mitigation:**
- PPL degradation: expected +0.3–1.5 PPL on Wikitext-2 at INT4/g=128. Acceptance criterion: <+1.0 PPL.
- If degradation exceeds threshold: switch to AWQ with protection of top-1% salient weights, or increase group size to 64.
- Calibration sensitivity: run quantization on 3 different calibration sets; report variance.

---

### Modification 2: Per-Layer Adaptive Expert Count (Idea 1.3, Static Variant)

**What to change:**
Apply LExI-style sensitivity profiling to Baseline B (Qwen3.5-397B-A17B). This is a pure routing configuration change — all 512 expert weights per layer remain loaded; only the per-layer `top_k` parameter changes.

Procedure:
1. Run Monte Carlo sensitivity profiling: for each of L=60 MoE layers, sample N=256 synthetic Gaussian inputs; compute Frobenius norm of output perturbation for each candidate k_l ∈ {4, 6, 8, 10, 11, 13}. Record sensitivity curve S_l(k_l).
2. Apply evolutionary algorithm (or simple greedy allocation) subject to global compute budget constraint: Σ k_l ≤ target_budget. Target: k̄_l ≈ 7 (36% FLOPs reduction from 11); alternatively k̄_l ≈ 9 (18% FLOPs reduction, lower risk).
3. Instantiate per-layer `top_k` array in vLLM's FusedMoE layer constructor. No kernel change required; vLLM accepts per-layer `top_k` at construction time.

**Parameter count impact:** Zero (routing configuration only; no weight changes).

**Expected TPOT/TTFT change vs. Baseline B BF16:**
At k̄_l=10: Net TPOT ~0.91× (9% improvement). At k̄_l=9: Net TPOT ~0.84× (16% improvement). At k̄_l=7: Net TPOT ~0.68× (32% improvement). [derived: (k̄_l/11) × 0.88 + 0.12; expert weight fraction = 277/315 ≈ 0.88 at k=11, per research_1_3.md §4.3]

Correction applied from research_1_3.md: the net TPOT bound accounts for unchanged gating (4 MB/layer) and DeltaNet attention weights (~33.6 MB/layer) alongside reduced expert weight loads. The MoE-FFN-only bound (0.89–0.64×) overstates net improvement by ~5–10 percentage points.

**Expected quality delta:**
- At k̄_l=9: Expected <1% quality degradation on MMLU/HumanEval per Alloc-MoE and GRAPE published results at comparable reductions.
- At k̄_l=7: Expected 1–3% quality degradation. Acceptance criterion: MMLU delta <2%.

**Risk mitigation:**
- Run at k̄_l=9 first (safer); only proceed to k̄_l=7 if k̄_l=9 shows <1% quality delta.
- 512-expert scale has not been validated in any published paper for static per-layer k. The LExI paper tested up to DeepSeek-V2-Lite (160 experts). The 512-expert scale is the primary residual novelty.
- If quality cliff appears: fall back to k̄_l=10 (9% TPOT improvement, near-zero quality risk).

---

### Modification 3: TurboQuant KV Cache Quantization (Idea 5.1)

**What to change:**
Apply post-hoc TurboQuant (Zandieh et al., ICLR 2026, arXiv:2504.19874) to the KV cache of A2 (Qwen3-32B). Use the PolarQuant stage (rotation + INT4 scalar quantization) for immediate deployment; optionally add QJL correction for research-grade results.

For vLLM deployment: use the built-in FP8 KV cache feature (vLLM 0.4+) as an approximation, or integrate TurboQuant's rotation + INT4 quantization as a post-processing step on the KV cache tensor before storage.

TurboQuant is training-free and calibration-free. Apply at serving time; no model weight change.

**Parameter count impact:** Zero. KV cache is runtime memory, not model parameters.

**Expected TPOT/TTFT change vs. A2 BF16 baseline:**
- TPOT at 32K context: ~1.13× improvement (KV cache from ~8.59 GB to ~2.15 GB INT4; but weight BW ~16.5 GB dominates at this context → modest TPOT gain).
- TPOT at 262K context: ~1.63× improvement (KV cache from ~68.7 GB to ~17.2 GB INT4; KV bandwidth begins to exceed weight bandwidth at this context length for A2).
- TTFT: Unchanged (KV quantization affects decode-time KV reads, not prefill compute).

**Risk mitigation:**
- Quality risk: TurboQuant achieves near-lossless 3.5-bit compression per ICLR 2026 results on LongBench. Acceptance criterion: LongBench score within 1% of BF16 KV baseline.
- Implementation availability: Use vLLM FP8 KV cache (FP8 = ~2× compression from BF16) as a validated first step; TurboQuant's full rotation+INT4 stack is the optimal target.

**Combined effect with Modification 1:**
5.7 + 5.1 applied simultaneously to A2 at 262K context:
- Weight BW: 64 → 16.5 GB (INT4)
- KV BW at 262K: 68.7 → 17.2 GB (INT4 KV)
- Net TPOT: (16.5 + 17.2) / (64 + 68.7) = 33.7 / 132.7 = **0.254×** → **~3.9× combined improvement**

---

### Modification 4: Static-Frequency Compressed LM Head (Idea 4.7)

**What to change:**
Replace the full vocabulary LM head (output projection W_out, shape V×d) with a subset-indexed version selecting only the top-V' most frequent tokens from frequency statistics.

Procedure:
1. Compute token frequency histogram from a 10M-token corpus sample (Wikitext or OpenWebText subset).
2. Sort V=151,936 tokens (A2) or V=248,320 tokens (A1/B) by frequency.
3. Build W_out_small = W_out[top_V' tokens, :] where V'=32,768 (maintains 99.7% of token probability mass per nucleus sampling statistics).
4. At inference, compute logits over W_out_small; project back to full vocabulary via a lookup for the <0.3% of steps where the next token is outside V'.

Simultaneously apply INT4 LM head quantization to W_out_small using AWQ:
- A2: W_out_small (32K × 5120) at INT4 = ~82.0 MB vs. ~1,488 MB (~1.49 GB) BF16 full head (151,936 × 5120 × 2 bytes = 1,555,824,640 bytes).

**Parameter count impact:** Zero (W_out_small is a view of the full W_out; inference uses only the subset).

**Expected TPOT/TTFT change vs. baseline:**
- A2 standalone: ~2.3% TPOT improvement (LM head is ~2.4% of total A2 weight BW; V' reduction to 32K eliminates ~79% of LM head BW). TTFT: ~0% (LM head is <0.001% of 8K prefill FLOPs).
- B standalone: ~5.7% TPOT improvement (LM head is ~6.0% of active weight BW for B).
- Combined with INT4 LM head: additive ~0.5–1% additional saving from quantizing the reduced head.

**Note on 4.7+2.1 hybrid architecture:**
Add a simple fallback for rare tokens (V_fallback = full W_out), triggered when the argmax of the small-head logits falls within a "boundary" confidence threshold. The combined architecture uses W_out_small for the fast path (~99.7% of steps) and W_out_full for rare token generation. This is the production architecture recommended in research_4_7.md.

**Risk mitigation:**
- Hard-miss failure: rare token outside V' produces argmax on W_out_small with wrong token. Mitigation: fallback to full W_out for any generation step where top-1 logit over W_out_small is below a threshold (or simply for any output token frequency rank > V'). Acceptance criterion: fallback trigger rate <0.5%.
- Distortion of top-k sampling: under top-p=0.9 nucleus sampling, V'=32K is effectively never the constraint. Under top-k=50, V'=32K has zero effect on sampling.

---

### Modification 5: Pre-Attention Expert Router V1 (Idea 6.5)

**What to change:**
This modification is an A/B quality characterization experiment, not a TPOT-improvement intervention. It changes where the MoE router reads its input (from post-attention residual to pre-attention residual x) in Baseline B.

Implementation for V1-parallel:
1. In the transformer block forward pass of Baseline B, move the router forward call from `h = x + Attn(LN(x)); g = Router(h)` to `g = Router(x)` (where x is the pre-attention residual), running the router concurrently with attention.
2. The expert dispatch proceeds identically after the routing decision; only the input to the router changes.
3. The change requires modifying one line in the MoE transformer block forward function. FlashAttention 2/3 is fully compatible.
4. No new parameters, no new FLOPs, no kernel changes required.

This is strictly an A/B comparison training run at small scale (1–3B model) to characterize:
(a) Expert activation entropy: does pre-attention routing produce better load balance?
(b) Expert specialization patterns: do experts cluster differently on syntactic vs. semantic features?
(c) Quality delta: MMLU, GSM8K, HumanEval on a 1–3B MoE model trained from scratch with V1 vs. standard router.

**Parameter count impact:** Zero. Router weights are unchanged; only their input changes.

**Expected TPOT/TTFT change vs. Baseline B:**
Zero TPOT change. Router FLOPs are <0.5% of total MLP FLOPs and run in parallel with attention for V1-parallel. The V1 parallel variant adds zero critical-path latency.

**Risk mitigation:**
- Worse load balance: if pre-attention representations are more homogeneous than post-attention, load balance may degrade. Mitigation: monitor expert activation entropy during training; if entropy drops below post-attention baseline, fall back to standard router.
- Training instability: V1 uses the unmodified residual x as router input — same gradient path as standard routing (x → LN(x) → Attn; x → Router). No stop-gradient needed. Low gradient coupling risk.
- SwitchHead negative result risk: SwitchHead tested Q/K MoE for attention head selection and found unnecessary. However, FFN expert routing on Q/K projections (V2/V3) is a distinct task — the SwitchHead result does not directly generalize. V1 (using residual x) is unaffected by this risk.

---

## Combined Architecture Diagram (ASCII)

### A2 Phase 1 Stack (5.7 + 5.1 + 4.7)

```
Input tokens
     |
[Embedding lookup]
     |
 +---v---------+  x64 layers
 | LN + QKV    |
 | (INT4 W4A16)|  <-- Modification 1: INT4 weights
 | Attention   |
 | + KV store  |  <-- Modification 3: INT4 KV cache (262K ctx)
 | LN + MLP    |
 | (INT4 W4A16)|  <-- Modification 1: INT4 weights
 +-------------+
     |
[LM Head W_out]
[W_out_small:  ]  <-- Modification 4: V'=32K subset + INT4
[32K × 5120    ]
     |
[Top-1 argmax  ]
[+ rare fallback] <-- Modification 4: <0.3% fallback rate
     |
Output token
```

### Baseline B Phase 1 Stack (1.3 + 6.5 V1)

```
Input tokens
     |
[Embedding lookup]
     |
 +---v---------+  x60 layers
 | LN + QKV    |
 | Attn (15 GatedAttn) or DeltaNet (45 layers)
 |             |
 | Router(x)   |  <-- Modification 5: router now reads x (pre-attention)
 |    |        |      runs in parallel with Attn
 |    v        |
 | TopK experts|  <-- Modification 2: k_l varies by layer (LExI profiling)
 |  [k_l=4–14] |      k̄_l ≈ 7–9 depending on aggressiveness
 +-------------+
     |
[LM Head W_out_B]  (standard; 4.7 applicable here too)
     |
Output token
```

---

## Training Recipe

Phase 1 is inference-only. All modifications are post-hoc:

| Modification | Toolchain | Steps | Estimated Time |
|---|---|---|---|
| 5.7 INT4 (A2) | autoawq 0.2+ | (1) Calibrate on 512 samples, (2) Quantize with g=128, (3) Export to MARLIN format | 2–4 GPU-hours on 1× A100-80G |
| 1.3 LExI (B) | Custom Python (sensitivity profiling) | (1) Profile per-layer sensitivity with N=256 synthetic samples, (2) Solve budget allocation via DP, (3) Configure vLLM per-layer top_k | 1–2 engineer-days |
| 5.1 TurboQuant (A2) | vLLM FP8 KV (immediate) or custom TurboQuant (optimal) | Enable vLLM FP8 KV flag | 1 hour (FP8 path) or 2–3 weeks (full TurboQuant rotation) |
| 4.7 LM head (all) | Python + AWQ | (1) Compute frequency histogram, (2) Build W_out_small, (3) Apply INT4 quantization, (4) Add fallback lookup | 1–2 engineer-days |
| 6.5 V1 A/B (B) | PyTorch forward modification + 1B MoE training run | Modify router call location; train 1–3B MoE from scratch twice (V1 vs. standard) | 2–4 days training on 8× H100 |

No hyperparameter search required for modifications 1–4. Modification 5 requires a training run.

---

## Evaluation Protocol

### Latency Benchmarks (all modifications)

Measure on H100-SXM5 (80 GB) or A100-80G (80 GB):
- **TPOT**: batch=1, context lengths {4K, 8K, 32K, 64K, 128K, 262K}. Report token/second.
- **TTFT**: batch=1, prompt lengths {1K, 2K, 4K, 8K, 32K}. Report time-to-first-token in ms.
- **Throughput**: batch sizes {1, 4, 16, 64} at 32K context. Report total tokens/second.

### Quality Benchmarks

| Task | Metric | Acceptance Criterion |
|------|--------|---------------------|
| MMLU (5-shot) | Accuracy | Delta vs. BF16 baseline ≤ 1.0 pp |
| GSM8K (8-shot CoT) | Accuracy | Delta ≤ 2.0 pp |
| HumanEval | Pass@1 | Delta ≤ 2.0 pp |
| LongBench (6 tasks) | Average score | Delta ≤ 1.0 pp (especially for 5.1+5.5) |
| RULER (128K) | Average score | Delta ≤ 2.0 pp |

### Combined Stack Evaluation

After validating each modification individually (acceptance criteria above), evaluate the combined stack:
1. A2 with 5.7+5.1+4.7 (three orthogonal optimizations)
2. B with 1.3+6.5 (routing modifications only)
3. Optionally: A2 with 5.7+5.1+4.7+2.2 (post-hoc SVD on MLP, 1 day additional effort)

Report combined TPOT/quality table comparing: (a) BF16 baseline, (b) each modification individually, (c) combined stack.

---

## Baseline C Considerations

**Can the combined architecture scale to 72.55B (Baseline C)?**

For modifications 5.7, 5.1, and 4.7: YES, directly applicable.
- 5.7 (INT4): ~145 GB BF16 → ~37 GB INT4. TPOT improvement: ~3.0–3.5× (higher than A2 due to larger weight BW). INT4 fits on 1× H100-SXM5 (80 GB).
- 5.1 (TurboQuant): applicable; ~10 GiB KV at 32K provides ~1.05× TPOT at 32K, ~1.33× at 262K.
- 4.7 (LM head): applicable; LM head is ~2.8% of Baseline C weight BW.

For modifications 1.3 and 6.5: NOT DIRECTLY APPLICABLE (Baseline C is dense, not MoE).
- To apply 1.3/6.5 to Baseline C, sparse upcycling is required first (~50% of pretraining compute per Komatsuzaki et al. 2023, arXiv:2212.05055). This is out of scope for Phase 1.
- Alternative for Baseline C: prioritize 2.2 (training-native low-rank) and 4.3-C (pure A·B inference) as the high-impact TPOT improvements for the dense architecture.

**Recommended Baseline C Phase 1 stack (if prioritized over A2):**
- 5.7 INT4: ~3.0–3.5× TPOT
- 5.1 TurboQuant at 262K context: ~1.33× additional TPOT at long context
- 4.7 static LM head: ~2.7% additional TPOT
- Combined at 262K: (37.4 + 20.0)/(145.1 + 80.0) = 57.4/225.1 = 0.255× → **~3.9× combined TPOT improvement**

The Baseline C combined stack delivers essentially the same 3.9× combined TPOT improvement as the A2 stack, but at 72.55B scale, which is the more practically deployable target for production use cases requiring maximum model quality.
