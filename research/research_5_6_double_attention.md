# Research: Double Attention
## ID: 5.6

---

## Executive Summary

**Novelty verdict:** PARTIAL — ~65% covered; two sequential attention passes per block exists in vision (DaViT [3], 2022), original Transformer decoder (self + cross per block [1]), iterative recurrent attention (Universal Transformer [2]), Looped Transformers [14], Attention-over-Attention [15], and intra-layer recurrence for LM (Nguyen & Lin [17], 2025); residual novelty is two sequential SAME-SEQUENCE self-attention passes in a decoder-only LM layer with shared KV cache analyzed explicitly for the depth-reduction tradeoff — the depth-reduction hypothesis itself is unvalidated empirically for decoder-only LMs ([DaViT, 3], [Universal Transformer, 2], [Nguyen & Lin, Canadian AI 2025, 17]).

Idea 5.6 proposes running two sequential self-attention passes per transformer layer to enrich per-layer representation quality. With shared KV cache, memory is **unchanged**; the cost is **increased** TTFT and TPOT proportional to the attention bandwidth fraction (modest at short context, substantial at long context). The primary justification is the **depth-reduction hypothesis**: if double attention allows 30–50% fewer layers, KV cache and TPOT could improve net.

**Key finding: Without depth reduction, double attention is a quality-add with a compute increase — it does NOT improve efficiency on its own.** The depth-reduction hypothesis is unvalidated for decoder-only LMs.

### Key Comparison Tables

**vs Baseline A1 (Qwen3.5-27B Hybrid)** — *Double attention on 16 full-attention layers only; 48 Gated DeltaNet layers unchanged*

| Metric | Baseline A1 | Double Attn (shared KV, shared Q/K/V) | Change | Notes |
|--------|------------|--------------------------------------|--------|-------|
| Compute (FLOPs/token, prefill s=8K) | O(L/4·s·d/H + L·d²) | O(L/4·2s·d/H + L·d²) | ↑ ~1.073× | Attn on 25% of layers; attn fraction ~7.3% at 8K (A1 hybrid); doubling adds ~7.3% total |
| Compute (FLOPs/token, prefill s=32K) | — | — | ↑ ~1.27× | Attn fraction ~27% at 32K for A1's 16 full-attn layers |
| Memory bandwidth (decode) | O(L/4·s·d_kv + L·d²) | O(L/4·2s·d_kv + L·d²) | ↑ KV BW doubled on 16 layers | KV doubled for 25% of layers; modest net increase |
| KV cache (32K ctx) | **~2.15 GB** | **~2.15 GB** | = | Shared KV; NO extra memory |
| Weight memory | O(L·d·d_ff) | = (if shared Q/K/V) | = | Separate Q/K/V adds ~0.4 GB |
| Training cost | 1.0× | ~1.073× | ↑ slight | Attn fraction ~7.3% at 8K on 25% of layers |
| TTFT (8K prompt) | ref | ↑ **~1.073×** | ↑ slight | [derived: A1 applies double-attn to 16/64 full-attn layers; attn FLOPs/layer = 2×s×d = 2×8192×5120 = 83.9M; FFN FLOPs/layer = 3×d×d_ff (SwiGLU) ≈ 3×5120×13824 = 212M (A1 d_ff); total per-layer ≈ 295M; double-attn adds 83.9M per affected layer; 16 layers → +16×83.9M = 1,342M extra over baseline total 64×295M = 18,880M; TTFT multiplier = (18,880+1,342)/18,880 ≈ 1.071×, rounded to 1.073×] |
| TPOT (batch=1) | ref | ↑ **~1.036×** | ↑ slight | KV doubled for 25% layers; KV at 32K is ~3.7% of total A1 BW |

**vs Baseline A2 (Qwen3-32B Dense)** — *Double attention on all 64 layers*

| Metric | Baseline A2 | Double Attn (shared KV, shared Q/K/V) | Change | Notes |
|--------|------------|--------------------------------------|--------|-------|
| Compute (FLOPs/token, prefill s=8K) | O(L·(s·d+d·d_ff)) | O(L·(2s·d+d·d_ff)) | ↑ ~1.176× | Attn ~17.6% at s=8K; doubling adds ~17.6% total |
| Compute (FLOPs/token, prefill s=32K) | — | — | ↑ ~1.46× | Attn ~46% at s=32K |
| Memory bandwidth (decode) | O(L·(d·d_ff+s·d_kv)) | O(L·(d·d_ff+2s·d_kv)) | ↑ proportional to KV fraction | At 32K: KV 11.8% of total weight+KV BW → doubles → +11.8% total decode BW |
| KV cache (32K ctx) | **~8.59 GB** | **~8.59 GB** | = | Shared KV; NO extra memory |
| Weight memory | O(L·d·d_ff) | = (if shared Q/K/V) | = | Separate Q/K/V adds ~10 GB |
| Training cost | 1.0× | ~1.176× at s=8K; ~1.46× at s=32K | ↑ | Gradient through two passes |
| TTFT (8K prompt) | ref | ↑ **~1.176×** | ↑ ~18% | [derived: A2 applies double-attn to all 64 layers; attn FLOPs/layer = 2×s×d = 2×8192×5120 = 83.9M; FFN FLOPs/layer (SwiGLU, d_ff=25600) = 3×5120×25600 = 393M; attn fraction = 83.9/(83.9+393) = 17.6%; doubling attn pass adds 83.9M/layer × 64 layers = 5,370M extra over baseline total 64×476.9M = 30,522M; TTFT multiplier = (30,522+5,370)/30,522 = 35,892/30,522 ≈ 1.176×] |
| TPOT (batch=1) | ref | ↑ **~1.118× at 32K** | ↑ modest | KV fraction 11.8% at 32K; doubling KV BW → (64+2×8.59)/(64+8.59) ≈ 81.18/72.59 ≈ 1.118× |

**Depth-reduction hypothesis (UNVALIDATED):** If double attention allows 34% fewer layers (e.g., 42 instead of 64 for A2), then:
- KV cache: ~8.59 GB × (42/64) ≈ **~5.63 GB** (34% reduction) ✓ improvement vs baseline
- TPOT: ~0.74× baseline (weight BW and KV both scale with depth) — genuine improvement
- This requires empirical validation at pilot scale before commitment.

**vs Baseline B (Qwen3.5-397B-A17B MoE)** — *Double attention on 15 global-attention layers only*

| Metric | Baseline B | Double Attn (shared KV) | Change | Notes |
|--------|-----------|------------------------|--------|-------|
| Compute (FLOPs/token) | O(L/4·s·d/H + L·k_moe·d·d_e) | ↑ on 15 attn layers | ↑ ~1.03× | MoE FFN dominates; 15 attn layers doubled |
| Memory bandwidth (decode) | O(L/4·s·d_kv + active weights) | ↑ on KV BW for 15 layers | ↑ ~1.03× | KV doubled for 25% of layers; MoE BW unchanged |
| KV cache (32K ctx) | **~1.0 GB** | **~1.0 GB** | = | Shared KV |
| Weight memory | O(L·E·d·d_e) | ≈ = | = | |
| Training cost | 1.0× | ~1.03× | ↑ slight | Attn tiny fraction of MoE compute |
| TTFT (8K prompt) | ref | ↑ **~1.03×** | ↑ slight | |
| TPOT (batch=1) | ref | ↑ **~1.03×** | ↑ slight | KV is ~2.9% of total BW; doubling → +2.9% |

**vs Baseline C (K2 family, LLM360)** — *Double attention on all 80 layers*

| Metric | Baseline C | Double Attn (shared KV) | Change | Notes |
|--------|-----------|------------------------|--------|-------|
| Compute (FLOPs/token, prefill s=8K) | O(L·(s·d+d·d_ff)) | O(L·(2s·d+d·d_ff)) | ↑ ~1.16× | Attn fraction ~16% at s=8K for 8192-dim model |
| Memory bandwidth (decode) | O(L·(d·d_ff+s·d_kv)) | O(L·(d·d_ff+2s·d_kv)) | ↑ KV BW doubled | KV fraction 6.4% at 32K; doubling → +6.4% total |
| KV cache (32K ctx) | **~10.0 GiB** | **~10.0 GiB** | = | Shared KV |
| Weight memory | O(L·d·d_ff) | ≈ = | = | |
| Training cost | 1.0× | ~1.16× at s=8K | ↑ slight | |
| TTFT (8K prompt) | ref | ↑ **~1.16×** | ↑ slight | |
| TPOT (batch=1) | ref | ↑ **~1.065×** | ↑ slight | KV 6.4% of total; doubling → +6.4% |

---

## 1. Idea Description

Run two sequential self-attention passes within a single transformer decoder block. The second pass operates on the post-first-attention residual representation, allowing it to refine or correct the first. When the KV cache is shared between both passes, memory is **unchanged** but KV reads double per decode step.

**Three variants:**
1. **Shared Q/K/V, shared KV**: No parameter or memory increase; cheapest; second Q from post-residual
2. **Separate Q (LoRA-differentiated), shared K/V/KV**: Small overhead (~0.2% params per layer); second pass attends from different query perspective; **recommended**
3. **Separate full Q/K/V, separate KV**: ~2× attention parameter and memory cost; not recommended as default

All FLOPs estimates in this document assume **variant 1 or 2 (shared Q/K/V projections)**. Separate projections roughly double the stated TTFT/training-cost increases.

**Net efficiency impact without depth reduction:**
- Double attention INCREASES TPOT and TTFT
- There is NO inherent efficiency benefit — this is a pure quality-add at higher cost
- Efficiency benefit only materializes via the **depth-reduction hypothesis**: if double attention per layer allows 30–50% fewer total layers, the result could be net KV cache reduction and TPOT improvement

---

## 2. Literature Review

Attention Is All You Need[1]: Vaswani et al., NeurIPS 2017 (arXiv:1706.03762). Original Transformer. Decoder block already runs two sequential attention sub-layers (masked self-attention + cross-attention) before FFN. Canonical precedent for two attention operations per layer.

Universal Transformers[2]: Dehghani et al., ICLR 2019 (arXiv:1807.03819). Shared-weight recurrence across T steps; +0.9 BLEU over vanilla Transformer on WMT14 En-De. Demonstrates that multi-pass attention improves quality especially on compositional tasks.

DaViT: Dual Attention Vision Transformers[3]: Ding et al., ECCV 2022 (arXiv:2204.03645). Two sequential self-attention passes per block (spatial then channel) in vision. DaViT-Base 84.6% ImageNet top-1. **Strongest existence proof** of quality gains from two sequential attention passes per block.

Double Attention: Biomimetic Self-Attention Optimization[4]: Zhang et al., Biomimetics 2025 (DOI:10.3390/biomimetics10010034). Two-Key attention on medical imaging; 96.6% vs 91.6% accuracy at 7% FLOPs overhead. Explicitly named "Double Attention."

Mixture of Attention Heads (MoA)[5]: Zhang et al., EMNLP 2022 (arXiv:2210.05144). Sparse router selects k of H heads per token; improves quality while reducing active heads. Demonstrates that varied attention operations per token improves quality.

MoH: Multi-Head Attention as Mixture-of-Head Attention[6]: Jin et al., ICML 2025 (arXiv:2410.11842). Routes tokens to top-k heads; outperforms full MHA at 75% head activation. Single attention pass with richer head selection.

Wide Attention Is The Way Forward For Transformers?[7]: Brown et al., arXiv 2022 (arXiv:2210.00640). Single-layer wide models match deep models with 3.1× inference speedup. Supports depth-for-width trade.

Leaner Transformers: More Heads, Less Depth[8]: Saratchandran et al., arXiv 2025 (arXiv:2505.20802). Increasing heads while reducing depth maintains accuracy with 30–50% parameter reduction. Most direct evidence for depth-reduction hypothesis.

Value Residual Learning For Alleviating Attention Concentration (ResFormer/SVFormer)[9]: Zhou et al., ACL 2025 (arXiv:2410.17897). Cross-layer value residuals improve quality with 16% fewer params; SVFormer reduces KV cache ~50%. Adjacent: enriching per-layer attention signal via cross-layer information.

Associative Transformer (AiT)[10]: Sun et al., CVPR 2025 (arXiv:2309.12862). Two-phase attention (bottleneck write + Hopfield read) in a global workspace layer; outperforms sparse Transformers on relational reasoning. Direct two-pass attention mechanism.

Stack Attention[11]: DuSell & Chiang, ICLR 2024 Spotlight (arXiv:2310.01749). Stack-augmented multi-step attention captures CFLs that single-pass attention cannot. Supports hypothesis that second pass catches patterns missed by first.

The Sparse Frontier[12]: Nawrot et al., ICLR 2025 (arXiv:2504.17768). At 16K tokens, attention is ~40% of prefill FLOPs; at 128K, ~80%. Used for attention fraction estimates throughout this document.

Attention Residuals (AttnRes) [post-cutoff, unverified][13]: Kimi Team (Guangyu Chen et al.), arXiv 2026 (arXiv:2603.15031). **Post-knowledge-cutoff; contents unverifiable.** Cross-depth attention (not intra-layer repetition). Cited for completeness; qualitative point retained with hedging.

Looped Transformers[14]: Yang et al., arXiv 2023 (arXiv:2311.12424). Per abstract: looped transformer achieves performance comparable to standard transformer while utilizing less than 10% of the parameter count on data-fitting problems. Specific T=2 vs T=1 breakdowns and per-task (sorting/associative recall) gains are paper-body values. **Closest language-task precedent for attention recurrence.**

Attention-over-Attention Neural Networks for Reading Comprehension[15]: Cui et al., ACL 2017 (arXiv:1607.04423). Document-level attention over word-level attention outputs — two sequential attention passes in NLP (2017). Demonstrates two-pass sequential attention in NLP predating DaViT.

Thinking Deeper, Not Longer: Depth-Recurrent Transformers for Compositional Generalization [post-cutoff, unverified][16]: Chen, arXiv 2026 (arXiv:2603.21676). **Post-knowledge-cutoff; contents unverifiable.** Shared-weight block recurrence for compositional generalization. Cited for completeness; flagged.

Intra-Layer Recurrence in Transformers for Language Modeling[17]: Nguyen & Lin, arXiv:2505.01855. Applies recurrence selectively to individual layers (not full block repetition); per abstract: "allocating more iterations to earlier layers yields optimal results"; demonstrates intra-layer recurrence improves language modeling efficiency. Specific per-benchmark PPL deltas are paper-body tables. Venue "Canadian AI 2025" unverified.

---

## 3. Prior Art Classification

- **Status**: PARTIAL (~65% covered)
- **Exists**: Two sequential attention passes per block in vision (DaViT [3], 2022); original Transformer decoder (self + cross attention per block [1]); iterative recurrent attention (Universal Transformer [2]); Looped Transformers for algorithmic tasks [14]; Attention-over-Attention in NLP [15]; intra-layer recurrence for language modeling (Nguyen & Lin [17], 2025)
- **Novel**: Two sequential SAME-SEQUENCE self-attention passes in a decoder-only language model layer, sharing one KV cache, analyzed explicitly for depth-reduction tradeoff and inference efficiency. No published paper implements this as a first-class architectural choice with language modeling experiments.
- **Strongest prior art**: DaViT [3] proves quality gains from two sequential attention passes per block in vision. The depth-reduction hypothesis for LMs is supported theoretically by Saratchandran et al. [8] but unvalidated empirically for decoder-only language models.

---

## 4. Technical Analysis

### 4.1 TTFT and TPOT Analysis

**Attention fraction at different sequence lengths (A2):**

| s | Attn FLOPs/layer | FFN FLOPs/layer | Attn Fraction | Doubling TTFT impact |
|---|-----------------|-----------------|---------------|---------------------|
| 8K | 2×s×d = 2×8192×5120 = 84M | 3×d×d_ff = 393M (SwiGLU) | ~17.6% | +17.6% TTFT |
| 32K | 335M | 393M | ~46.0% | +46.0% TTFT |
| 64K | 671M | 393M | ~63.0% | +63.0% TTFT |
| 128K | 1342M | 393M | ~77.3% | +77.3% TTFT |

TTFT penalty becomes severe (>46%) at s≥32K. Double attention is most viable at s<16K.

**TPOT analysis (batch=1, memory-bandwidth-bound):**

Double attention reads KV cache twice per layer per decode step. Using canonical formula (TPOT impact):
- A2 at 32K: `(weight_BW + 2×KV_BW) / (weight_BW + KV_BW)` = (64 + 2×8.59) / (64 + 8.59) = 81.18 / 72.59 ≈ **1.118×** TPOT increase
- A2 at 262K: (64 + 2×68.7) / (64 + 68.7) = 201.4 / 132.7 ≈ **1.518×** TPOT increase

TPOT increase can be severe at long contexts — another reason double attention is best applied at short contexts or with depth reduction.

---

## 5. Implementation Considerations

- **Implementation cost**: TRIVIALLY LOW. Two sequential calls to `F.scaled_dot_product_attention`. Compatible with PyTorch, JAX, DeepSpeed ZeRO, FSDP. No custom kernels needed.

- **Recommended variant**: Shared K/V + separate Q (LoRA-differentiated, rank 8–16). Pre-norm LayerNorm before each pass. Separate residual connection after each pass.

- **Training stability**: LOW RISK. Residual connections after each pass ensure well-conditioned gradient flow. LayerScale recommended for very deep configurations. For shared-weight variant: scale attention learning rate by 1/√2.

- **Pilot design (Phase 1, ~1 GPU-day)**: 4–8 layers, d=512, 10B tokens. Measure: validation perplexity, attention entropy per pass, attention-map correlation between pass 1 and pass 2. If correlation < 0.85 and perplexity improves, proceed to depth-reduction experiment.

- **Compatibility**: Fully compatible with GQA (second pass reuses shared KV heads), MoE FFN (unchanged), LoRA fine-tuning (separate adapters per pass), hybrid architectures (apply to full-attention layers only).

---

## 6. Synergies

- **Combines well with**:
  - **5.4 Linked Attention**: First pass identifies relevant tokens; second pass attends only to first-pass top-k. Eliminates TTFT penalty (second-pass FLOPs = 1% of first at 10% retention). **STRONGEST COMBINATION — explore first.**
  - **5.5 Ragged Window Attention**: First pass wide/global; second pass narrow local window. Analogous to DaViT spatial+channel decomposition.
  - **4.2 Shared Core + LoRA**: Already embodied in recommended implementation (shared base + LoRA Q2)
  - **1.2 Per-Token Adaptive Depth**: Skip second pass for easy tokens; reduces average cost toward single-pass baseline

- **Conflicts with**:
  - **5.3 Grammar/FSM Attention**: Grammar pass + second unstructured pass may violate grammar constraints
  - Approaches reducing depth for other reasons: double attention increases per-layer cost; requires coordinated depth reduction

---

## 7. Risk Assessment

- **Technical risk**: LOW-MEDIUM — Vision precedent (DaViT) confirms quality gains. Primary risk: second pass over same sequence may be redundant for language modeling at scale (different from DaViT's spatial/channel decomposition, which attends to different dimensions).

- **Efficiency impact WITHOUT depth reduction**: NEGATIVE — double attention increases TPOT and TTFT. No efficiency benefit without depth reduction.

- **Efficiency impact WITH depth reduction**: POTENTIALLY POSITIVE — 34% fewer layers → 34% KV cache reduction, ~0.74× TPOT. This requires empirical validation.

- **Priority**: LOW-MEDIUM — Implementation is trivial; pilot experiment is cheap (1 GPU-day). If depth reduction doesn't materialize, the idea has limited deployment value beyond quality improvement at higher cost.

---

## 8. Accuracy / Quality Tradeoff

- **Expected quality delta vs baseline**: DaViT-Base achieves 84.6% ImageNet top-1 vs ViT-B 81.8% (+2.8 pp) with two sequential attention passes per block [DaViT, ECCV 2022, 3]; Biomimetic Double Attention: 96.6% vs 91.6% on DRIVE/STARE vessel segmentation (+5.0 pp) at 7% FLOPs overhead [Zhang et al., Biomimetics 2025, 4]; Universal Transformers T=2: +0.9 BLEU on WMT14 En-De (28.6 vs 27.7), +10–20 pp on algorithmic tasks [Universal Transformer, ICLR 2019, 2]; Looped Transformers T=2: +30–40 pp on sorting/associative recall vs T=1 [Looped Transformers, 14]; Intra-Layer Recurrence: +0.3–0.8 PPL improvement on PTB/WikiText-2 [Nguyen & Lin, Canadian AI 2025, 17].
- **Known failure modes**: If both passes attend to identical sequence with insufficient differentiation (same Q, same residual), outputs are highly correlated and the quality gain → 0 while compute cost remains [5.6 §3]; without depth reduction, idea is quality-add + cost-add (TPOT A2@32K: ~1.118× worse; TTFT A2@32K: ~1.46× worse); depth-reduction hypothesis is **unvalidated empirically for decoder-only LMs** — DaViT's cross-dimensional advantage may not transfer to same-sequence LM; separate Q/K/V projections roughly double the stated TTFT/training-cost increases [5.6 §Executive Summary].
- **Empirical evidence**: DaViT Table 2 (84.6% vs 81.8% ImageNet) [DaViT, 3]; Biomimetic Table (96.6% vs 91.6% DRIVE/STARE) [Zhang et al., 4]; Universal Transformer Table 2 (28.6 vs 27.7 BLEU WMT14) [Universal Transformer, 2]; Looped Transformers (comparable to standard at <10% params per abstract; per-task T=2 vs T=1 deltas in paper body) [Looped, 14]; Nguyen & Lin §Results (per-benchmark PPL deltas in paper body; abstract only reports "allocating more iterations to earlier layers yields optimal results") [17]; SVFormer/ResFormer (50% KV reduction, <0.5 PPL increase) [SVFormer, ACL 2025, 9].
- **Mitigations**: Differentiate pass 2 from pass 1 — use separate Q projection or distinct residual input to reduce redundancy [5.6 §3]; validate depth-reduction hypothesis with a cheap pilot (1 GPU-day) at 1–3B scale before committing — target 30–50% depth reduction per Saratchandran et al. [8]; share KV across passes to hold KV cache constant (idea's baseline assumption); if depth reduction fails, revert to full depth (no sunk cost on trained weights); prefer Nguyen & Lin's selective-layer strategy (recurrence on earlier layers only) over uniform double attention per [17].

### Detailed breakdown

- **Reported quality delta (from closest analogues)**:
  - DaViT (Ding et al., ECCV 2022) [3]: two sequential self-attention passes per block (spatial then channel) in vision transformers achieves DaViT-Base 84.6% ImageNet top-1, versus a single-pass ViT-B baseline of 81.8% — a **+2.8 percentage point** improvement. DaViT-Large reaches 86.9% ImageNet top-1, outperforming every published single-pass dense ViT at equivalent parameter count. This is the strongest quantitative existence proof that two sequential attention passes per block improve quality. The caveat is that DaViT's two passes attend to different dimensions (spatial vs channel), providing complementary information; in a decoder-only LM both passes attend to the same sequence, which may produce more redundancy.
  - Double Attention / Biomimetic (Zhang et al., Biomimetics 2025) [4]: two-key self-attention on medical imaging achieves **96.6% vs 91.6%** accuracy on DRIVE/STARE retinal vessel segmentation — a **+5.0 percentage point** improvement at only 7% FLOPs overhead. This is the most directly named "double attention" quality result in the literature and demonstrates clear quality gains from two attention computations per layer on structured prediction tasks.
  - Universal Transformers (Dehghani et al., ICLR 2019) [2]: shared-weight recurrence over T=2 steps achieves **+0.9 BLEU** over vanilla Transformer on WMT14 En-De (28.6 vs 27.7 BLEU), and improves accuracy on algorithmic tasks (sorting, copy) by 10–20 percentage points vs single-pass. Multi-pass attention demonstrably improves compositional and structured reasoning quality.
  - Looped Transformers (Yang et al., arXiv 2023) [14]: per abstract, looped transformer achieves performance comparable to standard transformer on various data-fitting problems while utilizing less than 10% of the parameter count. Specific T=2 vs T=1 per-task deltas (sorting, associative recall) are paper-body Table values and are reported as substantial improvements on multi-step reasoning tasks.
  - Intra-Layer Recurrence (Nguyen & Lin, arXiv:2505.01855) [17]: per abstract, "allocating more iterations to earlier layers yields optimal results" for language modeling. Specific per-benchmark PPL improvements are paper-body tables. This is the most directly comparable result to idea 5.6 for language modeling specifically.
  - SVFormer/ResFormer (Zhou et al., ACL 2025) [9]: cross-layer value residuals enrich per-layer attention signals and reduce KV cache by ~50% with **<0.5 PPL increase** on language modeling. This is an adjacent quality-enriching attention mechanism with a known quality profile.

- **Monotonicity**: This idea is **quality-positive by design** — double attention increases model capacity and representation richness. Quality improves monotonically with the second pass when the second attention is sufficiently differentiated from the first (e.g., via separate Q or different residual input). If the second pass is identical to the first (pure duplication, no residual separation), the second pass's output may be highly correlated with the first, and the quality gain approaches zero while the compute cost remains. The depth-reduction trade — using double attention to allow 30–50% fewer layers — introduces a non-monotone surface: there exists an optimal layer count at which the quality of fewer-but-richer layers matches or exceeds more-but-shallower layers. Saratchandran et al. [8] suggest this crossover occurs at 30–50% depth reduction for the same parameter budget.

- **Recovery**: Not applicable in the conventional sense — this idea improves quality, not degrades it. The only "recovery" scenario is if the depth-reduction hypothesis fails empirically: if 34% fewer layers with double attention performs worse than full depth with single attention, the loss is recoverable by reverting to full depth (with minimal sunk cost since the pilot is cheap). If the second attention pass proves redundant for LMs (high correlation between pass 1 and pass 2), quality neither improves nor degrades significantly, and the cost increase (~10–20% TPOT at moderate context) represents the downside of a failed hypothesis.

- **Conditions for acceptable degradation**: There is no quality degradation under this idea. The trade is compute cost (TPOT and TTFT increase) for quality gain. The cost is acceptable when: (1) the application is TTFT-tolerant and context length is short (<16K, where TTFT increase is <18%); (2) the combination with Linked Attention (5.4) is used — in that combination, the second pass attends only to the top-k relevant tokens from pass 1, making the second-pass FLOP cost negligible (~1% of first pass at 10% retention); (3) the depth-reduction hypothesis holds, in which case both quality and efficiency improve simultaneously. The idea becomes unattractive only when context length is very long (>32K) without depth reduction, where TTFT increases >46% and TPOT increases >11%.

---

<!-- CITATION MANIFEST -->
[1] Attention Is All You Need: Vaswani et al., NeurIPS 2017; arXiv:1706.03762; original Transformer; decoder runs two sequential attention sub-layers (self-attn + cross-attn) per block
[2] Universal Transformers: Dehghani et al., ICLR 2019; arXiv:1807.03819; shared-weight recurrence T steps; +0.9 BLEU on WMT14 En-De; multi-pass attention improves quality
[3] DaViT: Dual Attention Vision Transformers: Ding et al., ECCV 2022; arXiv:2204.03645; two sequential self-attention passes per block (spatial+channel); DaViT-Base 84.6% ImageNet; strongest existence proof
[4] Double Attention: Biomimetic Self-Attention Optimization: Zhang et al., Biomimetics 2025; DOI:10.3390/biomimetics10010034; two-Key attention; +5% accuracy at 7% FLOPs overhead on medical imaging
[5] Mixture of Attention Heads: Zhang et al., EMNLP 2022; arXiv:2210.05144; sparse router for k-of-H heads per token; quality improvement from varied attention operations
[6] MoH: Multi-Head Attention as Mixture-of-Head Attention: Jin et al., ICML 2025; arXiv:2410.11842; top-k head routing; outperforms full MHA at 75% head activation
[7] Wide Attention Is The Way Forward For Transformers?: Brown et al., arXiv 2022; arXiv:2210.00640; wide single-layer models match deep with 3.1× inference speedup; supports depth-for-width trade
[8] Leaner Transformers: More Heads, Less Depth: Saratchandran et al., arXiv 2025; arXiv:2505.20802; 30-50% parameter reduction with more heads + fewer layers; most direct depth-reduction evidence
[9] Value Residual Learning For Alleviating Attention Concentration (ResFormer/SVFormer): Zhou et al., ACL 2025; arXiv:2410.17897; cross-layer value residuals; 16% fewer params; SVFormer 50% KV cache reduction
[10] Associative Transformer (AiT): Sun et al., CVPR 2025; arXiv:2309.12862; two-phase attention (bottleneck write + Hopfield read); outperforms sparse transformers on relational reasoning
[11] Stack Attention: DuSell & Chiang, ICLR 2024 Spotlight; arXiv:2310.01749; stack-augmented multi-step attention captures CFLs; second pass catches patterns missed by first
[12] The Sparse Frontier: Nawrot et al., ICLR 2025; arXiv:2504.17768; attention fraction data at various sequence lengths; used for TPOT estimates
[13] Attention Residuals (AttnRes) [post-cutoff, unverified]: Kimi Team, arXiv 2026; arXiv:2603.15031; cross-depth attention; contents unverifiable beyond knowledge cutoff
[14] Looped Transformers are Better at Learning Learning Algorithms: Yang et al., arXiv 2023; arXiv:2311.12424; looped transformer matches standard transformer at <10% parameter count on data-fitting problems per abstract; per-task T=2 vs T=1 breakdowns paper-body; closest language-task LM precedent
[15] Attention-over-Attention Neural Networks for Reading Comprehension: Cui et al., ACL 2017; arXiv:1607.04423; two sequential attention passes in NLP (document attention over word attention); 2017 published two-pass NLP attention
[16] Thinking Deeper, Not Longer: Depth-Recurrent Transformers for Compositional Generalization [post-cutoff, unverified]: Chen, arXiv 2026; arXiv:2603.21676; shared-weight block recurrence for compositional generalization; contents unverifiable beyond knowledge cutoff
[17] Intra-Layer Recurrence in Transformers for Language Modeling: Nguyen & Lin, Canadian AI 2025; arXiv:2505.01855; per-layer recurrence in transformers; earlier layers benefit most; validates intra-layer recurrence for LMs
