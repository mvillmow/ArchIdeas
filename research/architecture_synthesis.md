# Architecture Synthesis — 39 Ideas Across 6 Groups

## Executive Summary

This synthesis covers 39 architectural research ideas organized into 6 thematic groups. The corpus spans ideas ranging from immediately deployable inference optimizations (1.3, 5.7, 4.7) to multi-year research programs (6.4, 1.6, 3.6). The 2026-04-24 DeepSeek-V4 release adds a new external evidence point: DeepSeek-V4-Pro is a 1.6T-parameter MoE with 49B active parameters, 1M-token context, CSA/HCA hybrid compressed attention, mHC residual mixing, Muon training, FP4 MoE expert weights, and FP8-dominant inference storage. Across all groups, several systemic patterns emerge:

1. **Memory bandwidth is the primary inference bottleneck** in all dense and MoE baselines at batch=1 decode. The majority of the most impactful ideas address this: weight quantization (5.7), expert routing reduction (1.1, 1.3), KV cache compression (5.1, 5.4, 5.5), and low-rank factorization (2.2, 4.3-C). Ideas that improve FLOPs without reducing memory bandwidth (e.g., 3.8, 4.4 strong form) deliver no TPOT benefit and are deprioritized.

2. **Adaptive computation is the dominant theme** across Groups 1, 3, and 6. The corpus repeatedly arrives at the same architectural principle: not all tokens require the same amount of compute; mechanisms that route compute proportionally to token difficulty are both well-motivated and empirically validated.

3. **Compression and sparsity ideas form a composable stack.** Weight quantization (5.7), block sparsity (5.8), low-rank (2.2), and KV quantization (5.1) are all orthogonal and can be applied simultaneously to the same model. The combined stack is expected to yield ~5–10× weight bandwidth reduction on dense baselines.

4. **The hybrid architecture paradigm (Group 4) is mature but lacks a principled optimization framework.** Two-primitive hybrids (attention + SSM) are production-deployed; the key gap is automated NAS for 4-primitive assignment (1.6) and learned structural decisions (1.4, 4.5) rather than hand-designed rules.

5. **Generation paradigm innovation (Group 6) is the highest-novelty frontier.** BD3-LM (6.3's basis) is ICLR 2025 Oral; the combination 6.4 (prefill-decode split + block-diffusion + learned stopping) is unstudied and constitutes the strongest publication target in the corpus.

**Baseline C specific:** The K2 family (72.55B dense, d=8192) is the largest baseline considered. Its dominant characteristic — large dense weight footprint (~145 GB) — makes weight-compression ideas (2.2, 4.3-C, 5.7) exceptionally impactful at this scale while making MoE routing ideas (1.1, 1.3, 6.5) inapplicable without sparse upcycling.

**DeepSeek-V4 specific:** DeepSeek-V4-Pro should be treated as a new evidence baseline, not substituted for A/B/C. Its most important corpus impact is long-context attention: CSA compresses sequence KV by 4× and then performs sparse block selection, while HCA compresses KV by 128× and performs dense attention over the compressed stream, both with a local sliding-window branch. This validates a training-native version of the 5.3/5.4/5.5 KV-compression cluster at 1M context and narrows novelty for generic "compressed sparse attention" claims. V4 does not implement block diffusion, learned halting, or an architectural prefill/decode split, so 6.4 remains novel only as that combined system.

---

## Thematic Analysis

### Theme 1: Adaptive Computation (covers Groups 1, 3.x)

**Core principle:** The amount of compute a transformer applies should vary based on token or sequence difficulty, not be fixed by the architecture.

**Group 1 — Adaptive Sparsity (Ideas 1.1–1.7):** This group focuses on adapting the compute profile within a fixed layer count. Ideas 1.1 (per-token k) and 1.3 (per-layer k) represent the most direct inference speedup mechanisms in the corpus for MoE models. Together they form a 2D adaptive routing grid — 1.3 sets the per-layer budget, 1.1 varies allocation per token within that budget. Both are independently published at ICLR/ACL/NeurIPS level; both are PURSUE verdicts.

Idea 1.2 (per-token adaptive depth) extends the principle to layer count. The "Diminishing Returns" finding (arXiv:2603.23701) is a critical adverse signal: modern LLMs with better training recipes have reduced layer redundancy, and hybrid/MoE architectures are specifically called out as less suitable for early exit. This degrades 1.2 from PURSUE to INVESTIGATE. The three-zone (prefix/dynamic/postfix) structure remains a genuine novelty.

Ideas 1.4–1.6 operate at the architectural level: which layer type to use (1.6), whether it should be dense or sparse (1.4), and which sparsity modality to apply (1.5). These are training-time decisions with inference-time consequences. All three are INVESTIGATE due to training complexity (2× overhead for safe two-stage pipelines, DARTS collapse risk). The key value proposition is discovering configurations that human ablations would miss.

DeepSeek-V4 changes the evidence base for 1.4 specifically. DeepSeek-V3's 3-dense + 58-MoE dense-first rule was previously the most important published baseline for early-layer stabilization. V4 instead uses MoE in all Transformer blocks and stabilizes the first three MoE layers with Hash routing, which weakens any blanket claim that early dense FFNs are required. The updated question for 1.4 is not simply dense-first vs. MoE-later, but whether learned assignment can outperform "all-MoE plus deterministic early routing" at scale.

Idea 1.7 (dynamic vocabulary) is a modest but practical TPOT improvement (~2–6% for large models, 30–50% for speculative decoding draft models) that requires careful handling of the hard missing-token failure mode.

**Group 3 — Dynamic Routing/State (Ideas 3.1–3.8):** This group explores adaptive compute via routing and internal state. Ideas 3.1 (layer-level MoE — routing between full attention blocks rather than FFN experts) and 3.7/4.1 (learnable state machine — routing based on explicit discrete state) are the most novel ideas in this group. 3.1 enables per-token selection of distinct attention patterns, a qualitative capability gain beyond FFN-only MoE; however, the KV cache inflation (k× per full-block layer) is a fundamental architectural constraint requiring Stage 1 experimental characterization before infrastructure investment.

Ideas 3.4–3.6 form a hierarchy of increasing complexity: recurrent depth (3.4) → gated DAG (3.5) → recursive DAG with persistent state (3.6). Implement in sequence; never implement 3.6 before 3.4 and 3.5 succeed. The validated existence proof (Huginn 3.5B, Ouro 2.6B) makes 3.4 low-risk at the prototype level.

Idea 3.8 (trainable activation) is deprioritized: no inference speedup without deliberate sparsity training, and the windowed variant is confirmed infeasible. Its most practical reframing is as PACT-style per-neuron quantization-aware clipping — a secondary benefit, not a primary mechanism.

**Group 6 — New Proposals (Ideas 6.1–6.5):** These ideas push adaptive compute to the generation level. Idea 6.1 (in-architecture AR loop — learned stopping) is technically sound but overlaps substantially with LoopFormer (arXiv:2602.11451); the incremental novelty is the hybrid model application. Idea 6.2 (prefill/decode split) is the most practically impactful standalone idea in Group 6, with YOCO as closest prior art. Ideas 6.1+6.2+6.3 compose multiplicatively in 6.4 — the highest-novelty combination target. Idea 6.5 (pre-attention expert router) is a zero-cost-to-prototype novelty that could meaningfully change what information drives MoE routing, with confirmed synergies with 3.1, 3.3, and 1.1.

---

### Theme 2: Memory Efficiency (covers Groups 2, 5.x)

**Core principle:** At batch=1 decode, the model is memory-bandwidth-bound. Every idea in this theme reduces bytes loaded per generated token.

**Group 2 — Compression/Dictionary (Ideas 2.1–2.2):** Idea 2.2 (compressed dense layers via SVD/low-rank) is among the highest-impact ideas in the entire corpus for dense baselines. Post-hoc SVD (SVD-LLM/CALDERA) delivers quick wins (hours to apply); training-native low-rank delivers ~3.4–5.0× TPOT improvement for A2/C with <1.1 PPL degradation at 32% parameter count per published results. The learned adaptive sparse residual is a longer-term research direction.

Idea 2.1 (hierarchical frequency dictionary) is essentially adaptive softmax with an open-source reference implementation. Impact is low-to-medium for large primary models (2.4–5.9% TPOT) but HIGH for speculative decoding draft models, where LM head can be 30–52% of total weight.

**Group 5 — Attention/Quant/Compression (Ideas 5.1–5.10):** This group contains the most diverse set of mechanisms but the most immediate deployment candidates.

The KV compression ideas (5.1 TurboQuant, 5.4 linked attention, 5.5 ragged window) form a natural stack for long-context serving. Their value is context-length-dependent: at 32K context, KV bandwidth is modest compared to weight bandwidth; at 128K–262K, KV becomes the dominant bottleneck and these ideas deliver their largest gains. The recommended deployment is 5.4+5.5 (structural KV reduction) followed by 5.1 (quantize the retained KV entries) — a two-stage compression pipeline.

DeepSeek-V4 provides the strongest new validation for this theme. Its CSA/HCA stack is not a post-hoc KV eviction method; it is a trained attention architecture that combines compressed KV entries, sparse retrieval over compressed blocks, dense attention over heavily compressed blocks, FP4 indexer QK computation, mixed FP8/BF16 KV storage, and a sliding-window branch for local detail. In DeepSeek's reported 1M-token setting, V4-Pro uses 27% of DeepSeek-V3.2's single-token inference FLOPs and 10% of its KV cache, while V4-Flash uses 10% FLOPs and 7% KV. Future updates should benchmark 5.3/5.4/5.5 proposals against CSA/HCA, not against full-attention GQA alone.

The weight compression ideas (5.7 block compressed, 5.8 block sparse, 5.9 dynamic numeric type, 5.10 block dynamic compression) address the weight bandwidth bottleneck directly. Idea 5.7 Tier 1 (INT4 quantization) is the single most immediately deployable idea in the corpus with the largest short-term TPOT impact (2.4–3.5×). Ideas 5.8–5.10 add research novelty over the established INT4 baseline at the cost of higher engineering complexity and unvalidated quality recovery at 32B scale.

DeepSeek-V4 strengthens the low-precision direction but with a different deployment point: routed MoE expert weights use FP4 in instruct checkpoints, while most non-expert parameters use FP8. The model also applies FP4 QAT to the CSA indexer QK path. This supports the 5.7/5.10 claim that 4-bit formats are viable at frontier scale, but the concrete evidence is for MXFP4-style expert-weight and indexer-path quantization rather than arbitrary per-element dynamic type flags.

Ideas 5.2 (LSTM-gated attention), 5.3 (grammar attention), and 5.6 (double attention) are architecture-level ideas that change how attention computes, rather than compressing existing attention. 5.2 has the strongest TPOT case at long contexts (by eliminating the KV cache entirely for recurrent layers); 5.3 reduces to NSA at the practical limit; 5.6 offers quality improvement at higher compute cost unless combined with depth reduction.

---

### Theme 3: Hybrid Architectures (covers Groups 4.x)

**Core principle:** No single computational primitive (full attention, SSM, linear attention, MoE FFN) is optimal at all layer positions, sequence lengths, and tasks. The corpus explores several dimensions of this hypothesis.

Ideas 4.2 (shared core + per-layer LoRA) and 4.3 (LoRA everywhere) address the parameter efficiency dimension: instead of L independent weight matrices, use one shared matrix differentiated by small adapters. The inference benefit (4.3-C: ~2× weight memory reduction, ~1.64× TPOT) is compelling if quality can be maintained at 27B+ scale — which remains unvalidated beyond 7B (CoLA, Liu et al. 2025 provides the strongest evidence). The training-time optimizer memory reduction (~50%) is a free benefit for any variant.

Idea 4.4 (skip-list layers) is split: the strong form is fundamentally infeasible; the hybrid form (Hyper-Connections initialization) is worth a brief ablation for potential +1–6 benchmark point quality improvement with zero inference speedup. Ideas 4.5 (learned residual flow) and 4.6 (mixture of models) both explore structural routing at the layer or model level — 4.5 is more tractable (train-time layer gating for 25% TPOT reduction) while 4.6 is a multi-year engineering commitment with high routing collapse risk.

DeepSeek-V4 upgrades the residual-enrichment evidence. Its Manifold-Constrained Hyper-Connections (mHC) extend Hyper-Connections by constraining residual mixing matrices to the Birkhoff polytope via Sinkhorn normalization, with input/output mappings bounded by sigmoid gates. This does not rescue the strong skip-list replacement form, but it makes the hybrid residual-enrichment path more than a brief ablation: it is now a serious scale-tested stabilization and capacity mechanism.

Ideas 4.7 (compressed dictionary) is a practical win with clear deployment path: implement immediately (1–2 days) for the static-frequency variant; scale to context-conditioned for speculative decoding draft models.

---

### Theme 4: New Training Paradigms (covers Group 6.x)

**Core principle:** The standard autoregressive (one-token-per-forward-pass, all-layers-every-token) inference paradigm has structural inefficiencies that architectural changes can address.

Idea 6.2 (prefill/decode split) is the most practically grounded departure from standard AR: separate layers optimized for prefill FLOPs vs. decode bandwidth, connected by a compressed state (DeltaNet O(d²) recurrent state or compressed KV). This is adjacent to serving-level disaggregation (Splitwise, DistServe, Mooncake) but operates at the model-architecture level, enabling regime-specific optimization without separate model checkpoints.

Idea 6.3 (block diffusion AR decoder) is fully published as BD3-LM (ICLR 2025 Oral); its standalone deployment has no novelty. Its value in this corpus is as a component of 6.4.

Idea 6.4 (combined AR loop + split + diffusion) is the most ambitious combination target in the corpus: 4× decode FLOP reduction (bandwidth-bound) from the multiplicative composition of 6.2 (2× from layer reduction) and 6.3 (2× from tokens-per-pass). No published work achieves this combination. The three components (BD3-LM, YOCO-style split, LoopFormer-style stopping) have solid ICLR/NeurIPS/OSDI prior art separately, making a well-engineered integration publishable as an engineering contribution even without per-component novelty.

DeepSeek-V4 narrows the novelty language around "efficient 1M context" and "long-horizon agents": those are now demonstrated in an open MoE family through CSA/HCA plus agent-specific post-training. The remaining 6.4 contribution should be framed as a generation-mechanism integration — prefill/decode architectural split, block-diffusion decoding, and learned stopping — not as the first efficient long-context AR model.

---

## Cross-Theme Interactions

The strongest cross-theme interactions identified across all 39 research documents:

1. **Adaptive computation × Memory efficiency (1.1+1.3 ↔ 5.7+2.2):** Weight compression (5.7, 2.2) reduces bytes-per-expert-load; adaptive routing (1.1, 1.3) reduces number of experts loaded. These are fully orthogonal and multiplicatively beneficial. A model with INT4 expert weights and average k̄=7 (from k=11) achieves approximately 1.6× (quantization) × 1.57× (k reduction) = ~2.5× combined MoE FFN bandwidth reduction.

2. **KV compression stack (5.1+5.4+5.5):** Three independently validated KV compression mechanisms that compose multiplicatively: structural reduction (5.4: 5–10× capacity), sparse windows (5.5: bandwidth reduction ∝ w_avg/s), and quantization (5.1: 4× for INT4). Combined 64× KV memory reduction at w_avg=2K, INT4 is achievable. This is the primary mechanism for extending context to 262K+ without proportional memory growth.

3. **Recursive depth × Shared weights (3.4+4.2):** The combination of a shared middle block (4.2) applied T times (3.4) yields multiplicative bandwidth savings: T iterations × (1/rank_factor) bytes per iteration. With T=4, r=256 (rank factor ~16), the effective bandwidth cost of the loop body is ~64× lower than a full L-layer forward pass. This combination has strong theoretical grounding and multiple published implementations.

4. **FSM routing × Attention compression (3.7+5.3):** The FSM state (3.7) naturally structures what attention patterns are needed; grammar-constrained attention (5.3) implements these constraints. Both ideas target the same phenomenology (structured token dependencies) from different angles; joint design should yield a cleaner implementation than either alone.

5. **Generation paradigm × Recurrent state (6.2+3.4):** The prefill/decode split (6.2) requires a compact interface between the prefill and decode sub-networks. Idea 3.4's recurrent state (O(d²) for DeltaNet) is the natural compressed prefill output — it summarizes the entire input context in a fixed-size matrix, independent of input length. This is a clean technical composition with no conflicting mechanisms.

6. **CSA/HCA × KV compression stack (DeepSeek-V4 ↔ 5.1/5.3/5.4/5.5):** DeepSeek-V4 combines the core ingredients that were previously separate in this corpus: compressed KV entries, learned/sparse block retrieval, dense attention over a heavily compressed stream, sliding-window locality, and low-precision indexer/KV storage. Treat CSA/HCA as the new reference design for training-native long-context attention.

7. **mHC × residual-flow research (DeepSeek-V4 ↔ 4.4/4.5):** mHC validates learned residual mixing at scale while preserving a stable residual highway. It supports residual enrichment and learned flow, but it conflicts with any proposal that removes per-layer residual paths without a constrained replacement.

---

## Baseline C (K2 Family) Impact Analysis

Baseline C (K2 family, 72.55B dense, d=8192, d_ff=28672, KV d_kv=128, 80 layers, ~145 GB BF16 weight BW, ~10 GiB KV at 32K context) has a substantially different optimization profile from the other baselines.

**How a 72.55B dense model changes which ideas are most impactful:**

The primary characteristic of Baseline C is its dense weight footprint. At batch=1 decode, the model loads ~145 GB of weights per token — approximately 2.3× more than A2 (~64 GB). This makes weight bandwidth the dominant bottleneck at nearly all context lengths, not KV cache bandwidth (which only equals weight bandwidth at ~180K context for Baseline C vs. ~50K for A2).

**Baseline C has larger d (8192 vs 5120 for A2), so ideas that scale with d are more impactful:**

- **LoRA rank ideas (4.2, 4.3-C):** Weight matrices scale as d² or d×d_ff. At d=8192, the low-rank approximation at fixed rank r achieves proportionally more compression: compression ratio d²/(r×2d) = d/(2r). For r=64, this is 8192/128 = 64× per matrix (vs 5120/128 = 40× for A2). The full-model storage reduction for 4.2 at r=64 is approximately 27× for Baseline C vs. 17× for A2.

- **2.2 (Compressed Dense Layers):** At d_ff=28672, the MLP weight matrices are substantially larger (~112 GB for the MLP layers alone at BF16). At r=256, compression is ~96% — nearly all MLP weight bandwidth is eliminated. The practical net TPOT speedup (accounting for attention bandwidth) is ~3.5–5.0× — higher than any other baseline.

**Baseline C's larger weight BW (~145 GB) makes weight-compression ideas (5.7, 5.8, 5.9, 5.10) more attractive:**

The absolute benefit of INT4 quantization (5.7) scales linearly with weight size. For Baseline C, INT4 reduces the ~145 GB weight BW to ~36 GB — a savings of ~109 GB per token. This is the single largest absolute bandwidth reduction of any idea in the corpus, delivered by a production-ready mechanism with native vLLM/TRT-LLM support.

Block sparsity (5.8) at 50% BSR on Baseline C reduces weight BW to ~73 GB — comparable to A2's full BF16 weight BW. Combined with INT4 (5.7), Baseline C could in principle achieve ~18 GB weight BW per token — a ~8× reduction from BF16 baseline.

**MoE routing ideas become inapplicable to Baseline C:**

Ideas 1.1 (learnable top-k), 1.3 (per-layer adaptive expert count), 3.1 (layer-level MoE), 3.2 (swappable experts), 3.3 (dynamic expert router), and 6.5 (pre-attention expert router) all require a MoE architecture as a prerequisite. Applying these to Baseline C requires sparse upcycling first (~50% of pretraining compute per Komatsuzaki et al. 2023). The tradeoff: upcycled Baseline C + 1.1 vs. 2.2 applied to Baseline C is approximately (post-upcycling quality penalty) + (1.57× MoE routing speedup) vs. (3.5–5.0× dense weight compression speedup). For most practical purposes, weight compression wins on Baseline C without upcycling.

**Recommended Baseline C optimization stack (ordered by implementation priority):**
1. INT4 quantization (5.7 Tier 1) — immediate, 2.4–3.5× TPOT
2. TurboQuant (5.1) at long context — immediate, ~1.33× TPOT at 262K
3. Low-rank MLP compression (2.2, post-hoc SVD) — 1 day, ~93–96% MLP BW reduction
4. 4.3-C (pure A·B inference) training — medium-term, ~2× weight memory, ~1.64× TPOT
5. 5.4+5.5 KV compression stack — medium-term, enables 262K+ serving
6. 6.4 (combined AR+split+diffusion) — long-term, largest architectural novelty for dense model class

---

## Systemic Risks

**Risk 1: Training instability cascades.** Multiple ideas in Groups 1, 3, and 4 require training-time architectural changes (gates, discretization, Gumbel-Softmax). Each introduces a potential instability source. When combined, instabilities can interact non-linearly. The recommended mitigation: implement and validate ideas individually at 1–3B scale before combining.

**Risk 2: Hardware efficiency for irregular dispatch.** Ideas 1.1 (variable-k), 3.5 (DAG gating), 3.1 (full-block MoE), and 3.7 (hard FSM routing) all generate irregular per-token computation graphs that conflict with standard GPU batching. Naive implementations recover no hardware speedup despite correct algorithmic behavior. Custom Triton/CUTLASS kernels are required and represent 2–8 weeks of engineering work each.

**Risk 3: Quality cliffs at scale.** Several ideas show quality degradation that is model-size-dependent: ideas that work well at 1–3B (3.7, 4.2, 4.3) may fail catastrophically at 27B+ due to architectural sensitivity, gradient interference, or training dynamics that emerge at scale. The "Diminishing Returns" paper (arXiv:2603.23701) is an exemplar of this risk for early-exit ideas.

**Risk 4: Novelty erosion.** The research corpus moves fast. Several ideas that were novel at time of initial documentation have been partially or fully published by the merge date (6.3/BD3-LM, 1.2/LoopFormer, 3.4/AdaPonderLM). DeepSeek-V4 further erodes novelty for generic long-context KV compression, sparse compressed attention, Hyper-Connections-style residual enrichment, and 4-bit expert-weight QAT at frontier scale. Continuous literature monitoring is required; 6.5 and the 6.4 integration remain the lowest-overlap novelty targets as of 2026-04-24.

**Risk 5: KV cache assumptions for hybrid models.** Ideas in Group 5 (5.4, 5.5) that target KV cache compression are inapplicable to DeltaNet and RWKV/Mamba layers (which have no KV cache). For hybrid baselines A1 and B, only the fraction of layers with full attention (16/64 for A1) benefit from these ideas. This limits their TPOT impact on hybrid models to ~25% of the full-model improvement rate.

---

## Recommended Research Agenda

**Quarter 1 (0–3 months): Validate high-confidence inference optimizations**
- Deploy 1.3 (LExI static) + 5.7 (INT4/FP4 where hardware supports it) + 4.7 (LM head) + 5.1 (TurboQuant) on all baselines
- Run 6.5 V1 A/B test on Baseline B (<1 GPU-day)
- Rebaseline the 5.3/5.4/5.5 KV stack against DeepSeek-V4-style CSA/HCA at 262K–1M context, rather than against dense GQA alone
- Benchmark 1.1 (learnable top-k) training on Baseline B with AdaMoE/ReMoE recipe; compare against DeepSeek-V4's fixed 6-routed-expert + 1 shared-expert routing design

**Quarter 2 (3–6 months): Architecture research at 1–3B scale**
- Train 4.3-C (pure A·B inference) at 1B–7B with non-uniform rank allocation
- Run 3.4 (recursive internal state) prototype at 3.5B scale following Huginn recipe
- Run 4.5 (learned residual flow) ablation: compare gate variants vs. ShortGPT post-hoc pruning
- Start 3.7+4.1 (joint FSM) soft prototype at 1–3B scale

**Quarter 3–4 (6–12 months): Scaling and novel architecture**
- Scale best 4.3-C result to A2/C (27B–72B) if 7B validation succeeds
- Initiate 2.2 (training-native low-rank) on A2/C — highest TPOT potential
- Begin 6.4 (combined AR+split+diffusion) custom training from scratch at 3B scale
- Characterize 3.1 (layer-level MoE) KV semantics at <1B scale (Stage 1 experiment)

**Year 2 (12–24 months): Long-horizon research**
- Scale 6.4 to full A1/A2 size if 3B validation succeeds
- Run 1.6 (4-primitive NAS) at 7B scale using best 1B proxy result
- Evaluate 3.6 (recursive DAG) only if 3.4 and 3.5 both succeed at 3.5B scale
- Consider 4.6 (mixture of models) full system only if 5-step ablation sequence validates routing signal
