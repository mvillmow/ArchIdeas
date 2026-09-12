# Research: Recursive Internal State (Internal CoT)
## ID: 3.4

## 1. Idea Description

Internal chain-of-thought on the latent state — a while-loop within the forward pass that iteratively refines activations. Has a **fixed maximum iteration count (T_max)** and a **learned early-exit condition**. The model does "thinking" in latent space without emitting tokens.

The architecture is structured as three segments:
- **Prelude**: a fixed prefix of L_prefix layers that embed the input into latent space
- **Recurrent middle block**: L_mid shared-weight layers applied iteratively up to T_max times, with a per-token learned halting gate that can exit early
- **Coda**: a fixed suffix of L_suffix layers that decode the final latent state to token logits

The inferred intent is nuanced: the idea does NOT reduce FLOPs or memory bandwidth. Instead, it trades increased compute per token (T iterations of the middle block) for improved reasoning quality per token, potentially allowing the model to generate fewer total tokens (shorter CoT traces or more direct answers), thereby reducing wall-clock latency on reasoning tasks that would otherwise require long verbose outputs. The value proposition is **quality-per-compute-dollar**, not raw TTFT/TPOT reduction.

---

## 2. Executive Summary

**Novelty verdict:** EXISTS — all three defining components (T_max cap, learned per-token early exit, latent-only iteration) are simultaneously present at LLM scale in published work; remaining research gaps are narrow (hybrid linear+full-attention recurrent block, RLTT retrofit, per-iteration MoE) ([Universal Transformers, 2019], [Huginn-3.5B, 2025], [Ouro-2.6B, 2025], [AdaPonderLM, 2025]).

Recursive internal state (idea 3.4) is a well-validated EXISTS mechanism at LLM scale. The core prior art (Universal Transformers ICLR 2019, Huginn-3.5B at 795B tokens, Ouro-2.6B at 7.7T tokens, AdaPonderLM at 2.8B parameters) covers all three defining components simultaneously: T_max cap, learned per-token early-exit, and latent-only operation.

Worked examples: TTFT at T=4, f_mid=0.5 → **↑2.5×**; TPOT at T̄=2, f_mid=0.5 → **↑1.50×**. Both follow the full multiplier 1+(T−1)·f_mid rather than the additive increment (T−1)·f_mid alone.


---

## 3. Literature Review

### Universal Transformers
Dehghani, Gouws, Vinyals, Uszkoreit, Kaiser — ICLR 2019, arXiv:1807.03819[1]

Applies shared-weight transformer blocks recurrently in depth with per-position ACT halting. Achieves +0.9 BLEU on WMT14 En-De (Table 7, §3.6). All three defining components of idea 3.4 are simultaneously present: (a) T_max cap, (b) per-token learned early-exit via ACT halting probability accumulation, (c) operation in latent space without emitting tokens. Predates idea 3.4 by 5+ years.

**[Dehghani et al., 2019]** — §2.1 "The Universal Transformer", §2.2 "Dynamic Halting", §3.6 "Machine Translation", Table 7.

### Adaptive Computation Time for Recurrent Neural Networks
Graves — arXiv:1603.08983, 2016[2]

Introduces ACT: a learned scalar halting probability is accumulated at each step. When sum exceeds 1, computation halts with outputs as the weighted average of states. A ponder cost regularizes against unnecessary computation. Foundational halting mechanism for all descendant recurrent-depth work.

**[Graves, 2016]** — §2 "Adaptive Computation Time".

### Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach (Huginn)
Geiping, McLeish, Jain, Kirchenbauer, Singh, Bartoldson, Kailkhura, Bhatele, Goldstein — arXiv:2502.05171, 2025[3]

Introduces Huginn-3.5B with a **(2,4,2) prelude/recurrent/coda** structure (h=5280, ~3.5B total, ~1.5B recurrent core). Trained on 795B tokens. At 32 recurrent iterations the model matches reasoning performance of models up to 50B parameters. Supports early exit via KL-divergence or acceleration-based halting. Training used T̄=32 with truncated BPTT at k=8 — meaning effective backward-pass training cost for the middle block scales with k=8, giving approximately (1+(k−1)·f_mid) ≈ 4.5× at k=8, f_mid=0.5, not 16× as a naive forward-only estimate would suggest.

**[Geiping et al., 2025]** — §3 "A scalable recurrent architecture" (2,4,2) configuration, §4 "Training", §5 "Benchmark Results" (50B-class matching claim, Table N not independently verified — requires specific table annotation before publication).

### Reasoning with Latent Thoughts: On the Power of Looped Transformers
Saunshi, Dikkala, Li, Kumar, Reddi — ICLR 2025, arXiv:2502.17416[4]

**Theorem 5.2** (§5.3): looped transformers can simulate any same-depth non-looped model. Table 1: 1-layer looped 12× achieves 100.0% on addition at n=8 operands. Table 2: 1-layer looped 8× reaches 73.2% on i-GSM math problems. Proves that looped models can simulate CoT reasoning (§5.4). Theoretical justification for why iterative latent refinement improves reasoning quality.

**[Saunshi et al., 2025]** — §5.3 "Looped models can simulate non-looped models", Theorem 5.2; Table 1 (addition); Table 2 (i-GSM).

### Scaling Latent Reasoning via Looped Language Models (Ouro)
Zhu, Wang, Hua, Zhang et al. (33 authors incl. Bengio, Eshraghian) — arXiv:2510.25741, 2025[5]

Ouro-2.6B pretrained on 7.7T tokens with entropy-regularized depth allocation. Table 8: Ouro-2.6B-Thinking-R4 achieves 90.85% MATH500 vs Gemma3-12B's 83.20%. The advantage stems from "superior knowledge manipulation capabilities," not increased knowledge capacity.

**[Zhu et al., 2025]** — §3 "Learning Adaptive Latent Reasoning with LoopLM", §4 "Training", §5 "Experiments", Table 8.

### Training Large Language Models to Reason in a Continuous Latent Space (COCONUT)
Hao, Sukhbaatar, Su, Li, Hu, Weston, Tian — COLM 2025, arXiv:2412.06769[6]

COCONUT feeds the last hidden state back as the next input embedding — **horizontal recurrence** (each continuous thought occupies a new sequence position), not **vertical recurrence** (in-place depth iteration). Outperforms CoT on logical reasoning tasks requiring BFS-like search. Complementary, not identical, to idea 3.4.

**[Hao et al., 2024]** — Abstract: "outperforms CoT on logical reasoning tasks that require substantial search during planning." COLM 2025.

### Mixture-of-Recursions (MoR)
Bae, Kim, Bayat, Kim, Ha, Schuster, Fisch, Harutyunyan, Ji, Courville, Yun — NeurIPS 2025, arXiv:2507.10524[7]

Expert-choice and token-choice routing assigning different recursion depths per token. Figure 4(a): 2.06× inference throughput over vanilla transformers at similar accuracy (§3.3). §3.2 IsoFLOP: 19% training time reduction, 25% peak memory reduction. MoR's routing is purely learned, matching the "learned early-exit condition" specification.

**[Bae et al., 2025]** — §2.2 "Mixture-of-Recursions", §2.2.1 "Routing Strategies", §2.2.2 "KV Caching Strategies", §3.2, §3.3, Figure 4(a).

### AdaPonderLM: Gated Pondering Language Models with Token-Wise Adaptive Depth
Song, Li, Wang, Zeng, Song, Wang, Xu, He, Lin — arXiv:2603.01914, 2026[8]

Self-supervised recurrent LM with iteration-specific MLP gates and a **monotonic halting mask** (once halted, always halted). A **KV reuse mechanism** (Algorithm 2, §3.2) avoids recomputing KV for halted tokens. Table 3 (§4.3): 2.8B zero-shot avg 59.6% (+2.2% over Pythia baseline), five-shot 61.1% (+3.5%). Closest paper to the exact specification of idea 3.4.

**[Song et al., 2026]** — §3.1 "MLP-based Gate", §3.2 "KV Alignment" (Algorithm 2), Table 1 (perplexity), Table 3 (downstream tasks).

### PonderNet: Learning to Ponder
Banino, Balaguer, Blundell — ICML Workshop 2021, arXiv:2107.05407[9]

Reformulates halting as a Bernoulli halt variable, providing lower-variance gradient estimates compared to ACT. Alternative probabilistic halting formulation that could replace ACT in idea 3.4.

**[Banino et al., 2021]** — Abstract: "PonderNet dramatically improves performance over previous adaptive computation methods." ICML Workshop 2021.

### Two-Scale Latent Dynamics for Recurrent-Depth Transformers
Pappone, Crisostomi, Rodolà — NeurIPS 2025, arXiv:2509.23314[10]

Acceleration-based early-exit criterion: exit when `a⁽ᵏ⁾ = ||Δ⁽ᵏ⁾ − Δ⁽ᵏ⁻¹⁾||₂ < τ`. Figure 4(i) (§4 "Empirical comparison"): latency ~580ms/token → ~360ms/token (38% reduction). **Note: tested on GPT-2-scale model (12 layers, h=768, 300M tokens pretraining) — not LLM scale.**

**[Pappone et al., 2025]** — §3 "Empirical observations", §4 "Geometry-inspired early exit", §4 "Empirical comparison", Figure 4(i). NeurIPS 2025.

### Latent Chain-of-Thought? Decoding the Depth-Recurrent Transformer
Lu, Yang, Lee, Li, Liu — COLM 2025 Workshop, arXiv:2507.02199[11]

Critical negative result. Table 1: Huginn-3.5B GSM8K 3.11% (r=4) → 4.93% (r=32) — both well below explicit CoT (24.87%). Marginal gains without explicit latent reasoning supervision. This confirms that depth recurrence alone without RLTT-style trajectory reward is insufficient.

**[Lu et al., 2025]** — §2 "Method" (logit lens, coda lens probing), §3 "Experiment", Table 1 (GSM8K accuracy at r=4/r=32/CoT).

### Prioritize the Process, Not Just the Outcome: Rewarding Latent Thought Trajectories Improves Reasoning in Looped Language Models (RLTT)
Williams, Tureci — arXiv:2602.10520, 2026[12]

**Unreviewed two-author arXiv preprint.** RLTT distributes reward across the full latent reasoning trajectory rather than only the final output. Applied to Ouro-2.6B-Thinking, Table 2 (§5 "Experiments"): **+14.4% MATH-500, +16.6% AIME24, +10.0% BeyondAIME** vs GRPO baseline [UNREVIEWED PREPRINT — results carry lower evidentiary weight]. Note: gains are over GRPO, not over chain-of-thought.

**[Williams and Tureci, 2026]** — §3 "RLTT: Reward Latent Thought Trajectories", §5 "Experiments", Table 2. arXiv:2602.10520. Full title: "Prioritize the Process, Not Just the Outcome: Rewarding Latent Thought Trajectories Improves Reasoning in Looped Language Models".

### Teaching Pretrained Language Models to Think Deeper with Retrofitted Recurrence
McLeish, Li, Kirchenbauer, Singh Kalra, Bartoldson, Kailkhura, Schwarzschild, Geiping, Goldstein, Goldblum — arXiv:2511.07384, 2025[13]

Studies converting existing pretrained non-recurrent LMs into depth-recurrent variants via a curriculum of increasing recurrences. Abstract: "better performance at a given compute budget than simply post-training the original non-recurrent language model" on mathematics. Demonstrates that recurrent-depth capability can be retrofitted onto existing checkpoints (practical path for applying to Qwen3.5-27B without full retraining from scratch).

**[McLeish et al., 2025]** — Abstract. arXiv:2511.07384. November 2025.

### Loop, Think, & Generalize: Implicit Reasoning in Recurrent-Depth Transformers
Kohli, Parthasarathy, Sun, Yao — arXiv:2604.07822, 2026[14]

Identifies a "three-stage grokking process" (memorization → generalization) and depth extrapolation: models trained to 5-hop reasoning generalize to 10-hop via increased recurrence. Also identifies **"overthinking" pathology** where excessive recurrence degrades predictions — directly motivates the T_max cap in idea 3.4. Under review as of April 2026.

**[Kohli et al., 2026]** — Abstract and §1 (grokking process, depth extrapolation). §Results (overthinking). arXiv:2604.07822. Under review.

### SpiralFormer: Looped Transformers Can Learn Hierarchical Dependencies via Multi-Resolution Recursion
Yu, Shu, Wang, Zhang, Wu (Haoyi), Wu (You), Long, Chen, Xu, Su, Zheng — arXiv:2602.11698, 2026[15]

Multi-resolution recursion: each iteration processes a downsampled sequence before upscaling back to token resolution. At 1.4B scale: perplexity 7.14 vs 7.44 (LoopedFormer), 5-shot accuracy 54.37% vs 51.93%, FLOPs reduction 7–11%.

**[Yu et al., 2026]** — §Method "Multi-resolution recursion", §Experiments Table 1 (perplexity, accuracy), §Experiments "FLOPs reduction". arXiv:2602.11698.

### Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA
Bae, Fisch, Harutyunyan, Ji, Kim (Seungyeon), Schuster — ICLR 2025, arXiv:2410.20672[16]

Converts existing pretrained LLMs into recursive models by tying layer weights across loop iterations, then relaxing weight tying with iteration-specific LoRA modules (initialized via truncated SVD on residual matrices). Recursive Gemma 1B recovers most performance of original Gemma 2B. Continuous Depth-wise Batching with early exit yields potential 2–3× throughput. Directly informs the LoRA-based relaxation path for idea 4.2 synergy (§7).

**[Bae et al., 2025b]** — §3.1 "Relaxed Recursive Transformers", §4 "Continuous Depth-wise Batching", §5 "Experiments". ICLR 2025. arXiv:2410.20672.

### MeSH: Memory-as-State-Highways for Recursive Transformers
Yu, Shu, Wang, Zhang, Wu (Haoyi), Li (Jiaang), Long, Chen, Xu, Su, Zheng — arXiv:2510.07739, 2025[17]

Identifies two failure modes in recursive transformers: (a) undifferentiated computation — the shared block repeats similar patterns at each iteration; (b) information overload — persistent and temporary information share a single hidden state. MeSH introduces an external memory buffer with lightweight iteration-level routers, separating persistent state from transient activations. At 1.4B scale: +1.06% average downstream accuracy with 33% fewer non-embedding parameters vs same-depth non-recursive baseline. Directly addresses the "persistent internal state" framing of idea 3.4.

**[Yu et al., 2025]** — §2 "Method", §3 "Experiments", Table 2 (downstream accuracy). arXiv:2510.07739.

---

## 4. Prior Art Classification

**Status: EXISTS**

All three defining components — (a) T_max cap, (b) learned per-token early-exit gate, (c) latent-only operation without token emission — are **simultaneously present** in at least three published works: Universal Transformers (ICLR 2019)[1], Huginn-3.5B[3], and AdaPonderLM[8]. The idea is EXISTS in the most comprehensive sense.

**What is NOT covered (genuine research targets):**
- Applying recurrent-depth middle blocks to an existing **hybrid linear+full-attention stack** (Gated DeltaNet linear attention in the recurrent block): no paper has demonstrated this combination
- **Post-hoc retrofitting with RLTT-style trajectory reward** on a hybrid architecture
- **Per-iteration MoE routing within a depth-recurrent block**
- **Separation of persistent vs. transient hidden state across iterations** is partially addressed by MeSH [17] but not at LLM scale or in a hybrid-attention context

---

## 5. Technical Analysis

### 5.1 Theoretical Complexity

**Variable definitions:**
- L_prefix = prelude layers (unique); L_mid = shared-weight recurrent layers; L_suffix = coda layers
- T = actual iterations (1 ≤ T ≤ T_max); T̄ = mean after early-exit
- L_total = L_prefix + L_mid + L_suffix (total unique layers; weight memory unchanged)
- f_mid = L_mid / L_total (fraction of compute in recurrent block at T=1)

**NOTE — TTFT AND TPOT BOTH INCREASE WITH T. THE MULTIPLIER IS 1+(T−1)·f_mid, NOT (T−1)·f_mid.**

- TTFT increase factor ≈ **(1 + (T−1)·f_mid)** vs same-depth non-recurrent baseline. At T=4, f_mid=0.5: TTFT **↑2.5×**.
- TPOT increase factor ≈ **(1 + (T̄−1)·f_mid)** where T̄ is average iterations after early exit. At T̄=2, f_mid=0.5: TPOT **↑1.50×**. Without early exit, T=32 (as in Huginn training): TPOT **↑16×** for middle-block fraction.

### 5.2 Key Comparison Tables

**vs. Baseline A1 (Qwen3.5-27B Hybrid)**

| Metric | Baseline A1 | Idea 3.4 | Change | Notes |
|--------|------------|----------|--------|-------|
| Compute (FLOPs/token) | O(L·(d²+s·d/4)) | O((L_pre+T·L_mid+L_suf)·(s·d+d·d_ff)) | ↑ (T−1)·f_mid+1 = **2.5×** at T=4,f_mid=0.5 | Formula assumes shared middle block applied T times |
| KV cache (262K ctx) | ~17.2 GB | = ref (no cross-iteration KV accumulation) | = | KV not accumulated across iterations |
| Weight memory | O(L·d·d_ff) | O(L·d·d_ff) — unchanged (weight sharing) | = | L_total unique layers regardless of T |
| TTFT | ref | ↑ ~(1+(T−1)·f_mid) = **2.5×** at T=4, f_mid=0.5 | ↑ | Cost increase; early exit does not help TTFT |
| TPOT | ref | ↑ ~(1+(T̄−1)·f_mid) = **1.50×** at T̄=2, f_mid=0.5 (worst-case T=T_max; T̄=2 with early exit) | ↑ | Early exit helps TPOT; ~38% reduction achievable via acceleration threshold [Pappone et al., 2025] |
| Training cost | ref | ↑ ~2.5× at T_train=4, f_mid=0.5 | ↑ | With truncated BPTT k=8 and T_max=32: effective ~4.5× (not 16×) |
| Quality (reasoning) | ref | ↑ significant (RLTT +14.4% MATH-500) | ↑ | Requires RLTT-style training; depth recurrence alone insufficient |

**Note on A1 TTFT formula scope:** For A1's hybrid baseline at long contexts, if the recurrent block uses full attention replacing Gated DeltaNet linear-attention layers, TTFT increase exceeds 1+(T−1)·f_mid prediction at large s. The formula understates TTFT overhead for A1 hybrid when s is large.

**vs. Baseline A2 (Qwen3-32B Dense)**

| Metric | Baseline A2 | Idea 3.4 | Change | Notes |
|--------|------------|----------|--------|-------|
| Compute (FLOPs/token) | O(L·(s·d+d·d_ff)) | O((L_pre+T·L_mid+L_suf)·(s·d+d·d_ff)) | ↑ **2.5×** at T=4, f_mid=0.5 | Full multiplier 1+(T−1)·f_mid |
| KV cache (40K ctx) | ~10.74 GB | = ref | = | |
| TTFT | ref | ↑ **2.5×** at T=4, f_mid=0.5 | ↑ | Early exit does not help TTFT |
| TPOT | ref | ↑ **1.50×** at T̄=2, f_mid=0.5 (worst-case T=T_max; T̄=2 with early exit) | ↑ | Early exit reduces T̄ toward 1 |
| Training cost | ref | ↑ ~2.5× at T_train=4, f_mid=0.5 | ↑ | Same full multiplier applies to training forward pass |

**vs. Baseline B (Qwen3.5-397B-A17B MoE)**

| Metric | Baseline B | Idea 3.4 | Change | Notes |
|--------|-----------|----------|--------|-------|
| Compute (FLOPs/token) | O(L·(d²+k·d·d_e)) | Context-dependent | ↑ likely | Depends on middle-block attention type; MoE interaction unknown |
| KV cache (262K ctx) | ~8.0 GB | Context-dependent | Context-dependent | Hybrid attention layers complicate comparison |
| Weight memory | O(L·E·d·d_e) | O(L·d·d_ff) — shared weights | ↓ significantly | Weight sharing provides large memory advantage vs all-expert storage |
| TTFT | ref | ↑ (unknown) | ↑ | Depends on architectural choices |
| TPOT | ref | ↑ (unknown) | ↑ | Not an efficiency mechanism |

**vs. Baseline C (K2 family, 72.55B dense Llama-arch)**

| Metric | Baseline C (K2) | Idea 3.4 | Change | Notes |
|--------|----------------|----------|--------|-------|
| Compute (FLOPs/token) | O(80·(s·d+d·d_ff)), d=8192, d_ff=28672 | ↑ by (1+(T−1)·f_mid) | ↑ **2.5×** at T=4, f_mid=0.5 | K2 is dense; formula applies |
| KV cache (32K ctx) | ~10.0 GiB | = ref (no KV accumulation across iterations) | = | |
| MLP FLOPs/token/layer | ~4.70×10⁸ | Same per iteration | ↑ ×T for recurrent layers | At T=4: ~1.88×10⁹ for recurrent layers |
| Weight memory | ~145.1 GB | ↓ at same effective depth | ↓ | Weight sharing with shared middle block reduces unique parameters |
| TTFT | ref | ↑ ~2.5× at T=4, f_mid=0.5 | ↑ | Not a speedup idea |
| TPOT | ref | ↑ ~1.50× at T̄=2, f_mid=0.5 (worst-case T=T_max; T̄=2 with early exit) | ↑ | Tradeoff: higher per-token latency, potentially better quality per token |

---

## 6. Implementation Considerations

- **Hardware requirements**: Standard CUDA/Triton; no custom kernels required for correctness. Python loop over shared middle block. `torch.compile()` can trace fixed-iteration loops; adaptive loops require eager mode or `torch.jit.script`.
- **Training stability**: Known risks: gradient vanishing/exploding over T unrolled steps, mitigated by LayerScale + identity-biased residuals [Geiping et al., 2025, §3]. ACT ponder cost can conflict with task loss — Ouro's entropy regularization addresses this [Zhu et al., 2025, §3]. Monotonic halting mask in AdaPonderLM provides clean train-test consistency [Song et al., 2026, §3.1]. Overthinking at T > optimal is an identified pathology [Kohli et al., 2026] — motivates T_max cap.
- **Truncated BPTT**: Huginn used k=8 with T_max=32. At k=8, f_mid=0.5: effective backward training cost ≈ (1+(8−1)×0.5) = 4.5×, not 16×. This is the correct figure for training budget estimates at high T.
- **KV cache at prefill**: K and V entries change between iterations (because h changes), requiring T KV recomputation passes at prefill. This is captured in the TTFT formula but not separately called out.
- **Ragged-batch inference**: Per-token early exit in batched serving creates uneven depth distribution. Padding to T_max wastes compute; batch-level exit or bin-packing are practical workarounds.
- **Implementation effort**: LOW for research prototype (multiple open-source implementations: Huginn, Ouro, MoR). MEDIUM for production batched serving with per-token early exit. MEDIUM-HIGH for retrofitting onto Gated DeltaNet hybrid baseline (open design question with no published answer).

---

## 7. Synergies

- **3.6 (Recursive Internal DAG)**: Idea 3.4 provides the while-loop foundation; 3.6 adds conditional branching within each iteration.
- **3.7 (Learnable State Machine)**: The recurrent loop can carry explicit FSM state alongside activations, providing cross-token routing information.
- **4.2 (Shared Core Weights + Per-Layer LoRA)**: The shared middle block is the "shared core"; per-iteration LoRA adapters add expressiveness without breaking weight sharing. Relaxed Recursive Transformers [16] provide a direct implementation template: iteration-specific LoRA modules initialized via truncated SVD on residual matrices.
- **1.2 (Per-Token Adaptive Depth)**: Complementary; both are token-level adaptive compute mechanisms at different granularities. Combining requires design care to avoid redundancy.
- **2.2 (Compressed Dense Layers via Matrix Decomposition)**: Low-rank middle-block weights reduce per-iteration DRAM BW by rank factor r, with savings multiplied by T iterations.

**Conflicts:**
- 1.2 as a direct full-replacement: both address the same compute-allocation problem; combining without care creates redundant mechanisms.
- 3.1 (Layer-Level MoE): if the router selects between entire blocks, the fixed shared-middle-block structure is disrupted.

---

## Risk Assessment

**Technical risk: LOW** — validated at LLM scale in three independent systems (Huginn, Ouro, AdaPonderLM). Training challenges have established mitigations. Primary risk: Lu et al. 2025 negative result showing depth recurrence without trajectory supervision yields marginal GSM8K improvement (3.11% → 4.93%).

**Potential impact: MEDIUM** — Huginn at r=32 matching 50B-class models [Geiping et al., 2025, §5]. RLTT +14.4% MATH-500 [Williams & Tureci, 2026, §5, Table 2 — unreviewed]. Benefits are narrow (reasoning-dedicated endpoints, relaxed TTFT SLAs, or smaller recurrent model replacing larger dense one).

**Implementation effort: LOW** for research prototype; **MEDIUM** for production batched-serving with per-token early exit; **MEDIUM-HIGH** for retrofitting onto Gated DeltaNet hybrid baseline.

---

## 8. Accuracy / Quality Tradeoff

- **Reported quality delta (from closest analogues)**:
  - [Geiping et al., 2025][3] (Huginn-3.5B): at r=32 recurrent iterations, the model matches reasoning performance of models up to 50B parameters — a substantial quality-positive outcome. No quality degradation is reported at any tested iteration count up to r=32, only improvements (§5, Table N).
  - [Zhu et al., 2025][5] (Ouro-2.6B): Table 8 reports 90.85% MATH500 at 2.6B parameters versus Gemma3-12B's 83.20% — a +7.65 pp gain. The advantage grows with additional recurrent depth allocation, confirming monotone quality improvement with increased compute.
  - [Williams & Tureci, 2026][12] (RLTT — unreviewed preprint): applied to Ouro-2.6B-Thinking, RLTT delivers +14.4% MATH-500, +16.6% AIME24, +10.0% BeyondAIME versus GRPO baseline (Table 2, §5). These are gains over GRPO, not over chain-of-thought, but illustrate the quality upside when trajectory-level supervision is added.
  - [Lu et al., 2025][11] (Latent CoT critical analysis): GSM8K accuracy goes from 3.11% (r=4) to 4.93% (r=32) — both well below explicit CoT at 24.87% (Table 1). This critical negative result shows that depth recurrence alone without RLTT-style supervision provides only marginal absolute gains (+1.82 pp), framing the critical dependency on proper training supervision.
  - [Song et al., 2026][8] (AdaPonderLM): Table 3 reports 2.8B zero-shot average 59.6% (+2.2% over Pythia baseline) and five-shot 61.1% (+3.5%), confirming quality improvements at LLM scale with token-wise adaptive depth.
  - [Saunshi et al., 2025][4] (Looped Transformers Theory): 1-layer looped 12× achieves 100.0% on addition at n=8 operands (Table 1), and 73.2% on i-GSM math problems at 8× looping (Table 2), versus substantially lower accuracy for non-looped counterparts — another quality-positive data point.

- **Monotonicity**: Quality improvement is **monotone with iteration count up to a critical T_max**, beyond which the "overthinking" pathology identified by [Kohli et al., 2026][14] can degrade predictions. Quality is not a simple monotone function of aggressiveness overall — it is concave: improving up to an optimal T, then declining. The T_max cap in idea 3.4's design directly addresses this. Early-exit mechanisms (ACT, acceleration-based halting) track the quality peak automatically per token.

- **Recovery**: This idea is **quality-positive**, not quality-negative. There is no degradation to recover from under normal operating conditions (T within the optimal range). If the overthinking regime is inadvertently entered (T >> optimal), recovery is straightforward: reduce T_max. The key training-side risk is Lu et al.'s negative result — depth recurrence without trajectory-level supervision may yield near-zero practical quality gains despite the theoretical promise, meaning the supervision design (RLTT-style reward) is what "recovers" the quality potential. Full recovery of quality potential requires RLTT or equivalent trajectory-reward training; without it, the system is functional but underperforms its theoretical ceiling.

- **Conditions for acceptable degradation**: There is no quality degradation in the nominal operating regime — the tradeoff is entirely in the **compute dimension** (increased TTFT and TPOT) in exchange for quality gains. The architecture is appropriate when: (1) reasoning quality is the primary objective and latency is secondary (reasoning-dedicated endpoints, offline evaluation pipelines); (2) a smaller recurrent model (3.5B–2.6B) can replace a larger non-recurrent model (50B+) at equivalent quality, reducing total serving cost; (3) TTFT SLAs are relaxed (streaming or interactive sessions where thinking latency is acceptable); (4) explicit chain-of-thought token emission would cost more wall-clock time than the recurrent compute overhead. The mechanism is not appropriate for latency-sensitive streaming inference where every millisecond of TTFT matters or where T_max overhead exceeds the cost of CoT verbosity.

---

<!-- CITATION MANIFEST -->
[1]: Universal Transformers — Dehghani et al., ICLR 2019 (arXiv:1807.03819). Shared-weight recurrent depth + ACT per-token halting + latent space. Predates idea 3.4 by 5+ years; all three components simultaneously present.
[2]: ACT — Graves, 2016 (arXiv:1603.08983). Foundational learned halting probability accumulation with ponder cost regularization.
[3]: Huginn — Geiping et al., 2025 (arXiv:2502.05171). LLM-scale (3.5B, 795B tokens), (2,4,2) structure, T_max iterations, KL/acceleration halting. 50B-class matching at r=32.
[4]: Looped Transformers (Saunshi et al.) — ICLR 2025 (arXiv:2502.17416). Theorem 5.2: looped transformers simulate any same-depth non-looped model; looped models simulate CoT reasoning.
[5]: Ouro — Zhu et al., 2025 (arXiv:2510.25741). 7.7T token pretraining; entropy-regularized depth allocation; 90.85% MATH500 at 2.6B parameters.
[6]: COCONUT — Hao et al., COLM 2025 (arXiv:2412.06769). Horizontal recurrence (new sequence positions per thought) — complementary to idea 3.4's vertical recurrence.
[7]: Mixture-of-Recursions — Bae et al., NeurIPS 2025 (arXiv:2507.10524). Learned per-token depth routing; 2.06× throughput; 19% training time reduction.
[8]: AdaPonderLM — Song et al., 2026 (arXiv:2603.01914). Closest exact match to idea 3.4 specification: monotonic halting mask + KV reuse + per-token early exit.
[9]: PonderNet — Banino et al., ICML Workshop 2021 (arXiv:2107.05407). Bernoulli halt variable; lower-variance gradients than ACT.
[10]: Two-Scale Latent Dynamics — Pappone et al., NeurIPS 2025 (arXiv:2509.23314). Acceleration-based early exit; 580ms→360ms latency on GPT-2-scale model.
[11]: Latent CoT analysis — Lu et al., COLM 2025 Workshop (arXiv:2507.02199). Critical negative result: GSM8K 3.11%→4.93% at r=4→32; CoT 24.87%. Supervision required.
[12]: RLTT — Williams & Tureci, 2026 (arXiv:2602.10520). UNREVIEWED PREPRINT. Full title: "Prioritize the Process, Not Just the Outcome: Rewarding Latent Thought Trajectories Improves Reasoning in Looped Language Models". +14.4% MATH-500, +16.6% AIME24, +10.0% BeyondAIME over GRPO. Affiliation unverified.
[13]: Retrofitted Recurrence — McLeish et al., 2025 (arXiv:2511.07384). Demonstrates retrofitting of depth-recurrence onto existing pretrained models.
[14]: Kohli et al., 2026 (arXiv:2604.07822). Three-stage grokking; depth extrapolation; overthinking pathology motivates T_max cap. Under review.
[15]: SpiralFormer — Yu, Shu, Wang, Zhang, Wu (Haoyi), Wu (You), Long, Chen, Xu, Su, Zheng, 2026 (arXiv:2602.11698). Multi-resolution recursion; perplexity 7.14 vs 7.44 (LoopedFormer); 54.37% vs 51.93% 5-shot accuracy; 7–11% FLOPs reduction at 1.4B scale.
[16]: Relaxed Recursive Transformers — Bae, Fisch, Harutyunyan, Ji, Kim (Seungyeon), Schuster, ICLR 2025 (arXiv:2410.20672). Weight-tied recursive LLMs relaxed with layer-wise LoRA; 2–3× potential throughput via depth-wise batching + early exit; Gemma 1B recursive recovers Gemma 2B performance.
[17]: MeSH — Yu et al., 2025 (arXiv:2510.07739). External memory buffer separating persistent from transient state in recursive transformers; +1.06% downstream accuracy with 33% fewer non-embedding parameters at 1.4B scale.
