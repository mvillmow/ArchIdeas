# Research: Recursive Internal DAG (Near-Turing-Complete Flow Control)
## ID: 3.6

## Executive Summary

**Novelty verdict:** PARTIAL — the loop component is fully solved at LLM scale, differentiable branching (IPA-GNN) and constructively proved loops+branches+state (Giannou et al.) exist, but the specific combination of a learned gated DAG inside each loop step + explicit persistent state tensor + end-to-end language-data training has no published embodiment ([Universal Transformers, 2019], [Huginn, 2025], [Ouro, 2025], [IPA-GNN, 2020], [Giannou et al., 2023]).

Recursive internal DAG (idea 3.6) is PARTIAL prior art — ~70% of the idea exists in literature. The loop component (3.4) is a solved problem at LLM scale (Huginn, Ouro, ANIRA, LoopFormer, AdaPonderLM). The novel value of 3.6 beyond 3.4 lies in the learned gated DAG within each loop step and the explicit persistent state tensor, jointly trained end-to-end on language data. This specific combination has no published embodiment.

Framing and scope: "near-Turing-complete (bounded)" terminology used throughout; KV cache bandwidth formula includes the T·L_mid·s·d_kv KV re-read term; KV cache vs A1 comparison is "↑ (full-attention middle block) or = (hybrid middle block)"; training cost is estimated under full BPTT with truncated BPTT (k=4→8) noted as the practical alternative; a novel cascading routing-state collapse failure mode is documented.

---

## 1. Idea Description

Combines the DAG structure (3.5) with recursion (3.4) to form a near-Turing-complete flow control representation within the model's forward pass — **loops, branches, and state** — all learned and executed without emitting tokens.

The inferred architecture contains:
1. **Loops**: A while-loop iterating a shared-weight computation block up to T_max times with a learned halting condition (inherited from 3.4).
2. **Branches**: Within each loop iteration, a learned DAG with conditional gates selects different computation paths per input (inherited from 3.5).
3. **Persistent state**: An explicit state tensor updated at each loop iteration, governing which branches are taken in subsequent iterations.

Together, loops + branches + persistent state constitute a near-Turing-complete computation model (modulo the T_max bound, making it bounded-Turing-complete). **Important framing note:** The Turing-completeness theorems cited (Pérez et al. 2019, Giannou et al. 2023) require either unbounded steps or arbitrary-precision activations, neither available in finite-precision hardware. The correct framing is "**near-Turing-complete under bounded computation**" — this is the terminology used throughout this document.

**Inference-cost note:** This is NOT an inference speedup idea. Both TTFT and TPOT increase with each additional loop iteration (T) and DAG depth (D). The value proposition is quality improvement — more expressive computation per forward pass at the cost of higher latency.

---

## 2. Literature Review

### Neural Turing Machines
Graves, Wayne, Danihelka — arXiv:1410.5401, 2014[1]

Differentiable analogue of a Turing machine via RNN + external memory + attentional read/write heads. Demonstrates copying, sorting, associative recall. Foundational loops-plus-state paradigm. Idea 3.6 internalizes the external memory and control flow into the forward pass weights.

**[Graves et al., 2014]** — §1 "Introduction", §3 "The Controller Network".

### Hybrid Computing Using a Neural Network with Dynamic External Memory (DNC)
Graves et al. (20 authors) — Nature 538, 471–476, 2016[2]

Extends NTM with temporal link matrices and usage-based memory allocation. Demonstrates persistent structured state (temporal link memory) updated across steps — one of the three required components of idea 3.6. Published title is "Hybrid computing using a neural network with dynamic external memory"; commonly referred to as the Differentiable Neural Computer (DNC).

**[Graves et al., 2016]** — §1 "Introduction", §2 "Differentiable Neural Computer", Nature 538.

### Adaptive Computation Time for Recurrent Neural Networks
Graves — arXiv:1603.08983, 2016[3]

ACT provides the **halting mechanism** for the loop component of idea 3.6. The learned scalar halting probability with ponder cost regularization.

**[Graves, 2016]** — §3 "Adaptive Computation Time".

### Universal Transformers
Dehghani, Gouws, Vinyals, Uszkoreit, Kaiser — ICLR 2019, arXiv:1807.03819[4]

Combines transformer parallelism with recurrent inductive bias via shared-weight self-attention applied recurrently with per-position ACT halting. Proved near-Turing-complete (under unbounded steps + arbitrary precision). Table 7 (§3.6): +0.9 BLEU on WMT14 En-De over standard Transformer.

**[Dehghani et al., 2019]** — §3 "Universal Transformer", §4 "Adaptive Universal Transformer", Table 2, Table 7.

### On the Turing Completeness of Modern Neural Network Architectures
Pérez, Marinković, Barceló — ICLR 2019, arXiv:1901.03429[5]

Proves Transformer and Neural GPU are Turing-complete given arbitrary-precision activations. Validates the theoretical basis for the near-Turing-complete framing of idea 3.6. **Caveat: requires arbitrary-precision arithmetic, unavailable in finite-precision hardware.**

**[Pérez et al., 2019]** — §1 "Introduction", §3 "Turing Completeness".

### Neural Programmer-Interpreters
Reed, de Freitas — ICLR 2016, arXiv:1511.06279[6]

Recurrent compositional network with task-agnostic recurrent core, persistent key-value program memory, and domain-specific encoders. Learned program pointer governs branching between sub-programs. NPI implements **branches + state** components of idea 3.6.

**[Reed and de Freitas, 2016]** — §2 "Neural Programmer-Interpreter", §3 "Execution Traces".

### Programming with a Differentiable Forth Interpreter
Bosnjak, Rocktäschel, Naradowsky, Riedel — ICML 2017, arXiv:1605.06640[7]

Differentiable interpreter for Forth (stack-based language with explicit loops and branches). Directly instantiates the **loops + branches + state** paradigm. The Forth abstract machine is an explicit implementation of the computation model idea 3.6 proposes to learn implicitly.

**[Bosnjak et al., 2017]** — §2 "Differentiable Forth", §3 "∂4 Machine", ICML 2017.

### Learning to Execute Programs with Instruction Pointer Attention Graph Neural Networks (IPA-GNN)
Bieber, Sutton, Larochelle, Tarlow — NeurIPS 2020, arXiv:2010.12621[8]

GNN with soft instruction pointer for executing programs with control flow graphs (conditional branches). Differentiable branch decisions. Directly implements the **DAG + branch** component. Outperforms RNN/standard GNN baselines on program execution tasks.

**[Bieber et al., 2020]** — §3 "IPA-GNN Architecture", §4 "Experiments".

### Looped Transformers as Programmable Computers
Giannou, Rajput, Sohn, Lee, Lee, Papailiopoulos — ICML 2023, arXiv:2301.13196[9]

Shows a constant number of encoder layers in a loop can emulate basic computing blocks including **conditional branches and state** (program counters, data registers). A frozen transformer instructed by a "punchcard" input can emulate a calculator, linear algebra library, and backpropagation algorithm. **Most complete theoretical match to idea 3.6 — but program structure is hand-specified, NOT learned from data.** The gap from "constructive proof with hand-specified program" to "learned program from data" is the precise novelty.

**[Giannou et al., 2023]** — §3 "Looped Transformers as Programmable Computers", §4 "Conditional Branching", Table 1.

### Sparse Universal Transformer
Tan, Shen et al. — EMNLP 2023, arXiv:2310.07096[10]

Combines Universal Transformer loop with sparse MoE routing and stick-breaking-based halting. ~50% compute reduction vs standard Universal Transformer at comparable WMT14 performance. Implements a loop + per-step routing (MoE) — partial instantiation of "branches within a loop."

**[Tan et al., 2023]** — §2 "Sparse Universal Transformer", §3 "Dynamic Halting".

### Reasoning with Latent Thoughts: On the Power of Looped Transformers
Saunshi, Dikkala, Li, Kumar, Reddi — ICLR 2025, arXiv:2502.17416[11]

Theorem 5.2: looped transformers can simulate any same-depth non-looped model. Tables 1/2: arithmetic and i-GSM math results with 100% accuracy at 1-layer looped 12×. Provides theoretical justification for why the loop component of idea 3.6 improves reasoning quality.

**[Saunshi et al., 2025]** — §2 "Setup", §3 "Main Results", Theorem 5.2, Table 1, Table 2.

### Simulation of Graph Algorithms with Looped Transformers
Back de Luca, Fountoulakis — ICML 2024, arXiv:2402.01107[12]

Looped transformers with extra attention heads simulate Dijkstra's shortest path, BFS, DFS, and Kosaraju's SCC using constant width. Validates the loop component for algorithmic reasoning. Two authors: Artur Back de Luca and Kimon Fountoulakis.

**[Back de Luca and Fountoulakis, 2024]** — §3 "Looped Transformer Architecture", §4 "Graph Algorithm Simulation", ICML 2024.

### Scaling Latent Reasoning via Looped Language Models (Ouro)
Zhu et al. — arXiv:2510.25741, 2025[13]

7.7T token pretraining; entropy-regularized depth allocation. 1.4B and 2.6B models match 12B dense models. Large-scale practical validation of the looped-transformer paradigm.

**[Zhu et al., 2025]** — §3 "Ouro Architecture", §4 "Experiments".

### Understanding Dynamic Compute Allocation in Recurrent Transformers (ANIRA)
Moosa, Lohit, Wang, Chatterjee, Yin — arXiv:2602.08864, February 2026[14]

Controlled experimental framework for per-token variable-depth computation in recurrent transformers. Distinguishes early-allocation vs online-halting. **Negative OOD-generalization result:** adaptive compute does not guarantee algorithmic generalization to unseen input sizes — an important limitation for idea 3.6's Turing-completeness framing.

**[Moosa et al., 2026]** — §2 "ANIRA Framework", §4 "Results".

### LoopFormer: Elastic-Depth Looped Transformers for Latent Reasoning
Jeddi, Ciccone, Taati — ICLR 2026, arXiv:2602.11451[15]

Shortcut-consistency training enabling budget-conditioned inference. Smooth quality improvement as budget grows. **Venue ICLR 2026 should be qualified as "ICLR 2026 (venue unverified)" until proceedings URL is confirmed.**

**[Jeddi et al., 2026]** — §3 "LoopFormer Architecture", §4 "Language Modeling Results", ICLR 2026 (unverified venue).

### SpiralFormer: Looped Transformers Can Learn Hierarchical Dependencies via Multi-Resolution Recursion
Yu, Shu, Wang, Zhang, Wu, Wu, Long, Chen, Xu, Su, Zheng — arXiv:2602.11698, February 2026[16]

Multi-resolution recursion (different resolution levels per iteration). Fixed (non-learned) DAG structure applied within each loop. Outperforms looped and non-looped baselines at 160M–1.4B on language modeling. **Structurally the closest published work to "loop + structured computation paths" but lacks conditional gating and persistent state.**

**[Yu et al., 2026]** — §3 "SpiralFormer Architecture", §4 "Multi-Resolution Recursion".

### Neural Algorithmic Reasoning for Hypergraphs with Looped Transformers
Huang, Liang, Shi, Song, Zhuang — arXiv:2501.10688, 2025[17]

Extends looped transformer algorithmic reasoning to hypergraphs. Authors: Zekai Huang, Yingyu Liang, Zhenmei Shi, Zhao Song, Zhen Zhuang.

**[Huang et al., 2025]** — §3 "Hypergraph Algorithm Simulation". arXiv:2501.10688.

### Turing Completeness of Bounded-Precision Recurrent Neural Networks
Chung, Siegelmann — NeurIPS 2021[18]

Proves a 54-neuron bounded-precision RNN with growing memory can simulate a Universal Turing Machine with linear time complexity. Supports the near-Turing-complete claim under finite-precision constraints if the model can grow its state representation.

**[Chung and Siegelmann, 2021]** — §1 "Introduction", §3 "Main Result", NeurIPS 2021.

### Efficient Turing Machine Simulation with Transformers
Li, Wang — arXiv:2512.00003, 2025[19]

Proves any (t(n),s(n))-bounded TM can be simulated by a constant-depth Transformer with O(s(n))-long context and O(s(n)^c) CoT steps. Bounds the overhead of near-Turing-complete computation in transformers. **Venue listed as ICLR 2026 — unverified; paper submitted September 2025.**

**[Li and Wang, 2025]** — §1 "Introduction", §3 "Main Theorem", ICLR 2026 (unverified venue).

### Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach (Huginn)
Geiping, McLeish, Jain, Kirchenbauer, Singh, Bartoldson, Kailkhura, Bhatele, Goldstein — arXiv:2502.05171, 2025[20]

3.5B-parameter depth-recurrent transformer (Huginn-0125) trained on 800B tokens. Prelude + shared recurrent core + coda architecture; unrolled to arbitrary depth at test time. Achieves reasoning performance equivalent to ~50B-parameter dense models at 3.5B parameter count. Primary practical demonstration of the loop component of idea 3.6 at LLM scale, with no explicit DAG routing or persistent state tensor.

**[Geiping et al., 2025]** — §2 "Architecture", §4 "Experiments", arXiv:2502.05171.

### Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation
Bae, Kim, Bayat, Kim, Ha, Schuster, Fisch, Harutyunyan, Ji, Courville, Yun — NeurIPS 2025, arXiv:2507.10524[21]

Unified framework combining parameter sharing with adaptive per-token recursion depth via lightweight routers. Shared layer stack reused across recursion steps; tokens receive different depths, with selective KV caching for active tokens. Forms a new Pareto frontier from 135M to 1.7B parameters on perplexity and few-shot accuracy vs training FLOPs. Implements token-level routing of recursion depth — a restricted form of the branching component of idea 3.6 (depth routing rather than within-step DAG gating), without a persistent state tensor.

**[Bae et al., 2025]** — §3 "MoR Architecture", §4 "Experiments", NeurIPS 2025.

### PonderLM: Pretraining Language Models to Ponder in Continuous Space
Zeng, Song, Huang, Wang, Li, He, Wang, Li, Lin — ICLR 2026, arXiv:2505.20674[22]

Introduces iterative pondering within a single token step: instead of emitting a token, model yields a weighted sum of token embeddings and feeds it back for k additional forward passes. Self-supervised training; no human annotations. Per abstract, PonderPythia-2.8B surpasses Pythia-6.9B and rivals Pythia-12B; specific "9 benchmarks" count is paper-body §4. Implements the loop component of idea 3.6 (iterative latent refinement) but without DAG routing or persistent state.

**[Zeng et al., 2025]** — §2 "PonderLM Architecture", §4 "Experiments", ICLR 2026.

### AdaPonderLM: Gated Pondering Language Models with Token-Wise Adaptive Depth
Song, Li, Wang, Zeng, Song, Wang, Xu, He, Lin — arXiv:2603.01914, March 2026[23]

Self-supervised recurrent LM with token-wise early exiting via iteration-specific MLP gates and monotonic halting mask. KV reuse for halted tokens. ~10% inference compute reduction at comparable perplexity across 70M–2.8B Pythia models. Implements the loop + per-token halting components. Learned gates allocate more compute to high-NLL tokens. No within-step DAG routing or persistent state tensor.

**[Song et al., 2026]** — §2 "Architecture", §4 "Results", arXiv:2603.01914.

---

## 3. Prior Art Classification

**Status: PARTIAL (~70% covered)**

- **Loop component (fully exists):** Universal Transformers[4], Huginn[20], Ouro[13], ANIRA[14], LoopFormer[15], Mixture-of-Recursions[21], PonderLM[22], AdaPonderLM[23] implement shared-weight recurrent depth with learned/heuristic halting at LLM scale.
- **DAG branching (partially exists):** IPA-GNN[8] implements differentiable branching over a given control flow graph. Giannou et al.[9] constructively prove loops+branches+state in a looped transformer (hand-specified, NOT learned). SUT[10] implements MoE routing within a loop (restricted per-step branching). SpiralFormer[16] implements fixed multi-resolution DAG within each loop step.
- **Persistent external state:** NTM[1], DNC[2] (external). NPI[6] implements learned program memory as persistent branching state (but external, and requires supervised traces).

**Novel (~30%):** The specific combination of a learned gated DAG controlling which sub-networks execute within each loop step + an explicit persistent state tensor + all jointly trained end-to-end on language data has not been demonstrated. The critical gap: "learned from data" rather than "hand-specified or constructively proved."

---

## 4. Technical Analysis

### 4.1 Theoretical Complexity

**Variable definitions:** L=layers, L_pre/L_mid/L_suf=prelude/mid/coda layers, T=iterations (T̄=mean), D=DAG depth, B=branching factor, d=hidden dim, d_ff=MLP dim, s=sequence length, d_kv=KV head dim, d_state=state tensor dim.

**Total compute per token (T iterations):**
O((L_pre+L_suf)·(s·d+d·d_ff) + T·(L_mid·(s·d+d·d_ff) + D·d² + d_state·d))

Relative to Baseline A2: T=4, f_mid=0.5 → TTFT/TPOT ↑**2.5×** [derived: multiplier = 1 + (T−1)·f_mid = 1 + 3·0.5 = 2.5×] + small DAG overhead T·D·d²/(L·d·d_ff).

**KV cache bandwidth formula:**
```
O((L_pre+L_suf)·(d·d_ff+s·d_kv) + T·L_mid·(d·d_ff+s·d_kv) + T·D·d²)
```
At canonical A1/B context (s=262,144), the missing T·L_mid·s·d_kv term dominates. For T=4, L_mid=32, A1 has H_kv=4 and head_dim=256, giving d_kv=2×H_kv×head_dim=2048 bytes/position/layer (K+V): KV re-read = 4×32×262,144×2048×1 bytes ≈ **68 GB per token** — comparable to weight loading cost and cannot be ignored.

### 4.2 Key Comparison Tables

**vs. Baseline A1 (Qwen3.5-27B Hybrid)**

| Metric | Baseline A1 | Idea 3.6 | Change | Notes |
|--------|------------|----------|--------|-------|
| Compute (FLOPs/token) | O(L·(d²+s·d/4)) | O((L_pre+T·L_mid+L_suf)·...) | ↑ **2.5×** at T=4, f_mid=0.5 [derived: 1 + (T−1)·f_mid = 1 + 3·0.5 = 2.5×] | |
| KV cache (32K benchmark; A1 max = 262,144) | ~2.15 GB (16 full-attn layers only) | **↑ (full-attention middle block) or = (hybrid middle block)** | ↑ or = | Depends on middle block attention type; canonical formula: 16×2×4×256×32768×2 |
| KV bandwidth (at 262K ctx) | ~2.15 GB static | +~68 GB per token for T·L_mid·s·d_kv re-read (T=4, L_mid=32, A1: 2×H_kv×head_dim=2048/layer) | ↑ dramatically | d_kv per layer = 2×4×256 = 2048 bytes/position |
| Weight memory | O(L·d·d_ff) | O(L·d·d_ff + D·d²) | ↑ small | DAG gates +<1% for typical D=4 |
| TTFT | ref | ↑ ~2.5× at T=4, f_mid=0.5 | ↑ | Not a speedup |
| TPOT | ref | ↑ ~2.5× at T=4 (or ~1.5× at T̄=2) | ↑ | Not a speedup |

**vs. Baseline A2 (Qwen3-32B Dense)**

| Metric | Baseline A2 | Idea 3.6 | Change | Notes |
|--------|------------|----------|--------|-------|
| Compute | O(L·(s·d+d·d_ff)) | ↑ ~2.5× at T=4 | ↑ | |
| KV cache | ~8.59 GB @32K (64×2×8×128×32768×2) | = (A2 full-attn; KV re-read at 40K context adds ~2.50 GB at T=4 [derived: 2×16×4×256×40960×2 = 2,684,354,560 bytes ≈ 2.50 GB]) | ↑ small | KV re-read at 40K context is smaller than at 262K |
| TTFT | ref | ↑ ~2.5× | ↑ | |
| TPOT | ref | ↑ ~2.5× (or ~1.5× at T̄=2) | ↑ | |

**vs. Baseline B (Qwen3.5-397B-A17B MoE)**

| Metric | Baseline B | Idea 3.6 | Change | Notes |
|--------|-----------|----------|--------|-------|
| Compute | O(L·(d²+k·d·d_e)) | ↑ significantly | ↑ | 3.6 targets quality improvement, not efficiency |
| KV bandwidth (262K ctx) | ~8.0 GB | ↑ dramatically with T·L_mid·s·d_kv | ↑ | B's 32Q/2KV gives lower KV BW; 3.6's overhead vs B is understated if only A2-based formula is used |
| Weight memory | O(L·E·d·d_e) | O(L·d·d_ff) | ↓ | Weight sharing advantage vs all-expert storage |
| TTFT | ref | ↑ | ↑ | |
| TPOT | ref | ↑ | ↑ | Quality idea, not efficiency |

**vs. Baseline C (K2 family, 72.55B dense Llama-arch)**

| Metric | Baseline C (K2) | Idea 3.6 | Change | Notes |
|--------|----------------|----------|--------|-------|
| Compute | O(80·(s·d+d·d_ff)) | ↑ ~2.5× at T=4, f_mid=0.5 | ↑ | |
| KV cache (32K) | ~10.0 GiB | = + T·L_mid·s·d_kv re-read term | ↑ significant | At 32K with L_mid=40, C: d_kv=2×8×128=2048 bytes/layer → T·L_mid·s·d_kv = 4×40×32,768×2048 ≈ 10.0 GiB additional re-read per token. |
| MLP FLOPs/token/layer | ~4.70×10⁸ (2×d×d_ff = 2×8192×28672) | ~1.88×10⁹ at T=4 for recurrent layers | ↑ | Canonical formula from SHARED_PRELUDE: 2×d×d_ff; 2×8192×28672 = 469,762,048 ≈ 4.70×10⁸ | |
| Weight memory | ~145.1 GB | ↓ at same effective depth | ↓ | Weight sharing advantage |
| TTFT, TPOT | ref | ↑ | ↑ | Quality idea |

---

## 5. Implementation Considerations

- **Framework support:** PyTorch and JAX support via Python-level loops. `torch.compile()` supports fixed-T loops; variable-T hard routing requires eager mode or `torch.cond` (PyTorch 2.x).
- **Training stability:** Three interacting instability sources: (1) recurrent gradient flow (LayerScale + identity residuals), (2) learned halting (ponder cost), (3) DAG routing collapse (load-balancing auxiliary loss + Gumbel-softmax temperature annealing). Novel failure mode from review: **cascading routing-state collapse** — if the persistent state collapses (→ 0 or fixed vector), all subsequent routing decisions collapse in a cascade (unlike standard MoE where gates are independent). Mitigation: state magnitude regularization, non-trivial state initialization, monitoring state entropy during training.
- **Staged implementation (strongly recommended):**
  - Phase 1 (2–4 weeks): Loop only (3.4-style), fixed T, LayerScale + TBPTT. Validate training stability.
  - Phase 2 (4–8 weeks): Add persistent state tensor with soft state updates (all branches active). Validate state does not collapse.
  - Phase 3 (2–4 months): Add learned DAG routing. Monitor for cascading routing-state collapse.
  - Phase 4 (3–6 months): Full integration at LLM scale. Production GPU execution engineering.
- **Truncated BPTT:** Training cost estimate of 2.5× assumes full BPTT at T=4. With truncated BPTT at k=4 (similar to Huginn's k=8 approach), actual backward cost is (1+(k−1)·f_mid) ≈ 2.5× at k=4, f_mid=0.5 — same estimate holds for k=T. At higher T_max with small k: effective training cost ≈ 1+(k−1)·f_mid regardless of T_max.

---

## 6. Synergies

- **3.4 (Recursive Internal State):** 3.6 is a strict superset — the loop from 3.4 is a prerequisite. Implement 3.4 first.
- **3.5 (Gated Internal DAG):** 3.6 is a strict superset — the DAG from 3.5 must prove viability before adding the recurrence of 3.6.
- **3.7 (Learnable State Machine):** The explicit state tensor of 3.6 is the FSM state of 3.7. Highly synergistic.
- **4.2 (Shared Core Weights + Per-Layer LoRA):** Per-iteration LoRA adapters allow the shared middle block to specialize at each loop step.
- **2.2 (Compressed Dense Layers via Matrix Decomposition):** Low-rank middle-block weights reduce per-iteration loading, multiplied by T.

**Conflicts:**
- 3.1 (Layer-Level MoE Full Block Routing): router selecting entire blocks disrupts the fixed shared-middle-block structure.
- 3.4 as standalone: redundant if 3.6 is deployed.

---

## 7. Risk Assessment

**Technical risk: HIGH** — Three interacting instability sources; cascading routing-state collapse is a novel failure mode. The combined loops+DAG+state system has not been trained at any scale on any task.

**Potential impact: MEDIUM** — Strong theoretical motivation (near-Turing-complete); unknown empirical delta vs simpler loop-only (3.4). The critical unknown: whether DAG branching provides measurably better quality than loop-only at equivalent compute.

**Implementation effort: HIGH** — 6–12 months for production system; no existing implementation or checkpoint.

**Novelty risk (scooped):** MEDIUM — Loop+DAG (sans persistent state) is likely to appear in literature within 12 months given current research trends.

---

## 8. Accuracy / Quality Tradeoff

- **Reported quality delta (from closest analogues)**:
  - [Geiping et al., 2025][20] (Huginn-3.5B): At r=32 recurrent iterations, the loop-only component of idea 3.6 matches reasoning performance of models up to 50B parameters (§4 "Experiments") — establishing a strong quality-positive baseline from the loop component alone. The additional DAG and persistent state layers of 3.6 are expected to yield further gains over loop-only, but this is undemonstrated. The Huginn result bounds the quality floor for the loop component of 3.6.
  - [Zhu et al., 2025][13] (Ouro): 90.85% MATH500 at 2.6B parameters (vs Gemma3-12B at 83.20%, +7.65 pp) using loop-only recurrence. The additional expressiveness of DAG routing and persistent state in idea 3.6 over Ouro's pure loop should yield further gains on structured reasoning tasks where cross-iteration routing context is necessary — but the magnitude is unknown until ablation.
  - [Dehghani et al., 2019][4] (Universal Transformers): Table 7, §3.6 reports +0.9 BLEU on WMT14 En-De over the standard Transformer — the baseline quality gain from adding shared-weight recurrence.
  - [Tan et al., 2023][10] (Sparse Universal Transformer): ~50% compute reduction vs standard Universal Transformer at comparable WMT14 performance — confirming that a restricted form of "loop + per-step routing" (MoE within a loop) preserves quality while halving compute. Idea 3.6's DAG routing is a more expressive generalization of SUT's MoE routing, suggesting quality should be at least as good per unit compute.
  - [Bae et al., 2025][21] (Mixture-of-Recursions): per-token recursion depth routing establishes a Pareto frontier at 135M–1.7B parameters on perplexity and few-shot accuracy vs training FLOPs (§4) — demonstrating routing-augmented loops improve quality-per-FLOP over plain loops.
  - [Lu et al., 2025 — critical][11] (Latent CoT analysis): GSM8K 3.11% (r=4) → 4.93% (r=32), both well below CoT at 24.87% (Table 1). This critical negative result for loop-only without supervision shows that 3.6's additional complexity magnifies training risk: without proper supervision design the joint loop+DAG+state system may deliver near-zero quality gains over simpler architectures.

- **Monotonicity**: Quality improvement from increased compute (more iterations T and deeper DAG D) is **concave and non-monotone at extremes**. The "overthinking" pathology for loop-only models applies to idea 3.6 as well. The additional DAG routing dimension introduces a second non-monotone axis: overly deep or over-routed DAG paths can amplify cascading routing-state collapse, a novel failure mode (§5) that can sharply degrade quality at high DAG depth D. Quality is monotone within safe T and D operating ranges but has hard cliffs near collapse regimes.

- **Recovery**: Two recovery forms: (1) **Reduce T or D** — scaling back iteration count or DAG depth recovers quality by returning to the loop-only regime validated by Huginn and Ouro; (2) **Retrain with stronger supervision** — RLTT-style trajectory reward applied to the full 3.6 system is the theoretically principled path. State magnitude regularization and non-trivial state initialization (§5) mitigate cascading routing-state collapse. Knowledge distillation from a loop-only teacher (3.4-class model) during Phase 3 DAG-routing addition is the recommended practical recovery strategy.

- **Conditions for acceptable degradation**: The compute cost increase of 3.6 over baseline (↑2.5× at T=4, f_mid=0.5) plus the KV re-read bandwidth penalty (≈68 GB per token at 262K context, A1 scale) is only acceptable when: (1) the quality gain from DAG routing over loop-only (3.4) is measurably positive in controlled ablation — otherwise 3.4 alone is preferable; (2) the task requires structured branching behavior (program execution, multi-step planning, formal language generation) that loop-only transformers cannot express within the T_max budget; (3) contexts are short-to-medium where the KV re-read overhead is manageable, or the middle block uses hybrid linear attention; (4) the staged implementation (§5 Phase 1–4) confirms positive marginal value of DAG and state components before full system investment.

---

<!-- CITATION MANIFEST -->
[1]: NTM — Graves et al., 2014 (arXiv:1410.5401). Foundational loops-plus-state via external differentiable memory. §1 "Introduction", §3 "The Controller Network".
[2]: DNC — Graves et al. (20 authors), 2016 (Nature 538). Title: "Hybrid computing using a neural network with dynamic external memory." Persistent structured state with temporal link matrices. §1 "Introduction", §2 "Differentiable Neural Computer".
[3]: ACT — Graves, 2016 (arXiv:1603.08983). Halting mechanism for the loop component. §3 "Adaptive Computation Time".
[4]: Universal Transformers — Dehghani et al., ICLR 2019 (arXiv:1807.03819). Loop component at transformer scale with ACT; near-Turing-complete under unbounded steps. §3 "Universal Transformer", §4 "Adaptive Universal Transformer", Table 2, Table 7.
[5]: Turing Completeness — Pérez et al., ICLR 2019 (arXiv:1901.03429). Theoretical foundation; requires arbitrary-precision arithmetic. §1 "Introduction", §3 "Turing Completeness".
[6]: NPI — Reed & de Freitas, ICLR 2016 (arXiv:1511.06279). Branches + state via program pointer and program memory. §2 "Neural Programmer-Interpreter", §3 "Execution Traces".
[7]: Differentiable Forth — Bosnjak et al., ICML 2017 (arXiv:1605.06640). Explicit loops+branches+state in differentiable interpreter. §2 "Differentiable Forth", §3 "∂4 Machine".
[8]: IPA-GNN — Bieber et al., NeurIPS 2020 (arXiv:2010.12621). Full title: "Learning to Execute Programs with Instruction Pointer Attention Graph Neural Networks." Differentiable branching over given control flow graphs. §3 "IPA-GNN Architecture", §4 "Experiments".
[9]: Looped as Programmable — Giannou et al., ICML 2023 (arXiv:2301.13196). Constructive proof of loops+branches+state in looped transformer — hand-specified, not learned. §3 "Looped Transformers as Programmable Computers", §4 "Conditional Branching", Table 1.
[10]: Sparse Universal Transformer — Tan et al., EMNLP 2023 (arXiv:2310.07096). Loop + MoE routing (restricted branching); ~50% compute reduction. §2 "Sparse Universal Transformer", §3 "Dynamic Halting".
[11]: Looped Transformers Theory — Saunshi et al., ICLR 2025 (arXiv:2502.17416). Theorem 5.2: looped models simulate non-looped; looped simulate CoT. §2 "Setup", §3 "Main Results", Theorem 5.2, Table 1, Table 2.
[12]: Graph Algorithm Simulation — Back de Luca & Fountoulakis, ICML 2024 (arXiv:2402.01107). Two authors; simulate Dijkstra, BFS, DFS, Kosaraju with looped transformers. §3 "Looped Transformer Architecture", §4 "Graph Algorithm Simulation".
[13]: Ouro — Zhu et al., 2025 (arXiv:2510.25741). 7.7T token pretraining; matches 12B dense models. §3 "Ouro Architecture", §4 "Experiments".
[14]: ANIRA — Moosa et al., 2026 (arXiv:2602.08864). Controlled halting study; negative OOD-generalization result. §2 "ANIRA Framework", §4 "Results".
[15]: LoopFormer — Jeddi et al., ICLR 2026 (arXiv:2602.11451). Budget-conditioned inference; ICLR 2026 venue UNVERIFIED. §3 "LoopFormer Architecture", §4 "Language Modeling Results".
[16]: SpiralFormer — Yu et al., 2026 (arXiv:2602.11698). Fixed multi-resolution DAG within loop; closest to "loop + structured paths." §3 "SpiralFormer Architecture", §4 "Multi-Resolution Recursion".
[17]: Hypergraph Neural Algorithmic Reasoning — Huang, Liang, Shi, Song, Zhuang (arXiv:2501.10688). Extends looped transformer algorithmic reasoning to hypergraphs. §3 "Hypergraph Algorithm Simulation".
[18]: Bounded-Precision TC — Chung & Siegelmann, NeurIPS 2021. Near-Turing-complete under finite precision with growing memory. §1 "Introduction", §3 "Main Result".
[19]: Efficient TM Simulation — Li & Wang, arXiv:2512.00003, 2025 (ICLR 2026 unverified). Bounds on near-Turing-complete computation cost in transformers. §1 "Introduction", §3 "Main Theorem".
[20]: Huginn — Geiping et al., 2025 (arXiv:2502.05171). 3.5B depth-recurrent LM; 800B tokens; recurrent core only, no DAG or state. §2 "Architecture", §4 "Experiments".
[21]: Mixture-of-Recursions — Bae et al., NeurIPS 2025 (arXiv:2507.10524). Per-token recursion depth routing; Pareto frontier 135M–1.7B; no within-step DAG gating. §3 "MoR Architecture", §4 "Experiments".
[22]: PonderLM — Zeng et al., ICLR 2026 (arXiv:2505.20674). Iterative latent pondering without token emission; loop only, no DAG or persistent state. §2 "PonderLM Architecture", §4 "Experiments".
[23]: AdaPonderLM — Song et al., 2026 (arXiv:2603.01914). Token-wise adaptive depth via gated halting; loop + halting only. §2 "Architecture", §4 "Results".
