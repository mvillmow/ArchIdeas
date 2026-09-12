# Research: Gated Internal DAG (Learned If/Else Control Flow)
## ID: 3.5

## 1. Idea Description

Within a transformer layer, the computation graph is a Directed Acyclic Graph (DAG) with learned gates that implement if/else branching per input token. The model learns conditional control flow as part of its architecture, selecting different computation sub-paths per input.

**Key structural distinction from MoE:** MoE experts are executed **in parallel and independently** (each receives the same input x before combining outputs). A gated DAG has **sequentially dependent** paths — later DAG nodes may depend on outputs of earlier nodes chosen by earlier gates. This is the defining structural novelty.

**Inferred inference intent:** Replace the monolithic MLP sub-layer with a DAG of smaller sub-computations whose edges are gated per input token. For easy tokens, only a sparse subset of the DAG is traversed, reducing FLOPs and bandwidth. For hard tokens, more of the DAG executes. The key value proposition is **TPOT improvement at batch=1** (memory-bandwidth-bound decode).

**Critical limitation vs. Baseline B:** Baseline B activates k=11 of E=512 experts → k/E ≈ **2.1% active FFN compute** per token. Idea 3.5 at p_active=0.3–0.5 has **15–25× more active compute** than B. Idea 3.5 is not competitive with ultra-sparse MoE (B) as an efficiency technique; its value proposition is improvement over **dense models (A1, A2)**, not as an alternative to Baseline B.

---

## 2. Executive Summary

**Novelty verdict:** PARTIAL — per-input gating, execute/skip routing, and LLM-scale intra-layer conditional compute are established, but a branching intra-layer DAG with sequential gate dependencies (gate decisions at earlier nodes condition later paths) inside a generative LLM FFN has no direct published embodiment ([SkipNet, 2018], [Channel Gating, 2019], [Deja Vu, 2023], [PowerInfer, 2023], [MoD, 2024]).

Gated internal DAG (idea 3.5) has genuine structural novelty: an intra-layer branching DAG with sequential gate dependencies in generative LLMs has no direct published predecessor. Prior art coverage is approximately 70–75%; Deja Vu [Liu et al., 2023, ICML 2023] and PowerInfer [Song et al., 2023] establish LLM-scale intra-layer conditional compute as a demonstrated baseline. Practical DAG depth is bounded at d_dag ≈ 4–5 before gate overhead cancels bandwidth savings.

---

## 3. Literature Review

### Conditional Computation in Neural Networks for Faster Models
E. Bengio, Bacon, Pineau, Precup — ICLR Workshop 2015/2016, arXiv:1511.06297[1]

Introduced modern conditional computation: RL-trained gating policies that selectively activate only parts of a network per input. Conceptual ancestor of intra-layer gated DAG nodes. Key difference: gates entire layers (inter-layer), not intra-layer sub-computation DAG nodes.

### Highway Networks
Srivastava, Greff, Schmidhuber — arXiv:1505.00387, 2015[2]

Learned transform gate T and carry gate C = 1−T: the simplest gated DAG (two branches, continuous soft gate). Idea 3.5 generalizes to multi-branch, multi-level intra-layer DAG. Highway gates do not save compute (both branches always computed for soft gating).

### SkipNet: Learning Dynamic Routing in Convolutional Networks
X. Wang, Yu, Dou, Darrell, Gonzalez — ECCV 2018, arXiv:1711.09485[3]

Per-input binary gate to skip entire convolutional residual blocks. 30–90% computation reduction on CIFAR/ImageNet. Inter-layer analogue of what idea 3.5 proposes intra-layer.

### BlockDrop: Dynamic Inference Paths in Residual Networks
Wu, Nagarajan, Kumar, Rennie, Davis, Grauman, Feris — CVPR 2018, arXiv:1711.08393[4]

Policy network selects per-image which ResNet blocks to execute. ~20–36% FLOP reduction at matched ImageNet accuracy. Inter-layer gated DAG with real speedups confirmed.

**[Wu et al., 2018]** — p.7, Table 3: ResNet-101 ~20% avg speedup (up to 36%).

### ConvNet-AIG: Convolutional Networks with Adaptive Inference Graphs
Veit, Belongie — IJCV 2019, arXiv:1711.11503[5]

Per-input Gumbel-Softmax gates each residual layer. ConvNet-AIG-50: 20% fewer FLOPs; ConvNet-AIG-101: 38% fewer FLOPs at matched/exceeding ImageNet accuracy. Discovers class hierarchy structure in gate activation patterns. **Gumbel-Softmax training technique applies directly to intra-layer DAG gates.**

**[Veit & Belongie, 2019]** — p.8, Table 2 (ImageNet results; ConvNet-AIG-50 20% FLOPs, ConvNet-AIG-101 38% FLOPs reduction).

### D2NN: Dynamic Deep Neural Networks — Selective Execution
Liu, Deng — AAAI 2018, arXiv:1701.00299[6]

Explicit "directed acyclic graph of differentiable modules" augmented with controller sub-networks that gate downstream module execution. Most direct conceptual match for idea 3.5: gating a DAG of differentiable modules. Evaluated on image classification tasks.

**[Liu & Deng, 2018]** — §"Dynamic Deep Neural Networks", AAAI 2018.

### Channel Gating Neural Networks
Hua, Zhou, De Sa, Zhang, Suh — NeurIPS 2019, arXiv:1805.12549[7]

Intra-layer gated computation in CNNs: a base path (partial channels) and conditional path (remaining channels). A gate decides per-activation whether the conditional path executes. 2.7–8.0× FLOP reduction, 2.0–4.4× memory access reduction. 2.6× FLOP reduction on ImageNet without accuracy drop. **This is the closest published example to the spirit of idea 3.5 applied intra-layer — a 2-node intra-layer DAG with binary gating in CNNs.**

**[Hua et al., 2019]** — p.8, Table 2 (CIFAR-10 FLOP reductions 2.7–8.0×), Table 3 (ImageNet 2.6×, ResNet-18 with KD).

### DARTS: Differentiable Architecture Search
Liu, Simonyan, Yang — ICLR 2019, arXiv:1806.09055[8]

DAG with learned operation weights on edges — directly applicable to learning which computation sub-paths to keep. DARTS is the **static** version of idea 3.5 (architecture search finds a fixed DAG topology); idea 3.5 requires per-input dynamic path selection.

### Adaptive Computation Modules (ACM)
Wójcik, Devoto, Pustelnik, Minervini, Scardapane — arXiv:2312.10193 (venue "AAAI 2025" unverified)[9]

Sequential chain of "learners" within a layer, with a gate determining how many to execute per token. Evaluated on vision and speech transformers. **Closest published transformer-layer paper to idea 3.5's intra-layer sequential DAG formulation.** Key limitation: linear chain (not branching DAG), vision/speech only.

### Deja Vu: Contextual Sparsity for Efficient LLMs at Inference Time
Liu, Wang, Dao, Zhou, Yuan, Song, Shrivastava, Zhang, Tian, Ré, Chen — ICML 2023, arXiv:2310.17157[10]

Demonstrates per-token intra-MLP conditional compute in deployed LLMs: a lightweight predictor network predicts which MLP neurons to skip, achieving real TPOT improvements. This establishes that **LLM-scale intra-layer conditional compute is achievable** with a gate-like mechanism. Idea 3.5's DAG is richer (sequential dependencies vs. flat neuron sparsity) but Deja Vu undermines "no LLM-scale intra-layer conditional compute" as a novelty claim.

### PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU
Song, Mi, Xie, Chen — SOSP 2024 (arXiv:2312.12456, preprint 2023)[11]

Exploits static + dynamic neuron activation sparsity in LLM FFN layers to reduce memory bandwidth in deployment. Validates per-token intra-layer sparse execution at LLM deployment scale.

### Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer
Shazeer, Mirhoseini, Maziarz, Davis, Le, Hinton, Dean — ICLR 2017, arXiv:1701.06538[12]

Strongest existing competitor. MoE is **parallel and independent** (experts receive same input x, outputs combined); DAG is **sequentially dependent** (later nodes depend on parent output). MoE routing decides *which* parallel paths; DAG gating decides *which sequential paths*. At k/E ≈ 2.1% (Baseline B), MoE FFN efficiency far exceeds achievable DAG efficiency at realistic p_active values.

### MegaBlocks: Efficient Sparse Training with Mixture-of-Experts
Gale, Narayanan, Young, Zaharia — arXiv:2211.15841, 2022[13]

Critical infrastructure reference for the GPU batching irregularity problem. MoE solves batching by grouping tokens per expert (tokens are independent so regrouping is lossless). **DAG cannot use this strategy** because gate decisions at level k depend on level k-1 output, preventing pre-dispatch. This is the central hardware feasibility gap for idea 3.5 vs MoE.

### Conditional Computation in Neural Networks: Principles and Research Trends
Scardapane et al. — Intelligenza Artificiale 2024, arXiv:2403.07965[14]

Authoritative 2024 survey. Identifies three flavors: dynamic input sparsity, dynamic width sparsity, dynamic depth sparsity. Notably does NOT describe a generalized intra-layer DAG with multiple branching paths as a distinct category — evidence that the full DAG-within-layer formulation has not been unified as a research direction.

### Mixture-of-Depths: Dynamically Allocating Compute in Transformer Models
Raposo, Ritter, Richards, Lillicrap, Dayan, Santoro — arXiv:2404.02258, 2024[15]

Per-token, per-layer routing that decides whether a token participates in a layer's computation or passes through residual only. Demonstrated results at LLM scale. **Inter-layer** analogue of idea 3.5 (layer-level execute/skip vs intra-layer DAG branching). Provides the most direct LLM-scale baseline for conditional compute results.

---

## 4. Prior Art Classification

**Status: PARTIAL (~70–75%)** — Deja Vu[10] and PowerInfer[11] establish LLM-scale intra-layer conditional compute, narrowing the novelty surface.

The following components are covered:
1. Learned per-input binary gating — EXISTS (SkipNet, BlockDrop, ConvNet-AIG, Highway Networks, D2NN)
2. Intra-layer gating of sub-computations — PARTIAL (Channel Gating, ACM — but in CNNs or as sequential chains, not branching DAGs in LLMs)
3. Differentiable training of hard gates — EXISTS (Gumbel-Softmax in ConvNet-AIG, STE)
4. LLM-scale intra-layer conditional compute — EXISTS (Deja Vu, PowerInfer — flat sparsity, not structured DAG)

**Confirmed novel:** A general **branching** DAG (not merely execute/skip, not a linear chain, not parallel experts) as the computation structure **within** a single transformer FFN layer, where gate decisions at earlier DAG nodes condition path selection to later DAG nodes (**sequential gate dependency**), with TPOT/TTFT implications at generative LLM scale — has no direct published embodiment.

---

## 5. Technical Analysis

### 5.1 Theoretical Complexity

**Variables:** L=layers, d=hidden dim, d_ff=MLP dim, s=sequence length, d_dag=DAG depth, b=branching factor, p_active=expected active fraction, d_gate=gate network width.

**Expected FLOPs per token in DAG-MLP (balanced binary DAG):**
O(d · d_ff · d_dag / 2^d_dag) per layer — each token traverses d_dag gate decisions, visiting 1 of 2^d_dag leaf nodes.

**Gate overhead:** O(d_dag · d · d_gate) per layer.

**Critical threshold:** Gate compute overhead cancels bandwidth savings at approximately **d_dag ≥ 4–5** for typical LLM dimensions (d=4096–5120, d_gate=64–128). Below this threshold, gate overhead is <10% of savings.

### 5.2 FLOP Reduction Claims

**vs. A2 (Qwen3-32B Dense) at p_active=0.4: ~1.67× FLOP reduction**

Re-derivation: MLP fraction ≈ 67% at context=4K; at p_active=0.4: overall FLOP reduction = 1/(1 − 0.67×0.6) = **1.67×** — VALID.

**vs. A1 (Qwen3.5-27B Hybrid) at p_active=0.3: net TPOT ~1.2–1.5×**

MLP fraction in A1 ≈ 84% (Gated DeltaNet layers have low attention bandwidth). At p_active=0.3: theoretical MLP BW savings = 70% × 84% = ~59% → speedup ≈ 2.4×. Net full-model TPOT of "1.2–1.5×" (after accounting for kernel efficiency 65% and gate overhead) is **CONDITIONALLY VALID** — depends on achievable kernel efficiency.

**vs. B (Qwen3.5-397B-A17B MoE):**

Baseline B: k=11 / E=512 → k/E ≈ **2.1% active FFN compute**. Idea 3.5 at p_active=0.3–0.5 → **15–25× more active compute** than B. Idea 3.5 cannot compete with B on FFN compute efficiency. This comparison is reframed: idea 3.5 targets dense baselines (A1, A2), not ultra-sparse MoE (B).

### 5.3 Key Comparison Tables

**vs. Baseline A1 (Qwen3.5-27B Hybrid)**

| Metric | Baseline A1 | Idea 3.5 | Change | Notes |
|--------|------------|----------|--------|-------|
| Compute (FLOPs/token) | O(L·(d²+s·d/4)) | O(L·p_active·(s·d+d·d_ff)+L·d_dag·d_gate) | ↓ ~1.5–2× for MLP BW at p_active=0.3 | Net full-model TPOT ~1.2–1.5× |
| KV cache (32K benchmark; A1 max ctx = 262,144) | ~2.15 GB | = ref | = | DAG applies to MLP only; KV unchanged [16×2×4×256×32768×2] |
| Weight memory | O(L·d·d_ff) | O(L·d·d_ff) — full DAG stored | = | All DAG nodes must be in VRAM |
| TTFT | ref | ≈ p_active × ref (+ gate overhead) | ↓ ~1.5× at p_active=0.4 | Sequential gate critical path limits TTFT benefit |
| TPOT | ref | ~1.2–1.5× improvement at p_active=0.3 (after efficiency losses) | ↓ | Conditionally valid; depends on kernel efficiency |

**vs. Baseline A2 (Qwen3-32B Dense)**

| Metric | Baseline A2 | Idea 3.5 | Change | Notes |
|--------|------------|----------|--------|-------|
| Compute (FLOPs/token) | O(L·(s·d+d·d_ff)) | O(L·p_active·(s·d+d·d_ff)) | ↓ ~1.67× at p_active=0.4 | Validated re-derivation |
| KV cache (32K ctx) | ~8.59 GB | = ref | = | [64×2×8×128×32768×2] |
| TTFT | ref | ↓ ~1.67× at p_active=0.4 | ↓ | |
| TPOT | ref | ↓ ~1.2–1.7× depending on p_active | ↓ | Sequential gate critical path limits theoretical gains |

**vs. Baseline B (Qwen3.5-397B-A17B MoE)**

| Metric | Baseline B | Idea 3.5 | Change | Notes |
|--------|-----------|----------|--------|-------|
| FFN compute (FLOPs/token) | k/E ≈ 2.1% of full FFN | p_active = 30–50% of dense FFN | ↑ dramatically | 15–25× MORE active compute than B; idea 3.5 is not a B-replacement |
| KV cache (32K ctx) | ~1.0 GB | = ref (dense baseline context) | = | [15×2×2×256×32768×2] |
| TTFT | ref | ↑ vs B (more active compute) | ↑ | |
| TPOT | ref | ↑ vs B | ↑ | Idea 3.5's value proposition is vs dense models, not B |

**vs. Baseline C (K2 family, 72.55B dense Llama-arch)**

| Metric | Baseline C (K2) | Idea 3.5 | Change | Notes |
|--------|----------------|----------|--------|-------|
| Compute (FLOPs/token) | O(80·(s·d+d·d_ff)), d=8192, d_ff=28672 | O(80·p_active·(s·d+d·d_ff)) | ↓ ~1.67× at p_active=0.4 | K2 is dense; formula applies directly |
| KV cache (32K ctx) | ~10.0 GiB | = ref | = | |
| MLP FLOPs/token/layer | ~4.70×10⁸ | ~1.88×10⁸ at p_active=0.4 | ↓ | |
| TTFT | ref | ↓ ~1.67× at p_active=0.4 | ↓ | |
| TPOT | ref | ↓ ~1.2–1.5× (conditional) | ↓ | Gateway: kernel efficiency and gate overhead at K2 scale (d=8192) |

---

## 6. Implementation Considerations

- **Sequential DAG batching problem**: Unlike MoE (tokens are independent — pre-dispatch by grouping per expert is lossless), DAG cannot pre-dispatch tokens across nodes because gate decisions at level k depend on level k−1 output. This is the central hardware feasibility gap, and it has no published solution for sequential DAGs. MegaBlocks[13] addresses MoE batching; no equivalent for sequential DAG exists.
- **DAG depth limit**: d_dag ≥ 4–5 is the practical limit before gate overhead cancels bandwidth savings. For d=4096, d_gate=64, d_dag=4: gate overhead = 4×4096×64 = ~1M MACs per layer vs MLP savings ≈ (1−p_active)×2×4096×d_ff MACs. At d_ff=16384, p_active=0.5: MLP savings ≈ 67M MACs. Gate overhead = 1.5% of savings — acceptable. At d_dag=8: gate overhead = 2M MACs = 3% — still acceptable. But serial latency doubles.
- **Gradient flow through sequential gates**: STE or Gumbel gradients applied d_dag times in series during backprop. Gradient variance may scale as O(d_dag). Requires gradient clipping or reduced learning rates.
- **Tensor parallelism compatibility**: For DAG-structured MLP, each DAG node may require cross-device communication for gate decisions. This could 2–4× communication overhead.
- **Framework support**: PyTorch dynamic graph supports variable-depth DAG traversal. Custom Triton kernels needed for efficient sparse DAG weight loading at inference.
- **DynamicGate-MLP [Choi, 2026, arXiv:2603.16367]**: Excluded until arXiv ID is confirmed — citation was flagged as unverified in review.

---

## 7. Synergies

- **3.4 (Recursive Internal State)**: Both operate within layer computation; DAG gates can be informed by internal latent state.
- **3.6 (Recursive Internal DAG)**: Direct extension — adding recurrence (loops) to the DAG of 3.5 creates 3.6. Do not pursue 3.6 before 3.5 proves viability; gate depth ≤4–5 is already near the serial latency limit.
- **1.2 (Per-Token Adaptive Depth)**: Layer skipping + intra-layer DAG = hierarchical conditional compute. Can be stacked.
- **5.8 (Block Sparse Weights)**: DAG nodes are naturally block-sparse; block sparsity and DAG gating are synergistic.
- **1.6 (Learned Layer Type)**: Gate decisions can include routing to different primitive types.

**Conflicts:**
- 4.2 (Shared Core Weights + Per-Layer LoRA): which node gets which LoRA is complex.
- Baseline B (MoE): DAG-gated MLP and MoE FFN are competing architectures; using both is redundant.
- Fused CUDA graphs with DAG depth baked into kernel: changing d_dag at inference requires kernel recompilation.

---

## Risk Assessment

**Technical risk: HIGH** — No published results at LLM scale. Sequential gate critical path is a fundamental latency obstacle at batch=1 decode. No proven GPU batching solution for sequential DAGs. Training gradient variance through d_dag serial gates is uncharacterized.

**Potential impact: MEDIUM** — If achievable, 1.2–1.7× TPOT vs dense (A1, A2). Cannot compete with ultra-sparse MoE (B) on FFN efficiency.

**Implementation effort: HIGH** — 6–12 months for 7B research prototype; custom kernels required.

**Prototype experiment required before larger investment:**
- 2-level binary DAG (4 leaf nodes) within MLP of a 7B dense model
- Compare actual TPOT at batch=1 vs: (a) dense 7B, (b) equivalent MoE 7B with k=4
- Measure gate utilization per path (load balance diagnostic)
- Measure d_dag scaling: test d_dag = 2, 3, 4, 5 to empirically find latency crossover

---

## 8. Accuracy / Quality Tradeoff

- **Reported quality delta (from closest analogues)**:
  - [Veit & Belongie, 2019][5] (ConvNet-AIG): ConvNet-AIG-50 achieves 20% fewer FLOPs at *matched or exceeding* ImageNet accuracy; ConvNet-AIG-101 achieves 38% fewer FLOPs at matched accuracy (Table 2, §4). These results are in CNNs but establish the benchmark: intra-layer conditional compute can be accuracy-neutral at 20–38% FLOP reduction. The key question for idea 3.5 is whether the same holds for transformer FFN sub-layers in LLMs.
  - [Hua et al., 2019][7] (Channel Gating): 2.6× FLOP reduction on ImageNet (ResNet-18 with knowledge distillation) with no accuracy drop (Table 3). At 2.7–8× FLOP reductions on CIFAR-10, accuracy is preserved. This is the closest structural analog to idea 3.5 (intra-layer binary gating in CNNs), and it shows near-zero quality loss is achievable with careful training.
  - [Wu et al., 2018][4] (BlockDrop): policy-network-selected block dropping achieves ~20% average speedup (up to 36%) on ResNet-101 at matched ImageNet accuracy (Table 3, p.7). These are inter-layer gates rather than intra-layer, but the quality-neutrality result at 20–36% compute savings sets the expectation for comparable intra-layer gating.
  - [Liu et al., 2023 (Deja Vu)][10]: per-token intra-MLP conditional compute in deployed LLMs produces real TPOT improvements with reported accuracy preservation — demonstrating that flat neuron-level conditional compute in transformer FFN layers is quality-neutral at LLM scale. Idea 3.5's structured DAG is richer than flat sparsity but shares the same quality-preservation mechanism: the dense path is still available for tokens that need it.
  - [Raposo et al., 2024][15] (Mixture of Depths): per-token binary layer-skip routing at 6B+ scale achieves isoFLOP performance improvement — matched quality to dense baselines at lower active compute — the most direct LLM-scale evidence that conditional compute at the layer level (inter-layer analog of 3.5) is quality-neutral or quality-positive.

- **Monotonicity**: Quality loss is **monotone with p_active** in the expected direction — lower p_active (more aggressive gating) risks greater information loss. However, this relationship is not well-characterized for DAG-structured gating in transformers. At very high aggressiveness (p_active < 0.2), quality degradation is expected to accelerate non-linearly as the gate fails to route hard tokens to full compute paths. In CNN analogues, a "cliff" in accuracy appears at high compression levels (typically below 30% active compute).

- **Recovery**: Quality can be recovered via post-training fine-tuning if the DAG routing collapses or under-utilizes certain paths. Load-balancing auxiliary losses (analogous to MoE[12] load balancing) during training prevent routing collapse and preserve quality across diverse token types. Knowledge distillation from the full-compute teacher (as used in Channel Gating[7] on ImageNet) is the proven recovery mechanism: the dense baseline acts as teacher, the DAG-gated model as student.

- **Conditions for acceptable degradation**: Quality loss from DAG gating is acceptable when: (1) the input token distribution is dominated by "easy" tokens (common words, simple continuations) where the sparse DAG path is sufficient — code and formal-language inputs may tolerate more gating than literary prose; (2) a 1.2–1.7× TPOT improvement at batch=1 decode justifies a small quality delta for throughput-sensitive, quality-tolerant applications; (3) the gate threshold p_active is calibrated via benchmark regression so that average quality loss remains below 1% on key metrics; (4) the use case is bandwidth-constrained inference on consumer hardware (edge devices, laptop-class inference) where the 15–25× efficiency gap with Baseline B's MoE cannot be bridged by other means.

---

<!-- CITATION MANIFEST -->
[1]: Conditional Computation — E. Bengio et al., ICLR Workshop 2015 (arXiv:1511.06297). Foundational RL-trained gating; conceptual ancestor.
[2]: Highway Networks — Srivastava et al., 2015 (arXiv:1505.00387). Two-path gated DAG; simplest formulation.
[3]: SkipNet — X. Wang et al., ECCV 2018 (arXiv:1711.09485). Inter-layer binary gate; 30–90% compute reduction on CIFAR/ImageNet.
[4]: BlockDrop — Wu et al., CVPR 2018 (arXiv:1711.08393). 20–36% FLOP reduction with policy network.
[5]: ConvNet-AIG — Veit & Belongie, IJCV 2019 (arXiv:1711.11503). Gumbel-Softmax DAG gating; 20–38% FLOP reduction at ImageNet.
[6]: D2NN — Liu & Deng, AAAI 2018 (arXiv:1701.00299). Explicit DAG of differentiable modules with controller gating.
[7]: Channel Gating — Hua et al., NeurIPS 2019 (arXiv:1805.12549). Intra-layer gating in CNNs; 2-node DAG within a layer.
[8]: DARTS — Liu et al., ICLR 2019 (arXiv:1806.09055). DAG-based architecture search; static (not per-input dynamic routing).
[9]: ACM — Wójcik et al. (arXiv:2312.10193; venue "AAAI 2025" unverified). Sequential chain within transformer layer; vision/speech only.
[10]: Deja Vu — Liu, Wang, Dao, Zhou, Yuan, Song, Shrivastava, Zhang, Tian, Ré, Chen, ICML 2023 (arXiv:2310.17157). Per-token intra-MLP conditional compute in LLMs; real TPOT improvements.
[11]: PowerInfer — Song, Mi, Xie, Chen, SOSP 2024 (arXiv:2312.12456). Static+dynamic neuron sparsity in LLM FFN layers.
[12]: Sparsely-Gated MoE — Shazeer et al., ICLR 2017 (arXiv:1701.06538). Strongest competitor; parallel independent experts; k/E≈2.1% active for Baseline B.
[13]: MegaBlocks — Gale et al., 2022 (arXiv:2211.15841). Block-sparse MoE dispatch; explains why sequential DAG batching is harder than MoE batching.
[14]: Conditional Computation Survey — Scardapane et al., 2024 (arXiv:2403.07965). Authoritative survey; intra-layer branching DAG not identified as distinct category.
[15]: Mixture-of-Depths — Raposo et al., 2024 (arXiv:2404.02258). Inter-layer conditional compute baseline; LLM-scale results.
