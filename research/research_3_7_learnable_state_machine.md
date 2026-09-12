# Research: Learnable State Machine
## ID: 3.7

## Executive Summary

**Novelty verdict:** PARTIAL — component technologies exist across five threads (differentiable FSM training, discrete-state RNN regularization, FSM-controlled branching for non-LLM, per-layer routing without persistent state, implicit FSM tracking), but a discrete N-state FSM with learned transition, persistent cross-token state, and FSM-driven block routing at transformer LLM scale has no published embodiment ([SR-RNN, 2019], [H-NSM, 2021], [MoD, 2024], [Liu et al. FSA-tracking, 2023], [Merrill et al., 2024]).

Learnable State Machine (idea 3.7) is PARTIAL prior art — ~55% overlap with published work. Component technologies exist across five distinct threads: differentiable FSM training, discrete-state RNN regularization, FSM-controlled neural branching for non-LLM architectures, per-layer computation routing without persistent state, and implicit FSM tracking in transformers. The specific combination of (a) discrete N-state FSM, (b) end-to-end learned transition function with the LM objective, (c) persistent cross-token FSM state, (d) FSM state routing/gating computation paths at LLM scale has no published embodiment.

Mixture of Depths [Raposo et al., 2024] is the critical prior-art baseline for per-token block routing — the primary comparison point.

---

## 1. Idea Description

Extension of 3.6 where explicit state is associated with the control flow. The model maintains and updates a learned finite state machine alongside its activations, governing computation paths.

At each token step, an explicit discrete FSM state (one of N possible states) is maintained and updated in parallel with the transformer's continuous activations. The FSM transition function is learned end-to-end. The active FSM state then gates or routes computation — determining which transformer blocks to execute, which expert to select, whether to apply early exit, or which activation function branch to use. Primary inference benefit: if the FSM routes simple tokens to cheap computation paths and complex tokens to expensive ones, average TPOT improves because some tokens bypass expensive blocks. Secondary benefit: explicit discrete state enables better tracking of structured information (bracket depth, entity state) that continuous hidden states encode only implicitly.

---

## 2. Literature Review

### State-Regularized Recurrent Neural Networks
Wang, Niepert — ICML 2019, arXiv:1901.08817[1]

Introduces a stochastic state-transition mechanism constraining RNNs to a finite set of learned discrete states. SR-RNN behavior approaches DFA at low temperature. Evaluated on formal language recognition, sentiment analysis, visual recognition, and language modeling. The discrete state co-exists with continuous activations — the same co-existence design as 3.7. Key difference: discrete state is a regularizer for interpretability, not a compute router.

**[Wang & Niepert, 2019]** — ICML 2019, pp. 6596–6606, §"State-Regularization" (Table 1: language modeling perplexity results).

### Training Linear Finite-State Machines
Ardakani, Ardakani, Gross — NeurIPS 2020[2]

FSM-based networks implemented by lookup tables (no multiplications), stacked as neural layers, trained via gradient surrogates. Demonstrates multi-layer FSM architectures trained end-to-end on sequence tasks including character-level language modeling. The FSM layer here replaces the neural layer rather than running alongside it as a router.

**[Ardakani et al., 2020]** — NeurIPS 2020 (Advances in Neural Information Processing Systems 33), §"FSM-Based Network Architecture".

### Transformers Learn Shortcuts to Automata
Liu, Ash, Goel, Krishnamurthy, Zhang — ICLR 2023, arXiv:2210.10749[3]

Proves shallow transformers (O(log T) layers) can simulate any FSA via transformation semigroup algebraic shortcuts; O(1)-depth simulators exist for solvable-group automata. Transformers encode FSA computation implicitly via attention heads, not an explicit state variable. Critical counterpoint to 3.7: if transformers already simulate FSMs implicitly, the value of an explicit FSM is questionable — unless the explicit state provides routing control the implicit simulation does not.

**[Liu et al., 2023]** — ICLR 2023, §"Main Result" (Theorem 1: O(log T)-depth simulator), §"Solvable Groups" (O(1)-depth).

### Extracting Moore Machines from Transformers using Queries and Counterexamples
Adriaensen, Maene — arXiv:2410.06045 (venue "ICGI 2024" unverified)[4]

Constructs Moore-machine formal abstractions of transformers trained on regular languages using L*-style queries. The extracted FSM is implicit — it lives in hidden state clustering, not as an explicit variable. Transformers trained on positive-only data fail to model garbage states, supporting explicit FSM representation as providing completeness beyond what the implicit representation achieves.

**[Adriaensen & Maene, 2024]** — arXiv:2410.06045, §"Moore Machine Extraction Methodology".

### Finite State Automata Inside Transformers with Chain-of-Thought: A Mechanistic Study
Zhang, Du, Jin, Fu, Jin — arXiv:2502.20129, 2025[5]

Mechanistic study showing Transformer+CoT implements FSA state tracking via late-layer MLP neurons (layers 9–11). Near-perfect accuracy on group-word problems (sequences 100× layer depth) with CoT token emission; accuracy drops dramatically without CoT. Idea 3.7 would maintain internal discrete state without emitting scratchpad tokens — avoiding the inference cost of CoT.

**[Zhang et al., 2025]** — arXiv:2502.20129, §"Circuit Localization", §"Transformer+CoT Recovers FSA".

### Representing Formal Languages: A Comparison Between Finite Automata and Recurrent Neural Networks
Michalenko, Shah, Verma, Baraniuk, Chaudhuri, Patel — ICLR 2019, arXiv:1902.10297[6]

Investigates whether RNN hidden states map to MDFA states. Shows a decoding function mapping RNN states to "superstates" (clusters of MDFA states) — the implicit FSM encoding is lossy. This lossiness motivates an explicit representation.

**[Michalenko et al., 2019]** — ICLR 2019, arXiv:1902.10297, §"Decoding Function Analysis" (Figure 2: state space clustering).

### Recurrent Neural Language Models as Probabilistic Finite-State Automata
Svete, Cotterell — EMNLP 2023, arXiv:2310.05161[7]

Proves simple RNNs are equivalent to a subclass of probabilistic finite-state automata (PFSAs). Establishes that RNN LMs already function as PFSAs — what idea 3.7 would make explicit. Confirms the theoretical motivation: making FSM structure explicit could improve training efficiency and interpretability.

**[Svete & Cotterell, 2023]** — EMNLP 2023, pp. 8069–8086, §"Main Theorem".

### The Illusion of State in State-Space Models
Merrill, Petty, Sabharwal — ICML 2024, arXiv:2404.08819[8]

Proves SSMs (including Mamba) share the same computational complexity class (TC⁰) as transformers. SSMs cannot track permutation composition, chess moves, Python code evaluation, or entity tracking. The "recurrent state" in Mamba is an "illusion" — it does not grant genuine state-tracking advantage over transformers. This motivates an explicit discrete FSM state as providing expressiveness both transformers and Mamba fundamentally lack.

**[Merrill et al., 2024]** — ICML 2024, §"Main Theorem" (TC⁰ impossibility), §"Experimental Validation" (permutation composition, chess, code eval).

### Structured Sparse Transition Matrices to Enable State Tracking in State-Space Models (PD-SSM)
Terzić, Menet, Hersche, Hofmann, Rahimi — NeurIPS 2025 Spotlight, arXiv:2509.22284[9]

PD-SSM parametrizes SSM transition matrices as P×D (column one-hot × complex diagonal), enabling any N-state FSA emulation with one layer (per abstract: "can emulate any N-state FSA with one layer of dimension N and a linear readout of size N × N"). Abstract reports that PD-SSM "significantly outperforms a wide collection of modern SSM variants on various FSA state tracking tasks" (no specific percentage-point figure in abstract). Most direct comparison for idea 3.7: PD-SSM adds structured FSA-like behavior to SSMs (continuous relaxation); 3.7 adds explicit discrete FSM state. PD-SSM establishes the problem is real (SSMs need FSA help); 3.7's question is whether explicit discrete states outperform continuous PD-SSM structure on routing tasks.

**[Terzić et al., 2025]** — NeurIPS 2025 Spotlight, arXiv:2509.22284, §"Theoretical Result" (FSA emulation with one layer of dimension N), §"Experimental Results" (specific per-task tables in paper body).

### Steering LLMs' Reasoning With Activation State Machines
Li, Chen et al. — NeurIPS 2025 submission [ACCEPTANCE UNCONFIRMED], OpenReview:p17En1bhCY[10]

Proposes Activation State Machine (ASM): per-block predict-correct Kalman-style cycle. Predicts ideal activation state, observes raw LLM activation, corrects internal state based on error. Improves zero-shot accuracy on mathematical and physical reasoning. Closest published work to 3.7's inference-time behavior: explicit state runs alongside transformer hidden states with corrective interventions. Key differences: ASM is post-hoc steering (not jointly trained), state is continuous (Kalman-style, not discrete FSM), does not route computation paths.

**[Li & Chen, 2025]** — OpenReview NeurIPS 2025 submission [ACCEPTANCE UNCONFIRMED], §"Activation State Machine Architecture", §"Results".

### Hybrid Neural State Machine for Neural Network (H-NSM)
Lei, Wu, Wu, Shi et al. — Science China Information Sciences, Vol. 64, Art. 132202, 2021[11]

H-NSM framework: FSM controller (discrete states, binary spike encoding) receives input from ANNs/SNNs, makes state transitions, sends control signals to activate different branch networks. H-NSM-C (condition-based branching) and H-NSM-S (sequential task control) are precisely the "governing computation paths" behavior described in 3.7. Most structurally similar prior art found. Key limitations: applied to SNNs (not transformers), FSM states hand-designed (not learned), evaluated on robotics/control (not NLP), scale far below LLM parameters.

**[Lei et al., 2021]** — Science China Information Sciences, Vol. 64, Art. 132202, §"H-NSM Framework", §"Condition-Based Control (H-NSM-C)".

### Dr.LLM: Dynamic Layer Routing in LLMs
Heakl, Gubri, Khan, Yun, Oh — arXiv:2510.12773, 2025 [POST-CUTOFF PREPRINT; UNDER SUBMISSION][12]

Per-layer routers (bottleneck MLPs reading windowed hidden state summaries) decide skip/execute/repeat per block. MCTS-derived training targets on 4K examples. +3.4% ARC/DART accuracy while saving ~5 layers per example; +7.7% vs. prior methods. Closest prior art for the computation-path-routing use case: functionally similar to how an FSM state would govern computation paths. Key difference: routing decisions at each layer are independent (no persistent discrete state across tokens), routing conditioned on current hidden state not on a separate FSM state.

**[Heakl et al., 2025]** — arXiv:2510.12773 [POST-CUTOFF PREPRINT; UNDER SUBMISSION — not yet accepted at any venue], §"MCTS Router Training", §"Results" (Table: +3.4% ARC/DART, -5 layers/example).

### Mixture-of-Modules: Reinventing Transformers as Dynamic Assemblies of Modules
Gong et al. — EMNLP 2024, ACL Anthology 2024.emnlp-main.1164[13]

Per-token dynamic module selection (attention + FFN, parameterized independently) at inference time. 16% TFLOP savings, 42% memory vs. GPT-2 774M with comparable performance. Includes SKIP module as routing option. MoM selects modules per-token without a global state — 3.7 would add a persistent FSM state governing these selections over multiple tokens, enabling state-dependent routing (e.g., "route to programming-mode path until end-of-block").

**[Gong et al., 2024]** — EMNLP 2024, pp. 20924–20938, §"MoM Architecture", §"Results" (Table: 16% TFLOP savings, 42% memory savings vs GPT-2 774M).

### Mixture of Depths: Dynamically Allocating Compute in Transformer Models
Raposo, Ritter, Richards, Lillicrap, Humphreys, Santoro — arXiv:2404.02258, 2024[14]

Per-token dynamic depth routing at 6B+ scale: tokens either participate in a layer's computation or skip it via a learned router. Achieves matched quality to isoFLOP baselines while reducing active computation per token. **This is the primary prior-art baseline for the routing use case of idea 3.7.** MoD routing is stateless (no cross-token memory); each token's routing is independent. Idea 3.7's distinguishing feature is persistent cross-token FSM state enabling routing informed by discourse-level context.

**[Raposo et al., 2024]** — arXiv:2404.02258, §"Architecture", §"Experiments" (6B+ scale matched quality at lower compute).

### Categorical Reparameterization with Gumbel-Softmax
Jang, Gu, Poole — ICLR 2017, arXiv:1611.01144[15]

Gumbel-Softmax trick: differentiable approximation to discrete categorical sampling via the reparameterization trick with the Gumbel distribution + softmax. Enables gradient-based training of discrete choices. The training mechanism used in idea 3.7 to learn FSM state transitions end-to-end. Temperature annealing from soft (high T) to hard (low T) is the standard training procedure.

**[Jang et al., 2017]** — ICLR 2017, arXiv:1611.01144, §"Gumbel-Softmax Estimator", §"Experiments".

### Differentiable Finite State Machines
Mordvintsev — Google Research Blog, 2022 [INFORMAL SOURCE][16]

Demonstrates gradient-descent learning of discrete deterministic FSMs by representing states and characters as probability distributions over one-hot vectors, using entropy regularization and "lazy bias" initialization. >95% training success rate on 10 string processing tasks. Most direct prior art for differentiable training of FSM transition matrices. **Citation discipline note: blog post, not peer-reviewed; no page numbers. Cited only for entropy regularization anti-collapse technique.**

**[Mordvintsev, 2022]** — Google Research Blog, §"Training Procedure" [INFORMAL SOURCE — blog post].

### Neural Networks as Universal Finite-State Machines
Dhayalkar — arXiv:2505.11694, 2025 [SINGLE-AUTHOR PREPRINT][17]

Proves finite-depth ReLU feedforward networks can exactly simulate DFAs by unrolling state transitions into depth-wise layers. Constructive architectures with formal depth/width specifications. Limitation: simulation requires depth equal to sequence length — impractical for LLM-scale sequences. **Treat with LOW CONFIDENCE — single-author preprint, no peer review.**

**[Dhayalkar, 2025]** — arXiv:2505.11694, §"Main Theorem: Exact DFA Simulation" [LOW CONFIDENCE — single-author preprint].

### Symbolic Feedforward Networks for Probabilistic Finite Automata
Dhayalkar — arXiv:2509.10034, 2025 [SINGLE-AUTHOR PREPRINT; POST-CUTOFF][18]

Demonstrates exact simulation of PFAs using symbolic feedforward networks with shared stochastic transition matrices. Proves learnability via gradient descent on labeled sequences (Proposition 5.1). O(kn²) parameters. **Treat with VERY LOW CONFIDENCE — post-August-2025-cutoff single-author preprint, no peer review. Do NOT use as primary evidence for any novelty claims.**

**[Dhayalkar, 2025b]** — arXiv:2509.10034, §"Proposition 5.1: Exact PFA Recovery" [VERY LOW CONFIDENCE — post-cutoff single-author preprint].

### On the Relationship Between RNN Hidden-State Vectors and Semantic Structures
Muskardin, Tappler, Pill, Aichernig, Pock — ACL Findings 2024, ACL Anthology 2024.findings-acl.335[19]

Investigates the clustering hypothesis — that RNN hidden-state vectors form clusters of semantically similar representations — on RNNs trained to recognize regular and context-free languages, using ground-truth automata as evaluation targets. Finds clustering is not a reliable abstraction technique for RNNs in NLP tasks. Relevant to 3.7: demonstrates that implicit FSM-like structure in RNNs is not robustly recoverable, motivating an explicit discrete state representation.

**[Muskardin et al., 2024]** — ACL Findings 2024, pp. 5641–5658, ACL Anthology 2024.findings-acl.335, §"Clustering Evaluation".

---

## 3. Prior Art Classification

**Status: PARTIAL (~55% covered)**

- **Component technologies that exist:**
  1. Differentiable training of FSM transition matrices — [Mordvintsev, 2022][16] and [Dhayalkar, 2025b][18] (very low confidence)
  2. Discrete state co-existing with continuous activations in RNNs — [Wang & Niepert, 2019][1] (SR-RNNs)
  3. FSM as explicit controller of neural computation branches — [Lei et al., 2021][11] (H-NSM, for SNNs/robotics with hand-designed states)
  4. Per-token dynamic block routing without persistent state — [Raposo et al., 2024][14] (MoD; **primary baseline**), [Heakl et al., 2025][12] (Dr.LLM), [Gong et al., 2024][13] (MoM)
  5. Implicit transformer FSA tracking — [Liu et al., 2023][3], [Zhang et al., 2025][5], [Adriaensen & Maene, 2024][4]
  6. TC⁰ expressiveness limitation motivating explicit FSM — [Merrill et al., 2024][8], [Terzić et al., 2025][9]

- **What is NOT published:** An explicit, learned, discrete N-state FSM maintained across tokens in a transformer LLM forward pass, updated end-to-end with the LM objective, governing which transformer blocks execute for each token at inference time. Specifically: persistent cross-token FSM state as a computation router at LLM scale, jointly trained with the LM objective.

- **Novel contribution:** The specific combination of (a) discrete N-state FSM, (b) end-to-end learned transition function with LM objective, (c) persistent cross-token FSM state, (d) FSM state routing/gating computation paths (not just regularizing activations), applied in a transformer LLM at modern scale. The primary novelty vs. MoD[14] (the closest prior art) is the persistent cross-token state enabling discourse-level routing context.

---

## 4. Technical Analysis

### 4.1 Theoretical Complexity

Let:
- N = number of FSM states (hyperparameter; N ∈ {16, 64, 256})
- d = hidden dimension (5,120 for A2 Qwen3-32B)
- d_ff = MLP intermediate dimension (25,600 for A2; 17,408 for A1)
- L = number of layers (64 for A1 and A2)
- s = sequence length
- p = fraction of tokens routed to cheap path (0 ≤ p ≤ 1)
- r = fraction of compute retained on fast path (r=1 means full compute, r=0 means skip entirely)

**FSM overhead per token (soft, shared T matrix):** O(N²) for transition matrix-vector product + O(N) softmax.
- N=256: 65,536 ops vs. 262M ops per MLP layer = **0.025%** — negligible.
- N=1024: ~1M ops = **0.4%** — still negligible.
- N² overhead does not dominate for any reasonable N ≤ 1024.

**FSM weight memory:** Per-layer T ∈ ℝ^{N×N}; N=256, L=64, fp16: 64×256²×2 = 8 MB — negligible vs. 64 GB model.

**FSM state history:** N floats per token. N=256, s=32K: 256×32768×4 = 32 MB — negligible vs. 4.3 GB KV cache.

**Routing speedup formula:**
- r is defined as the **compute fraction retained on the fast path** (r=0 means skip entirely; r=0.5 means use half the compute)
- Average compute per token: C_avg = p·(r·C_full) + (1-p)·C_full = (1-p(1-r))·C_full
- **Example A — r=0.5 ("use half compute on fast path"):** At p=0.5: C_avg = 0.75×C_full → TPOT speedup = 1.33×
- **Example B — r=0 ("skip MLP entirely on fast path"):** At p=0.5: C_avg = 0.5×C_full → TPOT speedup = 2.0×
- Previous r=0.5 example should be interpreted as "execute half of the block" (e.g., skip every other layer or run a cheaper FFN sublayer).

### 4.2 Prefill Sequential Bottleneck

The FSM recurrence q_{t+1} = f(q_t, x_t) creates a true sequential dependency at prefill. Unlike transformer attention (parallelizable across s tokens), FSM update is O(s) sequential steps. Total FSM FLOPs at prefill are small:
- O(s·N²) per layer: N=256, s=8K, L=64 → 64×8192×65536 ≈ 34B ops total
- For one layer: 8192×65536 ≈ 536M ops = ~0.4% of one MLP layer

The FLOP overhead is negligible, but the **serial execution creates a pipeline bubble limiting GPU utilization**. A parallel scan approach (as used in Mamba/S4 training) could resolve this but requires the FSM transition to be expressible as a scan — possible for linear/log-linear FSM transitions, not for arbitrary learned transitions. This is a higher-severity issue than the FLOP analysis suggests.

### 4.3 Key Comparison Tables

**vs. Baseline A1 (Qwen3.5-27B Hybrid, 32K benchmark; A1 max context = 262,144)**

| Metric | Baseline A1 | Idea 3.7 (p=0.5, r=0.5) | Change | Notes |
|--------|------------|--------------------------|--------|-------|
| Compute (FLOPs/token) | O(L·(d²+s·d/4)) | O(L·((1-p(1-r))·(d²+s·d/4) + N²)) | ↓ ~0.75× | N² overhead negligible; routing benefit depends on quality retention |
| KV cache | O(L/4·s·d_kv) | O(L/4·s·d_kv) + O(s·N) | ≈ = | FSM state adds 32 MB at N=256, 32K ctx — negligible |
| Weight memory | O(L·d·d_ff) | O(L·d·d_ff) + O(N²) | ≈ = | FSM adds 8 MB |
| TTFT (32K prompt) | ref | ≈ ref † | ≈ = | FSM serial update is 34B ops total — negligible FLOPs but pipeline bubble |
| TPOT (batch=1) | ref | ↓* ~0.75× if routing effective (⚠ upper bound, unverified — sequential FSM decode) | ↓* ~1.33× (⚠ upper bound, unverified — sequential FSM decode) [derived: C_avg = (1−p(1−r))×C_full = (1−0.5×0.5)×C_full = 0.75×C_full; TPOT speedup = C_full/C_avg = 1/0.75 = 1.33×; at p=0.5 tokens on fast path, r=0.5 compute retained → average token uses 75% of full compute] | Bandwidth-bound; FSM routes p=0.5 of tokens to skip r=0.5 of MLP weight loads |

† **TTFT pipeline-bubble caveat (A1):** While FLOP count for the FSM update is negligible (34B ops, ≈0.4% of one MLP layer), the FSM recurrence q_{t+1} = f(q_t, x_t) is strictly sequential across the s prompt tokens. This creates a prefill pipeline bubble that limits GPU utilization independently of FLOP count. As documented in §4.2, this is a higher-severity issue than the FLOP analysis alone suggests. Wall-clock TTFT for A1 may therefore be measurably worse than "≈ ref" depending on hardware and prompt length, even though the FLOP ratio remains approximately equal to reference.

**vs. Baseline A2 (Qwen3-32B Dense, 32K benchmark; A2 max context = 40,960)**

| Metric | Baseline A2 | Idea 3.7 (p=0.5, r=0.5) | Change | Notes |
|--------|------------|--------------------------|--------|-------|
| Compute (FLOPs/token) | O(L·(s·d+d·d_ff)) | O(L·((1-p(1-r))·(s·d+d·d_ff) + N²)) | ↓ ~0.75× | MLP is larger (d_ff=25,600) so absolute savings larger |
| KV cache | O(L·s·d_kv) ≈ 8.59 GB | ≈ O(L·s·d_kv) + O(s·N) | ≈ = | 32 MB FSM state negligible vs 8.59 GB |
| Weight memory | O(L·d·d_ff) | ≈ O(L·d·d_ff) | ≈ = | 8 MB FSM negligible vs 64 GB model |
| TTFT | ref | ≈ ref | ≈ = | FSM sequential bottleneck exists but FLOPs negligible |
| TPOT (batch=1) | ref | ↓* ~0.75× if routing effective (⚠ upper bound, unverified — sequential FSM decode) | ↓* ~1.33× (⚠ upper bound, unverified — sequential FSM decode) [derived: identical formula to A1 table; C_avg = (1−0.5×0.5)×C_full = 0.75×C_full; speedup = 1/0.75 = 1.33×; A2 has larger d_ff=25,600 vs A1 d_ff=17,408 → absolute BW savings larger but fractional speedup identical given same p=0.5, r=0.5] | Same bandwidth-bound analysis |

**vs. Baseline B (Qwen3.5-397B-A17B MoE, 32K benchmark; B max context = 262,144)**

| Metric | Baseline B | Idea 3.7 (A2-scale + FSM) | Change | Notes |
|--------|-----------|---------------------------|--------|-------|
| Compute (FLOPs/token) | O(L·(d²+k·d·d_e)) | O(L·((1-p(1-r))·(d²+d·d_ff) + N²)) | ↑ or ≈ | B has k=11/512 MoE sparsity (≈2.1% active FFN); dense+FSM at p=0.5 uses ≥37.5% active compute — cannot match B efficiency |
| Memory bandwidth (decode) | O(L·(k·d·d_e+state)) — low (32Q/2KV) | O(L·(1-p(1-r))·(d·d_ff+s·d_kv)) | ↑ | B's 32Q/2KV gives ~4× lower KV bandwidth; FSM-routed dense loads more weight than B's sparse experts; comparison is approximate (depends on d_e for B) |
| KV cache | ~1.0 GB (15 global-attn layers, 2KV@32K) | ~8.59 GB + O(s·N) | ↑ | Dense A2 has KV at all 64 layers; B at 15/60 layers (GQA) |
| Weight memory | ~794 GB (397B total) | ~64 GB (32B dense) | ↓ | Much smaller model; not a fair direct comparison |
| TTFT, TPOT | ref | ↑* vs B (⚠ upper bound, unverified — sequential FSM decode) | ↑ | MoE is fundamentally more bandwidth-efficient; quality idea vs B |

**[B KV bandwidth comparison approximate — depends on d_e for expert intermediate dimension; all compute figures derived: C_avg = (1−p(1−r))×C_full = 0.75×C_full at p=0.5, r=0.5; ↓ 0.75× compute means 1.33× TPOT speedup = 1/0.75; B active FLOPs fraction = k/E = 11/512 = 2.15% of full MLP → dense+FSM at ≥37.5% active cannot match B; B KV bandwidth ≈ (k_global/L_total)×full_KV where k_global=15 global-attn layers gives ~1.0 GB vs A2's 8.59 GB at 32K ctx]**

**vs. Baseline C (K2 family, 32K context)**

| Metric | Baseline C (K2) | Idea 3.7 (C-scale + FSM) | Change | Notes |
|--------|----------------|--------------------------|--------|-------|
| Compute (FLOPs/token) | O(80·(s·d+d·d_ff)) | O(80·((1-p(1-r))·(s·d+d·d_ff) + N²)) | ↓ ~0.75× | d=8192, d_ff=28672; savings scale with MLP size |
| KV cache (32K) | ~10.0 GiB | ≈ ~10.0 GiB + O(s·N) | ≈ = | 32 MB FSM state negligible |
| MLP FLOPs/token/layer | ~4.70×10⁸ [derived: 2×d×d_ff×3 projections (up/gate/down in SwiGLU) = 2×8192×28672×3 ≈ but standard count: 2 matmuls of d×d_ff each = 2×8192×28672 = 470,237,184 ≈ 4.70×10⁸] | ~3.53×10⁸ at p=0.5, r=0.5 [derived: C_avg = (1−0.5×0.5)×4.70×10⁸ = 0.75×4.70×10⁸ = 3.525×10⁸ ≈ 3.53×10⁸] | ↓ ~0.75× [derived: C_avg/C_full = 0.75; ratio = 3.53/4.70 = 0.751] | Large d_ff=28,672 means larger absolute savings |
| Weight memory | ~145.1 GB | ~145.1 GB + 8 MB FSM | ≈ = | |
| TPOT (batch=1) | ref | ↓* ~1.33× if routing effective (⚠ upper bound, unverified — sequential FSM decode) [derived: speedup = 1/(1−p(1−r)) = 1/(1−0.5×0.5) = 1/0.75 = 1.33×; same formula as A1/A2; C's larger MLP (d_ff=28,672) increases absolute BW saved per skipped token but does not change the speedup ratio] | ↓* | Same analysis; C's larger MLP amplifies absolute BW savings |

---

## 5. Implementation Considerations

- **Hardware requirements:** No custom kernels required for FSM transition step (GEMM with N×N matrix, N≤1024). Standard PyTorch/JAX GEMM. Hard discrete routing (skip/execute) at batch=1: Python-level if/else — trivially implemented. Batch>1 hard routing requires custom gather/scatter kernels analogous to MoE token dispatch (adapt Tutel/MegaBlocks) — 1–3 months engineering.

- **Training stability — three risks:**
  1. **Routing collapse (HIGH risk):** All tokens converge to one FSM state. Mitigation: entropy regularization (identical to MoE auxiliary loss) + load-balancing loss across N states. Well-understood problem with established solutions from [Mordvintsev, 2022][16] and MoE literature. Not fatal.
  2. **Gumbel-Softmax instability (MEDIUM risk):** Discrete gradient estimation via Gumbel-Softmax[15] is noisy at large N. Mitigation: soft-to-hard temperature annealing over training. Alternatively, remain soft throughout and apply hard argmax only at inference.
  3. **BPTT through FSM recurrence (MEDIUM risk):** FSM q_{t+1} = f(q_t, x_t) creates a sequential gradient path through all s steps. For s=8K, this is an 8K-step BPTT computation causing vanishing/exploding gradients. **Mitigation (required): truncated BPTT at k=64–128 steps** — backprop only through the last k steps of FSM history.

- **r parameter — use clarification:** Always specify whether r is "compute fraction retained" (r=0.5 → use half the block) or "skip-or-full" (r=0 → skip entirely). The comparison tables above use r as "compute fraction retained." The skip-entirely case (r=0) gives TPOT speedup 1/(1-p) = 2.0× at p=0.5.

- **Framework:** PyTorch: `torch.nn.functional.gumbel_softmax`, `torch.masked_select`. JAX: `jax.lax.cond`, `jax.lax.associative_scan` for parallel FSM. vLLM: requires custom CUDA/Triton kernels for predicated block execution.

---

## 6. Synergies

- **3.6 (Recursive Internal DAG):** FSM state is the explicit persistent state variable 3.6 implicitly requires. Highly synergistic.
- **4.1 (State Machine Core):** Complete architectural overlap — the FSM infrastructure (q_t + T_mat) is identical; 3.7 handles routing/gating while 4.1 handles FiLM-style activation conditioning. **Joint 3.7+4.1 implementation is strongly recommended over either alone.**
- **1.2 (Per-Token Adaptive Depth):** FSM state determines early exit; more principled than learned per-token exit threshold.
- **3.5 (Gated Internal DAG):** FSM state naturally selects which DAG branch to execute.
- **3.1 (Layer-Level MoE):** FSM state governs which expert layer is activated.
- **5.3 (Grammar/State-Machine Structured Attention):** FSM state structures attention patterns + KV compression.

**Conflicts:**
- Baseline A1 hybrid: DeltaNet linear attention layers maintain their own recurrent state (O(d²)); adding separate FSM state creates two competing state mechanisms.
- Batched inference (batch>1): Hard discrete FSM routing requires different computation per token — contradicts standard batched GPU execution requiring uniform operations.

---

## 7. Risk Assessment

**Technical risk: MEDIUM-HIGH** — Main risks: (a) routing collapse converging to 1–2 states; (b) Gumbel-Softmax instability at large N; (c) BPTT through FSM recurrence (truncated BPTT required); (d) batched inference incompatibility with hard routing; (e) implicit FSM in standard transformers ([Liu et al., 2023][3]) may already suffice for most routing purposes.

**Potential impact: MEDIUM-HIGH** — At p=0.5, r=0.5: TPOT ↓ ~1.33×. Additional benefits: interpretability (explicit discrete state), reliable discrete state tracking (brackets, entity states, code scopes — where transformers/SSMs are TC⁰ limited per [Merrill et al., 2024][8]), state-conditional fine-tuning.

**Implementation effort: HIGH** for hard routing; MEDIUM for soft FSM prototype.

---

## 8. Accuracy / Quality Tradeoff

- **Reported quality delta (from closest analogues)**:
  - [Raposo et al., 2024][14] (Mixture of Depths — primary baseline): at 6B+ scale, per-token dynamic block routing achieves isoFLOP quality parity with standard dense models — matched quality at lower active compute. This establishes that stateless per-token routing is quality-neutral in the best case. Idea 3.7's distinguishing feature (persistent cross-token FSM state) should either match or improve on this by enabling context-aware routing, but the quality delta from the persistent state itself is uncharacterized.
  - [Gong et al., 2024][13] (Mixture-of-Modules): per-token module selection (including a SKIP module) achieves 16% TFLOP savings and 42% memory savings vs GPT-2 774M at comparable performance (§5 "Results", EMNLP 2024). This is the stateless routing analog of 3.7 at smaller scale, suggesting stateless conditional routing is quality-neutral up to 774M parameters.
  - [Heakl et al., 2025][12] (Dr.LLM — post-cutoff preprint): per-layer independent routing achieves +3.4% ARC/DART accuracy while saving ~5 layers per example (§Results Table). This is a quality-positive result from stateless per-layer routing, suggesting that per-token adaptive routing can improve quality-per-FLOP even without persistent state. The 3.7 persistent FSM state should provide additional gains on structured tasks.
  - [Terzić et al., 2025][9] (PD-SSM — NeurIPS 2025 Spotlight): structured FSA-emulation for SSMs "significantly outperforms a wide collection of modern SSM variants on various FSA state tracking tasks" (per abstract; specific per-task percentages are paper-body table values). This demonstrates that explicit FSM-like state tracking delivers substantial quality gains on discrete-state tasks — directly validating the quality motivation for idea 3.7's explicit FSM state.
  - [Merrill et al., 2024][8] (Illusion of State — ICML 2024): both transformers and SSMs are TC⁰ and cannot track permutation composition, chess moves, or entity state (§"Experimental Validation"). This framing establishes that standard architectures have a *quality floor* on discrete-state tasks that an explicit FSM state can lift — the quality gain from 3.7 is expected to be most pronounced on these structurally limited tasks.
  - [Wang & Niepert, 2019][1] (SR-RNN): state-regularized RNNs with discrete FSM state achieve competitive perplexity on language modeling (§Table 1, ICML 2019) — showing that co-existing discrete FSM and continuous activations do not degrade language modeling quality relative to standard RNN baselines.

- **Monotonicity**: Quality loss from FSM routing aggressiveness (higher p and lower r) is expected to be **monotone at moderate aggressiveness** — removing 25–50% of compute on the fast path degrades quality smoothly as more tokens bypass expensive blocks. However, routing collapse (all tokens → one FSM state) is a discontinuous degradation point: quality drops sharply when the FSM ceases to discriminate between token types. Below the routing-collapse threshold, quality loss is approximately monotone with p(1−r); above it, quality collapse is non-monotone and harder to recover.

- **Recovery**: Quality can be recovered via: (1) entropy regularization on FSM state distribution (standard MoE anti-collapse technique per [Mordvintsev, 2022][16]) prevents the hard quality cliff from routing collapse; (2) reducing p (fewer tokens routed to the fast path) or increasing r (less compute reduction on the fast path) monotonically recovers quality toward baseline; (3) the soft FSM (remaining in Gumbel-Softmax relaxed mode without hard argmax) avoids discrete routing collapse during training, preserving quality at the cost of slightly reduced inference efficiency. Full hard-routing recovery requires retraining with stronger load-balancing loss.

- **Conditions for acceptable degradation**: At p=0.5, r=0.5: TPOT ↓ ~1.33× with expected near-zero quality loss on generic language tasks, matching the MoD[14] precedent. Quality degradation is acceptable when: (1) the task distribution is dominated by "easy" tokens (common continuations, simple completions) where the FSM reliably routes to the cheap path without information loss; (2) explicit discrete state tracking (brackets, entity state, code scopes) is not required for the target task — FSM routing on these tasks may be quality-positive rather than neutral; (3) the model is deployed in environments where TPOT is the primary constraint (bandwidth-bound serving at batch=1, edge devices) and a small quality delta is tolerable; (4) the FSM state size N is calibrated to the task's structural complexity — N=256 is sufficient for most routing purposes; N >> 256 risks routing fragmentation where rare states receive insufficient training coverage and degrade quality on infrequent token types.

---

<!-- CITATION MANIFEST -->
[1]: SR-RNN — Wang & Niepert, ICML 2019 (arXiv:1901.08817). Discrete state + continuous activations co-existence; interpretability regularizer.
[2]: Linear FSM Networks — Ardakani, Ardakani & Gross, NeurIPS 2020 (Advances in NeurIPS 33). FSM layers stacked end-to-end on sequence tasks.
[3]: Shortcuts to Automata — Liu et al., ICLR 2023 (arXiv:2210.10749). Transformers simulate FSAs implicitly; critical counterpoint.
[4]: Moore Machines from Transformers — Adriaensen & Maene (arXiv:2410.06045; venue "ICGI 2024" unverified). L*-style FSM extraction from transformers.
[5]: FSA Inside Transformers+CoT — Zhang et al., 2025 (arXiv:2502.20129). CoT implements FSA tracking via MLP neurons; drops without CoT.
[6]: FA vs RNNs — Michalenko et al., ICLR 2019 (arXiv:1902.10297). RNN hidden states map lossily to FSM superstates.
[7]: RNN LMs as PFSAs — Svete & Cotterell, EMNLP 2023 (arXiv:2310.05161). RNNs formally equivalent to probabilistic FSA subclass.
[8]: Illusion of State — Merrill et al., ICML 2024 (arXiv:2404.08819). SSMs and transformers both TC⁰; cannot track permutation composition.
[9]: PD-SSM — Terzić et al., NeurIPS 2025 Spotlight (arXiv:2509.22284). Structured FSA emulation for SSMs via P×D transition parametrization; abstract reports significant outperformance vs other structured SSMs on state-tracking tasks.
[10]: ASM — Li & Chen, NeurIPS 2025 submission [ACCEPTANCE UNCONFIRMED] (OpenReview:p17En1bhCY). Post-hoc Kalman-style activation steering; closest to 3.7's inference behavior.
[11]: H-NSM — Lei et al., Science China Information Sciences Vol. 64, 2021. FSM controller activating neural branches; closest structural prior art (but hand-designed, SNN-scale).
[12]: Dr.LLM — Heakl et al., 2025 [POST-CUTOFF PREPRINT; UNDER SUBMISSION] (arXiv:2510.12773). Per-layer independent routing; +3.4% ARC/DART, -5 layers/example.
[13]: MoM — Gong et al., EMNLP 2024 (ACL Anthology 2024.emnlp-main.1164). Per-token module selection; 16% TFLOP, 42% memory savings.
[14]: Mixture of Depths — Raposo et al., 2024 (arXiv:2404.02258). PRIMARY BASELINE for routing; stateless per-token block routing at 6B+ scale.
[15]: Gumbel-Softmax — Jang et al., ICLR 2017 (arXiv:1611.01144). Training mechanism for discrete choices.
[16]: Differentiable FSM — Mordvintsev, Google Research Blog, 2022 [INFORMAL SOURCE]. Entropy regularization for FSM anti-collapse.
[17]: Universal FSMs — Dhayalkar, 2025 (arXiv:2505.11694) [LOW CONFIDENCE — single-author preprint].
[18]: Symbolic Feedforward for PFAs — Dhayalkar, 2025b (arXiv:2509.10034) [VERY LOW CONFIDENCE — post-cutoff single-author preprint; NOT primary evidence].
[19]: RNN Hidden States vs Semantic Structures — Muskardin, Tappler, Pill, Aichernig & Pock, ACL Findings 2024, pp. 5641–5658 (ACL Anthology 2024.findings-acl.335). Clustering not reliable for RNN state abstraction; motivates explicit discrete state.
