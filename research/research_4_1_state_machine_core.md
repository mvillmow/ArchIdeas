# Research: State Machine Core
## ID: 4.1

## Executive Summary

**Novelty verdict:** PARTIAL — FSA-structured SSMs (PD-SSM, SD-SSM) and FiLM-style conditioning are published, but a discrete N-state FSM maintained as a global variable across the forward pass with learned transitions and FiLM scale/shift conditioning applied to ALL layers of a transformer LLM, evaluated at scale on NLP benchmarks, is unpublished ([Merrill et al., 2024], [PD-SSM, 2024], [SD-SSM, 2024], [FiLM, 2018], [Raposo et al. MoD, 2024]).

Idea 4.1 proposes a learned discrete FSM state (N soft-categorical states, N=16–256) maintained as a global variable across the full transformer forward pass, updated at each token step via a learned transition function, and applied as FiLM-style (scale/shift) universal conditioning to all L layers. The theoretical motivation is strong: Merrill et al. (2024) prove that continuous recurrent states (Mamba, RWKV, Griffin) cannot reliably track discrete symbolic state due to TC⁰ complexity constraints, while FSA-structured models (PD-SSM, SD-SSM) achieve near-perfect FSA emulation where SSMs fail. Overhead is acceptable at N=256: ~1.14% TPOT (A1) [derived: 168M / 14.7B ≈ 1.14%], ~2.0% TPOT (A2) [derived: 168M / 8.4B ≈ 2.0%], ~336 MB additional weight bandwidth, ~170M additional parameters (~0.63% of A1, ~0.53% of A2).

Verdict: INVESTIGATE FURTHER. The ideas 4.1 and 3.7 are complementary (4.1 conditions HOW all layers compute; 3.7 gates WHICH layers execute) and should be pursued jointly in a single merged implementation.

> **Note:** Citation [15] is provisional with no confirmed arXiv ID. Claims relying on this citation should be treated as speculative until resolved.

## Key Comparison Tables

### vs. Baseline A1 (Qwen3.5-27B Hybrid)

| Metric | Baseline A1 | Idea 4.1 | Change |
|--------|------------|---------|--------|
| Compute (FLOPs/token) | O(L·(d²+s·d/4)) | +168M FLOPs conditioning (~1.14% overhead) [derived: 168M / 14.7B A1 baseline ≈ 1.14%] | ↑ ~1.14% |
| Memory bandwidth (decode) | O(L·(d²+s·d_kv/4)) | +336 MB/token conditioning weight load | ↑ ~0.5% |
| KV cache (32K ctx) | ~2.15 GB (16 full-attn layers) | +16 MB FSM state history | ↑ negligible |
| Weight memory | ~27B params | +~170M params (~0.63%) [derived: 170M / 27B ≈ 0.63%] | ↑ negligible |
| Training cost (wall-clock) | 1.0× | ~1.05–1.10× | ↑ |
| TTFT (8K prompt) | ref | ↑* +0.6% (⚠ upper bound, unverified — sequential FSM prefill at ≥262K context) [derived: sequential FSM chain adds ~0.6% to 8K-prompt prefill critical path vs A1 dense prefill] | ↑ negligible |
| TPOT (batch=1) | ref | +1.14% [derived: 168M / 14.7B A1 baseline ≈ 1.14%] | ↑ |
| NLP quality delta | — | Unknown at LLM scale | ? |

### vs. Baseline A2 (Qwen3-32B Dense)

| Metric | Baseline A2 | Idea 4.1 | Change |
|--------|------------|---------|--------|
| Compute (FLOPs/token) | O(L·(s·d+d·d_ff)) | +168M FLOPs conditioning [derived: 168M / 8.4B A2 baseline ≈ 2.0%] | ↑ ~2.0% |
| Memory bandwidth (decode) | O(L·(d·d_ff+s·d_kv)) | +336 MB/token | ↑ ~0.5% |
| KV cache (32K ctx) | ~8.59 GB (64 layers, GQA 8:1) | +16 MB FSM state history | ↑ negligible |
| Weight memory | ~32B params | +~170M params (~0.53%) [derived: 170M / 32B ≈ 0.53%] | ↑ negligible |
| Training cost (wall-clock) | 1.0× | ~1.05–1.10× | ↑ |
| TTFT (8K prompt) | ref | ↑* +0.6% (⚠ upper bound, unverified — sequential FSM prefill at ≥262K context) [derived: sequential FSM chain adds ~0.6% to 8K-prompt prefill critical path vs A2 dense prefill] | ↑ negligible |
| TPOT (batch=1) | ref | +2.0% [derived: 168M / 8.4B A2 baseline ≈ 2.0%] | ↑ |

### vs. Baseline B (Qwen3.5-397B-A17B MoE)

| Metric | Baseline B | Idea 4.1 | Change |
|--------|-----------|---------|--------|
| Compute (FLOPs/token) | O(L·(d²+k·d·d_e)) | +126M FLOPs conditioning (d=4096) [derived: 2·L·N·d = 2·60·256·4096 ≈ 126M; fraction ≈ 0.5–0.7% of B active-parameter MLP FLOPs] | ↑ ~0.5–0.7% |
| Memory bandwidth (decode) | O(L·(k·d·d_e+state)) | <0.1% overhead | ↑ negligible |
| KV cache (32K ctx) | ~1.0 GB (15 full-attn layers, 32Q/2KV, hd=256) | +16 MB | ↑ negligible |
| Weight memory | ~397B params | +~126M params (0.03%) | ↑ negligible |
| FSM→MoE routing synergy | — | Potential quality gain at zero net cost | + |

### vs. Baseline C (K2 family, 72.55B dense Llama-arch)

| Metric | Baseline C (K2) | Idea 4.1 | Change |
|--------|----------------|---------|--------|
| Compute (FLOPs/token) | O(L·(s·d+d·d_ff)), L=80, d=8192 | +2 × 80 × 256 × 8192 ≈ 336M FLOPs [derived: 336M / ~33.6B C per-token MLP FLOPs ≈ 1.0%] | ↑ ~1.0% |
| Memory bandwidth (decode) | ~145.1 GB weight BW | +672 MB FiLM conditioning weights | ↑ ~0.5% |
| KV cache (32K ctx) | ~10.0 GiB (80 layers, GQA 8:1) | +16 MB FSM state history | ↑ negligible |
| Weight memory | ~72.55B params | +~336M params (0.46%) | ↑ negligible |
| MLP FLOPs/token/layer | ~4.70 × 10⁸ | Unchanged per layer | = |
| Training cost (wall-clock) | 1.0× | ~1.02–1.05× | ↑ |
| NLP quality delta | — | Unknown at 72B scale | ? |

---

## 1. Idea Description

**From arch_research_ideas.md (Section 4, idea 4.1):**

> A learned state machine that the model explicitly modifies during the forward pass alongside activations. The FSM state influences computation and is updated as part of inference — not just implicit in hidden states.

**Inferred intent:** At each token step, an explicit FSM state (discrete or soft-categorical, N possible values) is maintained across the full forward pass. Unlike idea 3.7, which treats the FSM primarily as a **router** (governing which blocks execute), idea 4.1 treats the FSM as a **core computational primitive**: the FSM state modulates every layer's computation — biasing attention patterns, scaling MLP activations, or conditioning normalization — rather than selectively bypassing blocks. The FSM state is updated at the end of each token step (or each layer), is persistent across the entire sequence, and participates in all forward-pass operations.

**Relationship to idea 3.7:** 3.7 (routing) gates WHICH layers execute (compute-variable inference, TPOT reduction); 4.1 (conditioning) modulates HOW layers compute (compute-uniform, quality improvement). These are complementary rather than equivalent — a model implementing only 3.7 provides no FiLM conditioning benefit, and a model implementing only 4.1 provides no block-skipping speedup. The merged architecture (FSM state used for both routing AND conditioning simultaneously) is strictly superior and should be the implementation target.

---

## 2. Literature Review

### Neural Turing Machines (2014, arXiv)
Introduces a neural network controller coupled to an external differentiable memory bank via soft attention-based read/write heads. The NTM external memory is the most direct precursor to idea 4.1: it is an explicit structured state (N × W matrix) that the model reads from and writes to at every step, influencing every subsequent computation.

Neural Turing Machines[1]: arXiv:1410.5401, §"Neural Turing Machine Architecture", §"Copy Task", §"Sorting Task" (Table 1)

### Differentiable Neural Computer (2016, Nature)
Extends NTM to the Differentiable Neural Computer (DNC). Introduces temporal memory linkage and usage-based allocation. The DNC provides the clearest published instance of an explicit, persistent, differentiable state structure that actively modifies the forward pass at every step.

Hybrid Computing Using a Neural Network with Dynamic External Memory[2]: Nature 538:471–476, §"Dynamic memory allocation", §"Graph tasks" (Figure 4)

### State-Regularized Recurrent Neural Networks (2019, ICML)
Introduces a stochastic state-transition mechanism constraining RNNs to transition between a finite set of learned discrete states. The discrete state co-exists with continuous activations. Most direct prior art for "discrete FSM state alongside continuous activations" in idea 4.1.

State-Regularized Recurrent Neural Networks[3]: ICML 2019, pp. 6596–6606, §"State-Regularization" (Table 1: language modeling perplexity)

### RWKV (2023, EMNLP Findings)
RWKV's WKV state is a practical example of "explicit recurrent state influencing every forward-pass computation" at LLM scale (up to 14B parameters). The WKV accumulator is updated at each token step and directly conditions both time-mixing and channel-mixing at every layer. However, the RWKV state is a continuous exponential-weighted sum, not a discrete FSM.

RWKV: Reinventing RNNs for the Transformer Era[4]: EMNLP 2023 Findings, arXiv:2305.13048, §"Time Mixing", §"RWKV Architecture"

### Mamba (2023/2024, ICLR 2024)
Introduces Selective State Space Models (S6), where SSM parameters are input-dependent. The recurrent state is explicitly maintained and influences every output computation. Demonstrates that an explicit recurrent state at inference time is architecturally viable at 3B scale with 5× throughput improvement over transformers. Key distinction: Mamba's state is continuous, not discrete.

Mamba: Linear-Time Sequence Modeling with Selective State Spaces[5]: arXiv:2312.00752, §"Selective State Space Models", §"Language Model Experiments" (Table 3: Mamba-3B vs. Transformer-2×); ICLR 2024

### Mamba-2 (2024, ICML)
Establishes a theoretical duality between structured SSMs and masked self-attention. The SSM state is a compressed, lossy summary of the full KV cache. Mamba-2's analysis demonstrates the expressiveness ceiling of continuous recurrent states, motivating explicit discrete states for tasks requiring exact state tracking.

Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality[6]: ICML 2024, arXiv:2405.21060, §"Structured State Space Duality" (Theorem 1)

### Griffin (2024, arXiv)
Introduces the RG-LRU recurrent layer that maintains an explicit real-valued recurrent state at inference. Griffin-14B matches comparable-scale transformer baselines at competitive compute, validating that architectures with explicit recurrent state are competitive with transformers at production scale. The RG-LRU state is continuous (gated linear recurrence), not a discrete FSM.

Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models[7]: arXiv:2402.19427, §"RG-LRU Layer", §"Experimental Results" (Table 2: Griffin-14B competitive with comparable-scale baselines at same compute)

### The Illusion of State in State-Space Models (2024, ICML)
Proves that SSMs (including Mamba) belong to the TC⁰ complexity class, identical to transformers, meaning neither can reliably track discrete symbolic states (permutation composition, chess moves, entity tracking). This paper provides the strongest theoretical motivation for idea 4.1: continuous recurrent states cannot substitute for explicit discrete FSM state on tasks requiring exact discrete state tracking.

The Illusion of State in State-Space Models[8]: ICML 2024, arXiv:2404.08819, §"Main Theorem" (TC⁰ impossibility), §"Experimental Validation" (permutation composition, chess, entity tracking failures)

### PD-SSM (2025, NeurIPS Spotlight)
Proposes PD-SSM, parametrizing SSM transition matrices as the product of a column one-hot matrix (P) and a complex diagonal matrix (D). One PD-SSM layer of dimension N can emulate any N-state FSA (abstract). Per abstract: "significantly outperforms a wide collection of modern SSM variants on various FSA state tracking tasks"; specific per-task accuracy percentages are paper-body table values. PD-SSM is a direct differentiable approximation of 4.1's FSM core mechanism — validating that such structures are learnable.

Structured Sparse Transition Matrices to Enable State Tracking in State-Space Models[9]: NeurIPS 2025 Spotlight, arXiv:2509.22284, §"PD-SSM Architecture", §"Theoretical Result", §"Experimental Results" (per-task tables in paper body).

### SD-SSM (2025, AAAI)
Proposes the Selective Dense SSM, the first selective SSM achieving perfect length generalization on diverse regular language (FSA emulation) tasks in a single layer. The softmax selection over learned transition dictionaries is exactly the "soft discrete FSM update" that could be used in idea 4.1.

On the Expressiveness and Length Generalization of Selective State-Space Models on Regular Languages[10]: AAAI 2025, arXiv:2412.19350, §"SD-SSM Architecture", §"Perfect Length Generalization" (Table: 100% on diverse FSA tasks, single layer)

### Recurrent Memory Transformer (2022, NeurIPS)
Augments a transformer with special memory tokens carrying information across segments via segment-level recurrence. RMT implements a persistent explicit state (the memory token representations) that influences every subsequent segment's forward pass — a continuous analog of 4.1's FSM state.

Recurrent Memory Transformer[11]: NeurIPS 2022, arXiv:2207.06881, §"Recurrent Memory Mechanism", §"Experimental Results" (Transformer-XL comparison)

### Block-Recurrent Transformers (2022, NeurIPS)
Applies a transformer cell recurrently across blocks of tokens, maintaining explicit recurrent state vectors via cross-attention. Outperforms Transformer-XL while running 2× faster on PG19, arXiv, and GitHub datasets.

Block-Recurrent Transformers[12]: NeurIPS 2022, arXiv:2203.07852, §"Block-Recurrent Cell", §"Results" (2× training throughput vs Transformer-XL, PG19/arXiv/GitHub benchmarks)

### Titans (2024/2025, ICLR 2025)
Introduces a three-branch architecture with a neural long-term memory module updated during inference via gradient steps. Outperforms Mamba-2, Gated DeltaNet, and Transformer++ on language modeling; handles up to 2M token contexts. Demonstrates that architectures with explicit inference-time state modification are competitive at scale.

Titans: Learning to Memorize at Test Time[13]: ICLR 2025, arXiv:2501.00663, §"Neural Long-Term Memory Module", §"Persistent Memory Branch", §"Experiments" (Table 4: BABILong, language modeling)

### Transformers Learn Shortcuts to Automata (2023, ICLR)
Proves that shallow transformers (O(log T) layers) can simulate any finite-state automaton via algebraic structure of transformation semigroups. Establishes the key counterpoint to 4.1: transformers can already simulate any FSA implicitly, without any explicit state variable. The value of an explicit FSM state must be justified on reliability, sample efficiency, interpretability, or controllability grounds rather than expressiveness alone.

Transformers Learn Shortcuts to Automata[14]: ICLR 2023, arXiv:2210.10749, §"Main Result" (Theorem 1: O(log T) simulator), §"Solvable Groups" (O(1)-depth)

### Steering LLMs' Reasoning with Activation State Machines (2025, OpenReview)
Proposes Activation State Machine (ASM) [unverified post-cutoff preprint], a Kalman-filter-inspired inference-time steering mechanism that maintains an explicit continuous state variable alongside transformer activations and applies corrective interventions. Closest published work to "FSM state alongside activations, influencing every forward-pass step." Key distinctions from 4.1: ASM state is continuous, post-hoc (not jointly trained), not discrete.

Steering LLMs' Reasoning with Activation State Machines[15]: OpenReview NeurIPS 2025 submission (p17En1bhCY; no confirmed arXiv ID — PROVISIONAL citation), §"Activation State Machine Architecture", §"Results"

### Finite State Automata Inside Transformers with Chain-of-Thought: A Mechanistic Study on State Tracking (2025, arXiv)
Mechanistic study showing that Transformer+CoT implements FSA state tracking via late-layer MLP neurons (layers 9–11). Near-perfect accuracy on group-word problems with sequences 100× layer depth. Without CoT, accuracy drops dramatically. Idea 4.1 would maintain this state as an internal discrete variable rather than requiring scratchpad tokens.

Finite State Automata Inside Transformers with Chain-of-Thought: A Mechanistic Study on State Tracking[16]: arXiv:2502.20129, §"Circuit Localization", §"Transformer+CoT Recovers FSA" (near-100% accuracy on ℤ₆₀, A₄×ℤ₅, A₅ groups)

### Hybrid Neural State Machine (2021, Science China Information Sciences)
H-NSM: an explicit FSM controller receives inputs from neural networks, makes state transitions, and sends control signals to activate different branch networks. Most structurally similar prior art for an explicit FSM running alongside neural computation and controlling it. Key differences: FSM is hand-designed (not learned), applied to SNN robotics (not transformers), evaluated on control tasks (not NLP).

Hybrid Neural State Machine for Neural Network[17]: Science China Information Sciences, Vol. 64, Art. 132202, 2021, §"H-NSM Framework", §"Condition-Based Control (H-NSM-C)"

### DeltaFormer: Unlock the State Space of Transformer (2025, NeurIPS)
Reconceptualizes the Transformer from a state-space perspective using Delta Rule-based updates, breaking the TC⁰ expressivity ceiling that standard attention cannot escape. Demonstrates on language modeling benchmarks that DeltaFormer matches or exceeds standard Transformer quality while achieving greater expressivity for state-tracking tasks (entity tracking, chess, Python code evaluation). Both the standard Transformer and DeltaNet are special cases of the DeltaFormer general form. This is a competing approach to the same TC⁰ problem that motivates idea 4.1: where 4.1 augments transformers with an explicit external FSM state, DeltaFormer restructures the attention mechanism itself to subsume FSM-like state tracking. A comparison between these strategies is important prior art for 4.1's novelty positioning.

DeltaFormer: Unlock the State Space of Transformer[19]: NeurIPS 2025, OpenReview GSE3oaiDL2, §"State Space Perspective", §"TC⁰ Expressivity Analysis", §"Language Modeling Results"

### Mixture-of-Depths (2024, arXiv)
Demonstrates per-token dynamic depth routing at 6B+ scale. In the context of 4.1, MoD shows that per-token routing decisions are learnable and work at LLM scale. MoD's routing is stateless (each token's routing is decided independently without cross-token memory); idea 4.1's FSM state provides cross-token persistent signal for more structured routing.

Mixture-of-Depths: Dynamically Allocating Compute in Transformer-Based Language Models[18]: arXiv:2404.02258, §"Token Routing", §"Results" (isoFLOP quality match with per-token depth routing at 6B+ scale)

---

## 3. Prior Art Classification

- **Status**: PARTIAL
- **Overlap summary**: ~65% overlap with published work
- **Novel contribution**: The specific combination of (a) discrete N-state FSM, (b) end-to-end learned transition function, (c) persistent cross-token FSM state, and (d) FSM state as FiLM-style universal conditioning signal applied to ALL layers of a transformer LLM (not just routing blocks or external memory), evaluated at LLM scale on NLP benchmarks — is **not published** as of April 2026.
- **Novelty estimate**: ~30–40% of the idea is genuinely unpublished.
- **Gap for 3.7 vs. 4.1**: These are distinct unpublished gaps — "discrete FSM routing transformer blocks at LLM scale" (3.7) and "discrete FSM FiLM-conditioning transformer layers at LLM scale" (4.1) are separately unpublished. A merged implementation closes both gaps simultaneously.

---

## 4. Technical Analysis

### 4.1 Theoretical Complexity

Variables: N = number of FSM states (default N=256), d = hidden dimension, d_ff = MLP intermediate dimension, L = number of layers, s = sequence length.

**FSM overhead per token:**
- Transition update: O(N·d) dominant term (W_x · x_t = N×d projection ≈ 1.3M FLOPs at N=256, d=5120). The N² term (65K FLOPs) is negligible relative to N·d. Dominant FSM overhead: O((L+1)·N·d) per token.
- Conditioning overhead (FiLM-style, full scale+shift at every layer): 2 × L × N × d = 2 × 64 × 256 × 5120 ≈ 168M FLOPs per token.
- Net TPOT overhead: ~2.0% for A2 [derived: 168M / ~8.4B baseline]; ~1.14% for A1 [derived: 168M / 14.7B baseline]; ~0.5–0.7% for B (d=4096) [derived: 126M / B active-parameter baseline].

**KV cache denominator:**
A1 has only 16 full-attention layers (every 4th of 64) with 4 KV heads at head_dim=256. A1 KV at 32K context = 16 × 2 × 4 × 256 × 32,768 × 2 = ~2 GB. A2 KV at the same context (64 dense layers, 8 KV heads, hd=128) is ~8.59 GB, growing to ~68.7 GB at 262K under YaRN-extended context. The "16 MB FSM state history vs ~2 GB KV cache (A1)" comparison is the correct framing.

**Prefill sequential dependency:**
The FSM state update s_{t+1} = softmax(T·s_t + W_x·x_t) is inherently sequential — each step depends on the previous FSM state. At s=8K, the sequential cost is ~10.7G FLOPs (negligible compute). However, at s=262K (A1/B max context), this sequential chain may create a wall-clock TTFT bottleneck despite low FLOPs. For contexts ≤ 64K this is not a concern; for 262K, a parallel scan or chunked approximation may be needed.

**Weight memory:**
Full FiLM requires both scale and shift projections: 2 × L × N × d parameters = 2 × 64 × 256 × 5120 ≈ 168M parameters. Total with transition matrix and input projection: ~170M parameters. Percentage of A1 (27B): ~0.63%; percentage of A2 (32B): ~0.53%.

**Additional weight bandwidth:**
Scale projections: L × N × d × 2 bytes = 168 MB. Shift projections: 168 MB. Total full FiLM weight load: ~336 MB per decode token. As fraction of A2 model (64 GB): ~0.52%.

| Metric | This Idea (4.1) | A1 | A2 | B |
|--------|-----------------|----|----|---|
| Compute per token | O(L·(d²+d·d_ff) + 2·L·N·d) | O(L·(d²+s·d/4)) | O(L·(s·d+d·d_ff)) | O(L·(d²+k·d·d_e)) |
| KV cache memory | O(L·s·d_kv) + O(s·N) (negligible) | O(L/4·s·d_kv) | O(L·s·d_kv) | O(L/4·s·d_kv) |
| Weight memory | O(L·d·d_ff) + ~170M conditioning | O(L·d·d_ff) | O(L·d·d_ff) | O(L·E·d·d_e) |

---

## 5. Implementation Considerations

**FSM application mechanisms (ordered by implementation ease):**
1. **Post-norm conditioning** (replace static learned norm scales with FSM-state-conditional ones): EASY. Standard in diffusion models. No custom kernels.
2. **Post-attention conditioning** (element-wise multiply/add after attention): EASY.
3. **Attention bias injection** (FSM-state-dependent attention logit bias): MODERATE. Requires FlashAttention modification.

**Recommended starting point:** Post-norm / pre-residual FiLM conditioning with α warmup from 0.

**Training stability concerns:**
1. FSM collapse (all states converge to 1–2 states): mitigated by entropy regularization + load balancing (proven from MoE).
2. Discrete gradient: Gumbel-Softmax with temperature annealing (standard).
3. All-layer corruption from bad FSM state: residual scaling with α initialized to 0, warmed up over 1K steps.
4. Prefill sequential dependency: negligible at ≤ 64K context; chunked approximation needed at 262K.

**Redundancy with A1 Gated DeltaNet:** Gated DeltaNet layers in A1 maintain a competing O(d²) recurrent state. Training gradient competition is a real risk. Initial prototyping should use a dense baseline (A2-like) to cleanly measure the FSM conditioning benefit.

**Training cost clarification:** The "1.05–1.10× training cost" estimate is a wall-clock estimate. Raw FLOPs overhead is ~1.02–1.05×. The higher wall-clock estimate reflects entropy regularization, load balancing, and training instability overhead.

---

## 6. Synergies

- **Combines well with:**
  - **3.7 (Learnable State Machine):** Highly complementary and must be developed jointly. 3.7 handles FSM-conditioned routing decisions; 4.1 handles FiLM-style activation modulation. The synergy is strong: 3.7's block skipping (~30% compute reduction) offsets 4.1's conditioning overhead (~2%); 4.1's conditioning mitigates 3.7's quality risk from routing errors.
  - **3.1 (Layer-Level MoE):** FSM state as auxiliary input to MoE router at zero net cost (FSM state already computed, router input dimension increases only).
  - **1.2 (Per-Token Adaptive Depth):** FSM state as a depth-exit signal.
  - **3.5 (Gated Internal DAG):** FSM state naturally selects DAG branches based on sequence context.

- **Conflicts with:**
  - **Batched inference:** Hard discrete FSM routing requires different computation per token, conflicting with standard batched GPU execution. Soft FSM avoids this.
  - **Baseline A1 (Gated DeltaNet layers):** Linear attention layers already maintain O(d²) recurrent state; two competing explicit state mechanisms create gradient competition risk.

---

## 7. Key Tradeoffs

- **Gain:** Explicit discrete FSM state bypasses the TC⁰ expressiveness ceiling that prevents continuous recurrent states from reliably tracking discrete symbolic states — Merrill et al. (2024)[8].
- **Gain:** Enables interpretable, inspectable "mode" signal (bracket depth, code scope, entity state, dialogue phase) not easily extracted from continuous hidden states.
- **Gain:** FSM state can serve as auxiliary input to MoE routers, improving routing quality without additional per-token compute.
- **Gain:** If combined with block-skipping (3.7), the conditioning overhead is offset by blocks skipped; net effect at 30% skip rate is ~28% TPOT improvement.
- **Lose:** In the pure "universal modulator" form (no routing), 4.1 adds ~1.14% TPOT overhead on A1 [derived: 168M / 14.7B ≈ 1.14%] and ~2.0% on A2 [derived: 168M / 8.4B ≈ 2.0%] with no latency reduction.
- **Lose:** All-layer corruption from a bad FSM state is more severe than 3.7's single-block routing error.
- **Risk:** Routing/conditioning collapse: FSM converges to 1–2 active states. Entropy regularization + load balancing required.
- **Risk:** Transformers can already simulate any FSA implicitly via attention shortcuts — Liu et al. (2023)[14]. DeltaFormer (NeurIPS 2025)[19] further addresses the TC⁰ ceiling by restructuring attention itself via Delta Rule updates, representing a competing architectural response to the same problem. The practical benefit of 4.1's explicit external FSM over these implicit or restructured-attention approaches is unproven at LLM scale.

---

## 8. Accuracy / Quality Tradeoff

- **Reported quality delta:** No direct LLM-scale NLP experiments. PD-SSM "significantly outperforms a wide collection of modern SSM variants on various FSA state tracking tasks" (abstract; specific percentages paper-body tables) — Terzić et al. (2025)[9]. SD-SSM achieves perfect length generalization on diverse regular-language (FSA emulation) tasks in a single layer — Terzić et al. (2025b)[10]. SR-RNN improves RNN long-term memory on formal language tasks — Wang & Niepert (2019)[3]. None are transformer LLMs on standard NLP benchmarks.
- **Monotonicity:** Unknown for NLP tasks. On formal language tasks, quality improvement appears cliff-like.
- **Recovery:** Unknown. If FSM collapses, retraining from scratch with stronger entropy regularization is likely required.

---

## 9. Prior Art Gap

The combination of (a) discrete N-state FSM, (b) end-to-end learned transition function, (c) persistent cross-token FSM state, and (d) FSM state as a FiLM-style universal conditioning signal applied to ALL transformer layers, evaluated at LLM scale on NLP benchmarks, is **not published** (~35% of the idea is unpublished). The specific gap for 4.1 (universal FiLM conditioning) is distinct from the gap for 3.7 (routing/skipping), and a merged implementation fills both gaps simultaneously.

---

## 10. Analysis Notes

### Why investigate:
1. **Strong theoretical motivation:** Merrill et al. (2024)[8] proves the TC⁰ ceiling for continuous state; the discrete FSM is a theoretically sound bypass.
2. **FSA-structured models work on formal tasks:** PD-SSM[9] significantly outperforms other structured SSMs on FSA state-tracking (per abstract); SD-SSM[10] achieves perfect length generalization on regular-language tasks with a single layer.
3. **Overhead is acceptable:** ~2.0% TPOT (A2) [derived: 168M / 8.4B], ~1.14% TPOT (A1) [derived: 168M / 14.7B], ~0.5% memory bandwidth, ~170M additional parameters.
4. **Prior art gap is real:** 30–40% of the idea is genuinely unpublished.

### Why proceed carefully:
1. Zero NLP evidence at LLM scale — all validation is on formal language tasks.
2. 3.7 and 4.1 must be investigated jointly (shared FSM infrastructure, complementary benefits).
3. Test on dense baseline (A2-like) first — Gated DeltaNet in A1 creates gradient competition.
4. Start with soft FSM (τ=1.0, α warmup from 0) before adding hard routing.
5. DeltaFormer (NeurIPS 2025)[19] achieves TC⁰ bypass by restructuring attention — a competing approach without external FSM overhead; comparison experiments needed.

### Recommended next steps:
1. Implement merged 4.1+3.7 prototype at 1–3B scale on dense A2-like architecture
2. Soft FSM first (τ=1.0, α warmup from 0), FiLM conditioning post-norm
3. Measure: (a) perplexity on NLP benchmarks, (b) accuracy on formal language tasks, (c) FSM state entropy (must remain > 1 bit)
4. If Phase 1 succeeds: add hard routing (3.7 behavior), measure TPOT delta
5. If Phase 2 succeeds: scale to 7B, test on Baseline A1 hybrid

---

<!-- CITATION MANIFEST -->

[1] Neural Turing Machines: Graves, Wayne, Danihelka (Google DeepMind, 2014). arXiv:1410.5401. Canonical prior art for explicit differentiable state alongside neural activations. Controller conditioned on memory output at every step — foundational precursor to idea 4.1.

[2] Hybrid Computing Using a Neural Network with Dynamic External Memory: Graves et al. (DeepMind, 2016). Nature 538:471–476. Differentiable Neural Computer with temporal memory linkage; clearest published instance of explicit persistent differentiable state modifying the forward pass at every step.

[3] State-Regularized Recurrent Neural Networks: Wang & Niepert (ICML 2019, pp. 6596–6606). arXiv:1901.08817. Most direct prior art for discrete FSM state alongside continuous activations. Discrete N-state categorical alongside RNN hidden states as a regularizer for interpretability.

[4] RWKV: Reinventing RNNs for the Transformer Era: Peng et al. (EMNLP 2023 Findings). arXiv:2305.13048. Explicit continuous recurrent state influencing all forward-pass computations at LLM scale (up to 14B). Validates the pattern; continuous state only.

[5] Mamba: Linear-Time Sequence Modeling with Selective State Spaces: Gu & Dao (ICLR 2024). arXiv:2312.00752. Selective SSM with O(N) constant recurrent state per layer. Key reference for continuous recurrent state at scale. Per Merrill et al. (2024), Mamba cannot reliably track discrete symbolic states due to TC⁰ constraint.

[6] Transformers are SSMs — Structured State Space Duality: Dao & Gu (ICML 2024). arXiv:2405.21060. Establishes duality between structured SSMs and masked self-attention; SSM state is a compressed lossy summary of full KV cache. Motivates explicit discrete states for exact state tracking.

[7] Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models: De et al. (Google DeepMind, 2024). arXiv:2402.19427. RG-LRU recurrent state; Griffin-14B competitive with comparable-scale transformer baselines at same compute. Continuous state only.

[8] The Illusion of State in State-Space Models: Merrill, Petty, Sabharwal (ICML 2024). arXiv:2404.08819. **Core theoretical motivation for 4.1.** Proves SSMs belong to TC⁰ complexity class — cannot reliably track permutation composition, chess moves, entity tracking. Discrete FSM state bypasses this ceiling.

[9] Structured Sparse Transition Matrices to Enable State Tracking in SSMs (PD-SSM): Terzić et al. (NeurIPS 2025 Spotlight). arXiv:2509.22284. PD parametrization enables one-layer FSA emulation of any N-state FSA; abstract reports significant outperformance of modern SSM variants on FSA state-tracking. Validates discrete FSM state learning is feasible.

[10] On the Expressiveness and Length Generalization of Selective SSMs on Regular Languages (SD-SSM): Terzić et al. (AAAI 2025). arXiv:2412.19350. SD-SSM achieves 100% perfect length generalization on diverse FSA tasks in a single layer. Softmax over learned transition dictionaries — directly implementable as 4.1's update rule.

[11] Recurrent Memory Transformer: Bulatov, Kuratov, Burtsev (NeurIPS 2022). arXiv:2207.06881. Persistent explicit state (memory tokens) influencing forward pass across segments. Continuous analog of 4.1's FSM at segment granularity.

[12] Block-Recurrent Transformers: Hutchins et al. (NeurIPS 2022). arXiv:2203.07852. Explicit recurrent state vectors maintained across blocks via cross-attention. Close continuous-state implementation of 4.1 at block granularity.

[13] Titans: Learning to Memorize at Test Time: Behrouz, Zhong, Mirrokni (ICLR 2025). arXiv:2501.00663. Neural long-term memory updated during inference; outperforms Mamba-2 on BABILong at 2M context. Demonstrates explicit inference-time state modification competitive at scale.

[14] Transformers Learn Shortcuts to Automata: Liu et al. (ICLR 2023). arXiv:2210.10749. Proves O(log T) shallow transformers simulate any FSA. Key counterpoint: transformers can implicitly simulate FSA without explicit state. The benefit of 4.1 must be justified on reliability/sample-efficiency/interpretability grounds.

[15] Steering LLMs' Reasoning with Activation State Machines: Li & Chen (NeurIPS 2025 submission, PROVISIONAL). OpenReview p17En1bhCY. No confirmed arXiv ID. Kalman-filter ASM state alongside activations; post-hoc steering, continuous state. Treat as provisional pending arXiv publication.

[16] Finite State Automata Inside Transformers with Chain-of-Thought: A Mechanistic Study on State Tracking: Zhang et al. (2025). arXiv:2502.20129. Mechanistic study: transformer+CoT implements FSA tracking via late-layer MLP neurons. Near-100% accuracy with CoT on ℤ₆₀, A₄×ℤ₅, A₅; drops without CoT. 4.1 would internalize this state without scratchpad tokens.

[17] Hybrid Neural State Machine for Neural Network: Lei et al. (Science China Info. Sci. 2021, Vol. 64, Art. 132202). arXiv/DOI via Springer. Explicit non-learned FSM controlling neural branch activation. Most structurally similar non-transformer prior art; hand-designed FSM applied to SNN robotics.

[18] Mixture-of-Depths: Dynamically Allocating Compute in Transformer-Based Language Models: Raposo et al. (2024). arXiv:2404.02258. Per-token dynamic depth routing at 6B+ scale; matched quality to isoFLOP baselines. Validates per-token routing decisions learnable at LLM scale; MoD is stateless (no cross-token FSM).

[19] DeltaFormer: Unlock the State Space of Transformer: Xu, Ao, He, Lu, Shi, Zheng (NeurIPS 2025). OpenReview GSE3oaiDL2. Reformulates standard Transformer from state-space + kernel perspective with Delta Rule updates, breaking TC⁰ expressivity limits. Both Transformer and DeltaNet are special cases. Language modeling parity with standard Transformer plus improved state-tracking. Competing approach to the same TC⁰ problem as 4.1 — restructures attention rather than adding external FSM state.
