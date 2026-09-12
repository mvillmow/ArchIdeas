# Cross-Reference Matrix — 39 Architecture Ideas

## Methodology

This matrix is derived directly from the "Synergies" sections of each merged research document in the 39-idea corpus. Each document was read to extract stated synergies (positive interactions) and conflicts (negative interactions or redundancies). Synergy type codes used below:

- **COMP** — Complementary: the two ideas address orthogonal dimensions; applying both yields additive or multiplicative benefit.
- **MULT** — Multiplicative: the combined benefit is greater than the sum (super-additive).
- **NEST** — Nested/Sequential: one idea is a prerequisite for or generalization of the other.
- **IMPL** — Implementation convenience: one idea simplifies building the other.
- **OPER** — Operational synergy: both ideas improve production deployment, not raw latency.

Conflict type codes:

- **REDN** — Redundant: both ideas address the same optimization target; applying both is wasteful.
- **INCO** — Incompatible: the mechanisms contradict each other at the design level.
- **COMP_RISK** — Competing mechanisms: both work independently but compete for training stability or the same design budget.
- **ENGR** — Engineering incompatibility: both are sound but cannot be easily codeployed in a shared implementation.

---

## Synergy Map

| Idea A | Idea B | Type | Description |
|--------|--------|------|-------------|
| 1.1 Learnable Top-k | 1.3 Per-Layer Adaptive Expert Count | COMP | Orthogonal axes: 1.1 varies k per token, 1.3 varies k per layer. Combined creates a 2D adaptive routing grid. Alloc-MoE and DynaMoE partially explore this combination. [see research_1_1.md, research_1_3.md] |
| 1.1 Learnable Top-k | 1.4 Learned Dense/Sparse Layer | NEST | Sequential: 1.4 assigns layer types, 1.1 optimizes k within remaining MoE layers. Strong positive synergy. [see research_1_1.md, research_1_4.md] |
| 1.1 Learnable Top-k | 2.2 Compressed Dense Layers | MULT | Expert matrices compressed (low-rank) reduce per-expert weight-load cost independently of k reduction. Multiplicative bandwidth savings. [see research_1_1.md, research_2_2.md] |
| 1.1 Learnable Top-k | 5.1 TurboQuant | MULT | Quantized expert weights reduce bytes-per-expert; combined with lower k̄, delivers multiplicative bandwidth reduction. [see research_1_1.md, research_5_1.md] |
| 1.1 Learnable Top-k | 3.3 Dynamic Expert Router | COMP | Dynamic k naturally adapts to new expert capacity; new experts get low k-selection probability initially and grow as they become useful. [see research_3_3.md] |
| 1.1 Learnable Top-k | 6.5 Pre-Attention Expert Router | COMP | Q projection uncertainty may be a better proxy for routing confidence; combined routing signal from pre-attention representation and per-token k. [see research_6_5.md] |
| 1.2 Per-Token Adaptive Depth | 1.1 Learnable Top-k | COMP | Orthogonal dimensions: depth (layers) vs. breadth (experts). An early-exiting token also uses fewer experts in the layers it executes. [see research_1_2.md] |
| 1.2 Per-Token Adaptive Depth | 3.4 Recursive Internal State | COMP | 3.4's iterative refinement can use 1.2's dynamic middle section as the loop body; exit condition becomes the routing decision. [see research_1_2.md, research_3_4.md] |
| 1.2 Per-Token Adaptive Depth | 4.2 Shared Core + LoRA | COMP | Shared middle-layer weights reduce the cost of skipping; LoRA differentiates exit points. [see research_1_2.md, research_4_2.md] |
| 1.2 Per-Token Adaptive Depth | 5.1 TurboQuant | COMP | Adaptive depth reduces full-attention layer accesses; KV quantization reduces per-byte cost. Additive TPOT improvement. [see research_1_2.md, research_5_1.md] |
| 1.2 Per-Token Adaptive Depth | 5.4 Linked Attention | COMP | KV entries for easy tokens can be evicted after their exit layer, reducing KV memory impact. [see research_1_2.md, research_5_4.md] |
| 1.3 Per-Layer Adaptive Expert Count | 1.4 Learned Dense/Sparse Layer | NEST | Complete layer-level compute allocation policy: 1.4 decides dense/MoE, 1.3 decides k_l given MoE. [see research_1_3.md, research_1_4.md] |
| 1.3 Per-Layer Adaptive Expert Count | 2.2 Compressed Dense Layers | MULT | Reduces per-expert byte count independently of routing. Multiplicative bandwidth benefit. [see research_1_3.md, research_2_2.md] |
| 1.3 Per-Layer Adaptive Expert Count | 3.1 Layer-Level MoE | NEST | 3.1's per-block routing subsumes 1.3 in the limit; per-block k is a generalization of per-layer expert count. [see research_1_3.md, research_3_1.md] |
| 1.3 Per-Layer Adaptive Expert Count | 3.3 Dynamic Expert Router | COMP | Dynamic expert count at inference is the inference-time version of the add/remove capability. [see research_3_3.md] |
| 1.4 Learned Dense/Sparse Layer | 5.1 TurboQuant | COMP | Independent dimension; additive benefit. KV cache shrinks as more attention layers go MoE. [see research_1_4.md] |
| 1.5 Learned Sparsity Type | 1.1 Learnable Top-k | COMP | MoE-assigned layers can additionally use dynamic k routing; N:M layers unaffected. [see research_1_5.md] |
| 1.5 Learned Sparsity Type | 1.3 Per-Layer Adaptive Expert Count | NEST | Hierarchical: 1.3 configures the type after 1.5 decides it; nested per-layer decisions. [see research_1_5.md] |
| 1.5 Learned Sparsity Type | 5.8 Block Sparse Weights | COMP | N:M and block sparsity are both weight-sparsity types; gate could choose among three options (MoE/N:M/block) via 3-way Gumbel-Softmax. [see research_1_5.md, research_5_8.md] |
| 1.6 Learned Layer Type | 1.1 Learnable Top-k | COMP | Dynamic primitive selection and dynamic expert count are complementary adaptive compute forms. [see research_1_6.md] |
| 1.6 Learned Layer Type | 1.2 Per-Token Adaptive Depth | COMP | An early-exiting token skips both expensive attention and routing at later layers. [see research_1_6.md] |
| 1.6 Learned Layer Type | 5.1–5.5 KV Cache Compression | COMP | Applies to the attention-layer fraction's KV cache; reduces penalty for keeping some full-attention layers. [see research_1_6.md] |
| 1.7 Dynamic Vocabulary | 2.1 Hierarchical Frequency Dictionary | COMP | 2.1 provides frequency cluster structure; 1.7 provides context conditioning for which level to load. Implementation convenience. [see research_1_7.md] |
| 1.7 Dynamic Vocabulary | 4.7 Compressed Dictionary | MULT | HIGH VALUE: combined bandwidth reduction = (V'/V) × (bits_quantized/bits_baseline). For A1 at V'=1K, INT4: ~994× bandwidth reduction on W_out. [see research_1_7.md, research_4_7.md] |
| 1.7 Dynamic Vocabulary | 5.8 Block Sparse Weights | COMP | Additive and independent: block sparsity reduces MLP bandwidth; vocabulary subsetting reduces W_out bandwidth. [see research_1_7.md, research_5_8.md] |
| 2.1 Hierarchical Frequency Dictionary | 4.7 Compressed Dictionary | COMP | Complementary production architecture: 4.7's compressed active subset for ~95% head-vocabulary tokens; 2.1 cascade fallthrough for ~5% rare tokens. Recommended hybrid. [see research_2_1.md, research_4_7.md] |
| 2.1 Hierarchical Frequency Dictionary | 2.2 Compressed Dense Layers | COMP | D2/D3 weight matrices can be low-rank decomposed; reduces their cost when evaluated. [see research_2_1.md] |
| 2.2 Compressed Dense Layers | 4.2 Shared Core + LoRA | COMP | Shared core is a natural candidate for low-rank factorization; LoRA adapters are already low-rank by construction. [see research_2_2.md] |
| 2.2 Compressed Dense Layers | 4.3 LoRA Everywhere | COMP | 2.2's inference-efficiency framework makes base model low-rank; 4.3 adds fine-tuning deltas on top. [see research_2_2.md] |
| 2.2 Compressed Dense Layers | 5.7 Block Compressed Weights | COMP | 5.7 at block granularity (cross-layer) + 2.2 at matrix granularity (within layer): complementary compression scopes. [see research_2_2.md] |
| 3.1 Layer-Level MoE | 5.1 TurboQuant | COMP | Compresses the k× KV overhead from full-block experts; directly addresses the key bottleneck. [see research_3_1.md] |
| 3.1 Layer-Level MoE | 1.6 Learned Layer Type | COMP | Expert blocks could be of different architectural types (attention/linear/SSM), creating heterogeneous full-block MoE. [see research_3_1.md] |
| 3.1 Layer-Level MoE | 4.2 Shared Core + LoRA | COMP | Within each expert, shared core weights + per-expert LoRA adapters reduce E× weight growth to 1× core + E×LoRA. [see research_3_1.md] |
| 3.1 Layer-Level MoE | 6.5 Pre-Attention Expert Router | COMP | Orthogonal dimensions: 3.1 determines whether a MoE layer activates; 6.5 determines what input the activated layer's router uses. [see research_6_5.md] |
| 3.2 Swappable Experts | 3.3 Dynamic Expert Router | IMPL | 3.2 handles weight-management (which weights occupy which slots); 3.3 handles router adaptation to new experts. Together: complete modular MoE lifecycle system. [see research_3_2.md, research_3_3.md] |
| 3.2 Swappable Experts | 4.2 Shared Core + LoRA | COMP | Expert weights as shared core + LoRA delta; swapping an expert means only replacing the small LoRA delta (~100× smaller swap bandwidth). [see research_3_2.md] |
| 3.2 Swappable Experts | 4.3 LoRA Everywhere | COMP | Hot-swap becomes swapping only the LoRA component (<<1% of weight bytes). LoRA-Switch demonstrates this at inference time. [see research_3_2.md] |
| 3.3 Dynamic Expert Router | 6.5 Pre-Attention Expert Router | COMP | 3.3's variable-E routing architecture enables 6.5's dynamic capability — new attention-pattern experts added post-training to 6.5's pool without router retraining. [see research_3_3.md, research_6_5.md] |
| 3.4 Recursive Internal State | 3.6 Recursive Internal DAG | NEST | 3.6 is a strict superset; 3.4's loop is a prerequisite for 3.6. Implement 3.4 first. [see research_3_4.md, research_3_6.md] |
| 3.4 Recursive Internal State | 3.7 Learnable State Machine | COMP | Recurrent loop can carry explicit FSM state alongside activations, providing cross-token routing information. [see research_3_4.md, research_3_7.md] |
| 3.4 Recursive Internal State | 4.2 Shared Core + LoRA | COMP | Shared middle block is the "shared core"; per-iteration LoRA adapters add expressiveness without breaking weight sharing. [see research_3_4.md] |
| 3.4 Recursive Internal State | 2.2 Compressed Dense Layers | MULT | Low-rank middle-block weights reduce per-iteration DRAM bandwidth by rank factor r, multiplied by T iterations. [see research_3_4.md, research_2_2.md] |
| 3.4 Recursive Internal State | 6.1 InArch AR Loop | COMP | 6.1's loop module is exactly such a persistent hidden state; shared h_loop serves dual duty. [see research_6_1.md] |
| 3.5 Gated Internal DAG | 3.4 Recursive Internal State | COMP | DAG gates can be informed by internal latent state from 3.4. [see research_3_5.md] |
| 3.5 Gated Internal DAG | 3.6 Recursive Internal DAG | NEST | 3.6 is a strict superset — the DAG must prove viability before adding 3.6's recurrence. [see research_3_5.md] |
| 3.5 Gated Internal DAG | 5.8 Block Sparse Weights | COMP | DAG nodes are naturally block-sparse; block sparsity and DAG gating are synergistic. [see research_3_5.md] |
| 3.7 Learnable State Machine | 4.1 State Machine Core | NEST | Complete architectural overlap in FSM infrastructure (q_t + T_mat); joint 3.7+4.1 implementation strongly recommended over either alone. [see research_3_7.md, research_4_1.md] |
| 3.7 Learnable State Machine | 3.6 Recursive Internal DAG | COMP | Explicit FSM state tensor of 3.6 is the FSM state of 3.7. Highly synergistic. [see research_3_6.md, research_3_7.md] |
| 3.7 Learnable State Machine | 5.3 Grammar Attention | COMP | FSM state structures attention patterns and KV compression; natural joint design. [see research_3_7.md] |
| 4.1 State Machine Core | 3.1 Layer-Level MoE | COMP | FSM state as auxiliary input to MoE router at zero net cost (FSM state already computed). [see research_4_1.md] |
| 4.4 Skip List Layers (hybrid) | 5.2 LSTM-Gated Attention | COMP | Skip mechanisms apply equally to gated recurrent layers; hybrid Skip + LSTM-Gated is feasible. [see research_5_2.md] |
| 4.5 Learned Residual Flow | 4.2 Shared Core + LoRA | COMP | Static gating decides which LoRA adapters remain active in the surviving layers. [see research_4_5.md] |
| 4.5 Learned Residual Flow | 1.2 Per-Token Adaptive Depth | COMP | Static gating provides global skip topology; 1.2 provides per-token fine-tuning within that structure. [see research_4_5.md] |
| 4.6 Mixture of Models | 3.4 Recursive Internal State | COMP | MoMoRA's gated emission is a multi-model version of 3.4's looped latent computation. [see research_4_6.md] |
| 4.6 Mixture of Models | 1.1 Learnable Top-k | COMP | k (active models) made per-token adaptive. [see research_4_6.md] |
| 4.7 Compressed Dictionary | 2.2 Compressed Dense Layers | COMP | LM head stored in low-rank factorized form + context-conditioned row selection yields compounded savings. [see research_4_7.md] |
| 4.7 Compressed Dictionary | 5.1 TurboQuant | COMP | INT4/INT8 LM head quantization is a natural extension; LM head is permissive of aggressive quantization. [see research_4_7.md] |
| 5.1 TurboQuant | 5.7–5.10 Weight Compression | COMP | Fully orthogonal: 5.1 reduces KV cache bytes, 5.7–5.10 reduce weight bytes. Both applied simultaneously reduce bandwidth from two independent sources. [see research_5_1.md] |
| 5.2 LSTM-Gated Attention | 3.1 Layer-Level MoE | COMP | MoE FFN is orthogonal to attention mechanism; already the configuration of Baseline B. [see research_5_2.md] |
| 5.3 Grammar Attention | 5.1 TurboQuant | COMP | 5.3 selects which KV entries to retain; 5.1 compresses the retained ones. Natural two-stage compression. [see research_5_3.md] |
| 5.4 Linked Attention | 5.1 TurboQuant | MULT | Quantize retained entries after eviction; multiplicative compression. [see research_5_4.md] |
| 5.4 Linked Attention | 5.5 Ragged Window Attention | COMP | Local window (5.5) covers local tokens; linked attention (5.4) covers global distal tokens. Best combination; creates NSA-like architecture. [see research_5_4.md, research_5_5.md] |
| 5.5 Ragged Window Attention | 5.1 TurboQuant | MULT | (w_avg/s) × (bits/16) combined reduction; e.g., w_avg=2K and 4-bit KV: 16× × 4× = 64× KV memory reduction. [see research_5_5.md] |
| DeepSeek-V4 CSA/HCA | 5.4 Linked Attention | NEST | CSA is a trained compressed-block retrieval design: it compresses KV entries, uses an indexer to select top-k compressed blocks, and keeps a local sliding-window branch. This becomes the reference design for linked-attention variants. [see research_5_4.md] |
| DeepSeek-V4 CSA/HCA | 5.5 Ragged Window Attention | COMP | CSA/HCA use compressed global attention plus an uncompressed sliding-window branch, validating the local-retention requirement behind ragged-window proposals at 1M context. [see research_5_5.md] |
| DeepSeek-V4 CSA/HCA | 5.1 TurboQuant | COMP | V4 uses mixed BF16/FP8 KV storage and FP4 indexer QK computation; TurboQuant-style KV quantization remains orthogonal but must target heterogeneous CSA/HCA cache layouts. [see research_5_1.md] |
| DeepSeek-V4 mHC | 4.4 Skip List Layers (hybrid) | COMP | mHC validates residual enrichment while preserving stable residual highways; it supports the hybrid form but contradicts the strong residual-replacement form. [see research_4_4.md] |
| DeepSeek-V4 FP4 Experts | 5.7 Block Compressed Weights | COMP | FP4 MoE expert weights at frontier scale strengthen the case for 4-bit expert-weight compression, though V4 evidence is MXFP4/QAT-oriented rather than generic INT4 PTQ. [see research_5_7.md] |
| DeepSeek-V4 Hash Early MoE | 1.4 Learned Dense/Sparse Layer | COMP_RISK | V4 replaces early dense FFNs with Hash-routed MoE layers, creating a new baseline for layer-type assignment: learned dense/sparse must beat all-MoE plus deterministic early routing. [see research_1_4.md] |
| DeepSeek-V4 On-Policy Distillation | 4.6 Mixture of Models | OPER | V4 trains domain specialists and consolidates them through full-vocabulary OPD, providing an operational alternative to runtime mixture-of-model routing. [see research_4_6.md] |
| 5.6 Double Attention | 5.4 Linked Attention | COMP | First pass identifies relevant tokens; second pass attends only to first-pass top-k. Eliminates TTFT penalty from second pass. Strongest combination. [see research_5_6.md] |
| 5.6 Double Attention | 5.5 Ragged Window Attention | COMP | First pass wide/global; second pass narrow local window. Analogous to DaViT spatial+channel decomposition. [see research_5_6.md] |
| 5.7 Block Compressed Weights | 5.1 TurboQuant | COMP | Fully orthogonal: 5.7 reduces weight bytes, 5.1 reduces KV cache bytes. [see research_5_1.md] |
| 5.8 Block Sparse Weights | 5.7 Block Compressed Weights | COMP | Different compression axes within weight matrices; can be combined if designed carefully. [see research_2_2.md] |
| 5.9 Dynamic Numeric Type | 5.8 Block Sparse Weights | COMP | Orthogonal; quantization applies on top of either sparsity type (S→Q ordering must be respected). [see research_1_5.md] |
| 6.1 InArch AR Loop | 5.2 LSTM-Gated Attention | COMP | LSTM gate's recurrent state can be fused with loop module's GRU state, reducing redundant state maintenance. [see research_6_1.md] |
| 6.1 InArch AR Loop | 4.1 State Machine Core | COMP | Loop module halt state can be incorporated as a terminal state in the FSM, unifying architecture. [see research_6_1.md] |
| 6.2 Prefill/Decode Split | 3.4 Recursive Internal State | COMP | Decode sub-network operates entirely in recurrent mode, receiving compressed state from prefill sub-network as initial recurrent state. [see research_6_2.md] |
| 6.2 Prefill/Decode Split | 6.1 InArch AR Loop | COMP | Decode sub-network naturally hosts the loop module; all stopping logic lives in decode sub-network. [see research_6_2.md] |
| 6.2 Prefill/Decode Split | 6.3 Block Diffusion AR Decoder | MULT | FLOP savings multiply: (L_dec/L) × (D/k); multiplicative because optimizations act on orthogonal dimensions (layers vs. tokens-per-pass). [see research_6_2.md, research_6_4.md] |
| 6.3 Block Diffusion AR Decoder | 5.2 LSTM-Gated Attention | COMP | DeltaNet state cache provides O(d²) prior-context summary vs O(s×d) KV; high memory efficiency at long contexts. [see research_6_3.md] |
| 6.4 Combined AR+Split+Diffusion | 6.1 InArch AR Loop | NEST | 6.4 integrates 6.1+6.2+6.3 into a unified system; 6.1 provides learned stopping component. [see research_6_4.md] |
| 6.4 Combined AR+Split+Diffusion | 6.2 Prefill/Decode Split | NEST | 6.4 integrates 6.2; prefill-decode split is one of three components. [see research_6_4.md] |
| 6.4 Combined AR+Split+Diffusion | 6.3 Block Diffusion AR Decoder | NEST | 6.4 integrates 6.3; block-diffusion decoder is one of three components. [see research_6_4.md] |
| 6.5 Pre-Attention Expert Router | 3.1 Layer-Level MoE | COMP | Combined "Layer-Level Pre-Attention Router" addresses two orthogonal routing dimensions simultaneously. [see research_6_5.md] |
| 6.5 Pre-Attention Expert Router | 1.1 Learnable Top-k | COMP | Q projection may be a better-conditioned routing input; combined signal determines both which experts and how many. [see research_6_5.md] |
| 6.5 Pre-Attention Expert Router | 3.3 Dynamic Expert Router | COMP | Pre-attention routing confidence from Q projections determines both which experts activate AND how many. [see research_6_5.md, research_3_3.md] |

---

## Conflict Map

| Idea A | Idea B | Type | Description |
|--------|--------|------|-------------|
| 1.2 Per-Token Adaptive Depth | 3.1 Layer-Level MoE | COMP_RISK | Two-level routing hierarchy (per-layer MoE selection + per-token depth exit) may be difficult to train stably. [see research_1_2.md] |
| 1.5 Learned Sparsity Type | 4.3 LoRA Everywhere | INCO | N:M sparsity masks + LoRA delta matrices interact non-trivially; merging breaks N:M sparsity and requires re-pruning. MoE-assigned layers remain LoRA compatible. [see research_1_5.md] |
| 1.6 Learned Layer Type | 4.2 Shared Core + LoRA | INCO | Weight sharing across layers is incompatible if layers have different primitive types. [see research_1_6.md] |
| 1.6 Learned Layer Type | 3.4 Recursive Internal State | COMP_RISK | Looping computation assumes fixed layer structure; mixing primitives with a loop complicates state management. [see research_1_6.md] |
| 2.1 Hierarchical Frequency Dictionary | Standard Weight-Tied Embeddings | INCO | Cascade structure requires rethinking weight tying; must use partially-tied or untied weights. [see research_2_1.md] |
| 2.2 Compressed Dense Layers | 5.8 Block Sparse Weights | REDN | Both target MLP weight compression; applying both requires careful design to avoid double-counting. Best approach: use 5.8's sparse structure as R in 2.2 [see research_2_2.md]. |
| 3.1 Layer-Level MoE | 4.3 LoRA Everywhere | REDN | Sharing base weights across all expert blocks reduces between-expert differentiation. [see research_3_1.md] |
| 3.4 Recursive Internal State | 3.1 Layer-Level MoE | COMP_RISK | Router selecting entire blocks disrupts the fixed shared-middle-block structure of 3.4. [see research_3_4.md] |
| 3.5 Gated Internal DAG | 4.2 Shared Core + LoRA | ENGR | Which node gets which LoRA adapter creates a complex assignment problem. [see research_3_5.md] |
| 3.7 Learnable State Machine | Baseline A1 DeltaNet layers | COMP_RISK | DeltaNet linear attention already maintains O(d²) recurrent state; two competing state mechanisms create gradient competition risk. [see research_3_7.md] |
| 4.4 Skip List Layers (strong form) | Any idea | INCO | Strong form DEPRIORITIZED: gradient attenuation of ~10⁻¹⁰ between skip points makes training infeasible per He et al. 2016 analysis. [see research_4_4.md] |
| 4.6 Mixture of Models | 4.2 Shared Core + LoRA | INCO | MoMoRA requires heterogeneous models; single shared core undermines model diversity. [see research_4_6.md] |
| 5.2 LSTM-Gated Attention | 5.5 Ragged Window Attention | REDN | LSTM forgetting mechanism already compresses history; variable window is redundant for gated-recurrent layers. [see research_5_5.md] |
| 5.3 Grammar Attention | 5.4 Linked Attention | COMP_RISK | Both compress KV; apply to different layer ranges if combined to avoid interference. [see research_5_4.md] |
| DeepSeek-V4 CSA/HCA | Standalone 5.3/5.4/5.5 Novelty Claims | REDN | Generic compressed sparse attention, compressed-block retrieval, and compressed-global-plus-local-window claims now substantially overlap with DeepSeek-V4; novelty must target a distinct mechanism or post-hoc compatibility. |
| 5.6 Double Attention | 5.3 Grammar Attention | COMP_RISK | Grammar pass + second unstructured pass may violate grammar constraints. [see research_5_6.md] |
| 3.8 Trainable Activation | Fixed-Threshold Sparse Compute | INCO | If learned α ≠ 0, standard zero-detection sparse kernels need modification; re-parameterization tricks only valid for linear activations. [see research_3_8.md] |

---

## Per-Idea Synergy Summary

| ID | Name | Top Synergies | Key Conflicts |
|----|------|---------------|---------------|
| 1.1 | Learnable Top-k | 1.3, 2.2, 5.1, 1.4, 6.5, 3.3 | Expert Choice Routing (autoregressive incompatibility), SeqTopK (no TPOT gain at sequence level) |
| 1.2 | Per-Token Adaptive Depth | 1.1, 3.4, 4.2, 5.1, 5.4 | 3.1 (training stability risk), fixed-depth inference engines |
| 1.3 | Per-Layer Adaptive Expert Count | 1.1, 1.4, 2.2, 3.1, 5.1 | None stated |
| 1.4 | Learned Dense/Sparse Layer | 1.1, 1.3, 2.2, 5.1 | Training complexity (2× FLOPs in two-stage variant) is the primary risk, not cross-idea conflicts |
| 1.5 | Learned Sparsity Type | 1.1, 1.3, 5.8, 5.9 | 4.3 (LoRA+N:M masks incompatible) |
| 1.6 | Learned Layer Type | 1.1, 1.2, 1.4, 5.1–5.5 | 4.2 (weight sharing + type mixing), 3.4 (loop + type mixing) |
| 1.7 | Dynamic Vocabulary | 2.1, 4.7 (MULT), 5.8, speculative decoding | Standard weight-tied embeddings |
| 2.1 | Hierarchical Frequency Dictionary | 1.7, 4.7, 2.2, 5.1 | Standard weight-tied embeddings, speculative decoding (may be redundant) |
| 2.2 | Compressed Dense Layers | 1.1, 1.3, 4.2, 4.3, 5.1, 5.7, 3.4 (MULT) | 5.8 (redundant MLP compression if both applied naively) |
| 3.1 | Layer-Level MoE | 1.1, 5.1, 1.6, 4.2, 6.5 | 4.3 (expert differentiation), standard GQA (per-expert KV conflicts) |
| 3.2 | Swappable Experts | 3.3, 4.2, 4.3, 1.1 | Static compilation (TRT-LLM), full model quantization |
| 3.3 | Dynamic Expert Router | 3.2, 6.5, 1.1, 1.3 | Fixed-topology architectures |
| 3.4 | Recursive Internal State | 3.6, 3.7, 4.2, 2.2 (MULT), 6.1 | 1.2 (redundant if combined naively), 3.1 (disrupts fixed block structure) |
| 3.5 | Gated Internal DAG | 3.4, 3.6, 1.2, 5.8, 1.6 | 4.2 (LoRA assignment complexity), Baseline B MoE (competing FFN architectures) |
| 3.6 | Recursive Internal DAG | 3.4, 3.5, 3.7, 4.2, 2.2 | 3.1 (disrupts fixed shared structure), 3.4 (redundant if 3.6 deployed) |
| 3.7 | Learnable State Machine | 3.6, 4.1 (NEST — must co-develop), 1.2, 3.5, 3.1, 5.3 | Baseline A1 DeltaNet (competing state), batched inference (hard routing incompatibility) |
| 3.8 | Trainable Activation | Quantization-aware training (PACT reframe), MoE per-expert | Fixed-threshold sparse compute |
| 4.1 | State Machine Core | 3.7 (NEST), 3.1, 1.2, 3.5 | Batched inference (hard routing), Baseline A1 DeltaNet (competing state) |
| 4.2 | Shared Core + LoRA | 1.2, 3.4, 3.6, 3.1, 4.5, 2.2 | 4.6 (heterogeneous pool incompatible), 1.6 (type mixing) |
| 4.3 | LoRA Everywhere | 2.2, 3.2, 4.2 | 3.1 (shared base reduces expert differentiation), 1.5 (N:M mask+LoRA) |
| 4.4 | Skip List Layers (hybrid only) | 4.5, 5.2 | Strong form: fundamentally infeasible (gradient attenuation) |
| 4.5 | Learned Residual Flow | 4.2, 1.2, ShortGPT pruning baseline | 3.4/3.6 (training-phase gate-recursion interaction) |
| 4.6 | Mixture of Models | 3.4, 1.1, 3.2 | 4.2 (shared core incompatible with heterogeneous models) |
| 4.7 | Compressed Dictionary | 2.1 (hybrid), 2.2, 5.1, 1.7 (MULT), speculative draft models | None stated; value is highly deployment-scenario-dependent |
| 5.1 | TurboQuant | 1.2, 1.6, 5.4 (MULT), 5.5 (MULT), 5.7–5.10, 4.7 | None — fully orthogonal to all weight and attention ideas |
| 5.2 | LSTM-Gated Attention | 3.1, 4.4, 5.1, 6.3 | KV-based ideas become irrelevant if KV cache eliminated |
| 5.3 | Grammar Attention | 5.1, 5.2 | 5.4 (apply to different layer ranges), 5.6 (grammar+unstructured pass) |
| 5.4 | Linked Attention | 5.1 (MULT), 5.5, 1.6, 5.2 | 5.3 (different layer ranges), 3.4 (multiple passes compound eviction) |
| 5.5 | Ragged Window Attention | 5.1 (MULT), 5.4, 5.3 | 5.2 (LSTM makes it redundant), Baseline A1/B DeltaNet (no KV cache) |
| 5.6 | Double Attention | 5.4 (strongest combination), 5.5, 4.2, 1.2 | 5.3 (grammar violation risk), ideas increasing depth |
| 5.7 | Block Compressed Weights | 5.1, 2.2 | None stated (Tier 1 INT4 is production-ready today) |
| 5.8 | Block Sparse Weights | 5.7, 5.9, 1.5, 3.5 | 2.2 (redundant MLP compression if naively combined) |
| 5.9 | Dynamic Numeric Type | 5.8, 1.5 | Per-scalar approach (converges to SpQR/OWQ — already published) |
| 5.10 | Block Dynamic Compression | 5.7, 5.8 | Per-element k-level flags (b_eff=6.37 bits — not competitive vs INT4); k=2 interleaved layout is the viable path |
| 6.1 | InArch AR Loop | 3.4, 1.2, 4.1, 5.2 | None; overlaps heavily with LoopFormer (arXiv:2602.11451) |
| 6.2 | Prefill/Decode Split | 3.4, 6.1, 6.3 (MULT), serving disaggregation (Splitwise/DistServe) | None stated |
| 6.3 | Block Diffusion AR Decoder | 5.2, 1.2, 5.5, 3.1 | Fully published as BD3-LM (ICLR 2025 Oral); most synergy value in 6.4 context |
| 6.4 | Combined AR+Split+Diffusion | 6.1, 6.2, 6.3 (all NEST), serving infra | Requires full custom training; highest implementation cost in Group 6 |
| 6.5 | Pre-Attention Expert Router | 3.3 (direct), 1.1, 3.1 | SwitchHead Q/K result may partially bound expectations; V5 complexity is high |

---

## Notes on DeepSeek-V4 Interactions

DeepSeek-V4-Pro (1.6T total, 49B active, 1M context) adds a new evidence baseline distinct from A/B/C. The following interactions should be reflected in future per-idea updates:

**1. CSA/HCA compresses the 5.3/5.4/5.5 design space:** V4's hybrid attention combines 4× compressed sparse attention, 128× heavily compressed dense attention, and a sliding-window branch. This validates the broad KV-compression stack but erodes novelty for generic learned sparse/compressed attention claims.

**2. mHC changes residual-architecture framing:** V4 uses manifold-constrained residual mixing rather than sparse skip replacement. This strengthens residual enrichment and learned residual flow, while keeping strong skip-list replacement deprioritized.

**3. Fixed-k MoE remains a real frontier baseline:** V4 uses 384 routed experts with 6 active routed experts plus one shared expert, not variable-k routing. Ideas 1.1 and 1.3 remain open, but their evaluations should compare against V4's fixed-k, all-MoE, early Hash-routed design.

**4. FP4 is now frontier-scale evidence for expert compression:** V4's instruct checkpoints use FP4 routed expert parameters and FP8 for most other parameters. This supports 5.7/5.10 hardware-aware low-precision research, but does not validate arbitrary per-element type metadata.

**5. OPD is an alternative to runtime model mixtures:** V4 trains specialists and consolidates them with on-policy distillation. For 4.6, this is a lower-latency alternative to keeping multiple specialists live behind a router.

---

## Notes on Baseline C Interactions

Baseline C (K2 family, 72.55B dense, d=8192, d_ff=28672, ~145 GB weight BW) significantly shifts the relative attractiveness of several ideas compared to Baselines A1/A2/B:

**1. Weight-compression ideas (5.7, 5.8, 5.9, 5.10, 2.2) — MORE ATTRACTIVE at Baseline C:** The larger weight footprint (~145 GB vs ~64 GB for A2) means every percentage-point reduction in weight bandwidth yields ~2.3× larger absolute TPOT savings. At r=256 for Idea 2.2 on Baseline C, MLP compression is ~96% (vs ~94% for A2), delivering ~3.5–5.0× TPOT improvement vs ~3.4–4.0× for A2. INT4 block quantization (5.7 Tier 1) is especially attractive at this weight size [see research_2_2.md, research_5_7.md].

**2. LoRA/shared-weight ideas (4.2, 4.3) — MORE ATTRACTIVE at Baseline C:** At d=8192 vs d=5120 (A2), the weight matrices scale as d², so LoRA and shared-weight compression deliver proportionally larger storage reductions. The 17× full-model storage reduction at r=64 for A2 scales to approximately 27× for Baseline C at the same rank. This makes 4.3-C (pure A·B inference) more compelling as a training paradigm for Baseline C architecture design [see research_4_2.md, research_4_3.md].

**3. MoE routing ideas (1.1, 1.3, 1.4, 1.5, 3.1, 3.2, 3.3, 6.5) — NOT DIRECTLY APPLICABLE to Baseline C:** Baseline C is dense. All MoE routing ideas are inapplicable without a prior sparse upcycling step. This is the largest change in synergy picture for Baseline C: roughly 12 of the 39 ideas are either inapplicable or require a major architectural change before being relevant.

**4. KV cache ideas (5.1, 5.4, 5.5) — SIGNIFICANTLY MORE ATTRACTIVE at Baseline C:** KV cache at 32K context is ~10.0 GiB for Baseline C vs ~8.59 GB for A2. At 262K context, KV reaches ~80 GiB — making TurboQuant (5.1) deliver ~1.33× TPOT at 262K (vs ~1.63× for A2 at the same context, due to Baseline C's larger weight footprint dominating at short context). The relative attractiveness crosses over at ~50K context where KV bandwidth begins to approach weight bandwidth for C [see research_5_1.md].

**5. Ideas 6.2–6.4 (Generation paradigm) — APPLICABLE at Baseline C:** The prefill/decode split and block-diffusion ideas apply equally to dense architectures. Baseline C's dense DeltaNet-free structure may actually simplify the 6.2 prefill state interface (no recurrent state to handoff; compressed KV instead). The 6.4 combined system remains the highest-novelty direction for dense models of this class [see research_6_2.md, research_6_4.md].
