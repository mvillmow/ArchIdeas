# Research: Learned Residual Flow Control
## ID: 4.5

## Executive Summary

**Novelty verdict:** PARTIAL — ~70% component coverage; MoD [11] / GateSkip [10] / MUDDFormer [8] cover per-token dynamic routing, DenseFormer [4] / ELC-BERT [5] / Hyper-Connections SHC [6] cover soft blending frozen at inference, ShortGPT [9] and Reassessing Layer Pruning [19] cover post-hoc importance metrics; residual novelty is hard/binary + static (not per-token) + training-time gate discovery + physical model surgery on decoder-only LLMs at pretraining scale ([ShortGPT, ACL Findings 2025, 9], [GateSkip, ICLR 2026, 10], [MoD, Raposo et al., 2024, 11]).

Idea 4.5 proposes training scalar gates per layer to discover which residual updates to skip, then freezing gates after training. Zero-gate layers are physically removed from the model (model surgery), reducing TPOT and TTFT proportionally to the fraction of pruned layers — with zero inference-time gate overhead.

Two key distinctions from published work: (1) Unlike Mixture of Depths (MoD)[11] and GateSkip[10], the decisions are static (frozen, same for all tokens), not dynamic per-token. (2) Unlike DenseFormer[4] and Hyper-Connections SHC[6], the gate is hard binary (keep/remove), not a learned blend — enabling physical layer removal rather than soft weighting.

ShortGPT[9] establishes empirically that ~24–27% of transformer layers are genuinely redundant (BI ≈ 0). Lu et al.[19] further show that simple reverse-order pruning (last 25% of layers) outperforms complex importance metrics on modern LLMs — this sets a demanding post-hoc baseline. If training-time static gates can discover comparable or better pruning than the best post-hoc methods, the speedup is ~0.75× TPOT and TTFT with no per-inference overhead.


## Key Comparison Tables

### vs. Baseline A1 (Qwen3.5-27B Hybrid)

| Metric | Baseline A1 | This Idea (p=0.75, 25% pruned) |
|--------|------------|-------------------------------|
| Compute (FLOPs/token) | O(L·(d²+s·d/4+d·d_ff)) | O(p·L·(d²+s·d/4+d·d_ff)) |
| Memory bandwidth (decode) | O(L·(d²+d·d_ff+s·d_kv/4)) | O(p·L·(d²+d·d_ff+s·d_kv/4)) |
| KV cache (32K ctx) | ~2.15 GB (16 full-attn layers) | ~1.6 GB (modest — A1 already sparse) |
| KV cache (262K ctx) | ~17.2 GB | ~12.9 GB |
| Weight memory | O(L·(d·d_ff+d²)) | O(p·L·(d·d_ff+d²)+L·d) |
| Training cost | 1.0× | ~1.0× (scalar gates: negligible overhead; +5–10% for post-pruning fine-tune) |
| TTFT (8K prompt) | ref | ~0.75× (compute-bound; MLP=~85–90% of FLOPs) |
| TPOT (batch=1) | ref | ~0.75× (⚠ conditional on model surgery variant; soft-gate variant = ref) (bandwidth-bound; active weight bytes scale as p) |
| Quality (benchmark avg) | ref | −0.6 to −3.2% (ShortGPT range; training-time gating may improve) |

### vs. Baseline A2 (Qwen3-32B Dense)

| Metric | Baseline A2 | This Idea (p=0.75, 25% pruned) |
|--------|------------|-------------------------------|
| Compute (FLOPs/token) | O(L·(s·d+d·d_ff)) | O(p·L·(s·d+d·d_ff)) |
| Memory bandwidth (decode) | O(L·(d·d_ff+s·d_kv)) | O(p·L·(d·d_ff+s·d_kv)) |
| KV cache (32K ctx) | ~8.59 GB | ~6.4 GB (all layers full-attn; larger per-layer saving than A1) |
| Weight memory | ~64 GB bf16 | ~48 GB (physical removal of ~16 layers) |
| Training cost | 1.0× | ~1.0–1.10× |
| TTFT (8K prompt) | ref | ~0.75× |
| TPOT (batch=1) | ref | ~0.75× (⚠ conditional on model surgery variant; soft-gate variant = ref) |
| Quality (benchmark avg) | ref | −0.6 to −3.2% (ShortGPT range at 7B–13B) |

### vs. Baseline B (Qwen3.5-397B-A17B MoE)

| Metric | Baseline B | This Idea (p=0.75, 25% pruned) |
|--------|-----------|-------------------------------|
| Compute (FLOPs/token) | O(L·(d²+s·d/4+k·d·d_e)) | O(p·L·(d²+s·d/4+k·d·d_e)) |
| Memory bandwidth (decode) | O(L·(k·d·d_e+state)) | O(p·L·(k·d·d_e+state)) |
| KV cache (32K ctx) | ~1.0 GB (15 full-attn, 32Q/2KV, hd=256) | ~0.75 GB |
| Expert weight memory | O(L·E·d·d_e) | O(p·L·E·d·d_e+L·d) |
| Training cost | 1.0× | ~1.0–1.10× |
| TTFT (8K prompt) | ref | ~0.75× |
| TPOT (batch=1) | ref | ~0.75× (⚠ conditional on model surgery variant; soft-gate variant = ref) |

### vs. Baseline C (K2 family, 72.55B dense Llama-arch)

| Metric | Baseline C (K2) | This Idea (p=0.75, L=80) |
|--------|----------------|--------------------------|
| Compute (FLOPs/token) | O(L·(s·d+d·d_ff)), L=80 | O(p·L·(s·d+d·d_ff)) |
| Memory bandwidth | ~145.1 GB weight BW | ~108.8 GB (0.75×) |
| KV cache (32K ctx) | ~10.0 GiB | ~7.5 GiB |
| Weight memory | ~72.55B params | ~54.4B params (20 layers pruned) |
| MLP FLOPs/token/layer | ~4.70 × 10⁸ | ~4.70 × 10⁸ (per active layer; 20 layers removed) |
| TTFT (8K prompt) | ref | ~0.75× |
| TPOT (batch=1) | ref | ~0.75× (⚠ conditional on model surgery variant; soft-gate variant = ref) |

**Note:** For K2 (L=80), 25% pruning removes 20 layers → 60 active layers. This yields a model similar in depth to a K2 with 60 layers — comparable to A2 (64 layers) in depth but with K2's wider d=8192. TPOT benefit is proportionally the same.

---

## 1. Idea Description

**Mechanism:** Each transformer layer i has a learnable scalar gate g_i. During training:
```
x_{i+1} = x_i + g_i · F_i(RMSNorm(x_i))
```
where g_i is trained with a sparsity regularization loss to converge toward 0 or 1. After training, g_i ≤ threshold are set to exactly 0, and those layers are physically removed from the model checkpoint (model surgery). Remaining layers are fine-tuned on 5–10% of original training budget.

**Critical distinction from dynamic gating:** The gate values are static after training — every token sees the same pruned or active layer topology. This contrasts with MoD[11] (per-token routing) and GateSkip[10] (per-token threshold decisions) and delivers zero per-inference routing overhead.

**Important caveat:** Soft (blend) gating saves no compute at inference — F_i(x_i) must still be computed before multiplying by g_i ≈ 0. Only model surgery (physical layer removal) delivers the claimed speedup. Frozen soft gates ≠ speedup.

**Practical training recipe:**
1. Initialize gate logits to −2 (near-zero gate output)
2. Ramp sparsity regularization λ from 0 over 20% of training
3. Enforce p_active floor of 0.5 to prevent total gate collapse
4. Post-training: threshold gates at g_i < 0.5 → remove; g_i ≥ 0.5 → keep
5. Apply model surgery (physically remove pruned layers and their weight tensors)
6. Fine-tune remaining model for 5–10% of original training budget

---

## 2. Literature Review

### Highway Networks (Srivastava et al., 2015)
Gated residual mechanism[1] for very deep feedforward networks: y = H(x)·T(x) + x·(1−T(x)), T = sigmoid gate initialized to −1 to −3. Enables 900+-layer networks via SGD. Foundational prior art for learned gated residuals. Key distinction: gates are dynamic per-token at inference, not static/frozen after training. NeurIPS 2015, arXiv:1505.00387.

### Stochastic Depth (Huang et al., 2016)
Random per-batch layer dropping[2] during training (linearly decaying survival probability). At test time, all layers are active. Demonstrates networks can be trained robustly with layers skipped during training. Key distinction: random dropout only, not learned; full model used at inference. ECCV 2016, arXiv:1603.09382.

### LayerDrop (Fan et al., 2020)
Structured per-layer random dropout[3] during training: trains transformers to be robust to any subset of layers at inference. The user selects inference depth post-training. Key distinction: random training-time dropout (not learned optimal topology); user-selected (not auto-discovered) inference depth. Directly foundational for the "train resilience to layer dropping" concept. arXiv:1909.11556. ICLR 2020.

### DenseFormer (Pagliardini et al., NeurIPS 2024)
Static-at-inference learned aggregation[4]: DWA module after each block blends all past layer outputs with learned weights α (trained, then frozen). 48-block DenseFormer: PPL 17.84 vs 18.61 baseline on OpenWebText2. Throughput: 4.65 vs 5.94 batches/sec (78% = −22% overhead). Model preferentially routes from embedding and immediate preceding layers; some negative weights at depth. Key distinction: soft blending (not binary), O(L·d) memory overhead for storing all intermediate representations. NeurIPS 2024, arXiv:2402.02622.

### ELC-BERT (Charpentier and Samuel, 2023)
Static learned depth aggregation[5] in encoder-only transformers: each layer selects convex combination of past layer outputs (trained, frozen). Winner of BabyLM 2023 strict and strict-small tracks. Demonstrates static learned depth aggregation outperforms standard residuals on data-limited tasks. Key distinction: encoder-only; continuous blend (not binary). BabyLM/CoNLL 2023, arXiv:2311.02265.

### Hyper-Connections (Zhu et al., ICLR 2025)
Learnable matrix of connection strengths[6] across depths and widths. Static Hyper-Connections (SHC): trained, then frozen. Dynamic Hyper-Connections (DHC): per-token via small linear layer + tanh. On OLMoE-1B-7B: 1.8× faster convergence, +6 ARC-Challenge (41.8→47.8). On OLMo-7B: PPL 14.316→14.023. Key distinction: SHC is soft blending (not binary removal); expansion matrix overhead ~5–15% at inference (larger than scalar gates). ICLR 2025, arXiv:2409.19606.

### ResiDual (Xie et al., 2023)
Concurrent Pre-LN and Post-LN dual residual streams[7] (Pre-Post-LN). Outperforms both baselines on MT benchmarks. Direct prior art for multiple concurrent residual streams. Key distinction: fusion is static (sum), not learned routing; no mechanism to selectively update one stream. arXiv:2304.14802.

### MUDDFormer (Xiao et al., ICML 2025)
Per-position, per-stream (Q, K, V, residual) dynamic dense connection weights[8]: MUDDPythia-2.8B matches Pythia-6.9B on Pile PPL (6.29 vs 6.29) and zero-shot avg (55.0% vs 55.1%), adding only 0.23% parameters. Inference 88–94% of baseline. Key distinction: dynamic per-token at inference (not static); four separate aggregation modules per layer. ICML 2025, arXiv:2502.12170.

### ShortGPT (Men et al., 2024)
Block Influence BI_i = 1 − E[cos(X_i, X_{i+1})][9] measures each layer's residual contribution. LLaMA 2-7B: removing 27.1% of layers → MMLU 45.39→43.96 (−3.2%). LLaMA 2-13B: removing 24.6% → MMLU 55.00→54.69 (−0.6%). Layers 21–29 in LLaMA 2-7B are most redundant. Post-hoc metric, not a learned gate. Direct empirical evidence that ~25% of transformer layers are genuinely near-zero contribution — the target for idea 4.5's gate discovery. ACL Findings 2025, arXiv:2403.03853.

### GateSkip (Laitenberger et al., ICLR 2026)
Sigmoid-linear gates at residual exit of each attention/MLP module[10]; trained with sparsity penalty; inference quantile thresholds convert gate values to hard per-token skip decisions. Abstract: up to 15% compute saved on long-form reasoning while retaining over 90% of baseline accuracy; instruction-tuned models match baseline quality near 50% savings; tradeoff improves with model scale. Most similar published mechanism; key distinction: gates are dynamic per-token (threshold fixed, but gate evaluation per-token). arXiv:2510.13876.

### Mixture of Depths (Raposo et al., 2024)
Per-token binary layer-skip routing[11] at 6B+ scale via learned top-k router. Each token independently routes through or bypasses transformer blocks. Achieves isoFLOP performance improvement vs standard dense models. **This is the most directly relevant prior work.** Idea 4.5's novelty vs MoD: MoD routes per-token dynamically; 4.5 learns one static topology (same decision for all tokens), eliminating per-token routing overhead. arXiv:2404.02258. DeepMind.

### LLM-Pruner (Ma et al., NeurIPS 2023)
Structural pruning of LLMs[12] via Taylor expansion importance scores for layer-level removal. Directly relevant to the post-training model surgery path: identifies which layers/components to remove based on gradient-weighted activation magnitude. NeurIPS 2023, arXiv:2305.11627.

### Sheared LLaMA (Xia et al., ICLR 2024)
Structured pruning plus continued pre-training[13] from a larger model. Demonstrates recovery from structural pruning via targeted training (Dynamic Batch Loading). Directly relevant to the 5–10% fine-tuning step post-model surgery. ICLR 2024, arXiv:2310.06694.

### DeepCrossAttention (Heddes et al., ICML 2025)
Depth-wise cross-attention with learnable input-dependent weights[14]: 3× faster training (not inference) while matching dense quality; adds 0.2% parameters. Provides quality ceiling for dynamic cross-layer routing. Dynamic per-token at inference. ICML 2025, arXiv:2502.06785.

### Attention Residuals (Kimi Team, 2026)
AttnRes: softmax attention over all preceding layer outputs along the depth axis[15]. Kimi Linear (48B/3B active), 1.4T tokens: +7.5 GPQA-Diamond, +3.1 HumanEval, +3.6 MATH vs standard residual. Block AttnRes partitions layers into blocks. Dynamic per-token. arXiv:2603.15031.

### SkipNet (Wang et al., ECCV 2018)
Learned per-input layer skipping[16]: gating network decides per-input whether to execute or skip each block via REINFORCE. 30–90% compute reduction on four benchmarks while preserving accuracy. Key distinction: dynamic per-input (not static frozen); CNN domain. ECCV 2018, arXiv:1711.09485.

### Lottery Ticket Hypothesis (Frankle and Carbin, 2019)
Sparse subnetworks ("winning tickets")[17] exist in dense networks that, when trained from same initialization, match full-network accuracy. Conceptual foundation for "training discovers which connections to use." Key distinction: iterative magnitude pruning of weights, not architectural layer-level gating; no residual-stream analysis. ICLR 2019, arXiv:1803.03635.

### NAS Skip Connection Generalization (Zhu et al., NeurIPS 2022)
Theoretical framework[18] for NAS over skip connections via NTK eigenvalue bounds; train-free algorithm for architecture selection over skip/residual patterns. Theoretical justification that learned skip patterns outperform uniform residuals. NeurIPS 2022, arXiv:2209.07238.

### Reassessing Layer Pruning in LLMs (Lu et al., 2024)
Post-hoc layer pruning study[19] showing that pruning the final 25% of layers followed by fine-tuning the lm_head and remaining last three layers outperforms more complex layer-selection metrics on Llama-3.1-8B-Instruct. Challenges the assumption that sophisticated importance metrics (BI, Taylor expansion) are necessary — simple reverse-order pruning is a highly competitive baseline. Key distinction: post-hoc, not training-time gate discovery; no learned topology. Directly relevant as the simplest possible baseline for idea 4.5's ablation (variant 2 should include this reverse-order baseline, not only ShortGPT BI). arXiv:2411.15558, 2024.

---

## 3. Prior Art Classification

- **Status**: PARTIAL (~70% component coverage)
- **Novel contribution**: Training a gate to discover a single static topology (one binary keep/remove decision per layer, identical for all tokens) that is physically instantiated as a shorter model at inference — as opposed to:
  - MoD[11] (per-token dynamic routing), GateSkip[10] (per-token threshold), MUDDFormer[8] (per-token dynamic): all are dynamic
  - DenseFormer[4], ELC-BERT[5], Hyper-Connections SHC[6]: soft blending frozen at inference, not binary removal
  - ShortGPT[9], Reassessing Layer Pruning[19]: post-hoc importance metrics or simple reverse-order removal, not training-time gate discovery
  - LayerDrop[3]: random training-time dropout, user-selected inference depth

The specific combination of (a) hard/binary residual skip decisions, (b) frozen static decisions (not per-token), (c) discovered end-to-end via gradient-based training (not post-hoc), and (d) physically instantiated via model surgery for zero inference overhead is not demonstrated for decoder-only LLMs at pretraining scale.

---

## 4. Technical Analysis

### 4.1 Theoretical Complexity

Variables: L=layers, d=hidden, d_ff=MLP intermediate, s=seq length, p=fraction of active layers (0.75 for 25% pruned)

| Metric | This Idea | Baseline A1 | Baseline A2 | Baseline B |
|--------|-----------|------------|------------|-----------|
| Compute per token | O(p·L·(d²+s·d/4+d·d_ff)) | O(L·(d²+s·d/4+d·d_ff)) | O(L·(s·d+d·d_ff)) | O(L·(d²+s·d/4+k·d·d_e)) |
| KV cache | O(p·L/4·s·d_kv) [A1/B] or O(p·L·s·d_kv) [A2] | O(L/4·s·d_kv) | O(L·s·d_kv) | O(L/4·s·d_kv) |
| Weight memory | O(p·L·(d·d_ff+d²)+L·d) | O(L·(d·d_ff+d²)) | O(L·(d·d_ff+d²)) | O(L·E·d·d_e) |
| Memory bandwidth (decode) | O(p·L·(d²+d·d_ff+s·d_kv/4)) [A1] | O(L·(d²+d·d_ff+s·d_kv/4)) | O(L·(d·d_ff+s·d_kv)) | O(L·(k·d·d_e+state)) |
| TTFT (prefill) | ↓ p× vs baseline | ref | ref | ref |
| TPOT (decode, batch=1) | ↓ p× vs baseline | ref | ref | ref |

**Training overhead for scalar gates:** ~1.00× (negligible — scalar gate adds ~L×batch×s sigmoid ops per step vs ~10¹⁵ baseline FLOPs). Post-pruning fine-tuning adds 5–10% of original training budget.

### 4.2 Key Speedup Analysis

At p=0.75 (25% pruning, consistent with ShortGPT's 24–27% redundant layers):

- **TPOT:** Bandwidth-bound at batch=1. Weight bytes scale as p → TPOT ≈ 0.75×. KV cache access also reduces proportionally.
- **TTFT:** Compute-bound. MLP FLOPs dominate (85–90% of total at s=8K). Uniform 25% layer pruning → total FLOPs ≈ 0.75× → TTFT ≈ 0.75×.

**Hard-zero gating requirement:** Soft gates (g_i ≈ 0 but not 0) save nothing — F_i(x_i) is still computed. The 0.75× speedup requires model surgery (physical layer removal). Conditional execution (`if g == 0: skip`) avoids SIMD divergence in GPU execution (all threads take same branch), but model surgery is strictly preferred: produces a standard transformer with no gate infrastructure.

### 4.3 Memory Analysis

For A2 (d=5120, d_ff=25600, L=64):
- MLP weight per layer: 3 × 5120 × 25600 × 2 bytes ≈ **786 MB**
- Attention weight per layer: 4 × 5120² × 2 bytes ≈ **210 MB** (Q/K/V/O)
- Per-layer total: ~996 MB
- 16 pruned layers: ~15.9 GB weight reduction (from ~65 GB to ~49 GB)

---

## 5. Implementation Considerations

**Training:** Scalar gate + L1 sparsity regularization (GateSkip-style adaptive λ). Straight-through estimator or Gumbel-Softmax temperature annealing for hard binary gate training. Monitor active fraction per checkpoint; alert if below 0.50.

**Post-training surgery:**
1. Identify all layers with g_i < threshold (0.5)
2. Remove weight tensors for those layers from checkpoint
3. Update layer indices and forward pass graph
4. Fine-tune remaining model for 5–10% of original training budget

**Framework compatibility:** PyTorch `torch.jit.script` for conditional layer execution during training; JAX `jax.lax.cond`. Standard implementation post-surgery.

**Pre-norm interaction:** If g_i → 0 but pre-norm is not also skipped, the norm computation still runs (no compute saving). The full savings require: `if g_i < threshold: skip RMSNorm + sublayer entirely`. Ensure model surgery correctly removes both sublayer weights AND any associated pre-norm parameters.

---

## 6. Synergies

- **Combines well with:** 4.2 (Shared Core + LoRA — static gating decides which LoRA adapters remain active), 1.2 (Per-Token Adaptive Depth — static gating provides global skip topology; 1.2 provides per-token fine-tuning within that structure), LLM-Pruner[12] (complementary importance scoring for surgery guidance)
- **PARTIAL SYNERGY with 4.4 (Skip List Layers):** Requires careful composition order. Idea 4.5 gating should be applied first to determine surviving layers, then 4.4 applied to the resulting shorter architecture. Simultaneous or 4.4-first application creates layer-index conflicts for the exponential skip schedule.
- **PARTIAL CONFLICT with 3.4/3.6 (Recursive Internal DAG):** Training-phase interaction only — gates must stabilize before recursion depth is useful. Inference over pruned recursive model is compatible.

---

## 7. Risk Assessment

- **Technical risk:** MEDIUM. The static-at-inference constraint is the key unknown. GateSkip validates dynamic gating at 8B scale; ShortGPT validates ~25% redundancy. The critical question is whether static gates match post-hoc BI-based pruning — this is the blocking empirical question resolved by the recommended ablation.
- **Potential impact:** MEDIUM–HIGH. ~25% TPOT and TTFT reduction with zero inference overhead if 25% layers are stably pruneable.
- **Quality risk:** Model-scale dependent. LLaMA 2-13B shows −0.6% MMLU at 24.6% pruning; LLaMA 2-7B shows −3.2% at 27.1% pruning; 1B models show steep cliff (−26% relative at 25% savings). Behavior at 27B+ scale under training-time gating is unknown but likely more tolerant than 7B.
- **Multiple-stream variant:** DEPRIORITIZE. Memory doubles (~2× per stream), routing logic complex, no prior validation at LLM scale.

---

---

## 8. Accuracy / Quality Tradeoff

- **Expected quality delta vs baseline**: At 25% pruning, −0.6% MMLU on LLaMA 2-13B (55.00 → 54.69) and −3.2% on LLaMA 2-7B (45.39 → 43.96) via post-hoc BI-based layer removal [ShortGPT, 9, ACL Findings 2025]; GateSkip achieves >90% baseline accuracy at up to 15% compute savings on long-form reasoning via sigmoid residual gates, with instruction-tuned models matching baseline quality near 50% savings [GateSkip, 10, ICLR 2026]; MoD per-token dynamic routing achieves isoFLOP quality parity with dense baselines at 6B+ scale [MoD, 11, Raposo et al. 2024]; reverse-order pruning (last 25%) + lm_head/last-3-layer fine-tune outperforms BI on Llama-3.1-8B-Instruct [Lu et al., 19, arXiv 2024].
- **Known failure modes**: Scale-dependent cliff — 1B-class models show −26% relative at 25% savings (steep quality cliff) [4.5 §7]; at 7B, cliff is steeper (−3.2% MMLU) than at 13B (−0.6%) [ShortGPT, 9]; quality-aggressiveness relationship is concave with accelerating losses above 25–30% pruning; 27B+ scale behavior under training-time gating is uncharacterized; surviving layers cannot compensate for removed representations at >35% pruning [Sheared LLaMA, 13].
- **Empirical evidence**: ShortGPT Table (LLaMA 2-7B: 45.39 → 43.96 at 27.1% pruned; 13B: 55.00 → 54.69 at 24.6%) [ShortGPT, 9]; GateSkip §Results (>90% baseline accuracy at up to 15% long-form-reasoning compute savings; baseline match near 50% on instruction-tuned models) [GateSkip, 10]; Lu et al. §Experiments (reverse-order + lm_head fine-tune beats BI/Taylor) [Reassessing Layer Pruning, 19]; Sheared LLaMA §Results (Dynamic Batch Loading recovery after structural removal) [Sheared LLaMA, 13, ICLR 2024].
- **Mitigations**: Run 5–10% of original training budget as post-surgery fine-tune — matches Sheared LLaMA / LLM-Pruner recovery methodology [Sheared LLaMA, 13; LLM-Pruner, 12]; scope deployment to ≥13B models where ShortGPT shows <1% MMLU delta; benchmark training-time gate discovery against reverse-order-pruning baseline before adopting [Lu et al., 19]; cap pruning fraction at ≤25% to stay in the concave-plateau regime; avoid the multi-stream variant (memory doubles, no LLM-scale validation) [4.5 §7].

### Detailed breakdown

- **Reported quality delta (from closest analogues)**:
  - [Men et al., 2024][9] (ShortGPT — ACL Findings 2025): Removing 27.1% of LLaMA 2-7B layers → MMLU 45.39 → 43.96 (−3.2%). Removing 24.6% of LLaMA 2-13B layers → MMLU 55.00 → 54.69 (−0.6%). This is the primary empirical anchor: post-hoc layer removal at 25% pruning costs −0.6% (13B) to −3.2% (7B) on MMLU. Idea 4.5's training-time gate discovery is expected to find *better* pruning topologies than the post-hoc BI metric — but ShortGPT sets the quality floor that training-time gating must beat.
  - [Laitenberger et al., ICLR 2026][10] (GateSkip): dynamic per-token gating with sigmoid residual gates saves up to 15% compute while retaining over 90% of baseline accuracy on long-form reasoning; instruction-tuned models match baseline quality near 50% savings; tradeoff improves with model scale. This is the most directly relevant validation: sparsity-regularized residual gates do not substantially degrade quality up to 15–50% savings depending on task. Idea 4.5 uses static (frozen) rather than dynamic gates — which should be quality-neutral relative to GateSkip's dynamic regime, as static topology removes per-token routing overhead without changing the pruning decision quality.
  - [Raposo et al., 2024][11] (Mixture of Depths): per-token dynamic layer-skip routing at 6B+ scale achieves isoFLOP performance improvement — quality matched to dense baselines with lower active compute. This is the upper-bound quality reference: if dynamic per-token routing is quality-neutral, static per-layer routing (idea 4.5) eliminates routing overhead but may lose some adaptivity, costing a small quality delta on heterogeneous token distributions.
  - [Lu et al., 2024][19] (Reassessing Layer Pruning): simple reverse-order pruning (last 25% of layers) followed by lm_head + last-three-layer fine-tuning outperforms sophisticated metrics (BI, Taylor expansion) on Llama-3.1-8B-Instruct. This challenging baseline means that if training-time gate discovery merely matches reverse-order pruning, it provides no novelty benefit. The target quality outcome for idea 4.5 is: quality better than ShortGPT BI-based pruning at 25% removal, confirmed via ablation against the reverse-order baseline.
  - [Xia et al., ICLR 2024][13] (Sheared LLaMA): structured pruning plus continued pre-training (Dynamic Batch Loading) achieves quality recovery after aggressive structural removal. The 5–10% fine-tuning step post-surgery in idea 4.5 directly parallels Sheared LLaMA's recovery methodology, providing the strongest quality-recovery precedent.

- **Monotonicity**: Quality loss is **monotone with pruning aggressiveness (1−p)** in general: more layers removed → more quality lost. ShortGPT shows this relationship is scale-dependent — at 7B, the cliff is steeper (−3.2% MMLU at 27.1% removed) than at 13B (−0.6% at 24.6% removed). Risk Assessment (§7) notes "1B models show steep cliff (−26% relative at 25% savings)." The quality-aggressiveness relationship is concave: diminishing losses at low pruning rates (0–15%), accelerating losses above 25–30%. Training-time gate discovery is expected to find pruning topologies that are more favorable (less quality loss per unit compute saved) than post-hoc selection, flattening the monotone degradation curve.

- **Recovery**: Quality is recoverable via the 5–10% post-surgery fine-tuning step. Sheared LLaMA[13] and LLM-Pruner[12] provide validated recovery methodologies. The recovery mechanism is: (1) model surgery physically removes pruned layers, (2) remaining model is fine-tuned on 5–10% of original training budget, (3) remaining layers adapt to compensate for missing contributions. Recovery is most complete at moderate pruning rates (≤25%) and degrades at higher pruning (>35%), where surviving layers cannot compensate for the removed representations. Full recovery to baseline quality is not expected — the target is quality *competitive with* ShortGPT/reverse-order-pruning post-hoc methods, not exact recovery to the original model.

- **Conditions for acceptable degradation**: At 25% pruning (p=0.75): −0.6% MMLU (13B-class models per ShortGPT) in exchange for ~0.75× TPOT and TTFT with zero per-inference overhead is acceptable for: (1) throughput-optimized serving where latency SLAs are relaxed and quality tolerates a <1% absolute benchmark delta; (2) deployment on constrained hardware (fewer GPUs, lower VRAM) where the ~25% weight memory reduction enables serving a model that would otherwise not fit; (3) cost-reduction at large serving scale where 25% TPOT improvement compounds into substantial infrastructure savings at acceptable quality delta; (4) models at 13B+ scale where the ShortGPT precedent shows <1% MMLU delta, not at 7B scale where the −3.2% delta may exceed product quality thresholds.

---

<!-- CITATION MANIFEST -->

[1] Training Very Deep Networks (Highway Networks): Srivastava, Greff, Schmidhuber (NeurIPS 2015). arXiv:1505.00387. Gated carry/transform mechanism; dynamic per-token gates; foundational prior art.

[2] Deep Networks with Stochastic Depth: Huang, Sun, Liu, Sedra, Weinberger (ECCV 2016). arXiv:1603.09382. Random per-batch layer dropping; full model at inference; training regularizer concept.

[3] LayerDrop: Fan, Grave, Joulin (ICLR 2020). arXiv:1909.11556. Structured layer dropout trains robustness; user-selected inference depth; foundational for layer-dropping resilience.

[4] DenseFormer: Pagliardini, Mohtashami, Fleuret, Jaggi (NeurIPS 2024). arXiv:2402.02622. DWA static learned depth aggregation; PPL 17.84 vs 18.61; −22% throughput. Soft blend, not binary.

[5] ELC-BERT: Charpentier, Samuel (BabyLM/CoNLL 2023). arXiv:2311.02265. Static learned convex combination of past layers. Won BabyLM 2023. Encoder-only.

[6] Hyper-Connections: Zhu, Huang et al. (ICLR 2025). arXiv:2409.19606. SHC (static) and DHC (dynamic) learned multi-depth connectivity. +6 ARC-Challenge on OLMoE-1B-7B; 1.8× faster convergence.

[7] ResiDual: Xie, Zhang, Guo et al. (arXiv 2023). arXiv:2304.14802. Concurrent Pre-LN + Post-LN dual streams (Pre-Post-LN). Prior art for multiple concurrent residual streams.

[8] MUDDFormer: Xiao, Meng, Li, Yuan (ICML 2025). arXiv:2502.12170. Dynamic dense cross-layer connections per position, per Q/K/V/residual stream. 2.8B matches 6.9B quality. Dynamic.

[9] ShortGPT: Men, Xu, Zhang et al. (ACL Findings 2025). arXiv:2403.03853. BI metric identifies ~24–27% redundant layers. LLaMA 2-7B: −3.2% MMLU at 27.1% pruning. Empirical basis for 25% pruning target.

[10] GateSkip ("What Layers When: Learning to Skip Compute in LLMs with Residual Gates"): Laitenberger, Kopiczko, Snoek, Asano (ICLR 2026). arXiv:2510.13876. Sigmoid gates with sparsity penalty; up to 15% compute savings on long-form reasoning while retaining >90% baseline accuracy; instruction-tuned models match baseline near 50% savings; tradeoff improves with scale. Dynamic per-token.

[11] Mixture of Depths: Raposo, Ritter, Richards, Lillicrap, Humphreys, Santoro (DeepMind, 2024). arXiv:2404.02258. Per-token binary layer-skip routing at 6B+ scale; isoFLOP performance gains. Most directly relevant: dynamic per-token vs 4.5's static per-layer.

[12] LLM-Pruner: Ma et al. (NeurIPS 2023). arXiv:2305.11627. Structural LLM pruning via Taylor expansion importance scores. Relevant to post-training model surgery guidance.

[13] Sheared LLaMA: Xia et al. (ICLR 2024). arXiv:2310.06694. Structured pruning + continued pre-training (Dynamic Batch Loading). Relevant to post-surgery fine-tuning recovery.

[14] DeepCrossAttention: Heddes, Javanmard et al. (ICML 2025). arXiv:2502.06785. Learned input-dependent cross-layer depth attention; 3× faster training; 0.2% added params. Dynamic.

[15] Attention Residuals: Kimi Team (arXiv 2026). arXiv:2603.15031. Softmax attention over all preceding layers; Kimi Linear 48B/3A; +7.5 GPQA-Diamond, +3.1 HumanEval, +3.6 MATH. Dynamic.

[16] SkipNet: Wang, Yu, Dou, Darrell, Gonzalez (ECCV 2018). arXiv:1711.09485. Learned per-input layer skipping (REINFORCE); 30–90% compute reduction. Dynamic; CNN domain.

[17] Lottery Ticket Hypothesis: Frankle and Carbin (ICLR 2019). arXiv:1803.03635. Sparse winning-ticket subnetworks found via iterative pruning. Conceptual foundation for training-discovered sparse topology. (Note: author is Carbin, not Carlin.)

[18] NAS Skip Connection Generalization: Zhu, Liu, Chrysos, Cevher (NeurIPS 2022). arXiv:2209.07238. NTK-based theory for NAS over skip connection patterns; supports learned skip patterns outperforming uniform residuals.

[19] Reassessing Layer Pruning in LLMs: Lu, Cheng, Fang et al. (arXiv 2024). arXiv:2411.15558. Post-hoc pruning study showing reverse-order (last-25%) layer removal + lm_head fine-tune outperforms sophisticated metrics (BI, Taylor). Strong simple baseline for 4.5 ablation plan.
