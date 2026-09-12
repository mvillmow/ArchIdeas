# Research: LSTM-Gated Attention
## ID: 5.2

---

## Executive Summary

LSTM-Gated Attention proposes replacing the softmax normalization in standard self-attention with LSTM-style data-dependent gating mechanisms, eliminating the growing KV cache and replacing it with a fixed-size recurrent state (O(H × d_kv²) per layer, constant in sequence length). The primary benefit is TPOT reduction at long contexts by eliminating sequential KV cache memory reads.

**Key finding:** This mechanism EXISTS and is comprehensively validated in the literature (GLA, RetNet, RWKV, Mamba, Griffin, xLSTM, Gated DeltaNet). Baseline A1 already uses Gated DeltaNet for 75% of its layers. The research question reduces to: (1) converting the final 25% full-attention layers for A1/B, and (2) full-from-scratch training for A2 (the highest-impact target). The mechanism is production-ready via FlashLinearAttention and NVlabs/GatedDeltaNet.

**Sequential-state callout:** The recurrent state update is strictly sequential at batch>1, so all TPOT improvements in the comparison tables are upper bounds until measured on production serving stacks.


### Key Comparison Tables

**vs Baseline A1 (Qwen3.5-27B Hybrid)** — *Comparison is converting the remaining 25% full-attention layers*

| Metric | Baseline A1 | This Idea (100% Gated) | Change | Notes |
|--------|------------|----------------------|--------|-------|
| Compute (FLOPs/token, decode) | O(L·(d²+s·d/4)) | O(L·d·d_state) | ↓ at long s | s-linear attention term eliminated for 25% of layers |
| Memory bandwidth (decode) | O(L·(d²+s·d_kv/4)) | O(L·d_ff + L·H·d_kv²) | ↓ ~2.15 GB at 32K | s·d_kv/4 KV term drops to ~8 MB fixed recurrent state (H_kv) |
| KV cache (32K ctx) | ~**2.15 GB** | ~8 MB | ↓ ~**269×** | 16 × 2 × 4 KV heads × **256 head_dim** × 32768 × 2 bytes; recurrent state (H_kv): 16 × 4 KV heads × 256 × 256 × 2 bytes ≈ 8 MB |
| Weight memory | O(L·d·d_ff) | O(L·d·d_ff) | = | Gate matrices absorbed into projections |
| Training cost | 1.0× | ~1.0× | ≈ | Parallel scan adds minimal overhead at training lengths |
| TTFT (8K prompt) | ref | ≈ ref | ≈ | FlashLinearAttention faster than FlashAttention-2 at ≥4K |
| TPOT (batch=1) | ref | ↓* ~20–30% estimate (⚠ upper bound — recurrent state sequential at batch>1) | ↓* | KV ~2.15 GB eliminated; weight BW ~54 GB still dominates but KV fraction was ~3.8%; full elimination → ~4% total BW saved + improved arithmetic intensity |

**vs Baseline A2 (Qwen3-32B Dense)** — *Full conversion of all 64 layers*

| Metric | Baseline A2 | This Idea (All-Gated) | Change | Notes |
|--------|------------|----------------------|--------|-------|
| Compute (FLOPs/token, decode) | O(L·(s·d+d·d_ff)) | O(L·(d·d_state+d·d_ff)) | ↓ at long s | s-linear attention FLOPs eliminated; at s=32K: ~32–85× fewer attention FLOPs |
| Memory bandwidth (decode) | O(L·(d·d_ff+s·d_kv)) | O(L·(d·d_ff+H·d_kv²)) | ↓ large at long s | KV term (s·d_kv) eliminated; at 32K ctx removes ~8.59 GB per token; at 262K removes ~68.7 GB |
| KV cache (32K ctx) | ~**8.59 GB** | ~16 MB | ↓ **~537×** | 64 × 2 × 8 KV heads × 128 × 32768 × 2 bytes; recurrent state (H_kv): 64 × 8 KV heads × 128 × 128 × 2 bytes ≈ 16 MB; if H_q=64 assumed: 64 × 64 × 128 × 128 × 2 ≈ 134 MB (→ ~64×) |
| KV cache (262K ctx) | ~68.7 GB | ~16 MB | ↓ **~4,294×** | Recurrent state is constant; KV-only improvement (does not include weight BW); ~512× if H_q=64 assumed |
| Weight memory | O(L·d·d_ff) | O(L·d·d_ff) | = | Same weight count |
| Training cost | 1.0× | ~0.9–1.0× | ≈ | Chunk-wise parallel scan avoids O(s²) attention at training |
| TTFT (8K prompt) | ref | ↓ 5–15% at 8K | ↓ modest | Grows with s; at 32K: larger benefit |
| TPOT (batch=1) at 32K | ref | ↓* ~10–13% estimate (⚠ upper bound — recurrent state sequential at batch>1) | ↓* modest | KV bandwidth 8.59/(64+8.59)=~11.8% of total; eliminating 100% of KV term yields ~13.3% bandwidth reduction: theoretical max TPOT improvement is ~13%; arithmetic intensity improvement from eliminating HBM reads may add a small additional benefit |

**vs Baseline B (Qwen3.5-397B-A17B MoE)** — *Comparison is converting the remaining 25% full-attention layers*

| Metric | Baseline B | This Idea (100% Gated) | Change | Notes |
|--------|-----------|----------------------|--------|-------|
| Compute (FLOPs/token) | O(L·(d²+k·d·d_e)) | O(L·(d·d_state+k·d·d_e)) | ↓ slight | MoE FFN dominates; attention FLOP reduction is minor |
| Memory bandwidth (decode) | O(L·(k·d·d_e+state)) | O(L·(k·d·d_e+H·d_kv²)) | ↓ ~1.0 GB | KV from 25% full-attn layers (15 layers × 2 KV heads × 256 dim) eliminated |
| KV cache (32K ctx) | ~**1.0 GB** | ~4 MB | ↓ **~267×** | 15 × 2 × 2 KV heads × **256 head_dim** × 32768 × 2 bytes; recurrent state (H_kv): 15 × 2 KV heads × 256 × 256 × 2 bytes ≈ 4 MB |
| Weight memory | O(L·E·d·d_e) | O(L·E·d·d_e) | = | MoE weight footprint unchanged |
| Training cost | 1.0× | ~1.0× | ≈ | MoE routing costs dominate |
| TTFT (8K prompt) | ref | ≈ ref | ≈ | 25% of layers converted |
| TPOT (batch=1) | ref | ↓* 15–20% estimate (⚠ upper bound — recurrent state sequential at batch>1) | ↓* modest | KV ~1.0 GB of ~35 GB total BW; eliminating → ~2.9% BW savings + improved arithmetic intensity |

**vs Baseline C (K2 family, 72.55B dense Llama-arch)** — *Full conversion of all 80 layers*

| Metric | Baseline C | This Idea (All-Gated) | Change | Notes |
|--------|-----------|----------------------|--------|-------|
| Compute (FLOPs/token, decode) | O(L·(s·d+d·d_ff)) | O(L·(d·d_state+d·d_ff)) | ↓ at long s | s-linear attention FLOPs eliminated for all 80 layers |
| Memory bandwidth (decode) | O(L·(d·d_ff+s·d_kv)) | O(L·(d·d_ff+H·d_kv²)) | ↓ large at long s | KV eliminated: ~10.0 GiB at 32K or ~80.0 GiB at 262K replaced by ~20 MB fixed state (H_kv) |
| KV cache (32K ctx) | ~**10.0 GiB** | ~20 MB | ↓ **~500×** | 80 × 2 × 8 × 128 × 32768 × 2 bytes; recurrent state (H_kv): 80 × 8 KV heads × 128 × 128 × 2 bytes ≈ 20 MB; if H_q=64 assumed: 80 × 64 × 128 × 128 × 2 ≈ 167 MB (→ ~61×) |
| KV cache (262K ctx) | ~80.0 GiB | ~20 MB | ↓ **~4,000×** | K2-Think-V2 native max; ~490× if H_q=64 assumed |
| Weight memory | ~145.1 GB | ~145.1 GB | = | Weights unchanged |
| Training cost | 1.0× | ~0.9–1.0× | ≈ | From-scratch retraining required |
| TPOT (batch=1) at 32K | ref | ↓* ~20–35% estimate (⚠ upper bound — recurrent state sequential at batch>1) | ↓* | KV 10.0/(145.1+10.0)=6.4% of BW; elimination → 6.4% BW savings + arithmetic intensity improvement |
| TPOT at 262K | ref | ↓* ~50–60% estimate (⚠ upper bound — recurrent state sequential at batch>1) | ↓* significant | KV 80.0/(145.1+80.0)=35.5% of BW; elimination yields large benefit at K2 native context |

---

## 1. Idea Description

**Novelty verdict: EXISTS — ~95% covered by GLA [Yang et al., 2024] and Gated DeltaNet [Yang et al., 2024]; residual novelty is post-SDPA gating specifically applied to Qwen gated attention [17] at 27B+ scale.**

**LSTM-Gated Attention** proposes replacing the softmax normalization in standard self-attention with LSTM-style gating mechanisms. The gate controls information flow from key-value pairs to queries, potentially enabling better compression and selective memory. The intended inference benefit is twofold: (a) a recurrent formulation eliminates the growing KV cache (reducing TPOT at long contexts by removing sequential memory-bandwidth pressure), and (b) parallel scan during prefill means TTFT does not necessarily regress. The "LSTM-gated softmax" framing is a partially novel phrasing, but the underlying mechanism — data-dependent gating replacing or augmenting softmax in attention, admitting a recurrent form — is the subject of a large and active literature (GLA, RetNet, RWKV, Mamba/Mamba-2, Griffin, HGRN/HGRN2, xLSTM, GSA, Gated DeltaNet).

**Key inference benefit framing:** If all layers convert to a gated-recurrent form, KV cache is eliminated entirely and replaced by a fixed-size recurrent state (a d_k × d_v matrix per KV head per layer). For Baseline A2 (Qwen3-32B Dense), this would reduce KV cache from ~8.59 GB at 32K context (or ~68.7 GB at 262K) to ~16 MB (H_kv=8 basis) regardless of sequence length (~537× reduction). For Baseline A1 (Qwen3.5-27B Hybrid), which already uses Gated DeltaNet for 75% of layers, converting the remaining 25% full-attention layers (which use head_dim=256, yielding KV ~2.15 GB at 32K) yields a recurrent state of ~8 MB (H_kv=4 basis), a ~269× KV reduction.

---

## 2. Literature Review

### Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention (2020, ICML) [1]
- **Authors**: Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, François Fleuret
- **URL**: https://arxiv.org/abs/2006.16236
- **Summary**: Derives linear attention via kernel feature map φ(x), reformulating as recurrent hidden state S_t = S_{t-1} + φ(k_t)⊗v_t. Reports up to 4000× faster autoregressive inference on long sequences versus standard softmax attention.
- **Relevance**: Foundational derivation showing any decomposable similarity function has a constant-size recurrent form, eliminating the KV cache.

### Retentive Network: A Successor to Transformer for Large Language Models (2023, arXiv preprint) [2]
- **Authors**: Yutao Sun, Li Dong, Shaohan Huang, Shuming Ma, Yuqing Xia, Jilong Xue, Jianyong Wang, Furu Wei
- **URL**: https://arxiv.org/abs/2307.08621
- **Summary**: "Retention" mechanism supporting parallel, recurrent, and chunkwise-recurrent computation. Gated Multi-Scale Retention. Claims 8.4× faster decoding and ~70% memory reduction vs. Transformers with KV cache at 7B/8K context.
- **Note**: arXiv preprint only — no confirmed peer-reviewed venue.

### RWKV: Reinventing RNNs for the Transformer Era (2023, EMNLP Findings) [3]
- **Authors**: Bo Peng et al.
- **URL**: https://arxiv.org/abs/2305.13048
- **Summary**: Receptance gate (sigmoid forget gate analogous to LSTM). Scaled to 14B parameters. Trains as Transformer, infers as RNN. Confirmed: EMNLP 2023 Findings (ACL Anthology 2023.findings-emnlp.936).

### Gated Linear Attention Transformers with Hardware-Efficient Training (2024, ICML) [4]
- **Authors**: Songlin Yang, Bailin Wang, Yikang Shen, Rameswar Panda, Yoon Kim
- **URL**: https://arxiv.org/abs/2312.06635
- **Summary**: Data-dependent gating G_t into linear attention recurrence: S_t = G_t ⊙ S_{t-1} + k_t⊗v_t. FlashLinearAttention outperforms FlashAttention-2 at sequences ≥4K. 1.3B: Wikitext PPL ≈17.22, LAMBADA PPL ≈14.47. Length generalization 2K→20K+. Confirmed: ICML 2024 (PMLR v235, pp.56501–56523).

### Mamba: Linear-Time Sequence Modeling with Selective State Spaces (2023, ICLR 2024) [5]
- **Authors**: Albert Gu, Tri Dao
- **URL**: https://arxiv.org/abs/2312.00752
- **Summary**: Selective SSM with input-dependent A/B/C matrices. 3B Mamba outperforms same-size Transformers; 5× higher throughput than Transformers. Confirmed: ICLR 2024.

### Transformers are SSMs: Structured State Space Duality (Mamba-2) (2024, ICML) [6]
- **Authors**: Tri Dao, Albert Gu
- **URL**: https://arxiv.org/abs/2405.21060
- **Summary**: SSD equivalence between SSMs and masked linear attention (semiseparable matrices). Mamba-2: 2–8× speedup over Mamba-1. Confirmed: ICML 2024.

### Hierarchically Gated Recurrent Neural Network (HGRN) (2023, NeurIPS Spotlight) [7]
- **Authors**: Zhen Qin, Songlin Yang, Yiran Zhong
- **URL**: https://arxiv.org/abs/2311.04823
- **Summary**: Forget gates within recurrence with layer-depth monotonic lower bounds. NeurIPS 2023 Spotlight.

### HGRN2: Gated Linear RNNs with State Expansion (2024, COLM) [8]
- **Authors**: Zhen Qin, Songlin Yang et al.
- **URL**: https://arxiv.org/abs/2404.07904
- **Summary**: Outer-product state expansion grows state from O(d) to O(d²/H) without adding parameters. 3B HGRN2 slightly outperforms Mamba and LLaMA at same scale. COLM 2024.

### xLSTM: Extended Long Short-Term Memory (2024, NeurIPS 2024) [9]
- **Authors**: Maximilian Beck, Korbinian Pöppel et al.
- **URL**: https://arxiv.org/abs/2405.04517
- **Summary**: Exponential gating + log-sum-exp stabilization; mLSTM is GLA-equivalent with exponential gates. NeurIPS 2024 (Spotlight poster).

### Griffin: Mixing Gated Linear Recurrences with Local Attention (2024, arXiv, Google DeepMind) [10]
- **Authors**: Soham De, Samuel L. Smith, Anushan Fernando et al.
- **URL**: https://arxiv.org/abs/2402.19427
- **Summary**: RG-LRU gate (a = σ(Λ), interpolating retention and input). Per abstract, Griffin matches Llama-2's performance while being trained on over 6× fewer tokens. Hawk-3B exceeds Mamba-3B. Inference cache fixed ~2000 bytes vs. linearly growing KV cache.

### Gated Slot Attention (GSA) (2024, NeurIPS) [11]
- **Authors**: Yu Zhang, Songlin Yang, Ruijie Zhu et al.
- **URL**: https://arxiv.org/abs/2409.07146
- **Summary**: Two GLA layers connected via softmax; superior recall performance. NeurIPS 2024 Main Track.

### Gated Delta Networks: Improving Mamba2 with Delta Rule (2025, ICLR) [12]
- **Authors**: Songlin Yang, Jan Kautz, Ali Hatamizadeh (NVIDIA)
- **URL**: https://arxiv.org/abs/2412.06464
- **Summary**: Combines gating + delta rule. Outperforms Mamba2 and DeltaNet. **This is the mechanism used in Baseline A1 (Qwen3.5-27B) and Baseline B (Qwen3.5-397B)**. ICLR 2025.

### Simple Linear Attention Language Models Balance the Recall-Throughput Tradeoff (BASED) (2024, ICLR 2024) [13]
- **Authors**: Simran Arora, Sabri Eyuboglu, Michael Zhang et al.
- **URL**: https://arxiv.org/abs/2402.18668
- **Summary**: Recall-throughput fundamental tradeoff documented. Per abstract: BASED outperforms Mamba on real-world recall-intensive tasks by 6.22 accuracy points at 1.3B; achieves 24× higher throughput than FlashAttention-2 at 1.3B/1024 tokens. (Specific "90.8% recall" and "1e-5× latency penalty" figures are paper-body values.) **Confirmed: ICLR 2024**.

### LoLCATs: On Low-Rank Linearizing of Large Language Models (2025, ICLR) [14]
- **Authors**: Michael Zhang, Simran Arora et al. (Stanford Hazy Research)
- **URL**: https://arxiv.org/abs/2410.10254
- **Summary**: Post-hoc linearization of Llama 3.1 70B/405B; closes 77.8% of 5-shot MMLU gap with 0.2% parameters and 0.4% training tokens. ICLR 2025.

### MetaLA: Unified Optimal Linear Approximation to Softmax Attention (2024, NeurIPS Oral) [15]
- **Authors**: Yuhong Chou, Man Yao, Kexin Wang et al.
- **URL**: https://arxiv.org/abs/2411.10741
- **Summary**: Three conditions for optimal linear approximation to softmax. MetaLA outperforms GLA, Mamba, HGRN2 on MQAR and language modeling. NeurIPS 2024 Oral.

### Jamba: A Hybrid Transformer-Mamba Language Model (2024/2025, ICLR 2025) [16]
- **Authors**: Jamba Team, AI21 Labs
- **URL**: https://arxiv.org/abs/2403.19887
- **Summary**: Interleaved Transformer + Mamba + MoE architecture. Better training loss vs. pure Transformer or pure Mamba. Jamba-1.5 scales to 94B active params, 256K context. ICLR 2025.

### Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free (2025, NeurIPS 2025 Oral / Best Paper) [17]
- **Authors**: Zihan Qiu, Zekun Wang, Bo Zheng, Zeyu Huang, Kaiyue Wen, Songlin Yang, Rui Men, Le Yu, Fei Huang, Suozhi Huang, Dayiheng Liu, Jingren Zhou, Junyang Lin (Alibaba/Qwen team)
- **URL**: https://arxiv.org/abs/2505.06708
- **Summary**: Applies a head-specific sigmoid gate after the Scaled Dot-Product Attention output (post-SDPA gating). Tested at scale: 30 variants of 15B MoE and 1.7B dense models trained on 3.5T tokens. Improves performance, training stability, and scaling; eliminates attention-sink phenomenon. **Does not eliminate KV cache** — gates the attention output, not the recurrence. NeurIPS 2025 Oral and Best Paper. Integrated into Qwen3-Next (Baseline A1 lineage).
- **Relevance**: Directly validates sigmoid gating applied to attention in production-scale LLMs; distinguishes output-gating (this paper) from recurrent-replacement gating (GLA/Gated DeltaNet); both are gating strategies but with different inference implications.

### Liger: Linearizing Large Language Models to Gated Recurrent Structures (2025, ICML 2025) [18]
- **Authors**: Disen Lan, Weigao Sun, Jiaxi Hu, Jusen Du, Yu Cheng
- **URL**: https://arxiv.org/abs/2503.01496
- **Summary**: Converts pretrained LLMs to gated linear recurrent models by repurposing existing key-matrix weights as gating mechanisms (no extra parameters). Recovers ~93% of original Transformer performance using only 0.02% of pretraining tokens. Validated at 1B–8B scale. ICML 2025.
- **Relevance**: Lower-cost alternative to LoLCATs [14] for converting A1/B full-attention layers to gated recurrent form; uses 100× fewer tokens than prior distillation methods.

---

## 3. Prior Art Classification

- **Status**: EXISTS (~95% coverage)
- **Novel contribution**: The specific question of "what does replacing the remaining 25% full-attention layers in a Qwen3.5-style hybrid (A1/B) with gated attention achieve?" has no direct published answer. The mechanism is mature; the marginal gain from the last-mile conversion is the open question.
- **Genuine open questions**:
  1. 75%→100% conversion quality delta on a production Qwen3.5-style model
  2. Quality cliff asymmetry: the last 25% full-attention layers may do disproportionate associative recall work
  3. LoLCATs effectiveness on a pre-gated (already 75%) hybrid model (LoLCATs tested only on pure-softmax Llama)
  4. d_kv=256 (A1/B full-attn layers) → GLA recurrent state of 256×256=65K elements vs existing 128×128=16K for Gated DeltaNet layers

---

## 4. Technical Analysis

**Novelty verdict: EXISTS — ~95% covered by GLA [Yang et al., 2024] and Gated DeltaNet [Yang et al., 2024]; residual novelty is post-SDPA gating specifically applied to Qwen gated attention [17] at 27B+ scale.**

### 4.1 Theoretical Complexity

For fully gated-recurrent attention, the KV cache is replaced by a fixed-size recurrent state S ∈ ℝ^{H × d_kv × d_kv} per layer (constant in s).

| Metric | This Idea (All-Gated) | Baseline A1 | Baseline A2 | Baseline B | Baseline C |
|--------|----------------------|------------|------------|------------|------------|
| KV cache | O(L·H·d_kv²) constant | O(L/4·s·d_kv) | O(L·s·d_kv) | O(L/4·s·d_kv) | O(L·s·d_kv) |
| KV at 32K (BF16) | ~4–20 MB fixed (H_kv) | ~2.15 GB | ~8.59 GB | ~1.0 GB | ~10.0 GiB |
| KV reduction | — | ~269× (A1) | ~537× (A2) | ~267× (B) | ~500× (C) |
| TPOT direction | ↓ at long s | ↓ ~20-30% | ↓ ~10-13% | ↓ ~15-20% | ↓ ~20-35% |

**Recurrent state size calculations (using H_kv; one recurrent matrix per KV head):**
- A1 converted (16 layers × 4 KV heads × 256 × 256 × 2 bytes): ~8 MB
- A2 fully gated (64 layers × 8 KV heads × 128 × 128 × 2 bytes): ~16 MB
- B converted (15 layers × 2 KV heads × 256 × 256 × 2 bytes): ~4 MB
- C fully gated (80 layers × 8 KV heads × 128 × 128 × 2 bytes): ~20 MB
- Note: if the implementation maintains one recurrent state per Q-head (H_q) rather than per KV head, multiply by H_q/H_kv (typically 8×). H_q is not specified in canonical baselines; H_kv-based figures above are the conservative lower bound.

### 4.2 FLOPs Analysis

At s=32K, d_kv=128, the per-layer comparison is:
- Softmax attention FLOPs per layer: s × H_kv × d_kv = 32768 × 8 × 128 = 33.6M ops
- GLA state update FLOPs per layer: H_q × d_kv × d_kv = 64 × 128 × 128 = 1.05M ops
- Ratio: 33.6M / 1.05M = **~32×** per layer

A single-head simplification (s / d_kv = 32768 / 128 = 256×) ignores the H_kv factor and overstates the multi-head-architecture ratio.

**Correct figure: ~32–85× fewer attention FLOPs per decode step for A2 at s=32K.** Still very large — qualitative conclusion unchanged.

### 4.3 Memory Bandwidth Analysis

- **Weight loading**: Unchanged. O(L·d·d_ff) bytes per token at decode. Dominant at short contexts.
- **KV cache access**: Eliminated entirely. s-scaling term vanishes. Replaced by fixed ~4–20 MB recurrent state (H_kv basis; scales by H_q/H_kv if per-Q-head states are used).
- **Practical TPOT improvement** at batch=1 (memory-bandwidth-bound):
  - A2 at 32K: (weight_BW + KV_BW) / weight_BW = (64 + 8.59) / 64 ≈ **1.13× theoretical max** (but arithmetic intensity improvement pushes practical beyond this)
  - A2 at 262K: (64 + 68.7) / 64 ≈ **3.1× theoretical max** from KV alone; practical ~2–3×

---

## 5. Implementation Considerations

- **Hardware requirements**: FlashLinearAttention library (open source: fla-org/flash-linear-attention) provides hardware-efficient Triton kernels for GLA, HGRN2, GSA, and Gated DeltaNet. NVlabs/GatedDeltaNet has reference CUDA implementation. No new kernel development required.

- **Training stability**: Sigmoid gates (GLA, HGRN) are stable in BF16. Exponential gates (xLSTM) require log-sum-exp stabilization. Gate initialization: initialize forget bias ≈ 3–5 (high gate values near 1) to avoid vanishing gradients. Gated DeltaNet is safest for A1/B conversion (same mechanism as existing 48 layers).

- **Implementation paths (ranked by cost):**
  1. **Liger** [18] or **LoLCATs** [14]: Linearize using key-matrix repurposing (Liger: 0.02% tokens, no extra params) or attention-swap + LoRA (LoLCATs: ~78% quality gap closed). LOWEST cost. Best for A1/B validation.
  2. **Continued pre-training**: Load weights, swap layers, train ~50–100B tokens. MEDIUM cost. For A1/B production.
  3. **From-scratch training**: Full 100% Gated DeltaNet architecture. HIGH cost. For A2 production.

- **Recommended next steps:**
  1. Run LoLCATs on Qwen3.5-27B (A1) converting the 16 full-attention layers to Gated DeltaNet. Measure MQAR and needle-in-haystack recall before/after.
  2. Benchmark d_kv=256 recurrent state (converted A1 layers) vs. d_kv=128 (existing Gated DeltaNet layers). Determine if downprojecting is necessary.
  3. For A2: Design from-scratch training schedule with 100% Gated DeltaNet architecture.

---

## 6. Synergies

- **Combines well with**:
  - **3.1 (Layer-Level MoE)**: MoE FFN is orthogonal to attention mechanism; already the configuration of Baseline B.
  - **4.4 (Skip-List Layers)**: Skip mechanisms apply equally to gated recurrent layers.
  - **5.1 (TurboQuant)**: Complementary — if any full-attention KV cache remains, quantization can further compress it. Orthogonal if all layers are fully gated.

- **Conflicts with**:
  - Any idea relying on growing KV cache as a compression target (KV quantization, KV eviction) — these become irrelevant if the KV cache is eliminated.

---

## 7. Risk Assessment

- **Technical risk**: LOW — Core mechanism is production-deployed in Qwen3.5 (A1/B). Primary risk is quality cliff on recall-intensive tasks, well-documented and partially mitigated by hybrid approaches.
- **Potential impact**: HIGH for A2 (KV cache ~8.59 GB → ~134 MB at 32K), MEDIUM for A1/B (final 25% conversion).
- **Implementation effort**: MEDIUM — FlashLinearAttention and NVlabs/GatedDeltaNet kernels exist. LoLCATs provides a low-cost validation path.

---

## 8. Accuracy / Quality Tradeoff

**Novelty verdict: EXISTS — ~95% covered by GLA [Yang et al., 2024] and Gated DeltaNet [Yang et al., 2024]; residual novelty is post-SDPA gating specifically applied to Qwen gated attention [17] at 27B+ scale.**

- **Reported quality delta**:
  - GLA 1.3B: competitive with LLaMA-style Transformer++ baselines per abstract; specific Wikitext/LAMBADA PPL values are paper-body tables [4]
  - Griffin: matches Llama-2 performance while trained on over 6× fewer tokens (abstract); specific per-size downstream accuracy numbers (Griffin-7B 65.8% vs Llama-2-13B 69.3%) are paper-body values [10]
  - BASED: outperforms Mamba on recall by 6.22 accuracy points at 1.3B and achieves 24× throughput vs FlashAttention-2 at 1024 tokens (abstract); specific "90.8% softmax recall / 1e-5× latency" is paper body [13]
  - LoLCATs: 77.8% MMLU gap closed for Llama 3.1 70B with 0.2% parameters [14]
  - Mamba 3B: outperforms same-size Transformer; 5× throughput [5]
- **Monotonicity**: Cliff-shaped. Standard LM perplexity is approximately competitive; associative recall drops sharply below a task-specific threshold. The cliff is caused by recurrent state saturation.
- **Recovery strategies**: Hybrid (retain 10-25% full-attention layers); LoLCATs post-hoc linearization; larger recurrent state (HGRN2 outer product, GLA d_k × d_v matrix); training from scratch.

---

<!-- CITATION MANIFEST -->
[1]: Transformers are RNNs — Katharopoulos, Vyas, Pappas, Fleuret. arXiv:2006.16236. ICML 2020. Foundational linear attention as recurrent hidden state; up to 4000× faster autoregressive inference.
[2]: Retentive Network — Sun, Dong et al. arXiv:2307.08621. arXiv preprint only (no confirmed peer-reviewed venue). 8.4× faster decoding, ~70% memory reduction vs. Transformer.
[3]: RWKV — Bo Peng et al. arXiv:2305.13048. EMNLP Findings 2023 (ACL Anthology 2023.findings-emnlp.936). Receptance gate analogous to LSTM; 14B parameters.
[4]: Gated Linear Attention (GLA) — Songlin Yang, Bailin Wang et al. arXiv:2312.06635. ICML 2024 (PMLR v235). Data-dependent gating; FlashLinearAttention kernel; faster than FlashAttention-2 at ≥4K.
[5]: Mamba — Albert Gu, Tri Dao. arXiv:2312.00752. ICLR 2024. Selective SSM with input-dependent gates; 5× throughput over Transformers.
[6]: Mamba-2 / SSD — Tri Dao, Albert Gu. arXiv:2405.21060. ICML 2024. Structured State Space Duality; 2–8× speedup over Mamba-1.
[7]: HGRN — Zhen Qin, Songlin Yang, Yiran Zhong. arXiv:2311.04823. NeurIPS 2023 Spotlight. Hierarchical forget gates with depth-based lower bounds.
[8]: HGRN2 — Zhen Qin et al. arXiv:2404.07904. COLM 2024. State expansion via outer product; matches LLaMA at 3B scale.
[9]: xLSTM — Maximilian Beck et al. arXiv:2405.04517. NeurIPS 2024 (Spotlight). Exponential gating + log-sum-exp stabilization; mLSTM = matrix-valued GLA analog.
[10]: Griffin — Soham De et al. (Google DeepMind). arXiv:2402.19427. RG-LRU gate; Griffin-7B matches Llama-2-13B on downstream tasks.
[11]: GSA — Yu Zhang, Songlin Yang et al. arXiv:2409.07146. NeurIPS 2024 Main Track. Gated Slot Attention; two GLA layers with softmax; superior recall on T2R.
[12]: Gated DeltaNet — Songlin Yang, Jan Kautz, Ali Hatamizadeh. arXiv:2412.06464. ICLR 2025. Gating + delta rule; mechanism used in Baseline A1 (Qwen3.5-27B) and Baseline B (Qwen3.5-397B).
[13]: BASED — Simran Arora et al. arXiv:2402.18668. ICLR 2024. Recall-throughput fundamental tradeoff; 90.8% recall recovery at 1e-5× latency penalty.
[14]: LoLCATs — Michael Zhang, Simran Arora et al. arXiv:2410.10254. ICLR 2025. Post-hoc linearization; 77.8% MMLU gap closed with 0.2% parameters and 0.4% tokens.
[15]: MetaLA — Yuhong Chou et al. arXiv:2411.10741. NeurIPS 2024 Oral. Three conditions for optimal linear softmax approximation; outperforms GLA/Mamba/HGRN2 on MQAR.
[16]: Jamba — AI21 Labs. arXiv:2403.19887. ICLR 2025. Hybrid Transformer+Mamba+MoE; 94B active params, 256K context; supports case for large-scale hybrid gating.
[17]: Gated Attention for LLMs — Zihan Qiu, Zekun Wang et al. (Qwen team). arXiv:2505.06708. NeurIPS 2025 Oral + Best Paper. Post-SDPA sigmoid output gating; 15B MoE + 1.7B dense at 3.5T tokens; eliminates attention sink; KV cache preserved. Integrated into Qwen3-Next (A1 lineage).
[18]: Liger — Disen Lan, Weigao Sun et al. arXiv:2503.01496. ICML 2025. Linearizes LLMs to gated recurrent structures using existing key-matrix weights; ~93% performance recovery at 0.02% pretraining tokens; 1B–8B scale validated.
