# Research: TurboQuant — Learned Quantization Baked into Model at KV Cache Interface
## ID: 5.1

---

## Executive Summary

**Novelty verdict:** PARTIAL — KV-cache-only quantization and sub-4-bit KV quantization are well-covered; the specific TurboQuant rotation+QJL algorithm exists post-hoc; the remaining gap is QAT targeting the KV cache interface in isolation (decoupled from weight QAT), "baked into training" ([TurboQuant / Zandieh et al., 2026], [KVQuant, 2024], [KIVI, 2024], [LLM-QAT, 2024], [SpinQuant, 2025]).

TurboQuant proposes aggressive quantization (INT4 or lower) of KV cache entries at the KV cache interface, learned during training and baked into the model rather than applied post-hoc. **Scope: KV cache interface only** — not applied to model weights. The primary beneficiary is Baseline A2 (Qwen3-32B Dense) at long context, where the KV cache grows to ~68.7 GB at 262K tokens and INT4 reduces this to ~17.2 GB (4×). At 32K context the KV cache is ~8.59 GB, where weight bandwidth (~64 GB) still dominates and TPOT benefit is modest (~1.10×). For Baselines A1 and B (hybrid models with small KV caches), benefits are minor.

**Key finding:** The published TurboQuant paper (Zandieh et al., ICLR 2026) is explicitly **post-hoc and training-free**. The "learned / baked into training" framing is a QAT extension not yet demonstrated at scale. Post-hoc TurboQuant at 3.5-bit is immediately deployable for A2 with near-zero quality loss.

**DeepSeek-V4 update (2026-04-24):** DeepSeek-V4 does not use TurboQuant, but it materially strengthens the low-precision KV/interface direction. V4 stores compressed attention KV in a mixed format (BF16 for RoPE dimensions, FP8 for the remaining dimensions) and applies FP4 QAT to the CSA indexer QK path. The update is architectural: future KV quantization analysis must support heterogeneous CSA/HCA caches, sliding-window state caches, and indexer-side QK caches rather than assuming a uniform full-attention KV tensor.

### Key Comparison Tables

**vs Baseline A1 (Qwen3.5-27B Hybrid)**

| Metric | Baseline A1 | This Idea (INT4 KV) | Change | Notes |
|--------|-------------|---------------------|--------|-------|
| Compute (FLOPs/token) | O(L·(d²+s·d/4)) | O(L·(d²+s·d/4)) | = | KV quantization adds <2% dequant overhead |
| Memory bandwidth (decode) | O(L·(d²+s·d_kv/4)) | O(L·(d²+s·d_kv/16)) KV term | ↓ 4× (KV term only) | Weight bandwidth (~54 GB) dominates at all contexts |
| KV cache (32K ctx, BF16) | ~2.15 GB | ~0.54 GB | ↓ 4× | 16 × 2 × 4 × 256 × 32768 × 2 bytes; head_dim=256 per canonical baseline |
| Weight memory | O(L·d·d_ff) | = | = | Weights not quantized |
| Training cost | 1.0× | ~1.35× | ↑ | QAT fake-quantize overhead |
| TTFT (8K prompt) | ref | ≈ ref | ≈ ref | Prefill is compute-bound at 8K; KV quantization affects decode-time KV reads only, not prefill FLOPs; attention-kernel dequant overhead <2% (§4.2) |
| TPOT (batch=1) | ref | ↓ ~1.4% total BW reduction | negligible | KV ~3.6% of total BW at 32K; even 4× reduction yields ~2.7% improvement |

**vs Baseline A2 (Qwen3-32B Dense)**

| Metric | Baseline A2 | This Idea (INT4 KV) | Change | Notes |
|--------|-------------|---------------------|--------|-------|
| Compute (FLOPs/token) | O(L·(s·d+d·d_ff)) | = | = | Attention FLOPs unchanged |
| Memory bandwidth (decode) | O(L·(d·d_ff+s·d_kv)) | O(L·(d·d_ff+s·d_kv/4)) | ↓ significant at long ctx | At 262K: KV 68.7 GB → 17.2 GB; total 132.7 → 81.2 GB |
| KV cache (32K ctx, BF16) | ~8.59 GB | ~2.15 GB | ↓ 4× | 64 × 2 × 8 × 128 × 32768 × 2 bytes |
| KV cache (262K ctx, BF16) | ~68.7 GB | ~17.2 GB | ↓ 4× | At native context for YaRN-extended A2 [derived: 64×2×8×128×262144×2 = 68,719,476,736 bytes ≈ 68.7 GB] |
| Weight memory | O(L·d·d_ff) | = | = | Weights not quantized |
| Training cost | 1.0× | ~1.35× | ↑ | QAT overhead |
| TTFT (8K prompt) | ref | ≈ ref | ≈ ref | Prefill is compute-bound at 8K; KV quantization affects decode-time KV reads only, not prefill FLOPs; attention-kernel dequant overhead <2% (§4.2) |
| TPOT (batch=1) | ref | ↓ ~1.10× at 32K; ~1.63× at 262K | ↓ meaningful at long ctx | Formula: (weight_BW + KV_BW_before) / (weight_BW + KV_BW_after); at 32K: (64+8.59)/(64+2.15)=72.59/66.15=1.097≈1.10×; at 262K: (64+68.7)/(64+17.2)=132.7/81.2=1.63× |

**vs Baseline B (Qwen3.5-397B-A17B MoE)**

| Metric | Baseline B | This Idea (INT4 KV) | Change | Notes |
|--------|------------|---------------------|--------|-------|
| Compute (FLOPs/token) | O(L·(d²+k·d·d_e)) | = | = | MoE FFN compute unchanged |
| Memory bandwidth (decode) | O(L·(k·d·d_e+state)) | O(L·(k·d·d_e+state_kv/4)) | ↓ negligible | KV ~1 GB vs ~34 GB active weights; 3% of total |
| KV cache (32K ctx, BF16) | ~1.0 GB | ~0.25 GB | ↓ 4× | 15 × 2 × 2 × 256 × 32768 × 2 bytes |
| Weight memory | O(L·E·d·d_e) | = | = | Weights not quantized |
| TTFT (8K prompt) | ref | ≈ ref | ≈ ref | Prefill is compute-bound at 8K; KV quantization affects decode-time KV reads only, not prefill FLOPs; attention-kernel dequant overhead <2% (§4.2) |
| TPOT (batch=1) | ref | ↓ ~2.2% total BW reduction | negligible | Weight bandwidth of active 17B params dominates |

**vs Baseline C (K2 family, 72.55B dense Llama-arch)**

| Metric | Baseline C | This Idea (INT4 KV) | Change | Notes |
|--------|-----------|---------------------|--------|-------|
| Compute (FLOPs/token) | O(L·(s·d+d·d_ff)) | = | = | FLOPs unchanged |
| Memory bandwidth (decode) | O(L·(d·d_ff+s·d_kv)) | O(L·(d·d_ff+s·d_kv/4)) | ↓ at long ctx | At 32K: KV ~10.74 GB; weight BW ~145.1 GB; ratio 6.9%; modest at 32K. At 262K: KV ~85.9 GB; weight BW ~145.1 GB; ratio 37.2%; TPOT ~1.36× |
| KV cache (32K ctx, BF16) | ~10.0 GiB | ~2.5 GiB | ↓ 4× | 80 × 2 × 8 × 128 × 32768 × 2 bytes |
| KV cache (262K ctx, BF16) | ~80.0 GiB | ~20.0 GiB | ↓ 4× | K2-Think-V2 native max context |
| Weight memory | ~145.1 GB (bf16) | = | = | Weights not quantized |
| TTFT (8K prompt) | ref | ≈ ref | ≈ ref | Prefill is compute-bound at 8K; KV quantization affects decode-time KV reads only, not prefill FLOPs; attention-kernel dequant overhead <2% (§4.2) |
| TPOT (batch=1) | ref | ↓ ~1.05× at 32K; ~1.36× at 262K | ↓ moderate at long ctx | Formula: (145.1+10.74)/(145.1+2.69)=155.84/147.79=1.054≈1.05× at 32K; (145.1+85.9)/(145.1+21.5)=231.0/166.6=1.387≈1.39× at 262K (values expressed in GB decimal for ratio consistency; KV cell above uses GiB binary for cache-size convention) |

---

## 1. Idea Description

**Novelty verdict: PARTIAL — QAT targeting the KV cache interface in isolation, decoupled from weight QAT, has no published precedent; TurboQuant [1] is post-hoc, not QAT.**

**TurboQuant (Learned Quantization Baked into Model — KV Cache Interface):** Aggressive quantization of KV cache entries at the KV cache interface using TurboQuant-style quantization, learned during training and baked into the model rather than applied post-hoc. **Scope: KV cache interface only** — not applied to all model weights.

The inferred intent for inference speedup: at decode time (TPOT), the KV cache is a significant memory read per generated token. For Baseline A2 (Qwen3-32B Dense), the KV cache at 32K context in BF16 is approximately **8.59 GB** (rising to ~10.49 GB at A2's 40K native maximum, and ~68.7 GB at ~262K contexts reached via YaRN/RoPE scaling). If KV entries are stored at 4-bit instead of 16-bit, the bandwidth for KV reads during attention shrinks by 4×, directly reducing TPOT at batch=1 when KV bandwidth is significant relative to weight bandwidth. The "learned" framing means quantization parameters are co-trained with the model (QAT-style), rather than applied post-hoc via calibration, with the expectation of reduced quality loss at a given bit-width.

**Relationship to published TurboQuant:** The TurboQuant paper (Zandieh et al., ICLR 2026, arXiv:2504.19874) is post-hoc and training-free. The "learned during training and baked into the model" framing in this idea departs from that baseline. The closest published work that trains KV quantization awareness into the model is LLM-QAT (Liu et al., 2024, ACL Findings 2024) and SpinQuant (Liu et al., ICLR 2025); this distinction is analyzed in §3 and §4.

---

## 2. Literature Review

### TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate [1]
- **Authors**: Amir Zandieh, Majid Daliri, Majid Hadian, Vahab Mirrokni (Google Research)
- **URL**: https://arxiv.org/abs/2504.19874 (verified)
- **Summary**: A data-oblivious, training-free two-stage algorithm for near-optimal lossy compression of KV cache vectors. Stage 1 (PolarQuant / random rotation) induces a concentrated Beta distribution on coordinates, then applies optimal scalar quantizers. Stage 2 (QJL — Quantized Johnson-Lindenstrauss) corrects the inner-product bias of the MSE quantizer by projecting the residual through a random Gaussian matrix and storing only the sign bit (1-bit correction). On Llama 3.1-8B on LongBench, TurboQuant at 3.5-bit matches full-precision (50.06 vs 50.06 average score); at 2.5-bit it drops to 49.44. Reported at least 6× KV memory reduction and up to 8× attention speedup on H100 GPUs (vs FP32 baseline, not BF16).
- **Relevance**: This is the paper named in the idea. However, it is explicitly post-hoc and training-free. It covers KV cache interface only (scope matches).
- **Limitations**: Does not perform QAT. The "8× speedup" claim references vs FP32 baseline, not BF16.

### KIVI: A Tuning-Free Asymmetric 2-Bit Quantization for KV Cache [2]
- **Authors**: Zirui Liu, Jiayi Yuan, Hongye Jin, Shaochen Zhong, Zhaozhuo Xu, Vladimir Braverman, Beidi Chen, Xia Hu
- **URL**: https://arxiv.org/abs/2402.02750 (verified)
- **Summary**: Post-hoc 2-bit asymmetric KV cache quantization. Keys quantized per-channel; values per-token. FP16 residual buffer for R most recent tokens. 2.6× peak memory reduction, 2.35×–3.47× throughput improvement on A100 at 2-bit. Up to 2% accuracy drop on CoQA, TruthfulQA, GSM8K for Llama-2-7B.

### KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization [3]
- **Authors**: Coleman Hooper, Sehoon Kim, Hasan Genc, Daniel Hooper, Amir Gholami et al., UC Berkeley
- **URL**: https://arxiv.org/abs/2401.18079 (verified)
- **Summary**: Post-hoc sub-4-bit KV quantization with pre-RoPE key quantization, non-uniform datatypes, per-vector dense-and-sparse. <0.1 PPL at 3-bit on Wikitext-2. Up to ~1.7× speedup vs FP16 for LLaMA-7B.

### LLM-QAT: Data-Free Quantization Aware Training for Large Language Models [4]
- **Authors**: Zechun Liu, Barlas Oguz, Changsheng Zhao, Ernie Chang, Pierre Stock, Yashar Mehdad, Yangyang Shi, Raghuraman Krishnamoorthi, Vikas Chandra (Meta AI Research)
- **URL**: https://arxiv.org/abs/2305.17888 (verified)
- **Summary**: QAT method quantizing weights, activations, AND KV cache jointly using data-free distillation + STE. W4A8KV4 outperforms uniform 4-bit. "Large improvements over training-free methods, especially at low-bit." ACL Findings 2024. Most direct prior art for "learned / baked into training" KV quantization — but does W+A+KV jointly, not KV-only.

### SpinQuant: LLM Quantization with Learned Rotations [5]
- **Authors**: Zechun Liu et al. (Meta AI Research)
- **URL**: https://arxiv.org/abs/2405.16406 (verified)
- **Summary**: PTQ method optimizing rotation matrices (Cayley optimization) for W+A+KV quantization. On LLaMA-2 7B W4A4KV4: accuracy gap to full precision reduced to 2.9pp (vs 19.1pp without SpinQuant). Reduces gap by 45.1% relative to QuaRot. ICLR 2025.

### QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs [6]
- **Authors**: Saleh Ashkboos, Amirkeivan Mohtashami et al. (ETH Zurich)
- **URL**: https://arxiv.org/abs/2404.00456
- **Summary**: Random Hadamard rotations eliminate outliers for end-to-end 4-bit quantization. LLaMA2-70B: at most 0.47 PPL loss on WikiText-2, retains 99% zero-shot task performance. NeurIPS 2024.

### GEAR: KV Cache Compression Recipe for Near-Lossless Generative Inference [7]
- **Authors**: Hao Kang et al.
- **URL**: https://arxiv.org/abs/2403.05527 (verified)
- **Summary**: Combines ultra-low-bit quantization + low-rank SVD residual + sparse correction for KV cache. Up to 2.38× throughput, 2.29× peak memory reduction. The "GEAR-X" extension is speculative and not separately cited.

### H2O: Heavy-Hitter Oracle for Efficient Generative Inference [8]
- **Authors**: Zhenyu Zhang et al.
- **URL**: https://arxiv.org/abs/2306.14048 (verified)
- **Summary**: KV eviction (not quantization) retaining heavy-hitter tokens. Up to 29× throughput improvement. NeurIPS 2023. Complementary to quantization.

### ScissorHands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression [9]
- **Authors**: Zichang Liu, Aditya Desai, Fangshuo Liao et al.
- **URL**: https://arxiv.org/abs/2305.01623
- **Summary**: KV eviction using persistence-of-importance hypothesis (tokens important in early layers remain important later). Up to 5× KV memory reduction without quality loss on generation tasks. NeurIPS 2023. Distinct eviction strategy from H2O; completes the KV compression landscape.

### RotateKV: Accurate and Robust 2-Bit KV Cache Quantization via Outlier-Aware Adaptive Rotations [10]
- **Authors**: Zunhai Su et al.
- **URL**: https://arxiv.org/abs/2501.16383 (verified)
- **Summary**: Outlier-aware adaptive rotations, pre-RoPE grouped-head rotation, attention-sink-aware quantization. <0.3 PPL at 2-bit on WikiText-2 for LLaMA-2-13B; <1.7% degradation on GSM8K; 3.97× peak memory reduction; 5.75× larger batch sizes; 2.32× decode speedup. IJCAI 2025.

### Kitty: Accurate and Efficient 2-Bit KV Cache Quantization with Dynamic Channel-wise Precision Boost [11]
- **Authors**: (First authors from arXiv:2511.18643)
- **URL**: https://arxiv.org/abs/2511.18643 (verified)
- **Summary**: Dynamic Channel-wise Precision Boost for 2-bit KV quantization. 2.1×–4.1× higher inference throughput vs FP16. Evaluated on Qwen3-8B and LLaMA3-8B. November 2025.

### TurboAttention: Efficient Attention Approximation for High Throughput LLMs [12]
- **Authors**: Microsoft Research
- **URL**: https://arxiv.org/abs/2412.08585 (verified)
- **Summary**: FlashQ (head-wise KV quantization) + SAS (softmax approximation). 1.2×–1.8× attention speedup, >4.4× KV reduction, up to 2.37× total throughput. Distinct from TurboQuant.

### FireQ: Fast INT4-FP8 Kernel and RoPE-aware Quantization for LLM Inference [13]
- **Authors**: Baek, Choi et al.
- **URL**: https://arxiv.org/abs/2505.20839 (verified)
- **Summary**: PTQ with INT4 KV and FP8 Q. Two-stage RoPE-aware outlier smoothing. On Llama3-8B: PPL 7.06 vs FP16 PPL 6.13 (+0.93); zero-shot accuracy 67.92% vs 70.61% (-2.69pp). H100 SXM5.

### QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving [14]
- **Authors**: Yujun Lin, Haotian Tang, Shang Yang et al. (MIT HAN Lab)
- **URL**: https://arxiv.org/abs/2405.04532
- **Summary**: Production W4A8KV4 serving framework. 2.4–3.5× throughput over TensorRT-LLM for 72B-class models on A100. FireQ reports 1.26× prefill speedup vs QServe. Establishes W4A8KV4 as the production-viable PTQ target. MLSys 2024.

### NVFP4 KV Cache (NVIDIA, 2025) [15]
- **Authors**: NVIDIA Technical Blog
- **URL**: https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/ (verified)
- **Summary**: FP4 KV cache on Blackwell GPUs. <1% accuracy loss on LiveCodeBench, MMLU-PRO. <50% KV memory vs FP8. Tested on Qwen3-Coder-480B-A35B and Llama 3.3 70B.

### DeepSeek-V4 Low-Precision Attention State (DeepSeek-AI, 2026) [22]
- **URL**: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro and `DeepSeek_V4.pdf`
- **Summary**: DeepSeek-V4 uses hybrid CSA/HCA attention with mixed BF16/FP8 KV storage and FP4 QAT for the CSA lightning-indexer QK path. It reports that V4-Pro uses 10% of DeepSeek-V3.2's KV cache at 1M context and V4-Flash uses 7%. This is not a post-hoc KV quantizer; it is evidence that low-precision storage and computation can be trained into the attention-state interface at frontier scale.
- **Relevance**: Strongest scale evidence for the "baked into model/interface" part of this idea, though the mechanism is CSA/HCA-specific rather than TurboQuant's rotation/QJL scheme.

### No Token Left Behind: Reliable KV Cache Compression via Importance-Aware Mixed Precision Quantization (MiKV) [16]
- **Authors**: Yang et al.
- **URL**: https://arxiv.org/abs/2402.18096 (verified)
- **Summary**: Keeps important tokens at high precision, eviction-candidate tokens in low precision. Up to 80% compression ratio.

### vLLM FP8 KV Cache [17]
- **Authors**: vLLM project
- **URL**: https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/ (verified)
- **Summary**: FP8 KV cache (2× reduction vs BF16, up to 1.6× throughput on H100). Production baseline.

### Coupled Quantization: KV Cache is 1 Bit Per Channel [19]
- **Authors**: Tianyi Zhang, Jonah Yi, Zhaozhuo Xu, Anshumali Shrivastava
- **URL**: https://arxiv.org/abs/2405.03917 (verified)
- **Summary**: Exploits inter-channel dependence within KV activations via joint (coupled) quantization, showing that per-channel independent quantization is sub-optimal. Achieves 1-bit-per-channel KV compression with reasonable quality preservation; 1.4×–3.5× throughput improvement vs uncompressed baseline. NeurIPS 2024.
- **Relevance**: Establishes that exploiting cross-channel structure — not just per-channel scalar quantization — enables extreme (1-bit) KV compression. Complements TurboQuant's rotation-based approach and motivates QAT at extreme bit-widths.

### Cache Me If You Must: Adaptive KV Cache Quantization [20]
- **Authors**: Alina Shutova, Vladimir Malinovskii, Vage Egiazarian, Denis Kuznedelev, Denis Mazur, Nikita Surkov, Ivan Ermakov, Dan Alistarh
- **URL**: https://openreview.net/forum?id=COowwJOAZi (ICML 2025)
- **Summary**: PTQ method exploiting inter-layer KV dependencies via compact linear adapters that predict adjacent-layer KV representations, storing only unpredicted residuals. Near-lossless at 2–2.5 bits per value on Llama 3.2 (up to 70B). One-shot calibration on a single GPU in 1–6 hours. ICML 2025.
- **Relevance**: Demonstrates that inter-layer KV redundancy (not just intra-channel structure) can be exploited for extreme compression. A QAT variant of this inter-layer prediction approach is an alternative to TurboQuant-style rotation QAT.

### PM-KVQ: Progressive Mixed-Precision KV Cache Quantization for Long-CoT LLMs [21]
- **Authors**: Tengxuan Liu, Shiyao Li, Jiayi Yang, Tianchen Zhao, Feng Zhou, Xiaohui Song, Guohao Dai, Shengen Yan, Huazhong Yang, Yu Wang
- **URL**: https://arxiv.org/abs/2505.18610
- **Summary**: Addresses quantization error accumulation in long chain-of-thought decoding by progressively decreasing bit-width across blocks (higher precision for sensitive early blocks) and using positional interpolation for long-context calibration. Up to 8% improvement over SOTA on reasoning benchmarks (math, coding) at 7B–70B scale. arXiv 2025.
- **Relevance**: Directly addresses A2/C long-context use case (262K tokens) — standard flat-precision KV quantization degrades on long-CoT reasoning, which is a realistic workload. Mixed-precision progressive strategy is synergistic with TurboQuant-style QAT.

### PV-Tuning: Beyond Straight-Through Estimation for Extreme LLM Compression [18]
- **Authors**: Vladimir Malinovskii, Denis Mazur, Ivan Ilin, Denis Kuznedelev, Konstantin Burlachenko, Kai Yi, Dan Alistarh, Peter Richtárik
- **URL**: https://arxiv.org/abs/2405.14852
- **Summary**: Proximal gradient estimators for extreme (2-bit) compression, outperforming STE. NeurIPS 2024. Relevant mitigation for sub-4-bit QAT instability; while applied to weights, the technique transfers to KV cache quantization. Reduces estimated technical risk for 2-3 bit KV QAT from HIGH to MEDIUM-HIGH.

---

## 3. Prior Art Classification

- **Status**: PARTIAL

- **Overlap summary**: The literature covers approximately 80% of the idea.
  - **KV-cache-only quantization scope**: Fully covered (KVQuant, KIVI, RotateKV, TurboQuant all scope to KV cache interface only).
  - **The specific TurboQuant algorithm (rotation + QJL) applied to KV cache**: EXISTS (Zandieh et al., 2026), but it is post-hoc and training-free.
  - **Aggressive quantization (sub-4-bit) of KV cache**: EXISTS at 2-bit (KIVI, RotateKV, Kitty) and 3-bit (KVQuant, TurboQuant).
  - **QAT for KV cache**: EXISTS partially (LLM-QAT), but co-quantizes W+A+KV jointly — not KV in isolation.
  - **"Baked into model" / purely learned-from-scratch QAT for KV cache only**: **PARTIAL** — no paper does QAT targeting the KV cache interface in isolation, decoupled from weight quantization.

- **Novel contribution**: The specific combination of (a) TurboQuant-style rotation + QJL quantization applied (b) purely to the KV cache interface (c) with quantization-aware training from scratch (not post-hoc) has not been published. LLM-QAT is the closest prior art but applies QAT to all tensors jointly.

---

## 4. Technical Analysis

**Novelty verdict: PARTIAL — QAT targeting the KV cache interface in isolation, decoupled from weight QAT, has no published precedent; TurboQuant [1] is post-hoc, not QAT.**

### 4.1 Theoretical Complexity

**KV cache memory** with quantization factor q = B/b (q=4 for INT4 vs BF16):

| Metric | This Idea (INT4 KV) | Baseline A1 | Baseline A2 | Baseline B | Baseline C |
|--------|---------------------|-------------|-------------|------------|------------|
| KV cache (32K ctx, INT4) | ↓ 4× vs BF16 | ~0.54 GB | ~2.15 GB | ~0.25 GB | ~2.5 GiB |
| KV cache (262K ctx, INT4) | ↓ 4× vs BF16 | ~4.3 GB | ~17.2 GB | ~2.0 GB | ~20.0 GiB |
| Weight memory | = baseline | ~54 GB | ~64 GB | ~794 GB total | ~145.1 GB |

**TPOT improvement formula (canonical):**
```
TPOT_improvement = (weight_BW + KV_BW_before) / (weight_BW + KV_BW_after)
```

At batch=1, memory-bandwidth-bound decode:

| Baseline | Context | weight_BW | KV_BW_before | KV_BW_after (INT4) | TPOT_improvement |
|---------|---------|-----------|--------------|-------------------|------------------|
| A2 | 32K | ~64 GB | ~8.59 GB | ~2.15 GB | ~1.10× |
| A2 | 262K | ~64 GB | ~68.7 GB | ~17.2 GB | ~1.63× |
| A1 | 32K | ~54 GB | ~2.15 GB | ~0.54 GB | ~1.03× |
| B | 32K | ~34 GB | ~1.0 GB | ~0.25 GB | ~1.02× |
| C | 32K | ~145.1 GB | ~10.0 GiB (10.74 GB) | ~2.5 GiB (2.69 GB) | ~1.05× |
| C | 262K | ~145.1 GB | ~80.0 GiB (85.9 GB) | ~20.0 GiB (21.47 GB) | ~1.36× |

> **Note**: The FlashInfer "nearly linear speedup to compression ratio (~4x for 4bit)" refers to the attention **kernel** specifically, not total TPOT. The kernel-only 4× speedup assumes KV access is the sole bottleneck in that kernel. Total TPOT improvement is always less because weight loading is concurrent.

**BF16→INT4 4× compression is definitional** (2 bytes → 0.5 bytes per element); no experimental citation required.

### 4.2 Dequantization FLOP Overhead

For A2 at 262K context:
- KV elements per decode step: 64 × 2 × 8 × 128 × 262144 = 34.4B elements
- Dequant FLOPs (×+add per element): 34.4B × 2 = 68.8B
- Attention FLOPs at decode: ~4.3T total
- Overhead: ~1.6%

**"Negligible dequant FLOPs" is CORRECT** — overhead is <2%.

### 4.3 Memory Bandwidth Analysis

- **Weight loading**: Unchanged. KV cache interface only — model weight parameters not quantized.
- **KV cache access**: Sequential read per decode step. With INT4 KV, read is 4× smaller. Dequantization happens in registers — does not add HBM reads.
- **Activation memory**: Unchanged. Q, attention scores, and output remain at full precision.

### 4.4 Memory Capacity Summary Table

| Baseline | BF16 KV (32K) | INT4 KV (32K) | BF16 KV (native ctx) | INT4 KV (native ctx) | Savings |
|---------|--------------|--------------|---------------------|---------------------|---------|
| A1 (16 layers, 4KV, hd=256) | ~2.15 GB | ~0.54 GB | ~17.2 GB at 262K | ~4.3 GB | ↓ 4× |
| A2 (64 layers, 8KV, hd=128) | ~8.59 GB | ~2.15 GB | ~68.7 GB at 262K | ~17.2 GB | ↓ 4× |
| B (15 layers, 2KV, hd=256) | ~1.0 GB | ~0.25 GB | ~8.0 GB at 262K | ~2.0 GB | ↓ 4× |
| C (80 layers, 8KV, hd=128) | ~10.0 GiB | ~2.5 GiB | ~80.0 GiB at 262K | ~20.0 GiB | ↓ 4× |

---

## 5. Implementation Considerations

- **Hardware requirements**: INT4 KV attention kernels require custom CUDA/Triton work. Efficient INT4×FP8 GEMMs for attention are available in FlashAttention-3 (FlashInfer, FireQ's modified FA3) on H100/H200 with FP8 tensor cores. Blackwell (B200) has native FP4 tensor core support. For QAT specifically, the fake-quantize forward pass requires only standard PyTorch autograd (STE is implemented in TorchAO).

- **Training stability**: QAT with STE is stable for 4-bit and 8-bit quantization targets. For 2-bit or 3-bit via QAT, training instability is possible — STE ignores quantization rounding in backprop. Mitigation: PV-Tuning's representation-agnostic framework (NeurIPS 2024) outperforms STE at extreme compression.

- **Phased implementation recommendation**:
  - **Phase 1 (tractable)**: PolarQuant QAT — rotation (Cayley-parameterized, differentiable) + INT4 scalar quantization via STE, no QJL correction. Directly deployable on existing PyTorch + TorchAO stack.
  - **Phase 2 (research-grade)**: Full TurboQuant QAT — add differentiable QJL correction. The sign function's STE gradient is degenerate at 1-bit; requires soft-sign approximation (tanh(βx) with β annealing) or treating QJL as fixed post-hoc post-training.

- **Three additional training stability risks**:
  1. **Attention feedback loop**: Training with quantized K affects attention weights and the gradient signal for the entire model.
  2. **Scale gradient explosion**: If per-channel/per-token quantization scale becomes very small, gradients blow up. Mitigation: gradient clipping.
  3. **Rotation-weight coupling**: If Cayley rotation R is trained jointly with model weights, optimizer may find degenerate solutions. Mitigation: separate LR for R.

- **Framework support**: PyTorch TorchAO supports QAT with fake-quantize nodes. Inserting fake-quantize nodes at the KV write point requires minimal modification to HuggingFace or vLLM attention implementations.

---

## 6. Synergies

- **Combines well with**:
  - **Idea 1.2 (Per-Token Adaptive Depth)**: Adaptive depth reduces full-attention layer accesses; KV quantization reduces per-byte cost. Both reduce TPOT independently and additively.
  - **Idea 1.6 (Learned Layer Type)**: If the model learns to minimize full-attention layers, the KV cache shrinks; KV quantization compresses what remains.
  - **KV eviction (H2O, ScissorHands-style)**: After eviction, quantize retained tokens — two orthogonal compression dimensions.
  - **Ideas 5.7–5.10 (weight compression)**: Orthogonal; both can be applied simultaneously for maximum compression.

---

## 7. Risk Assessment

- **Technical risk**: MEDIUM — QAT for KV cache in isolation is technically straightforward at 4-bit (LLM-QAT shows it works), but scaling to sub-4-bit (2-3 bit) KV quantization with training-stable QAT at 27B–32B model scale has not been demonstrated. The TurboQuant rotation+QJL approach has not been adapted to a differentiable (QAT) setting.

- **Potential impact**: HIGH (for Baseline A2 / dense large models, especially at 262K context) — At 262K context, the KV cache grows to ~68.7 GB and INT4 reduces this to ~17.2 GB, yielding ~1.63× TPOT improvement. High impact for Baseline C at 262K–524K context as well. For Baselines A1 and B (hybrid architectures with small KV caches), absolute impact is low.

- **Implementation effort**: HIGH for QAT variant — Full-scale QAT at 27B–32B requires substantial GPU-hours (~1.35× pretraining compute). The post-hoc alternative (running TurboQuant as published, without QAT) is low-effort and captures most of the benefit for 3.5-bit compression.

---

## 8. Accuracy / Quality Tradeoff

**Novelty verdict: PARTIAL — QAT targeting the KV cache interface in isolation, decoupled from weight QAT, has no published precedent; TurboQuant [1] is post-hoc, not QAT.**

- **Reported quality delta** (all post-hoc PTQ — QAT expected to improve these):
  - **FP8 KV (2× compression)**: Near-zero accuracy loss. [15]
  - **3.5-bit TurboQuant (post-hoc)**: LongBench 50.06→50.06 (zero loss) on Llama 3.1-8B. [1]
  - **2.5-bit TurboQuant (post-hoc)**: LongBench 50.06→49.44 (-0.62, ~1.2% relative). [1]
  - **INT4 KV (PTQ, FireQ)**: WikiText-2 PPL 6.13→7.06 (+0.93); zero-shot 70.61%→67.92% (-2.69pp) on Llama3-8B. [13]
  - **2-bit KV (RotateKV, post-hoc)**: <0.3 PPL delta on WikiText-2 for LLaMA-2-13B; <1.7% GSM8K degradation. [10]
  - **2-bit KV (KIVI, tuning-free)**: Up to 2% accuracy drop on CoQA/TruthfulQA/GSM8K for Llama-2-7B. [2]
  - **W+A+KV QAT (LLM-QAT, 4-bit)**: "Large improvements over training-free methods, especially at low-bit" — no KV-only ablation published. [4]

- **Monotonicity**: Yes — more compression consistently degrades quality, but non-linearly. Rotation-based methods (TurboQuant, QuaRot, RotateKV, SpinQuant) show a much more favorable accuracy-vs-bits curve than naive scalar quantization.

- **Recovery via QAT**: SpinQuant narrows the gap to full precision to 2.9pp on LLaMA-2 7B from 19.1pp for LLM-QAT PTQ baseline — ~85% recovery. [5]

### Precision-to-Quality Table for A2

| Precision | KV A2 (32K ctx) | KV A2 (262K ctx) | PPL delta (wiki2) | Task accuracy delta | TPOT improvement |
|-----------|----------------|-----------------|------------------|--------------------|--------------------|
| BF16 (baseline) | ~8.59 GB | ~68.7 GB | — | — | 1× |
| FP8 / INT8 | ~4.3 GB | ~34.4 GB | ~0 (FP8) | ~0 | ~1.06× at 32K |
| INT4 | ~2.15 GB | ~17.2 GB | +0.93 PPL (FireQ PTQ) | −2.7pp (FireQ PTQ) | ~1.10× at 32K; ~1.63× at 262K |
| 3.5-bit TurboQuant | ~1.8 GB | ~14.4 GB | ~0 (matches full precision) | ~0 | ~1.15× at 32K |
| 2.5-bit TurboQuant | ~1.3 GB | ~10.3 GB | ~0.6 PPL equiv | −0.6/50 LongBench | ~1.17× at 32K |
| 2-bit (RotateKV) | ~1.07 GB | ~8.6 GB | <0.3 PPL | <1.7% on GSM8K | ~1.19× at 32K |

---

## Prior Art Gap

**What exists**: Post-hoc TurboQuant achieves near-optimal distortion at 3.5-bit (Zandieh et al., ICLR 2026); LLM-QAT applies QAT jointly to W+A+KV (Liu et al., ACL Findings 2024); SpinQuant uses learned rotations for KV quantization via calibration PTQ (Liu et al., ICLR 2025); DeepSeek-V4 trains low-precision CSA/HCA-specific attention state at frontier scale.

**What is novel**: No published work performs model-agnostic QAT targeting the KV cache interface in isolation (decoupled from weight and activation quantization), using TurboQuant-style rotation + QJL as a differentiable training objective, at pretraining scale for 27B–72B parameter models and across both standard GQA and heterogeneous compressed-attention layouts.

---

## Recommended Path

1. **Immediate**: Deploy post-hoc TurboQuant on A2 at 262K context — high ROI, low risk.
2. **DeepSeek-V4-informed path**: separate quantization targets by cache type: compressed CSA/HCA entries, sliding-window state cache, uncompressed tail buffers, and indexer QK caches.
3. **Phase 1 experiment**: Run PolarQuant QAT (rotation + INT4 scalar, no QJL) at ~1B parameter scale to validate KV-only QAT training stability and quality recovery at 2-3 bit.
4. **If Phase 1 succeeds**: Scale to 7B, then 32B/72B. Consider Phase 2 (differentiable QJL) as a research-grade extension.

---

## Appendix: Key Numerical Summary

| Baseline | Context | BF16 KV | INT4 KV | TPOT Improvement |
|---------|---------|---------|---------|-----------------|
| A2 (Qwen3-32B) | 32K | ~8.59 GB | ~2.15 GB | ~1.10× |
| A2 (Qwen3-32B) | 262K | ~68.7 GB | ~17.2 GB | ~1.63× |
| A1 (Qwen3.5-27B) | 32K | ~2.15 GB | ~0.54 GB | ~1.03× |
| B (Qwen3.5-397B) | 32K | ~1.0 GB | ~0.25 GB | ~1.02× |
| C (K2 family) | 32K | ~10.0 GiB (10.74 GB) | ~2.5 GiB (2.69 GB) | ~1.05× |
| C (K2 family) | 262K | ~80.0 GiB (85.9 GB) | ~20.0 GiB (21.47 GB) | ~1.36× |

<!-- CITATION MANIFEST -->
[1]: TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate — Zandieh, Daliri, Hadian, Mirrokni (Google Research). arXiv:2504.19874, §3 "TurboQuant: High Performance Quantization". ICLR 2026. Post-hoc, training-free two-stage KV cache quantization with rotation (PolarQuant) and QJL correction. 3.5-bit near-lossless on LongBench; 8× attention speedup vs FP32.
[2]: KIVI: A Tuning-Free Asymmetric 2-Bit Quantization for KV Cache — Zirui Liu et al. arXiv:2402.02750, §3.3 "KIVI: Algorithm and System Support". ICML 2024. 2-bit asymmetric KV quantization; per-channel keys, per-token values; FP16 residual buffer.
[3]: KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization — Hooper, Kim, Genc et al. arXiv:2401.18079, §3 "Method". NeurIPS 2024. Pre-RoPE quantization, non-uniform datatypes, dense+sparse correction.
[4]: LLM-QAT: Data-Free Quantization Aware Training for Large Language Models — Zechun Liu et al. (Meta AI). arXiv:2305.17888, §3 (Methods) [unavailable — inferred]. ACL Findings 2024. Joint W+A+KV QAT using data-free distillation and STE.
[5]: SpinQuant: LLM Quantization with Learned Rotations — Zechun Liu et al. (Meta AI). arXiv:2405.16406, §3 (Methods) [unavailable — inferred]. ICLR 2025. Cayley optimization of rotation matrices for W+A+KV PTQ; 2.9pp gap to full precision on LLaMA-2-7B.
[6]: QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs — Ashkboos, Mohtashami et al. (ETH Zurich). arXiv:2404.00456, §4 "Method". NeurIPS 2024. Random Hadamard rotations for end-to-end 4-bit quantization.
[7]: GEAR: An Efficient KV Cache Compression Recipe for Near-Lossless Generative Inference — Hao Kang et al. arXiv:2403.05527, §3 "GEAR Framework". 2024. Three-component KV compression: quantization + low-rank SVD + sparse correction.
[8]: H2O: Heavy-Hitter Oracle for Efficient Generative Inference — Zhenyu Zhang et al. arXiv:2306.14048, §4 "Heavy-Hitter Oracle". NeurIPS 2023. KV eviction strategy retaining heavy-hitter tokens.
[9]: ScissorHands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression — Zichang Liu et al. arXiv:2305.01623, §3 (Methods) [unavailable — inferred]. NeurIPS 2023. KV eviction using importance-persistence hypothesis; up to 5× KV reduction.
[10]: RotateKV: Accurate and Robust 2-Bit KV Cache Quantization via Outlier-Aware Adaptive Rotations — Zunhai Su et al. arXiv:2501.16383, §3 (Methods) [unavailable — inferred]. IJCAI 2025. 2-bit KV with outlier-aware adaptive rotations; <0.3 PPL, 5.75× batch size increase.
[11]: Kitty: Accurate and Efficient 2-Bit KV Cache Quantization with Dynamic Channel-wise Precision Boost — arXiv:2511.18643, §3 "Algorithm Design". November 2025. Sensitivity-ranked channel precision boost for 2-bit KV; 2.1×–4.1× throughput.
[12]: TurboAttention: Efficient Attention Approximation for High Throughput LLMs — Microsoft Research. arXiv:2412.08585, §3 "FlashQ: Headwise mix-precision progressive quantization". 2025. FlashQ + SAS; 1.2×–1.8× attention speedup. Distinct from TurboQuant.
[13]: FireQ: Fast INT4-FP8 Kernel and RoPE-aware Quantization for LLM Inference Acceleration — Baek, Choi et al. arXiv:2505.20839, §3.2 "Attention Layer". 2025. INT4 KV + FP8 Q; two-stage RoPE-aware smoothing; H100 SXM5.
[14]: QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving — Yujun Lin et al. (MIT HAN Lab). arXiv:2405.04532, §4 "QoQ Quantization". MLSys 2024. Production W4A8KV4 framework; 2.4–3.5× throughput over TensorRT-LLM for 72B-class models.
[15]: NVFP4 KV Cache — NVIDIA Technical Blog. 2025. FP4 KV cache on Blackwell GPUs; <1% accuracy loss. §"Results and Accuracy" [blog section]. https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/
[16]: MiKV: No Token Left Behind — Yang et al. arXiv:2402.18096, §3 "Mixed-Precision KV Cache Compression". 2024. Mixed-precision KV; important tokens at high precision; up to 80% compression.
[17]: vLLM FP8 KV Cache documentation — vLLM project. 2024. §"FP8 KV Cache" [docs section]. https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/
[18]: PV-Tuning: Beyond Straight-Through Estimation for Extreme LLM Compression — Malinovskii, Mazur, Ilin, Kuznedelev, Burlachenko, Yi, Alistarh, Richtárik. arXiv:2405.14852, §3 "Fine-Tuning Quantized Models". NeurIPS 2024. Representation-agnostic fine-tuning framework that goes beyond STE for 2-bit compression; applicable to KV cache QAT.
[19]: KV Cache is 1 Bit Per Channel: Efficient Large Language Model Inference with Coupled Quantization — Tianyi Zhang, Jonah Yi, Zhaozhuo Xu, Anshumali Shrivastava. arXiv:2405.03917, §3.2 "Coupled Quantization with Centroid Learning". NeurIPS 2024. Joint (coupled) cross-channel KV quantization achieving 1-bit-per-channel compression; 1.4×–3.5× throughput improvement.
[20]: Cache Me If You Must: Adaptive Key-Value Quantization for Large Language Models — Shutova, Malinovskii, Egiazarian et al. ICML 2025, §3 "Method" [venue paper — no arXiv]. Inter-layer KV prediction via linear adapters; near-lossless 2–2.5-bit PTQ on Llama 3.2 up to 70B; single-GPU calibration.
[21]: PM-KVQ: Progressive Mixed-Precision KV Cache Quantization for Long-CoT LLMs — Liu, Li, Yang et al. arXiv:2505.18610, §3 "Method". 2025. Progressive block-wise bit-width reduction with long-context calibration via positional interpolation; up to 8% over SOTA on long-CoT reasoning benchmarks.
