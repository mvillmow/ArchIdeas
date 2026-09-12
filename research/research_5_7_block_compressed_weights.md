# Research: Block-Level Compressed Weights
## ID: 5.7

---

## Executive Summary

Idea 5.7 proposes applying block-level compression (quantization, low-rank factorization, or both) to sub-matrix tiles within each weight matrix. This is the primary lever for **TPOT improvement** in autoregressive decoding: at batch=1, the model is memory-bandwidth-bound, and reducing weight bytes loaded per step directly reduces per-token decode latency.

**Three tiers of increasing novelty:**
1. **Block-wise INT4 quantization (Tier 1, Production-Ready):** GPTQ/AWQ/QServe/MARLIN deliver ~2.4–3.5× TPOT and ~3.9× weight memory reduction for models ≥7B. No custom kernel required; vLLM/TRT-LLM integrate these natively. Deploy today.
2. **Block low-rank (Tier 2, Research-to-Production):** BLAST-style per-tile UV factorization (B=128, r_b=16 for hardware-efficient 4× compression) requires a custom Triton/CUTLASS kernel (2–3 months). Published results stop at 7B; 32B validation needed.
3. **Combined per-tile Q+LR (Tier 3, Research):** Sub-tile `W_tile ≈ Q_tile + L_tile × R_tile` has no published precedent at 32B scale; 6–12 month research project.

**DeepSeek-V4 update (2026-04-24):** DeepSeek-V4 provides frontier-scale evidence for 4-bit expert-weight deployment, but not via GPTQ/AWQ-style generic INT4 PTQ. Its instruct checkpoints store routed MoE expert parameters in FP4 and most other parameters in FP8; FP4 QAT is also used for the CSA indexer QK path. This strengthens the "4-bit weights are viable at MoE frontier scale" claim while narrowing the concrete evidence to MXFP4/QAT-style expert weights and hardware-aware kernels.

**Key quantitative anchors:**
- A2 KV at 32K: **~8.59 GB** (8 KV heads GQA)
- Block LR FLOPs: **4× fewer FLOPs** at r_b/B = 1/8 (ratio 2r_b/B = 0.25)
- Combined Q+LR at 2.5 bits/param for 32B: **~10 GB** (<8 GB requires <2.0 bits/param)
- ATOM 7.7× throughput: multi-user serving metric (batch>1), not batch=1 TPOT
- CALDERA author list: arXiv:2405.18886 is Saha, Sagan, Srivastava, Goldsmith, Pilanci (Stanford/Princeton)
- Hardware-efficient tile minimum: r_b ≥ 16, B ≥ 128 for Tensor Core utilization


---

### Key Comparison Tables

**vs Baseline A1 (Qwen3.5-27B Hybrid)** — *27B params, 64 layers, ~54 GB BF16 weights; 16 full-attn + 48 Gated DeltaNet layers*

| Metric | Baseline A1 | INT4 Block Quant (g=128) | Block LR (B=128, r_b=16) | Change | Notes |
|--------|------------|--------------------------|--------------------------|--------|-------|
| Compute (FLOPs/token, decode) | O(L·d·d_ff) | O(L·d·d_ff) | O(L·d·d_ff/4) | = or ↓4× | W4A16: FLOPs unchanged. Block-LR: 4× FLOPs reduction (2r_b/B = 2×16/128 = 0.25) |
| Memory bandwidth (decode) | ~56 GB | ~14.4 GB | ~14.0 GB | ↓ **~3.9×** | Dominant gain is weight bytes; KV unchanged at ~2.15 GB |
| KV cache (32K ctx) | **~2.15 GB** | **~2.15 GB** | **~2.15 GB** | = | KV cache **unchanged** by weight compression |
| Weight memory | ~54 GB (BF16) | **~13.5 GB + ~0.3 GB meta** | **~13.5 GB** | ↓ ~**3.9×** | INT4: 13.5 GB weights + scales. LR: 2×d×d_ff×r_b/B bytes |
| Training cost | 1.0× | ~1.0× (post-hoc calibration) | ~1.0× (post-hoc SVD) | = | GPTQ/AWQ: ≤1 GPU-hour. BLAST: per-tile SVD ~minutes |
| TTFT (8K prompt) | ref | ~ref (W4A16); ↓~2× (W4A8) | ~ref to ↓~4× (compute-bound) | = or ↓ | W4A16: prefill compute-bound, FLOPs unchanged → TTFT unchanged |
| TPOT (batch=1) | ref | ↓ **~2.4–3.5×** practical | ↓ **~2–3×** estimated | ↓ **~2.4–3.5×** | MARLIN: 2.8× on RTX A6000; QServe: 2.4–3.5× (Qwen1.5-72B reference) |

**vs Baseline A2 (Qwen3-32B Dense)** — *32B params, 64 layers, ~64 GB BF16 weights*

| Metric | Baseline A2 | INT4 Block Quant (g=128) | Block LR (B=128, r_b=16) | Change | Notes |
|--------|------------|--------------------------|--------------------------|--------|-------|
| Compute (FLOPs/token, decode) | O(L·(s·d+d·d_ff)) | O(L·(s·d+d·d_ff)) | O(L·(s·d+d·d_ff/4)) | = or ↓4× MLP | FLOPs unchanged W4A16; block-LR reduces MLP FLOPs ~4× |
| Memory bandwidth (decode) | ~64 GB | ~16.5 GB | ~16 GB | ↓ **~3.9×** | Weight BW ~16.5 GB dominates KV BW ~8.59 GB at 32K |
| KV cache (32K ctx) | **~8.59 GB** | **~8.59 GB** | **~8.59 GB** | = | KV cache unchanged; weight BW **dominates** KV at batch=1, 32K ctx |
| Weight memory | ~64 GB (BF16) | **~16 GB + ~0.5 GB meta** | **~16 GB** | ↓ ~**3.9×** | INT4 fits single A100-40G (16.5 GB + 23.5 GB remaining) |
| Training cost | 1.0× | ~1.0× | ~1.0× | = | Post-hoc; calibration only |
| TTFT (8K prompt) | ref | ~ref (W4A16) | ~ref to ↓~4× | = or ↓ | W4A16: unchanged; W4A8: ~2× |
| TPOT (batch=1) | ref | ↓ **~2.4–3.5×** practical | ↓ **~2–3×** estimated | ↓ **~2.4–3.5×** | TPOT formula: baseline_weight_BW / compressed_weight_BW ≈ 64/16.5 = **3.88× ideal**; ~2.4–3.5× practical (dequant overhead) |

*TPOT formula for weight compression (canonical): `TPOT_compressed / TPOT_baseline = (compressed_weight_BW + KV_BW) / (baseline_weight_BW + KV_BW)` → (16.5 + 8.59) / (64 + 8.59) = 25.09 / 72.59 = **0.346×** → 2.89× improvement (batch=1, 32K)*

**vs Baseline B (Qwen3.5-397B-A17B MoE)** — *~17B active params, 60 layers, ~34 GB active weight BW*

| Metric | Baseline B | INT4 Block Quant (g=128) | Change | Notes |
|--------|-----------|--------------------------|--------|-------|
| Compute (FLOPs/token, decode) | O(L·(k·d·d_e+d²)) | O(L·(k·d·d_e+d²)) | = | W4A16: FLOPs unchanged |
| Memory bandwidth (decode, active weights) | ~34 GB | ~8.5 GB | ↓ **~4×** | Only active expert weights loaded; 4× byte reduction |
| KV cache (32K ctx) | **~1.0 GB** | **~1.0 GB** | = | KV from 15 global-attn layers; unchanged |
| Total weight memory | ~794 GB (BF16) | **~199 GB** | ↓ ~**4×** | Entire MoE weight storage; GPU cluster requirement drops from ~10× A100-80G to ~3× |
| Training cost | 1.0× | ~1.0× | = | Post-hoc |
| TPOT (batch=1) | ref | ↓ **~2.4–3.5×** | ↓ | Active expert weight bytes dominate TPOT for sparse MoE at batch=1 |

**vs Baseline C (K2 family, 80 layers)** — *~145.1 GB BF16 weights, 80 layers, 64Q/8KV GQA*

| Metric | Baseline C | INT4 Block Quant (g=128) | Change | Notes |
|--------|-----------|--------------------------|--------|-------|
| Compute (FLOPs/token, decode) | O(L·(s·d+d·d_ff)) | O(L·(s·d+d·d_ff)) | = | W4A16: FLOPs unchanged |
| Memory bandwidth (decode) | ~145.1 GB | ~37.4 GB | ↓ **~3.9×** | Weight compression dominates; ~37.4 GB INT4 + ~1.9 GB metadata |
| KV cache (32K ctx) | **~10.0 GiB** | **~10.0 GiB** | = | KV unchanged; compressed weights (~37.4 GB) still dominate KV at 32K |
| Weight memory | ~145.1 GB (BF16) | **~36.3 GB + ~1.1 GB meta** | ↓ ~**3.9×** | Enables 2× A100-80G for C (vs 2× for baseline; previously 2× were tight) |
| TPOT (batch=1) | ref | ↓ **~3.5×** (ideal ~3.88×) | ↓ **~3.5×** | TPOT formula: (37.4 + 10.0) / (145.1 + 10.0) = 47.4 / 155.1 = **0.306×** → 3.27× improvement ideal; ~3.0–3.5× practical |

---

## 1. Idea Description

**From arch_research_ideas.md (Section 5, idea 5.7):**

> Matrix decomposition applied at the block level rather than entire weight matrices. Each block of weights gets its own compressed representation (low-rank factorization, quantization, etc.).

**Inferred intent:** At decode time (batch=1), the dominant cost is memory bandwidth — every weight byte must be loaded from HBM for each generated token. By compressing weight matrices at the block level (sub-matrix granularity), the model reduces effective weight bytes loaded per token, reducing TPOT. "Block-level" means the matrix is partitioned into tiles; each tile gets its own compressed representation (its own quantization scale/zero-point, its own low-rank factors, or its own structured matrix representation).

This is distinct from:
- **Idea 5.8 (Block Sparse Weights):** 5.8 zeroes out entire blocks (structural sparsity — blocks are absent). 5.7 compresses every block densely — no zeroes, all weights retained in fewer bits or via low-rank factors. The distinction is "dense but compact" vs "sparse but zero."
- **Idea 2.2 (Compressed Dense Layers via Matrix Decomposition):** 2.2 applies SVD/low-rank to an entire weight matrix W = UV^T as a whole. 5.7 partitions W into sub-blocks B₁, B₂, …, Bₙ and applies the factorization separately to each Bᵢ, so each block gets its own rank and its own U_i, V_i. Per-block adaptation can improve accuracy at a given compression ratio by capturing local structure that global SVD misses, though global SVD (Eckart-Young theorem) remains Frobenius-optimal when dominant singular vectors span the full matrix.

---

## 2. Literature Review

### GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers[1]: Block-column second-order quantization
- **Authors**: Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh
- **URL**: https://arxiv.org/abs/2210.17323
- **Summary**: GPTQ quantizes weights layer-by-layer using approximate second-order (Hessian) information. Within each layer, weights are processed in columns grouped into blocks of 128 columns; Cholesky decomposition is used to efficiently compute second-order updates. Achieves 3–4 bit quantization of OPT-175B and BLOOM-176B in ~4 GPU hours. Reported inference speedup: 3.25× on NVIDIA A100, 4.5× on A6000 vs FP16, for decode-time GEMV.
- **Limitations**: Column-group blocking is for algorithmic efficiency, not per-block accuracy tuning. Does not apply separate low-rank factorization per block.

### AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration[2]: Activation-guided per-channel INT4 quantization
- **Authors**: Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, Song Han
- **URL**: https://arxiv.org/abs/2306.00978
- **Summary**: AWQ identifies "salient" weight channels by their corresponding activation magnitudes; protecting 1% of salient weights dramatically reduces quantization error. Uses per-channel scaling to preserve salient channels before quantization. Quantizes all weights to INT4 (W4A16). Achieves 3× speedup over HuggingFace FP16 on desktop and mobile GPUs. Enables 70B Llama-2 on mobile GPUs. MLSys 2024 Best Paper.
- **Limitations**: Per-channel scaling is still coarser than per-element. Does not apply low-rank factorization to sub-blocks.

### SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models[3]: Per-channel scale migration for W8A8
- **Authors**: Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, Song Han
- **URL**: https://arxiv.org/abs/2211.10438
- **Summary**: Smooths quantization difficulty from activations to weights via a mathematically equivalent per-channel scaling transformation. Enables W8A8 (INT8 both weight and activation) quantization. Achieves up to 1.56× speedup and 2× memory reduction with negligible accuracy loss.
- **Limitations**: W8A8 (2× compression vs FP16) is less aggressive than INT4 (4×). Channel-level, not sub-matrix-block level.

### OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models[4]: Per-transformer-block calibrated quantization
- **Authors**: Wenqi Shao et al. (OpenGVLab)
- **URL**: https://arxiv.org/abs/2308.13137
- **Summary**: Introduces Learnable Weight Clipping (LWC) and Learnable Equivalent Transformation (LET) for block-wise quantization error minimization. Quantizes parameters sequentially block by block (transformer block granularity), optimizing clipping thresholds and equivalent transformations via SGD on 128 calibration samples. All LLaMA-2 family models quantizable on a single A100-40G. Achieves near-lossless quantization for W4A16 and W4A8.
- **Limitations**: Block granularity is at the transformer-block level, not sub-matrix tile level.

### ATOM: Low-Bit Quantization for Efficient and Accurate LLM Serving[5]: Fused W4A8KV4 kernel with fine-grained group quantization
- **Authors**: Yilong Zhao, Chien-Yu Lin, Kan Zhu, Zihao Ye, Lequn Chen, Size Zheng, Luis Ceze, Arvind Krishnamurthy, Tianqi Chen, Baris Kasikci
- **URL**: https://arxiv.org/abs/2310.19102
- **Summary**: Combines mixed-precision, fine-grained group quantization (g=128 for INT4), dynamic activation quantization, and KV-cache quantization. Fuses quantization/dequantization operators into existing GEMM kernels to minimize overhead. Achieves 7.7× end-to-end **serving throughput** (batch>1) and 2.5× vs INT8. **Note: The 7.7× figure is a multi-user serving throughput result (batch>1), not a batch=1 TPOT speedup.** At batch=1, TPOT speedup follows the ~2.4–3.5× range from weight bandwidth reduction.
- **Limitations**: Group size = 128 chosen for hardware alignment. 7.7× figure is batch>1 serving throughput, not batch=1 TPOT.

### ASVD: Activation-aware Singular Value Decomposition for Compressing Large Language Models[6]: Per-layer activation-aware SVD with adaptive rank selection
- **Authors**: Zhihang Yuan, Yuzhang Shang, et al.
- **URL**: https://arxiv.org/abs/2312.05821
- **Summary**: Training-free LLM compression via SVD-based weight low-rank decomposition. Transforms weight matrices based on activation distributions before decomposition — absorbing activation outliers into the transformed weight matrix. Achieves 10%–30% network compression and 50% KV cache reduction without performance drop. Per-layer rank selection.
- **Limitations**: Per-weight-matrix (not sub-matrix tile) level. Compression limited to 10%–30% before performance degradation.

### SVD-LLM: Truncation-aware Singular Value Decomposition for Large Language Model Compression[7]: Per-layer SVD with calibration-guided rank allocation
- **Authors**: Xin Wang et al. (AIoT-MLSys-Lab)
- **URL**: https://arxiv.org/abs/2403.07378
- **Summary**: Addresses limitations in ASVD via truncation-aware data whitening, establishing a direct mapping between singular values and compression loss. Evaluated on 10 datasets across seven LLM families. Outperforms state-of-the-art SVD-based methods at high compression ratios.
- **Limitations**: Per-weight-matrix level. Struggles at >30% compression without additional quantization.

### CALDERA: Compressing Large Language Models using Low Rank and Low Precision Decomposition[8]: W ≈ Q + LR decomposition at per-weight-matrix level
- **Authors**: Rajarshi Saha, Naomi Sagan, Varun Srivastava, Andrea J. Goldsmith, Mert Pilanci (Stanford University / Princeton University)
- **URL**: https://arxiv.org/abs/2405.18886
- **Summary**: Post-training compression algorithm decomposing weight matrices as W ≈ Q + LR, where Q is a quantized matrix, and L, R are quantized low-rank factors. Outperforms existing post-training compression in the <2.5 bits/parameter regime on LLaMA-2 (7B, 13B, 70B) and LLaMA-3 (8B).
- **Limitations**: Per-layer weight matrix level, not sub-matrix block level. No TPOT improvement numbers — focuses on compression quality.

### LC-SVD: Layer Collaborative Low-Rank Decomposition with Automatic Rank Search for LLM Compression[9]: Transformer-block-granularity joint SVD with automatic rank search
- **Authors**: (OpenReview submission — ICLR 2025: openreview.net/forum?id=m2nupeHqV7; author names not available from abstract)
- **URL**: https://openreview.net/forum?id=m2nupeHqV7
- **Summary**: Block-wise collaborative SVD that jointly compresses all linear layers within a transformer block (not layer-by-layer in isolation). Uses error-driven rank search. Outperforms state-of-the-art SVD-based methods.
- **Limitations**: Block granularity is the full transformer block, not sub-matrix tile.

### Monarch: Expressive Structured Matrices for Efficient and Accurate Training[10]: Block-diagonal butterfly structured matrices
- **Authors**: Tri Dao, Beidi Chen, Nimit Sohoni, Albert Gu, Michael Ermon, et al.
- **URL**: https://arxiv.org/abs/2204.00595
- **Summary**: Parameterizes weight matrices as products of two block-diagonal matrices, enabling O(n log n) matrix-vector multiply. Achieves 2× training speedup vs dense on GPT-2 (Wikitext-103), BERT pretraining 23% faster than NVIDIA's optimized implementation.
- **Limitations**: Shared block-diagonal structure (not per-block independent adaptation). Limited to products of two block-diagonal matrices.

### BLAST: Block-Level Adaptive Structured Matrices for Efficient Deep Neural Network Inference[11]: Per-block adaptive structured matrices
- **Authors**: Changwoo Lee et al.
- **URL**: https://arxiv.org/abs/2410.21262
- **Summary**: Learns block-adaptive structured weight matrices that can represent any weight matrix with per-block adaptation. Achieves 2× compression on Llama-7B with lowest performance degradation among structured matrix methods. 70% and 40% complexity reduction for ViT and GPT-2. GitHub: github.com/changwoolee/BLAST.
- **Limitations**: Published results stop at 7B parameters. TPOT speedup numbers not reported in abstract. Uses shared bases across blocks with per-block diagonal coupling factors (not fully independent U_b, V_b per tile as in 5.7's proposed formulation).

### Building on Efficient Foundations: Effectively Training LLMs with Structured Feedforward Layers[12]: BlockDense structured MLP layers
- **Authors**: Xiuying Wei, Skander Moalla, Razvan Pascanu, Caglar Gulcehre
- **URL**: https://arxiv.org/abs/2406.16450
- **Summary**: Replaces dense MLP weight matrices with structured feedforward layers (block-diagonal + low-rank). At 32% FFN params: ~1.0 PPL increase. 2.6× FFN speed-up (NeurIPS 2024 Table 2) and 1.35× training speedup on compute-bound hardware.
- **Limitations**: Applies at training time (not post-hoc). 2.6× speedup is compute-bound prefill; batch=1 GEMV benefit would be smaller.

### MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models[13]: INT4 batch=1 GEMV kernel for A100/H100
- **Authors**: Elias Frantar, Roberto L. Castro, Jiale Chen, Torsten Hoefler, Dan Alistarh
- **URL**: https://arxiv.org/abs/2408.11743
- **Summary**: Mixed-precision GEMV kernel for FP16×INT4 weight-only quantization. Uses lop3 instruction for simultaneous INT4 unpacking, fused dequantize+MACC, and persistent thread blocks. Achieves ~4× speedup up to batch size 16–32; ~2.02× at batch=1 on A100 (Llama-2-70B); ~2.8× TPOT on RTX A6000. Integrated into vLLM.
- **Limitations**: A100-specific optimization. Speedup at batch=1 (~2.8×) is below ideal 4× due to ~30% dequantization overhead.

### QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving[14]: W4A8KV4 system with 20–90% dequantization overhead elimination
- **Authors**: Yujun Lin et al. (MIT HAN Lab)
- **URL**: https://arxiv.org/abs/2405.04532
- **Summary**: W4A8KV4 quantization with progressive quantization strategy. Eliminates 20–90% dequantization overhead via progressive quantization and SmoothAttention for KV4. Achieves 1.2–1.4× over TRT-LLM for Llama-3-8B; **2.4–3.5× for Qwen1.5-72B** (note: for Qwen3-32B, expected range ~2.0–3.0× given smaller model size).
- **Limitations**: Results cited are for Qwen1.5-72B; expected range for 32B models is 2.0–3.0×.

### DeepSeek-V4 FP4 Expert Weights (DeepSeek-AI, 2026)[22]: Frontier-scale MXFP4/QAT expert-weight deployment
- **URL**: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro and `DeepSeek_V4.pdf`
- **Summary**: DeepSeek-V4 instruct checkpoints use FP4 for routed expert parameters and FP8 for most remaining parameters. During post-training, FP4 quantization-aware training is applied to MoE expert weights; for forward compute, FP4 weights are dequantized losslessly to FP8 under their block-scale assumptions, while inference/rollout uses real FP4 quantized weights.
- **Relevance**: Direct evidence that 4-bit expert weights can be used in a 1.6T-total-parameter MoE family when the training and kernel stack are designed around the format.

### LoftQ: LoRA-Fine-Tuning-Aware Quantization for Large Language Models[15]: Combined quantization + low-rank initialization
- **Authors**: Yixiao Li et al. (Microsoft Research)
- **URL**: https://arxiv.org/abs/2310.08659
- **Summary**: Initializes LoRA adapters with alternating quantization and SVD to minimize initialization gap. W ≈ Q + LR alternating quantization + low-rank approximation. Outperforms QLoRA at 2-bit and mixed 2/4-bit regimes for fine-tuning.
- **Limitations**: Fine-tuning focused (not post-training-only). Low-rank factors are LoRA adapters, not permanent compressed weights.

### AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning[16]: Per-layer rank adaptation for fine-tuning
- **Authors**: Qingru Zhang, Minshuo Chen, Alexander Bukharin et al. (Microsoft Research)
- **URL**: https://arxiv.org/abs/2303.10512
- **Summary**: SVD-parameterized incremental updates with importance-based singular value pruning, enabling per-layer rank allocation during fine-tuning. Integrated into HuggingFace PEFT.
- **Limitations**: Fine-tuning focused; incremental updates only. Not directly applicable to post-training weight compression for TPOT.

### UniQL: Unified Quantization and Low-Rank Compression for Adaptive Edge LLMs[17]: Combined quantization + LR for edge deployment
- **Authors**: Hung-Yueh Chiang, Chi-Chih Chang, Yu-Chen Lu, Chien-Yu Lin, Kai-Chiang Wu, Mohamed S. Abdelfattah, Diana Marculescu
- **URL**: https://arxiv.org/abs/2512.03383
- **Summary**: Unified compression combining quantization and low-rank decomposition for adaptive edge LLMs. Claims 4×–5.7× memory reduction, 2.7×–3.4× throughput. Accuracy within 5% at 15% pruning. Supports Transformers, SSMs, and hybrid architectures; enables on-device configurable pruning up to 35%.
- **Limitations**: Edge-deployment focus; primarily targeting smaller on-device models rather than 30B+ server-side models. Results primarily on smaller model sizes.

### QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks[18]: Near-lossless 2-bit block vector quantization
- **Authors**: Albert Tseng, Jerry Chee, Qingyao Sun, Volodymyr Kuleshov, Christopher De Sa
- **URL**: https://arxiv.org/abs/2402.04396
- **Summary**: Extends QuIP (Chee et al., 2023, arXiv:2307.13304) with E8 lattice codebooks and improved Hadamard incoherence preprocessing. Achieves near-lossless 2-bit compression of LLaMA models (within 1.2 PPL of FP16 at 2 bits/param). Each block of weights gets its own codebook entry — this is block-level vector quantization, directly related to 5.7's per-block compressed representation.
- **Limitations**: E8 lattice codebook inference kernel is more complex than scalar INT4 dequantization. Less production-mature than GPTQ/AWQ.

### SqueezeLLM: Dense-and-Sparse Quantization[19]: Sparse outlier weights + INT4 non-outlier weights
- **Authors**: Sehoon Kim, Coleman Richard Hooper, Amir Gholami, Zhen Dong, Xiuyu Li, Sheng Shen, Michael W. Mahoney, Kurt Keutzer
- **URL**: https://arxiv.org/abs/2306.07629 (ICML 2024)
- **Summary**: Combines sparse storage of sensitive outlier weights (small number stored at FP16) with INT4 quantization of the remaining (non-sensitive) weights. Achieves better quality than pure INT4 at the same average bit rate. Directly implements the 5.7+5.8 hybrid.
- **Limitations**: Sparse outlier addressing requires custom sparse kernel; scatter-gather overhead at batch=1.

### SparseGPT: Massive Language Models Can Be Accurately Pruned in One Shot[20]: Second-order joint pruning for LLMs
- **Authors**: Elias Frantar, Dan Alistarh
- **URL**: https://arxiv.org/abs/2301.00774 (ICLR 2023)
- **Summary**: Post-hoc magnitude pruning of LLMs using same second-order (Hessian) framework as GPTQ. Achieves 50–60% sparsity with minimal quality loss on OPT, BLOOM, LLaMA families. Provides the prune-then-quantize baseline. Joint prune+quantize with GPTQ gives ~8× compression.
- **Limitations**: Sparse weight storage at batch=1 GEMV is hardware-inefficient unless 2:4 structured sparsity is used.

### SLiM: One-shot Quantization and Sparsity with Low-rank Approximation for LLM Weight Compression[21]: Unified one-shot Q + 2:4 sparsity + LR correction
- **Authors**: Mohammad Mozaffari, Amir Yazdanbakhsh, Maryam Mehri Dehnavi
- **URL**: https://arxiv.org/abs/2410.09615 (ICML 2025)
- **Summary**: One-shot compression framework integrating hardware-friendly quantization (SLIM-Quant), 2:4 structured sparsity, and a low-rank adapter for residual error correction in a unified process. Uses a novel saliency function to compute the value of low-rank adapters analytically. Achieves up to 5.66% accuracy improvement over W4+2:4 sparsity baselines (LLaMA-2-7B), 4.3× layer-wise speedup on RTX3060 and 3.8× on A100. Directly addresses the Tier 3 combined Q+LR direction at a hardware-efficient level via 2:4 sparsity kernels.
- **Limitations**: 2:4 sparsity hardware acceleration is NVIDIA-specific (Ampere+). Published results at 7B/13B scale; 32B validation needed. Low-rank correction is a post-compression adapter, not per-tile as in 5.7's Tier 3 formulation.

---

## 3. Core Mechanism

### Tier 1: Block-Wise Quantization (INT4, g=128)

Each weight matrix W ∈ ℝ^{m×n} is partitioned into groups of g=128 consecutive elements (per row or per column). Each group gets its own scale s_g and optional zero-point z_g at FP16 precision:

`W_compressed[i,j] = round(W[i,j] / s_g - z_g)` in INT4 (0–15 range)

During inference (GEMV at batch=1):
1. Load W_compressed tile (INT4, 0.5 bytes/param) + scales (FP16, 2 bytes per 128 params → 1/64 overhead)
2. Dequantize: `W_fp16 = (W_int4 + z_g) × s_g` — using fused lop3 instructions (MARLIN)
3. Compute GEMV: `y += W_fp16 × x`

**Key parameters**:
- Compression ratio: `2 bytes / (0.5 + 2/128 bytes) ≈ 3.88×` (scale-only) or `3.77×` (scale+zero-point)
- Bandwidth reduction: ~3.88× (matched by practical speedup ~2.4–3.5× after ~30% dequantization overhead)
- FLOPs: **unchanged** (dequantization adds negligible FLOPs at g=128)

**DeepSeek-V4 note:** for MoE expert weights, FP4/QAT should be considered alongside GPTQ/AWQ PTQ. PTQ remains the shortest path for existing checkpoints; QAT-style FP4 is the stronger path when training or post-training control is available.

### Tier 2: Block Low-Rank (Per-Tile UV Factorization)

W ∈ ℝ^{m×n} is partitioned into (m/B) × (n/B) tiles, each B×B. Each tile gets its own independent low-rank factorization:

`W_tile = U_b × V_b^T` where U_b ∈ ℝ^{B×r_b}, V_b ∈ ℝ^{r_b×B}

During GEMV (batch=1): `y[i*B:(i+1)*B] += U_b × (V_b × x[j*B:(j+1)*B])`

**Key parameters**:
- Compression ratio: `B / (2r_b)` — at B=128, r_b=16: **4×**
- FLOPs per tile: `2 × B × r_b` (two sequential GEMV steps) vs `B²` dense → ratio `2r_b/B` = **0.25× at r_b/B=1/8** (4× fewer FLOPs, NOT 8×)
- Hardware constraint: r_b ≥ 16 required for Tensor Core efficiency (MMA minimum: 16×8×16). At r_b=8, tiles underfit Tensor Core MMA atoms → warp underutilization.
- **Minimum hardware-efficient configuration: B=128, r_b=16** (not B=64, r_b=8)

Two-step GEMV adds sequential dependency (V_b×x must complete before U_b×result). At batch=1, this is bandwidth-bound so the dependency is not the primary bottleneck.

### Tier 3: Combined Sub-Tile Q + LR

Each B×B tile gets both quantization and low-rank correction:

`W_tile ≈ Q_tile + L_tile × R_tile`

where Q_tile is INT4-quantized and L_tile, R_tile are the residual low-rank factors. This is the sub-tile analog of CALDERA (which operates at the per-weight-matrix level).

**Key parameters**:
- Memory at 2.5 bits/param for 32B model: `32B × 2.5 / 8 = 10 GB` (not "<8 GB"; <8 GB requires <2.0 bits/param)
- No production kernel implementation for per-tile combined Q+LR

---

## 4. Prior Art Gap

Production-deployed existing work:
- Per-transformer-block quantization calibration: **OmniQuant [4]** — EXISTS
- Per-weight-matrix SVD with rank adaptation: **ASVD [6], SVD-LLM [7]** — EXISTS
- Per-weight-matrix Q+LR: **CALDERA [8], LoftQ [15]** — EXISTS
- Per-block structured matrices (block-diagonal, block-adaptive): **Monarch [10], BLAST [11]** — EXISTS
- Per-block codebook vector quantization with Hadamard preprocessing: **QuIP# [18]** — EXISTS
- Sparse + INT4 hybrid: **SqueezeLLM [19]** — EXISTS
- One-shot unified Q + 2:4 sparsity + LR correction: **SLiM [21]** — EXISTS

**What is novel (5.7 specific gap):** Fine-grained sub-tile independent low-rank factorization where each B×B tile of a single weight matrix gets its own U_b, V_b with per-tile rank r_b — this is more fine-grained than BLAST (which uses shared bases + per-block diagonal coupling) and LC-SVD (transformer-block granularity). The combination of fine-grained sub-tile LR + per-tile quantization (sub-tile combined Q+LR) has no direct precedent at the sub-matrix-tile level at 32B scale.

**Novelty verdict:** Tier 1 (INT4 block-wise quantization) is EXISTS — GPTQ [1] / AWQ [2] prior art is production-deployed. Tier 2 (block low-rank) is EXISTS — BLAST [11] establishes per-block adaptive structured matrices; per-tile UV rank variation is an incremental refinement of the same direction, not a new class. Tier 3 (combined per-tile Q+LR) is PARTIAL/NOVEL — whole-matrix Q+LR (CALDERA [8], LoftQ [15]) and block-level codebook quantization (QuIP# [18]) narrow the gap from both directions, but fused sub-tile Q+LR at 32B scale has no published precedent.

---

## 5. Implementation

Deployment proceeds in tiers matching the novelty structure of the idea. Tier 1 (INT4 block-wise quantization) is immediately production-ready via MARLIN, AWQ kernels, AutoGPTQ, TensorRT-LLM, and vLLM; no calibration or custom kernel work is needed to ship GPTQ/AWQ on Qwen3 family weights. Tier 2 (per-tile UV low-rank) requires a custom Triton or CUTLASS kernel (2–3 months engineering) because existing MARLIN backends do not support the two-step U_b×(V_b×x) pattern at block granularity; the BLAST reference codebase (github.com/changwoolee/BLAST) supplies a block-BMM starting point. Tier 3 (combined per-tile Q+LR) is a 6–12 month research project — no production kernel exists for fused sub-tile Q+LR and validation at 32B scale is open. Sparse+INT4 hybrids (in combination with 5.8) require a custom INT4+sparse kernel because cuSPARSELt supports only INT8+sparse. Detailed kernel notes, hardware constraints, and GQA asymmetry analysis appear in §6.

---

## 6. Synergies

Block-compressed weights compose with orthogonal compute- and KV-side techniques:
- **5.8 Block-sparse weights**: Tier 1 (INT4) can stack with 2:4 structured sparsity per SqueezeLLM [19] and SLiM [21], yielding ~8× theoretical and ~3–5× practical TPOT, subject to a custom INT4+sparse kernel (cuSPARSELt supports only INT8+sparse as of 2024).
- **5.1 TurboQuant / KV quantization**: INT4 weights and INT4/INT8 KV are independent dimensions; combining W4A8KV4 (QServe [14]) delivers ~2.4–3.5× over FP16 for Qwen1.5-72B.
- **2.2 Compressed dense layers**: Tier 2 block-LR shares the structured-matrix direction with 2.2's low-rank MLPs; per-tile r_b overlays on top of whole-matrix rank budgeting.
- **Baseline-specific interactions**: Gated DeltaNet recurrent state weights (A1 hybrid, 48/64 layers) are shape-agnostic for Tier 1 but uncharacterized for Tier 2/3. GQA asymmetry (A2, C) reduces Tier 2/3 benefit on small W_K/W_V matrices; Tier 1 remains applicable.

Implementation specifics for each kernel path are enumerated in §6.1–§6.5 below.

### 6.1 Existing Production Kernels (Tier 1 — INT4)

- **MARLIN** (github.com/IST-DASLab/marlin): Production FP16×INT4 GEMV for NVIDIA Ampere; integrated into vLLM
- **AutoGPTQ**: Python+CUDA library wrapping GPTQ kernels; HuggingFace integration
- **AWQ kernels** (github.com/mit-han-lab/llm-awq): INT4 GEMV with per-group scaling
- **TensorRT-LLM**: Native W4A16 and W4A8KV4 quantization
- **vLLM**: GPTQ/AWQ/MARLIN backends integrated; no calibration needed for GPTQ/AWQ

### 6.2 Block-LR Kernel Requirements (Tier 2)

For hardware-efficient block-LR:
- **Minimum tile size**: B=128, r_b=16 (4× compression, Tensor Core efficient)
- **Avoid**: B=64, r_b=8 (tiles underfit MMA 16×16 atom → warp underutilization)
- **Implementation options**: PyTorch BMM (prototype, 2–4 weeks), custom Triton kernel (production, 2–3 months), CUTLASS template (optimal, 3–4 months)
- **Reference**: BLAST codebase (github.com/changwoolee/BLAST) provides block-BMM reference implementation
- **vLLM integration**: Requires custom kernel; cannot use existing MARLIN backend

### 6.3 Sparse + INT4 (5.7 + 5.8 Combined)

- **cuSPARSELt**: Supports sparse + INT8 natively. **Does NOT support sparse + INT4 as of 2024** — custom kernel required for INT4+sparse combination
- **SqueezeLLM [19]**: Published reference for sparse outlier + INT4 non-outlier hybrid
- **Estimated combined TPOT** (50% sparse + 4× INT4): ~8× theoretical, ~3–5× practical due to sparse addressing overhead

### 6.4 Gated DeltaNet Layers (Baseline A1 Specific)

Qwen3.5-27B Hybrid uses Gated DeltaNet (linear attention) for 48/64 layers. Gated DeltaNet recurrent state update weights have non-standard shapes distinct from standard attention projections. Block-LR compression compatibility with Gated DeltaNet weights is not analyzed; Tier 1 (INT4) is shape-agnostic and compatible with any weight matrix.

### 6.5 GQA Asymmetry (A2, C)

Qwen3-32B uses 64Q/8KV heads. W_K and W_V are 5120×1024 (much smaller than W_Q 5120×8192). At B=128: K/V matrices have only (5120/128)×(1024/128) = 40×8 = 320 tiles vs Q's 40×64 = 2560 tiles. Per-block adaptation benefit is reduced for small K/V matrices; Tier 1 INT4 remains applicable to all matrices regardless of size.

---

## 7. Risk Assessment

- **Technical risk**:
  - Tier 1 (INT4 block quantization): LOW — GPTQ/AWQ/QServe/MARLIN are production-deployed on 70B-class models with near-lossless quality.
  - Tier 2 (per-tile block-LR): MEDIUM — BLAST [11] validates block-adaptive structured matrices at 7B; extrapolation to 27B/32B/72B is unproven and r_b/B ratio selection is model-dependent.
  - Tier 3 (combined per-tile Q+LR): HIGH — no production kernel exists for fused sub-tile Q+LR; quality at <2.5 bits/param has only whole-matrix precedents (CALDERA [8], LoftQ [15]).
- **Implementation effort**: LOW (Tier 1, days) → MEDIUM (Tier 2, 2–3 months custom Triton/CUTLASS) → HIGH (Tier 3, 6–12 months including kernel and quality validation).
- **Quality cliff risk**: INT2–3 without low-rank correction degrades sharply on all tested models (see §8). Tier 3 is the only path into the <2.5 bits/param regime at current quality bars; its validation is the primary research risk for this idea.
- **Hardware risk**: cuSPARSELt lacks INT4+sparse support as of 2024; any 5.7+5.8 combination requires a custom kernel. Tier 2 minimum hardware-efficient configuration (B=128, r_b=16) must be enforced to avoid Tensor Core underutilization.

---

## 8. Accuracy / Quality Tradeoff

**INT4 block quantization (4× compression, post-training):**
- GPTQ on OPT-175B at INT4: negligible accuracy degradation [Frantar et al., 2022 — arXiv:2210.17323, §1]
- AWQ on LLaMA-2-70B INT4: comparable to GPTQ with better generalization, 3× speedup on desktop GPUs [Lin et al., 2023 — arXiv:2306.00978, §4]
- ATOM on Llama/Mixtral INT4 group-128: negligible accuracy loss [Zhao et al., 2023 — arXiv:2310.19102, §5]
- QServe W4A8KV4 on Qwen1.5-72B: 2.4-3.5× throughput over TensorRT-LLM, quality maintained [Lin et al., 2024 — arXiv:2405.04532, §6]
- OmniQuant W4A16 on LLaMA-2 family: near-lossless quantization [Shao et al., 2023 — arXiv:2308.13137, §4]

**Block low-rank (2–4× compression):**
- Monarch on GPT-2 (Wikitext-103): comparable model quality at 2× training speedup [Dao et al., 2022 — arXiv:2204.00595, §4]
- BLAST on Llama-7B: 2× compression with lowest performance degradation among structured matrices tested [Lee et al., 2024 — arXiv:2410.21262, §5]
- BlockDense at 32% FFN params: ~1.0 PPL increase on Transformer-xl 1.3B [Wei et al., 2024 — arXiv:2406.16450, §4.1]
- ASVD on LLMs: 10–30% compression without performance drop [Yuan et al., 2023 — arXiv:2312.05821, §4]

**Combined Q+LR (<2.5 bits/param):**
- CALDERA on LLaMA-2 7B/13B/70B, LLaMA-3 8B: outperforms all existing post-training compression at <2.5 bits/param regime [Saha et al., 2024 — arXiv:2405.18886, §5]
- LoftQ: outperforms QLoRA especially in 2-bit and mixed 2/4-bit regimes for fine-tuning [Li et al., 2023 — arXiv:2310.08659, §4]
- QuIP# near-lossless 2-bit on LLaMA via Hadamard + E8 codebook [Tseng et al., 2024 — arXiv:2402.04396]
- SLiM one-shot Q+2:4sparsity+LR on LLaMA-2-7B/13B: up to 5.66% accuracy improvement over W4+2:4 baseline, 4.3× RTX3060 / 3.8× A100 speedup [Mozaffari et al., ICML 2025 — arXiv:2410.09615]

**Non-linear quality cliff:**
- At INT4 (4×) for models ≥7B: flat quality curve — minimal loss
- At INT2–3 without low-rank correction: severe degradation on all tested models
- At INT2–3 WITH low-rank correction (CALDERA, LoftQ, QuIP#): quality recovers to INT4-comparable
- Block low-rank at 2× (r_b/B = 0.5): near-lossless. At 4× (r_b/B = 0.125): ~1 PPL increase

**Note**: No Qwen3-family-specific published results; all figures extrapolated from LLaMA/OPT/Qwen1.5 families.

---

<!-- CITATION MANIFEST -->
[1]: GPTQ (Frantar et al., ICLR 2023) — arXiv:2210.17323: Block-column second-order post-training INT4 quantization for OPT-175B/BLOOM-176B; 3.25× A100 speedup
[2]: AWQ (Lin et al., MLSys 2024 Best Paper) — arXiv:2306.00978: Activation-aware per-channel INT4 with 1% salient weight protection; 3× speedup on desktop GPUs
[3]: SmoothQuant (Xiao et al., ICML 2023) — arXiv:2211.10438: Per-channel scale migration enabling W8A8; 1.56× speedup, 2× memory reduction
[4]: OmniQuant (Shao et al., ICLR 2024 Spotlight) — arXiv:2308.13137: LWC + LET per-transformer-block calibrated quantization; near-lossless W4A16/W4A8
[5]: ATOM (Zhao et al., MLSys 2024) — arXiv:2310.19102: W4A8KV4 fused kernel; 7.7× serving throughput (batch>1), not batch=1 TPOT
[6]: ASVD (Yuan et al., 2023) — arXiv:2312.05821: Activation-aware SVD with per-layer rank selection; 10–30% compression without PPL drop
[7]: SVD-LLM (Wang et al., ICLR 2025) — arXiv:2403.07378: Truncation-aware whitening SVD; outperforms ASVD at high compression ratios across 7 LLM families
[8]: CALDERA (Saha, Sagan, Srivastava, Goldsmith, Pilanci, NeurIPS 2024) — arXiv:2405.18886: W≈Q+LR post-training compression at per-weight-matrix level; best at <2.5 bits/param (Stanford/Princeton)
[9]: LC-SVD (ICLR 2025) — openreview.net/forum?id=m2nupeHqV7: Transformer-block-granularity joint SVD with automatic rank search
[10]: Monarch (Dao et al., ICML 2022) — arXiv:2204.00595: Block-diagonal butterfly structured matrices; 2× training speedup on GPT-2
[11]: BLAST (Lee et al., NeurIPS 2024) — arXiv:2410.21262: Per-block adaptive structured matrices; 2× compression on Llama-7B with lowest degradation among structured matrices
[12]: BlockDense / Building on Efficient Foundations (Wei, Moalla, Pascanu, Gulcehre, NeurIPS 2024) — arXiv:2406.16450: Structured FFN layers; 32% FFN params with ~1.0 PPL increase, 2.6× FFN speed-up
[13]: MARLIN (Frantar et al., NeurIPS 2024) — arXiv:2408.11743: INT4 GEMV kernel with lop3 unpacking; ~2.8× TPOT on A6000, integrated into vLLM
[14]: QServe (Lin et al., 2024) — arXiv:2405.04532: W4A8KV4 system; 2.4–3.5× over TRT-LLM for Qwen1.5-72B; 20–90% dequant overhead eliminated
[15]: LoftQ (Li et al., ICLR 2024) — arXiv:2310.08659: LoRA-aware Q+LR alternating initialization; outperforms QLoRA at 2-bit
[16]: AdaLoRA (Zhang et al., ICLR 2023) — arXiv:2303.10512: SVD-parameterized LoRA with importance-based rank pruning; HuggingFace PEFT integrated
[17]: UniQL (Chiang et al., 2025) — arXiv:2512.03383: Unified quantization + LR for adaptive edge LLMs; 4×–5.7× memory reduction, 2.7×–3.4× throughput; supports Transformers/SSMs/hybrid
[18]: QuIP# (Tseng et al., 2024) — arXiv:2402.04396: Hadamard incoherence + E8 lattice codebook; near-lossless 2-bit compression of LLaMA models
[19]: SqueezeLLM (Kim et al., ICML 2024) — arXiv:2306.07629: Sparse outlier + INT4 non-outlier hybrid; direct precedent for 5.7+5.8 combination
[20]: SparseGPT (Frantar & Alistarh, ICLR 2023) — arXiv:2301.00774: Second-order one-shot pruning; 50–60% sparsity; joint prune+quantize with GPTQ gives ~8× compression
[21]: SLiM (Mozaffari, Yazdanbakhsh, Dehnavi, ICML 2025) — arXiv:2410.09615: One-shot unified Q + 2:4 sparsity + LR correction; 4.3× RTX3060 / 3.8× A100 layer-wise speedup; directly addresses Tier 3 combined Q+LR direction
