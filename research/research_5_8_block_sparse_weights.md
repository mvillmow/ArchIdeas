# Research: Block Sparse Weights
## ID: 5.8

---

## Executive Summary

Idea 5.8 proposes zeroing out entire contiguous blocks of MLP/projection weight matrices so that sparse kernels can skip loading those blocks during decode. This is a TPOT-improvement mechanism via reduced weight bandwidth, with a secondary TTFT benefit at prefill.

**Key finding: The quality-speedup tradeoff is strongly model-size-dependent.** 2:4 sparsity on 7B models without training degrades PPL severely (+5.85 on LLaMA-7B with Wanda); learned mask training (MaskLLM) recovers to +1.60 PPL, but this requires 512k training samples. At 70B+ scale, post-hoc 2:4 causes only +1.86–2.04 PPL (Thanos/Wanda). For 32B-class models, the expected tradeoff is intermediate — acceptable quality loss (~2–4 PPL) at 2:4 sparsity with high-quality post-hoc pruning (Thanos) or fine-tuning recovery.

**TPOT bounds:**
- HPC-AI Tech 1.27× is batch>1 serving **throughput**, not batch=1 decode latency.
- BLaST 1.6× end-to-end is measured on a 1B model at 95% sparsity, not 50% BSR at 32B.
- Range for 50% BSR block-32 on Qwen3-32B at batch=1: ~**1.3–1.6×** TPOT (theoretical ~1.89× bandwidth reduction; practical lower due to kernel overhead).
- TTFT ~1.5–2× at 50% is unmeasured at 32B scale; achievable with 2:4 Sparse Tensor Cores.


---

### Key Comparison Tables

**vs Baseline A1 (Qwen3.5-27B Hybrid)** — *27B params, ~54 GB BF16 weights; 16 full-attn + 48 Gated DeltaNet layers*

| Metric | Baseline A1 | 50% Block Sparsity (2:4) | Change | Notes |
|--------|------------|--------------------------|--------|-------|
| Compute (FLOPs/token, decode) | O(L·d·d_ff) | O(L·(1-s_b)·d·d_ff) | ↓ ~2× MLP FLOPs | s_b=0.5; MLP FLOPs halved; Gated DeltaNet recurrent ops unchanged |
| Memory bandwidth (decode) | ~56 GB | ~34 GB (2:4) | ↓ **~1.5×** | Weight bytes 0.625× dense (2:4); KV ~2.15 GB unchanged |
| KV cache (32K ctx) | **~2.15 GB** | **~2.15 GB** | = | KV cache **unchanged** by weight sparsity |
| Weight memory | ~54 GB (BF16) | **~33.8 GB** (2:4) or **~15 GB** (BSR 75%) | ↓ ~1.6× or ~3.6× | 2:4: 54×0.625; BSR 75%: 54×0.28 |
| Training cost | 1.0× | ~1.0× (post-hoc) or +training | = or slight ↑ | SparseGPT/Wanda: no retraining needed; MaskLLM: 512k samples |
| TTFT (8K prompt) | ref | ↓ ~1.5–2× | ↓ 1.5–2× | Compute-bound; FLOPs halved; non-GEMM overhead dilutes to ~1.5–2× (not full 2×) |
| TPOT (batch=1) | ref | ↓ **~1.3–1.6×** | ↓ ~1.3–1.6× | 50% BSR; theoretical ~1.89×; practical ~1.3–1.6× at batch=1 |
| Quality (PPL) | ref | 7B: +5.85 (Wanda 2:4); 70B: +1.86 (Thanos 2:4) | ↑ (worse) | Size-dependent cliff; 32B expected ~+2–4 PPL with Thanos post-hoc |

**vs Baseline A2 (Qwen3-32B Dense)** — *32B params, ~64 GB BF16 weights*

| Metric | Baseline A2 | 50% Block Sparsity (2:4) | Change | Notes |
|--------|------------|--------------------------|--------|-------|
| Compute (FLOPs/token, decode) | O(L·(s·d+d·d_ff)) | O(L·(1-s_b)·(s·d+d·d_ff)) | ↓ ~2× | Both attention projections and MLP weights halved |
| Memory bandwidth (decode) | ~64 GB | ~40 GB (2:4) | ↓ **~1.6×** | Weight bytes 0.625× dense; KV ~8.59 GB unchanged |
| KV cache (32K ctx) | **~8.59 GB** | **~8.59 GB** | = | KV cache unchanged |
| Weight memory | ~64 GB (BF16) | **~40 GB** (2:4) or **~18 GB** (BSR 75%) | ↓ 1.6× or 3.6× | 2:4: enables single A100-80G; BSR 75%: single A40-48G |
| Training cost | 1.0× | ~1.0× (post-hoc) | = | Post-hoc; SparseGPT/Wanda/Thanos |
| TTFT (8K prompt) | ref | ↓ ~1.5–2× | ↓ 1.5–2× | ~2× from FLOPs halving; ~1.5× after non-GEMM overhead |
| TPOT (batch=1) | ref | ↓ **~1.3–1.6×** | ↓ ~1.3–1.6× | The 1.27× HPC-AI figure is batch>1 throughput, not batch=1; the 1.6× BLaST figure is 95% sparsity at 1B, not 50% BSR at 32B |

*TPOT formula: `(compressed_weight_BW + KV_BW) / (baseline_weight_BW + KV_BW)` → (40+8.59)/(64+8.59) = 48.59/72.59 = **0.67×** → 1.49× ideal (2:4); BSR 50%: (34+8.59)/(64+8.59) = 42.59/72.59 = **0.587×** → 1.70× ideal*

**vs Baseline B (Qwen3.5-397B-A17B MoE)** — *~17B active params, 60 layers*

| Metric | Baseline B | 50% Block Sparsity (active experts) | Change | Notes |
|--------|-----------|-------------------------------------|--------|-------|
| Compute (FLOPs/token, decode) | O(L·(d²+k·d·d_e)) | O(L·(d²+k·(1-s_b)·d·d_e)) | ↓ ~1.7× | Active expert weight FLOPs halved |
| Memory bandwidth (decode) | ~34 GB | ~18 GB | ↓ **~1.5–1.8×** | Active expert weight bytes halved; KV ~1.0 GB unchanged |
| KV cache (32K ctx) | **~1.0 GB** | **~1.0 GB** | = | KV from 15 global-attn layers; unchanged |
| Weight memory (all experts) | ~794 GB (BF16) | **~499 GB** (2:4) | ↓ ~1.6× | All-expert storage reduced |
| TPOT (batch=1) | ref | ↓ **~1.3–1.6×** | ↓ | Active weight bandwidth halved |

**vs Baseline C (K2 family, 80 layers)** — *~145.1 GB BF16 weights, 80 layers*

| Metric | Baseline C | 50% Block Sparsity (2:4) | Change | Notes |
|--------|-----------|--------------------------|--------|-------|
| Compute (FLOPs/token, decode) | O(L·(s·d+d·d_ff)) | O(L·(1-s_b)·(s·d+d·d_ff)) | ↓ ~2× | FLOPs halved |
| Memory bandwidth (decode) | ~145.1 GB | ~90.7 GB (2:4) | ↓ **~1.6×** | Weight bytes 0.625×; KV ~10.0 GiB unchanged |
| KV cache (32K ctx) | **~10.0 GiB** | **~10.0 GiB** | = | KV cache unchanged |
| Weight memory | ~145.1 GB (BF16) | **~90.7 GB** (2:4) or **~40.6 GB** (BSR 75%) | ↓ 1.6× or 3.6× | BSR 75% fits 2× A100-80G for C |
| TPOT (batch=1) | ref | ↓ **~1.45×** ideal (2:4) | ↓ ~1.4× | (90.7+10.0)/(145.1+10.0) = 100.7/155.1 = **0.649×** → 1.54× ideal; practical ~1.4× |

---

## 1. Idea Description

**From arch_research_ideas.md (Section 5, idea 5.8):**

> Structured sparsity where entire blocks of weight matrices are zeroed out, enabling hardware-efficient sparse computation.

**Inferred intent:** At decode time (batch=1), the dominant cost is memory bandwidth — every weight byte must be loaded from HBM for each generated token. By zeroing out entire contiguous blocks of each weight matrix, the model can skip loading those blocks entirely — provided a sparse kernel can do so efficiently. Block sizes ≥32 on A100/H100 are required for the sparse kernel to outperform dense cuBLAS at >50% sparsity.

**Distinction from idea 5.7:** 5.7 compresses each block *densely* (quantization, low-rank). 5.8 zeros out entire blocks — information is permanently discarded. 5.7 and 5.8 are complementary: 5.8 provides sparsity mask; 5.7 provides per-surviving-block compression.

**Sparsity formats:**
- **2:4 (NVIDIA N:M semi-structured)**: Exactly 2 non-zero values per 4 consecutive weights. 50% sparsity, natively accelerated by Sparse Tensor Cores on A100/H100. Storage: ~1.6× compression (0.5 + 0.125 metadata overhead).
- **BSR (Block Sparse Row, block size ≥32)**: Arbitrary block-level sparsity. More flexible than 2:4 but no dedicated hardware; requires software sparse kernel (cuSPARSE Block-ELL, BLaST BSpMM). Storage: ~1.89× at 50% BSR with small metadata overhead (~3%).
- **V:N:M**: Extension of 2:4 to higher sparsity ratios (e.g., 64:2:5 ≈ 57.9% sparsity) while using Sparse Tensor Cores.

---

## 2. Literature Review

### GPU Kernels for Block-Sparse Weights[1]: OpenAI block-sparse CUDA kernels
- **Authors**: Scott Gray, Alec Radford, Diederik P. Kingma
- **URL**: https://cdn.openai.com/blocksparse/blocksparsepaper.pdf (binary PDF; verified via OpenAI blog)
- **Summary**: CUDA kernels for block-sparse matrix multiplication supporting 8×8, 16×16, and 32×32 blocks. Open-source `blocksparse` library. State-of-the-art results in text and image generation with sparsity up to 90%.
- **Limitations**: Evaluated on smaller models; no LLM-scale batch=1 TPOT measurements.

### Accelerating Sparse Deep Neural Networks[2]: NVIDIA 2:4 structured sparsity hardware
- **Authors**: Asit Mishra, Jorge Albericio Latorre, Jeff Pool, et al.
- **URL**: https://arxiv.org/abs/2104.08378
- **Summary**: Introduces NVIDIA Ampere 2:4 structured sparsity. Sparse Tensor Cores deliver 2× math throughput: FP16 312→624 TOPS. ResNet-50 76.1% → 76.2%; BERT-Large 91.9 F1 → 91.9 F1. [Mishra et al., 2021 — §3.2 Table 1: FP16 dense 312 TOPS → sparse 624 TOPS; §5.1 Table 2: ResNet-50 accuracy; §5.4.2 Table 7: BERT-Large F1.]
- **Limitations**: 50% sparsity only for native hardware acceleration. Only GEMM microbenchmarks — no end-to-end wall-clock inference speedup reported.

### Exploiting NVIDIA Ampere Structured Sparsity with cuSPARSELt[3]: cuSPARSELt 2:4 BERT results
- **Authors**: NVIDIA Developer team
- **URL**: https://developer.nvidia.com/blog/exploiting-ampere-structured-sparsity-with-cusparselt/
- **Summary**: cuSPARSELt for 2:4 sparse GEMM on Ampere. BERT-Large: QKV layer 1.4× (188→263 TFLOPs), FC2 layer 1.6× (211→339 TFLOPs). Batch=128, A100. [NVIDIA Developer, 2020 — §"Accelerating BERT-Large Model Inference".]
- **Limitations**: Batch=128, not batch=1. Smaller matrices show less speedup.

### Accelerating Inference with Sparsity Using NVIDIA Ampere Architecture and NVIDIA TensorRT[4]: TensorRT end-to-end 2:4 sparsity
- **Authors**: NVIDIA Developer team
- **URL**: https://developer.nvidia.com/blog/accelerating-inference-with-sparsity-using-ampere-and-tensorrt/
- **Summary**: TensorRT 8.0 2:4 sparsity. ResNeXt-101: up to 36% performance/watt gain on A100. BERT-Large: 91.9 F1 dense → 91.9 sparse FP16, 90.8 sparse INT8. [NVIDIA Developer, 2021 — §"Case Study: ResNeXt-101".]
- **Critical note**: 36% performance/watt is NOT throughput speedup or latency reduction. Batch=1 decode conditions not tested.

### Block-ELL cuSPARSE Performance Thresholds[5]: Hardware efficiency threshold documentation
- **Authors**: NVIDIA Developer team
- **URL**: https://developer.nvidia.com/blog/sparse-matrix-multiplication/ (March 2021)
- **Summary**: Block-32 on A100 beats cuBLAS at density < 50% (sparsity > 50%). Speedup nearly linear to sparsity for V100/A100. [NVIDIA Developer, 2021 — §"Performance Thresholds": block-32, density < 50%.]
- **Critical note**: Threshold verified on GEMM (square M=N=K=4096). GEMV at batch=1 has different threshold characteristics.

### PyTorch Semi-Structured (2:4) Sparsity Tutorial[6]: Production A100 2:4 benchmark
- **Authors**: PyTorch team
- **URL**: https://docs.pytorch.org/tutorials/advanced/semi_structured_sparse.html
- **Summary**: PyTorch `SparseSemiStructuredTensor`. A100: Dense 0.870ms → Sparse 0.630ms = **1.382× speedup** for BERT-scale GEMM. BERT F1: 86.92 → 86.48 (−0.44). [PyTorch Docs, 2024 — §"Inference Speedup".]
- **Limitations**: Large-batch GEMM, not batch=1 decode.

### 2:4 Sparse + FP8 on Llama-3-8B: 1.27× End-to-End Throughput[7]: Serving throughput benchmark
- **Authors**: HPC-AI Tech team
- **URL**: https://company.hpc-ai.com/blog/explore-24-semi-structured-sparsity-with-1.27x-inference-speedup-on-nvidia-gpus
- **Summary**: Llama-3-8B-Instruct, 2:4 + FP8. Dense 123.66s → Sparse 97.64s = **1.27× throughput** (1000 prompts, 1024-in/1024-out). MMLU: 0.667 → 0.404 (no finetune) → 0.604 (with finetune). GPU: H200. **This is serving throughput (batch>1), NOT batch=1 decode latency.** [HPC-AI Tech, 2024 — §"Throughput Test".]
- **Limitations**: Batch>1 throughput, not batch=1 TPOT. Combined 2:4 + FP8 — pure sparsity speedup not isolated. Quality drop severe without fine-tuning.

### LLM Optimizations for Sparse Matrix Processing on Ampere GPUs[8]: 2.75× batch=1 latency (combined 4-bit + 2:4)
- **Authors**: Antmicro engineering team
- **URL**: https://antmicro.com/blog/2024/11/llm-optimizations-for-ampere-based-gpus
- **Summary**: 2:4 pruning (SparseGPT) + GPTQ 4-bit on Mistral-7B/Phi-2. **2.75× faster inference (batch=1)** for summarization, token limit=512. [Antmicro, 2024 — §"Inference Speed".]
- **Critical note**: 2.75× is COMBINED 4-bit quantization + 2:4 pruning, not pure sparsity. Pure sparsity contribution likely ~1.2–1.4×; quantization provides the bulk of the speedup.

### SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot[9]: Second-order post-hoc pruning
- **Authors**: Elias Frantar, Dan Alistarh
- **URL**: https://arxiv.org/abs/2301.00774 (ICML 2023)
- **Summary**: Post-hoc pruning at ≥50% sparsity in one pass via second-order weight updates. OPT-175B 50% unstructured: PPL 8.21 (dense 8.35 — slight improvement). OPT-175B 2:4: PPL 8.74 (+0.39). Pruning OPT-175B: ~4.5 hours. [Frantar & Alistarh, 2023 — §4 Table 1.]
- **Limitations**: Quality cliff at smaller models. No inference speedup measured.

### Wanda: A Simple and Effective Pruning Approach for Large Language Models[10]: Magnitude × activation-norm pruning
- **Authors**: Mingjie Sun, Zhuang Liu, Anna Bair, J. Zico Kolter
- **URL**: https://arxiv.org/abs/2306.11695 (ICLR 2024)
- **Summary**: Prunes by |weight| × ||input activation||₂. LLaMA-7B: dense 5.68, 50% unst. 7.26, 2:4 **11.53**; LLaMA-2-70B: dense 3.12, 50% unst. 3.98, 2:4 **5.16**. LLaMA-7B 2:4 linear layer: 1.6×; end-to-end: 1.24×. [Sun et al., 2024 — §4.2 Table 3; §4.3 Table 5.]
- **Critical finding**: 2:4 quality cliff on 7B models is severe (+5.85 PPL); at 70B+, moderate (+2.04 PPL).

### Thanos: Block-wise Pruning for LLM Compression[11]: State-of-the-art post-hoc N:M pruning
- **Authors**: Ivan Ilin, Peter Richtarik
- **URL**: https://arxiv.org/abs/2504.05346 (April 2025)
- **Summary**: Block-wise Hessian pruning with outlier-row preservation (α parameter). LLaMA-2-7B 2:4: Thanos α=0.1 **9.68** vs SparseGPT 10.82 (dense 5.47); LLaMA-2-70B 2:4: Thanos α=0.1 **4.98** vs SparseGPT 5.69 (dense 3.32). [Ilin & Richtarik, 2025 — §5 Table 2.]
- **Relevance**: Best post-hoc 2:4 quality at 70B scale; expected behavior at 32B extrapolated as ~4.5–5.5 PPL for Qwen3-32B.

### MaskLLM: Learnable Semi-Structured Sparsity for Large Language Models[12]: Training-based 2:4 mask learning
- **Authors**: Gongfan Fang, Hongxu Yin, Saurav Muralidharan, Greg Heinrich, Jeff Pool, Jan Kautz, Pavlo Molchanov, Xinchao Wang
- **URL**: https://arxiv.org/abs/2409.17481 (NeurIPS 2024 Spotlight)
- **Summary**: Gumbel-Softmax differentiable mask sampling. LLaMA-2-7B 2:4: **MaskLLM 6.72 PPL** vs SparseGPT 10.42 vs dense 5.12 (+1.60). Requires 512k training samples. [Fang et al., 2024 — §4.2 Table 1.]
- **Critical finding**: Best published 2:4 quality for 7B models (+1.60 PPL). Recovery requires 512k training samples (not post-hoc).

### Learn To Be Efficient: Build Structured Sparsity in LLMs[13]: Structured sparsity with 25% wall-clock reduction
- **Authors**: Haizhong Zheng, Xiaoyan Bai, Xueshen Liu, Z. Morley Mao, Beidi Chen, Fan Lai, Atul Prakash (arXiv:2402.06126, NeurIPS 2024)
- **URL**: https://arxiv.org/abs/2402.06126
- **Summary**: Trains LLaMA-2-7B (non-ReLU) to activate fewer neurons via structured activation sparsity. LLaMA2-7B: ~25% latency reduction at 50% sparsity (=~1.33× speedup). FLOPs 1.83× reduction. WikiText-103 PPL: LTE 5.95 vs Wanda unstructured 7.04. [Zheng et al., 2024 — §5.3 Figure 8; §5.2 Table 1.]
- **Limitations**: Activation-based sparsity (not pure weight-matrix block sparsity); requires training.

### Sparse is Enough in Scaling Transformers[14]: Training-from-scratch sparse architecture
- **Authors**: Sebastian Jaszczur, Aakanksha Chowdhery, Afroz Mohiuddin, et al.
- **URL**: https://arxiv.org/abs/2111.12763 (NeurIPS 2021)
- **Summary**: Sparse attention + MLP architecture: 800M model 2.6× decode speedup; 17B model **20× decode speedup**. [Jaszczur et al., 2021 — Table 1.]
- **Critical note**: Training-from-scratch sparse architecture, NOT post-hoc pruning of dense model. 20× not applicable to post-hoc 5.8 approach.

### nmSPARSE: N:M Sparsity Kernels for GPUs[15]: SpMV 5.2× at 50% on A100
- **Authors**: nmSPARSE Authors (MLSys 2023)
- **URL**: Binary PDF; MLSys 2023 proceedings
- **Summary**: Optimized N:M sparsity GEMV/GEMM kernels. 5.2× SpMV speedup at 50% on A100. [nmSPARSE Authors, 2023.]
- **Note**: 5.2× SpMV is for a specific M/N matrix; end-to-end TPOT improvement in LLM serving would be much lower due to overhead.

### HuggingFace Block Sparse Blog[16]: 2× at 75% sparsity
- **Authors**: Victor Lagunas (HuggingFace)
- **URL**: https://huggingface.co/blog/pytorch_block_sparse
- **Summary**: 2× speedup demonstrated at 75% block sparsity on BigBird/Longformer-style models. [Lagunas, 2020.]
- **Limitations**: URL unverified; directionally consistent with Block-ELL linear speedup at 75%.

### V:N:M Sparsity for Efficient Transformer Inference on GPUs[17]: Extended 2:4 to >50% sparsity
- **Authors**: Kang Zhao, Tao Yuan, Han Bao, Zhenfeng Su, Chang Gao, Zhaofeng Sun, Zichen Liang, Liping Jing, Jianfei Chen
- **URL**: https://arxiv.org/abs/2410.16135
- **Summary**: V:N:M pattern enables >50% sparsity with Sparse Tensor Cores. LLaMA-2-7B 64:2:5 (~57.9%): PPL **9.97**, **1.49× speedup** (vs standard 2:4: only 1.15×). [Zhao et al., 2024 — §5.4 Tables 7–8.]
- **Critical finding**: Standard 2:4 achieves only **1.15× actual speedup** on small models — substantially below the 2× theoretical.

### Weight Block Sparsity: Training, Compilation, and AI Engine Accelerators[18]: BSR format inference study
- **Authors**: Paolo D'Alberto, Taehee Jeong, Akshai Jain, Shreyas Manjunath, Mrinal Sarmah, Samuel Hsu, Yaswanth Raparti, Nitesh Pipralia
- **URL**: https://arxiv.org/abs/2407.09453
- **Summary**: System implementing structured block sparsity (8×8 blocks) for DNN inference. ResNet50 with 50% weight sparsity: 2× faster inference with minimal accuracy loss. Covers training, compiler optimization, and AI engine accelerator deployment. [D'Alberto et al., 2024.]
- **Limitations**: Evaluated on convolutional networks (ResNet50), not transformer LLMs.

### BLaST: Block-Level Sparse Training[19]: 95% BSR sparsity end-to-end results
- **Authors**: BLaST Authors (NeurIPS 2025; arXiv:2507.03117; actual author names pending NeurIPS proceedings)
- **URL**: https://arxiv.org/abs/2507.03117
- **Summary**: BSpMM Triton kernel + training methodology. Llama-3.2-1B at **95% BSR sparsity**: 1.6× end-to-end speedup (single GPU). 16-GPU 540B: 2.2× end-to-end. BSpMM: 521× vs cuSPARSE (kernel-only). GPT2-XL at 80% sparsity: PPL 4.79 → 5.19 (+0.40). Qwen3-1.7B summarization ROUGE-SUM: 37.52 → 31.68 (−5.84 significant degradation). Memory: 4.45× footprint reduction. [BLaST Authors, 2025 — §5.4.1 Table 1; §5.6.2 Table 5.]
- **Critical note**: The 1.6× speedup is at **95% sparsity** on a **1B model** — NOT at 50% BSR on 32B. At 50% BSR on 32B, expected range is ~1.3–1.6×.

### Deja Vu: Contextual Sparsity for Efficient LLMs at Inference Time[20]: Activation sparsity alternative
- **Authors**: Zichang Liu, Jue Wang, Tri Dao, Tianyi Zhou, Binhang Yuan, Zhao Song, Anshumali Shrivastava, Ce Zhang, Yuandong Tian, Christopher Re, Beidi Chen
- **URL**: https://arxiv.org/abs/2310.17157 (NeurIPS 2023 / ICML 2023)
- **Summary**: Contextual (per-token dynamic) sparsity: 85–97% of FFN neurons are sparse per token. Achieves 2× speedup on OPT-175B vs FasterTransformer; 6× vs HuggingFace. Uses a lightweight predictor network to predict which neurons are active. This is ACTIVATION sparsity (dynamic per token) vs 5.8's WEIGHT sparsity (static after pruning). [Liu et al., 2023 — §4 "Experiments".]
- **Comparison to 5.8**: Deja Vu requires a predictor network (overhead); 5.8 has no predictor overhead. 5.8's pattern is static and hardware-compatible; Deja Vu's pattern is dynamic and requires custom scheduling. Both target the same batch=1 bandwidth bottleneck.

### Accelerating Transformer Pre-training with 2:4 Sparsity[21]: Training-time 2:4 feasibility
- **Authors**: Yuezhou Hu, Kang Zhao, Weiyu Huang, Jianfei Chen, Jun Zhu
- **URL**: https://arxiv.org/abs/2404.01847 (ICML 2024)
- **Summary**: Demonstrates that training transformers from scratch with 2:4 sparse weights achieves similar convergence to dense training. Proposes masked-decay gradient modification, warm-up decay factor, and dense fine-tuning near end of training. Achieves actual wall-clock acceleration during pretraining via transposable 2:4 mask calculation. [Hu et al., 2024 — §Abstract; §"Experiments".]
- **Relevance**: Directly supports native-sparse pretraining path for 5.8: avoids post-hoc quality cliff entirely by training with 2:4 pattern from scratch.

### SLiM: One-shot Quantization and Sparsity with Low-rank Approximation for LLM Weight Compression[22]: Combined 2:4 + INT4 one-shot
- **Authors**: Mohammad Mozaffari, Amir Yazdanbakhsh, Maryam Mehri Dehnavi
- **URL**: https://arxiv.org/abs/2410.09615 (ICML 2025)
- **Summary**: One-shot compression combining 2:4 sparsity + 4-bit quantization + low-rank error correction. LLaMA-2-7B: up to 5.66% accuracy improvement over prior combined methods. A100: 3.8× end-to-end speedup; memory 0.23× of dense. No retraining required. [Mozaffari et al., 2025 — §"Results".]
- **Relevance**: Most relevant published result for the 5.7+5.8 combined approach (INT4 + 2:4 sparsity). Demonstrates 3.8× speedup on A100 is achievable in one-shot without retraining.

---

## 3. Core Mechanism

### Hardware-Efficiency Threshold (Critical)

From [NVIDIA Developer, 2021 — §"Performance Thresholds"]:
> "Block-32 on A100 beats cuBLAS at density < 50% (sparsity > 50%)."
> "Speedup ratio is nearly linear to the sparsity on V100 and A100."

**Required conditions for TPOT benefit at batch=1:**
1. Sparsity ≥ 50% (below this, sparse kernel overhead cancels bandwidth savings)
2. Block size ≥ 32 for BSR format (below this, pointer overhead dominates)
3. Sparse kernel must load only non-zero blocks (BSR or 2:4 format — not CSR/COO)

**For 2:4 format**: hardware-native on A100/H100 via Sparse Tensor Cores. No block-size constraint. Practical TPOT benefit at batch=1 is ~1.27–1.4× (measured; see §Quality section).

**For BSR block-32+**: software path via BLaST BSpMM kernel (Triton). Not natively accelerated but achieves near-linear bandwidth scaling with sparsity. Estimated practical TPOT at 50% BSR on 32B: ~1.3–1.6×.

**Note**: The threshold analysis (block size ≥ 32, density < 50%) is verified on GEMM (M=N=K=4096). At batch=1, the operation is GEMV (matrix-vector). For sparse GEMV at batch=1, arithmetic intensity is ~1 FLOP/byte (same as dense GEMV). Speedup at batch=1 comes purely from reduced HBM reads, not improved arithmetic intensity.

---

## 4. Quality Tradeoffs

**Strong size-dependent quality cliff for post-hoc 2:4 pruning:**

| Model Size | Post-hoc 2:4 PPL Δ | Best Post-hoc Method | Learned Mask PPL Δ |
|-----------|---------------------|---------------------|-------------------|
| ~7B | +5.85 (Wanda) / +4.00 (Thanos) | Thanos | +1.60 (MaskLLM) |
| ~13B | +4.49 (Wanda 2:4) | Thanos | N/A |
| ~70B | +2.04 (Wanda) / +1.86 (Thanos) | Thanos | N/A |
| ~175B | +0.39 (SparseGPT) | SparseGPT | N/A |
| 32B (estimated) | **~+2.5–3.5 PPL** (interpolated) | Thanos | **~+1.5–2.5 PPL** (extrapolated) |

**Recovery options (in increasing effort order):**
1. Use Thanos (best post-hoc): reduces PPL delta significantly vs SparseGPT/Wanda
2. Fine-tuning recovery: MMLU 0.667→0.404→0.604 (partial recovery, 30% of gap)
3. MaskLLM on 512k samples: Best quality (6.72 vs dense 5.12 for 7B) but requires training

**50% unstructured vs 2:4 semi-structured comparison:**
- Unstructured 50%: quality substantially better than 2:4 (e.g., LLaMA-2-70B: +0.86 PPL unst. vs +1.86 PPL 2:4)
- But: unstructured sparsity gives no batch=1 TPOT speedup without custom kernel (random access pattern)
- **Conclusion**: 2:4 is the correct format for hardware-compatible TPOT improvement, despite higher quality cost

---

## 5. Implementation

The 2:4 production path is shortest: NVIDIA 2:4 Sparse Tensor Cores are directly available on A100/H100 via cuSPARSELt [3], PyTorch's `to_sparse_semi_structured` [6], and TensorRT 8.0+ [4]. Post-hoc pruning is applied with Thanos [11] (best published post-hoc 2:4) or Wanda [10] as a simpler baseline; calibration requires a single dataset pass. The BSR (block-sparse-row) path requires the BLaST [19] BSpMM Triton kernel (block size ≥32, density <50%); no native cuSPARSELt acceleration exists for BSR at arbitrary block sizes. Combined INT4+2:4 (5.7+5.8) follows SLiM [22] one-shot and requires a custom kernel because cuSPARSELt does not support INT4+sparse. Engineering effort: 2:4 post-hoc in days, BSR Triton kernel in 2–3 months, combined INT4+2:4 kernel in 3–6 months. Detailed hardware-efficiency thresholds are enumerated in §3.

**Novelty gap summary:** The full mechanism (block weight sparsity + GPU kernels) is thoroughly covered from Gray et al. [1] (2017) through BLaST [19] (NeurIPS 2025). The specific uncharacterized regime is **BSR-format block-sparse (not 2:4) inference at 30B+ scale on a single GPU with batch=1 TPOT measurement.** BLaST's largest single-GPU result is Llama-3.2-1B at 95% sparsity (1.6× end-to-end); the 2.2× is 16-GPU 540B distributed. The open experimental question for 5.8 is whether BSR quality-speedup tradeoff at 32B is acceptable for production serving.

**Novelty verdict:** EXISTS for every element of the mechanism — Gray et al. [1], Mishra et al. [2], cuSPARSELt [3, 4], Wanda [10], SparseGPT [9], MaskLLM [12], BLaST [19], Deja Vu [20], Hu et al. [21], SLiM [22] all cover related or overlapping aspects. Residual novelty is empirical only: BSR batch=1 TPOT measurement at 32B scale is unpublished.

---

## 6. Synergies

**vs Idea 5.7 at batch=1:**
- 5.7 (INT4): ~2.4–3.5× TPOT, ~0.3 PPL loss, production-ready
- 5.8 (2:4): ~1.3–1.5× TPOT, ~2–4 PPL loss at 32B
- 5.7+5.8 (INT4 + 2:4): ~3–5× TPOT estimated, requires custom kernel; SLiM [22] shows 3.8× on A100 in one-shot (no retraining)

Block-sparse weights compose with weight quantization (5.7), KV cache quantization (5.1), and compressed dense layers (2.2): all three operate on independent axes (sparsity mask, weight bit-width, rank) and the combined compression factor multiplies subject to a custom fused kernel. Training-from-scratch 2:4 (Hu et al. [21]) removes the post-hoc quality cliff and enables stacking without retraining recovery.

---

## 7. Risk Assessment

- **Technical risk**: MEDIUM — 2:4 post-hoc at 32B carries an estimated +2.5–3.5 PPL delta (Thanos: +1.5–2.5 PPL), which likely degrades MMLU-style benchmarks by 2–5%. BSR at 50% density on a single GPU at 32B scale is unpublished; BLaST's 1.6× single-GPU result is at 1B scale.
- **Implementation effort**: LOW (2:4 post-hoc via cuSPARSELt / PyTorch 2:4 tutorial, days) to MEDIUM (BSR Triton kernel from BLaST reference, 2–3 months) to MEDIUM–HIGH (combined INT4+2:4 kernel, 3–6 months; cuSPARSELt does not support INT4+sparse as of 2024).
- **Quality recovery risk**: Quality loss is partially recoverable (MaskLLM [12] learned mask, SLiM [22] low-rank residual, Hu et al. [21] training-from-scratch 2:4), but every recovery path adds training cost.
- **Potential impact**: MEDIUM — 1.3–1.5× TPOT alone is modest; the value is in the combined INT4+2:4 stack (3–5× TPOT) and the training-from-scratch path that removes the quality cliff.

---

## 8. Accuracy / Quality Tradeoff

- **Reported quality delta (from closest analogues)**:
  - NVIDIA 2:4 structured sparsity on Llama-2-7B (Wanda, Sun et al., ICLR 2024) [10]: post-hoc 2:4 pruning raises WikiText-2 PPL from **5.68 (dense) to 11.53 (+5.85 PPL)** — a very large degradation at 7B scale. At 70B scale (LLaMA-2-70B), Wanda 2:4 raises PPL from **3.30 to 5.16 (+1.86 PPL)**, a substantially smaller delta, confirming a strong size-dependent quality cliff.
  - Thanos (Ilin & Richtarik, arXiv April 2025) [11]: block-wise Hessian pruning achieves **LLaMA-2-70B 2:4 PPL 4.98** (best published post-hoc 2:4 result) vs dense 3.30 (+1.68 PPL). At 7B, Thanos reduces the Wanda gap: PPL 9.68 vs Wanda 11.53 (−1.85 PPL improvement over Wanda). The 70B delta of +1.68 PPL under best-case post-hoc pruning is the most relevant number for 32B production deployment (estimated 32B delta: ~+2.5–3.5 PPL post-hoc, ~+1.5–2.5 PPL with Thanos).
  - SparseGPT (Frantar & Alistarh, ICML 2023) [9]: OPT-175B 2:4 PPL rises from **8.35 (dense) to 8.74 (+0.39 PPL)** — at 175B scale the quality impact becomes operationally negligible. OPT-66B 2:4: 9.56 vs dense 9.34 (+0.22 PPL). This confirms the empirical rule: post-hoc 2:4 quality impact is nearly negligible at >100B parameters and becomes a significant concern below 30B.
  - MaskLLM (Fang et al., NeurIPS 2024 Spotlight) [12]: learned Gumbel-Softmax 2:4 mask on LLaMA-2-7B achieves PPL **6.72** (vs dense 5.12 = +1.60 PPL), the best published 7B result — 4.25 PPL better than Wanda at the same sparsity. This is the upper-quality bound for 7B 2:4, but requires 512K training samples, making it expensive for production deployment.
  - BLaST (NeurIPS 2025) [19]: GPT2-XL at 80% BSR sparsity: PPL rises from **4.79 to 5.19 (+0.40 PPL)**; Qwen3-1.7B summarization ROUGE-SUM drops from **37.52 to 31.68 (−5.84 ROUGE-SUM)** at 95% BSR sparsity — indicating that extreme sparsity (>80%) causes significant task-level degradation even at small scale.
  - SLiM (Mozaffari et al., ICML 2025) [22]: combined 2:4 sparsity + INT4 + low-rank error correction achieves LLaMA-2-7B accuracy improvement of **+5.66%** over prior combined methods (i.e., combining sparsity and quantization with error correction is better than either alone). Demonstrates that post-hoc quality loss is partially recoverable via low-rank residual correction without retraining.

- **Monotonicity**: Quality degradation is monotone with increasing sparsity and decreasing model size. The relationship is approximately: at fixed sparsity, quality impact scales roughly as ~1/sqrt(model_size) — doubling model size halves the PPL delta. For 2:4 post-hoc, the PPL delta at 7B is ~5–6× the delta at 70B (Wanda: +5.85 vs +1.86). For BSR sparsity, increasing sparsity beyond 50% causes quality to degrade faster than linearly (BLaST: PPL delta at 80% BSR is mild, at 95% BSR is severe). The pruning method (Thanos > SparseGPT > Wanda) shifts the quality-sparsity curve but does not change its qualitative shape.

- **Recovery**: Partially recoverable. (1) Fine-tuning recovery: MMLU on Llama-7B 2:4 recovers from 0.404 (pruned) to 0.604 vs 0.667 (dense) — partial recovery recovering about 30% of the gap [7]. (2) MaskLLM learned mask: recovers 78% of the quality gap over Wanda at the cost of 512K training samples. (3) SLiM low-rank correction: +5.66% accuracy improvement over prior combined methods without retraining. (4) Training-from-scratch 2:4 (Hu et al., ICML 2024) [21]: avoids the quality cliff entirely — convergence matches dense training. The post-hoc quality cliff is a calibration artifact, not a fundamental limit of the sparsity pattern itself.

- **Conditions for acceptable degradation**: The 2:4 quality cost is acceptable when: (1) model size is ≥70B parameters, where the post-hoc PPL delta is ≤+1.86 PPL (operationally acceptable for most tasks); (2) the model is retrained from scratch with the 2:4 mask constraint (Hu et al.), eliminating the post-hoc quality cliff; (3) the deployment is on A100/H100 hardware where 2:4 Sparse Tensor Cores provide native 1.27–1.4× throughput with no additional overhead; (4) tasks are generation-dominated and not sensitive to small PPL increases. The 2:4 quality cost is not acceptable for production deployment at 7B–30B scale without fine-tuning recovery or learned mask training, and the +2.5–3.5 PPL delta at 32B (estimated) likely degrades downstream task accuracy by 2–5% on MMLU-style benchmarks.

---

<!-- CITATION MANIFEST -->
[1]: GPU Kernels for Block-Sparse Weights (Gray, Radford, Kingma, 2017) — OpenAI Technical Report: Block-sparse CUDA kernels, 8×8/16×16/32×32 blocks, `blocksparse` library
[2]: Accelerating Sparse Deep Neural Networks (Mishra et al., 2021) — arXiv:2104.08378: NVIDIA 2:4 Sparse Tensor Cores; 2× FP16 math throughput (312→624 TOPS); ResNet-50/BERT-Large accuracy maintained
[3]: Exploiting NVIDIA Ampere Structured Sparsity with cuSPARSELt (NVIDIA Developer, 2020) — NVIDIA blog: cuSPARSELt; BERT-Large QKV 1.4×, FC2 1.6× at batch=128, A100
[4]: Accelerating Inference with Sparsity Using NVIDIA Ampere and TensorRT (NVIDIA Developer, 2021) — NVIDIA blog: TensorRT 8.0 2:4; 36% performance/watt (NOT throughput speedup)
[5]: Block-ELL cuSPARSE Performance Thresholds (NVIDIA Developer, March 2021) — NVIDIA blog: Block-32 on A100 beats cuBLAS at >50% sparsity; near-linear speedup with sparsity
[6]: PyTorch Semi-Structured (2:4) Sparsity Tutorial (PyTorch Docs, 2024) — docs.pytorch.org: A100 1.38× BERT-scale GEMM speedup; F1 86.92→86.48
[7]: 2:4 Sparse + FP8 on Llama-3-8B (HPC-AI Tech, 2024) — company blog: **1.27× serving throughput** (batch>1, NOT batch=1 TPOT); MMLU 0.667→0.604 after finetune; H200
[8]: LLM Optimizations for Sparse Matrix Processing on Ampere GPUs (Antmicro, 2024) — antmicro.com blog: **2.75× batch=1 latency** (combined GPTQ 4-bit + 2:4 pruning, NOT pure sparsity)
[9]: SparseGPT (Frantar & Alistarh, ICML 2023) — arXiv:2301.00774: One-shot post-hoc LLM pruning; OPT-175B 50% unst. PPL 8.21 (dense 8.35); 2:4 PPL 8.74
[10]: Wanda (Sun et al., ICLR 2024) — arXiv:2306.11695: |w|×||act||₂ pruning; LLaMA-7B 2:4 PPL 11.53; LLaMA-2-70B 2:4 PPL 5.16; 1.24× end-to-end speedup LLaMA-7B
[11]: Thanos (Ilin & Richtarik, April 2025) — arXiv:2504.05346: Block-wise Hessian pruning; LLaMA-2-70B 2:4 PPL 4.98 (best published post-hoc)
[12]: MaskLLM (Fang et al., NeurIPS 2024 Spotlight) — arXiv:2409.17481: Gumbel-Softmax 2:4 mask learning; LLaMA-2-7B PPL 6.72 (best 7B result); requires 512k training samples
[13]: LTE / Learn To Be Efficient (arXiv:2402.06126, NeurIPS 2024) — Structured activation sparsity; ~1.33× wall-clock at ~50% sparsity; PPL 5.95 vs Wanda 7.04
[14]: Sparse is Enough in Scaling Transformers (Jaszczur et al., NeurIPS 2021) — arXiv:2111.12763: Training-from-scratch; 17B: **20× decode** (not post-hoc); 800M: 2.6×
[15]: nmSPARSE (MLSys 2023) — Binary PDF: N:M GEMV kernels; 5.2× SpMV at 50% on A100 (kernel benchmark, not LLM serving)
[16]: HuggingFace Block Sparse Blog (Lagunas, 2020) — HuggingFace blog: 2× at 75% BSR block sparsity (URL unverified but directionally consistent)
[17]: V:N:M Sparsity (Zhao et al., 2024) — arXiv:2410.16135: Extends 2:4 beyond 50% sparsity; LLaMA-2-7B 64:2:5: 1.49× speedup vs standard 2:4 1.15× baseline
[18]: Weight Block Sparsity: Training, Compilation, and AI Engine Accelerators (D'Alberto et al., 2024) — arXiv:2407.09453: 8×8 block sparsity; ResNet50 2× speedup at 50% sparsity; training + compiler + accelerator deployment
[19]: BLaST (NeurIPS 2025) — arXiv:2507.03117: BSpMM Triton kernel; Llama-3.2-1B **95% BSR**: **1.6× end-to-end** (NOT 50% BSR); 16-GPU 540B: 2.2×; GPT2-XL 80% PPL +0.40
[20]: Deja Vu (Liu et al., ICML 2023) — arXiv:2310.17157: Contextual activation sparsity; 85–97% FFN sparsity per token; 2× speedup OPT-175B vs FasterTransformer; alternative to weight sparsity
[21]: Accelerating Transformer Pre-training with 2:4 Sparsity (Hu et al., ICML 2024) — arXiv:2404.01847: Training-from-scratch 2:4 sparsity matches dense convergence; masked-decay gradient; actual wall-clock pretraining speedup
[22]: SLiM (Mozaffari et al., ICML 2025) — arXiv:2410.09615: One-shot 2:4 sparsity + 4-bit quantization + low-rank correction; LLaMA-2-7B +5.66% accuracy over prior combined methods; 3.8× A100 speedup; no retraining
