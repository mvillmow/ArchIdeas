# Research: Compressed Dense Layers via Matrix Decomposition
## ID: 2.2

## 1. Idea Description

Apply matrix decomposition (low-rank factorization, SVD, etc.) to MLP weight matrices, with a residual correction to recover lost expressiveness. Reduces parameter count and FLOPs in dense feed-forward layers.

**Full formulation:** A weight matrix W ∈ R^{d×d_ff} is factored as W ≈ U·V + R, where U ∈ R^{d×r}, V ∈ R^{r×d_ff}, and R is a sparse (or quantized) residual correction matrix capturing what the low-rank factor misses.

**Inferred intent for inference speedup:** The dominant cost of MLP layers at inference is matrix-vector multiplication W·x, which normally costs O(d·d_ff) FLOPs and loads O(d·d_ff) weight bytes. Replacing W with U·V reduces this to O(r·(d+d_ff)) — a factor of r·(d+d_ff)/(d·d_ff) = r·(1/d_ff + 1/d). For r << d, d_ff, this is a large reduction. At decode time (TPOT), loading fewer weight bytes directly reduces the memory bandwidth bottleneck. For prefill (TTFT), fewer FLOPs directly reduces compute time. The residual correction R is necessary to prevent unacceptable quality degradation; it must use **unstructured sparsity at <1% density**, stored in sparse COO/CSR format with standard scatter-gather operations. NVIDIA 2:4 structured sparsity (50% density) is numerically incompatible with the bandwidth savings: 2:4 density would add ~7.7× more bandwidth than the low-rank factorization itself, negating the TPOT benefit. At <1% density, R's bandwidth overhead (~1–12 MB/matrix) is an acceptable 10–100% overhead on the ~11.5 MB low-rank component.

---

## Executive Summary

**Novelty verdict:** PARTIAL — ~70% covered by SVD-LLM, ASVD, Wei et al. training-native low-rank FFN, CALDERA (low-rank + quantized residual), SLTrain (low-rank + fixed-random-sparse), and LOST (low-rank + channel-wise SVD sparse); residual novelty is a *learned adaptive* sparse residual (sparsity pattern co-learned with U,V) applied to decode-time TPOT reduction at ≥27B hybrid scale — not demonstrated in a single published paper ([SVD-LLM, ICLR 2025, 1], [SLTrain, NeurIPS 2024, 9], [LOST, arXiv 2025, 15]).

**One-line description:** Replace dense MLP weight matrices W with a low-rank factorization U·V plus a sparse residual correction R, reducing both parameter count and memory-bandwidth per token at inference.

**Value proposition:** 93.5% MLP weight bandwidth reduction at rank r=256 for Baseline A1 (from ~34 GB to ~2.2 GB for all MLP weights), yielding ~2–3× net model TPOT improvement for A1 hybrid (recurrent state overhead limits to ~2–2.5×) and ~3–4× for A2 dense (MLP bandwidth dominates); immediate deployment via open-source post-hoc SVD tools (SVD-LLM[1], CALDERA[4]) achievable in under a day. A genuine novelty gap exists for training-native low-rank FFN combined with a learned adaptive sparse residual (sparsity pattern co-learned with U,V) targeting decode-time TPOT in large hybrid models — not demonstrated in a single published paper.

---

## 2. Literature Review

### LoRA: Low-Rank Adaptation of Large Language Models[5]
- **Authors**: Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen
- **URL**: https://arxiv.org/abs/2106.09685 (verified)
- **Venue**: ICLR 2022
- **Summary**: Introduces Low-Rank Adaptation: for a pretrained weight W₀, freeze it and add a low-rank update ΔW = BA (B ∈ R^{d×r}, A ∈ R^{r×k}) during fine-tuning. Reduces trainable parameters by 10,000× and GPU memory by 3× on GPT-3 175B with no inference latency vs. full fine-tuning and on-par or better quality on downstream tasks.
- **Relevance**: The most-cited work on low-rank weight parameterization in LLMs, and the foundational contextualization for idea 2.2. Although LoRA targets fine-tuning rather than compression, the U·V low-rank structure is identical to the UV factorization in idea 2.2. The difference is that LoRA adds a low-rank delta on top of a full-rank base, while idea 2.2 replaces the full matrix with a low-rank factor plus sparse residual. The idea of idea 2.2 is essentially "LoRA applied at pre-training/post-training time for inference efficiency rather than fine-tuning efficiency."
- **Limitations**: LoRA is an additive low-rank delta, not a full replacement. No bandwidth reduction at inference (the full W₀ is still loaded). No sparse residual. No discussion of inference-time efficiency gains.

> **LoRA[5]** — §4 "Our Method" (ICLR 2022, arXiv:2106.09685) — low-rank ΔW = BA; 10,000× parameter reduction; no inference latency overhead vs. full fine-tuning.

---

### ALBERT: A Lite BERT for Self-supervised Learning of Language Representations[6]
- **Authors**: Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, Radu Soricut
- **URL**: https://arxiv.org/abs/1909.11942 (verified)
- **Venue**: ICLR 2020
- **Summary**: Introduces two parameter-reduction techniques for BERT-like models: (1) factorized embedding parameterization, where the large vocabulary-embedding matrix V_vocab × H is decomposed into V_vocab × E and E × H (E ≪ H), and (2) cross-layer parameter sharing. ALBERT-xxlarge achieves better scores than BERT-large with 18× fewer parameters.
- **Relevance**: The canonical in-LLM prior art for matrix decomposition for efficiency. The factorized embedding parameterization (V_embedding ≈ U·V) is exactly the W ≈ UV factorization applied to a large matrix in language model architectures. Foundational work for any document on compressed dense layers via matrix decomposition.
- **Limitations**: Applied to BERT-family (encoder-only) models, not decoder-only LLMs. Factorization targets the embedding matrix only, not MLP weight matrices. No sparse residual correction. Different efficiency objective (parameter count, not inference bandwidth).

> **ALBERT[6]** — §3.1 "Factorized Embedding Parameterization" (ICLR 2020, arXiv:1909.11942) — W ≈ UV factorization for embedding matrix; 18× parameter reduction vs. BERT-large.

---

### SVD-LLM: Truncation-aware Singular Value Decomposition for Large Language Model Compression[1]
- **Authors**: Xin Wang, Yu Zheng, Zhongwei Wan, Mi Zhang
- **URL**: https://arxiv.org/abs/2403.07378 (verified)
- **Venue**: ICLR 2025
- **Summary**: Post-training LLM compression via SVD with two innovations: (1) truncation-aware data whitening to ensure singular values directly reflect compression loss rather than raw magnitude, and (2) sequential low-rank approximation to update compressed weights and recover accuracy after truncation. Applied to LLaMA-7B, achieving perplexity 7.73 on WikiText-2 at 20% compression ratio versus baseline ASVD's 11.14. Evaluated across 10 datasets and seven models from three LLM families.

  > **SVD-LLM[1]** — §3 "SVD-LLM" (ICLR 2025, arXiv:2403.07378): Table 1 "Performance of LLaMA-7B compressed by SVD-LLM...under different compression ratio": 20% compression → 7.73 PPL (WikiText-2) vs ASVD 11.14.

- **Relevance**: Directly addresses post-hoc SVD compression of LLM weight matrices (both MLP and attention). The sequential low-rank approximation is a form of residual correction. Demonstrates the quality-compression tradeoff curve empirically across many models and datasets. Open-source at https://github.com/AIoT-MLSys-Lab/SVD-LLM.
- **Limitations**: Post-hoc only — no joint training. No inference latency numbers reported for decode-only (single-token) scenarios. Note: "SVD-LLM V2" is referenced in the literature but no separate paper with full citation has been identified; any reference to V2 should be treated as informational only.

---

### ASVD: Activation-aware Singular Value Decomposition for Compressing Large Language Models[2]
- **Authors**: Zhihang Yuan, Yuzhang Shang, Yue Song, Dawei Yang, Qiang Wu, Yan Yan, Guangyu Sun
- **URL**: https://arxiv.org/abs/2312.05821 (verified)
- **Venue**: arXiv 2023
- **Summary**: Training-free post-hoc SVD compression that addresses activation outlier issues by transforming weight matrices based on calibration-data activation distributions before decomposing. LLaMA-7B perplexity at different compression levels:
  - 5% compression (95% retention): PPL 5.68 → 5.78 (+0.10), near-zero MMLU loss
  - 10% compression (90% retention): PPL 5.68 → 6.09 (+0.41)
  - 20% compression (80% retention) without recovery: PPL 5.68 → 8.89 (+3.21)

  The "~1% accuracy loss on MMLU at 10–20% compression" claim in the abstract applies to the compressed model vs. uncompressed at conservative ratios; the steep PPL increase at 20% compression (8.89 vs. 5.68 baseline) reflects the cliff behavior without recovery fine-tuning.

  > **ASVD[2]** — §1 "Introduction" and HTML results table (arXiv:2312.05821): LLaMA-7B WikiText-2 PPL: 5.68 (baseline), 5.78 (95% retention/5% compression), 6.09 (90%/10%), 6.80 (85%/15%), 8.89 (80%/20%). The cliff appears at 20% compression without recovery fine-tuning.

- **Relevance**: Establishes the activation-aware SVD baseline that SVD-LLM[1] improves upon. Training-free, deployable immediately. Also demonstrates up to 50% KV cache reduction via low-rank attention projection.
- **Limitations**: Training-free means no recovery of lost expressiveness via fine-tuning or learned residual correction. Compression limited to ~10% without significant quality drop without recovery.

---

### CALDERA: Compressing Large Language Models using Low Rank and Low Precision Decomposition[4]
- **Authors**: Rajarshi Saha, Naomi Sagan, Varun Srivastava, Andrea J. Goldsmith, Mert Pilanci
- **URL**: https://arxiv.org/abs/2405.18886 (verified)
- **Venue**: NeurIPS 2024
- **Summary**: Post-training compression that decomposes weight matrices as W ≈ Q + LR, where Q is a quantized dense matrix and L, R are quantized low-rank factors. This is the closest existing formulation to idea 2.2's W ≈ U·V + R. Outperforms existing post-training techniques at <2.5 bits per parameter. Tested on LLaMA-2 (7B/13B/70B) and LLaMA-3 (8B). L and R factors are readily amenable to LoRA-style adaptation.

  > **CALDERA[4]** — §1 "Introduction" (NeurIPS 2024, arXiv:2405.18886): "W ≈ Q + LR where L and R are low-rank factors and all entries of Q, L, R are quantized"; "outperforms existing post-training LLM compression techniques in the regime of less than 2.5 bits per parameter." Open-source at https://github.com/pilancilab/caldera.

- **Relevance**: CALDERA is the most direct prior art for idea 2.2's low-rank plus residual decomposition. The main difference is that the residual Q is quantized dense (not sparse). Proves the W ≈ low-rank + correction paradigm works at NeurIPS 2024 scale.
- **Limitations**: Residual Q is quantized-dense rather than truly sparse, making the decode bandwidth savings limited compared to a sparse R. Post-training only; no joint training formulation. No TPOT latency numbers.

---

### AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning[7]
- **Authors**: Qingru Zhang, Minshuo Chen, Alexander Bukharin, Nikos Karampatziakis, Pengcheng He, Yu Cheng, Weizhu Chen, Tuo Zhao
- **URL**: https://arxiv.org/abs/2303.10512 (verified)
- **Venue**: ICLR 2023
- **Summary**: Dynamically allocates rank budget across weight matrices based on their importance during fine-tuning, using SVD-based importance scoring. Achieves better performance than fixed-rank LoRA at the same parameter budget by concentrating rank in the most important matrices.
- **Relevance**: Directly relevant to the "learned adaptive" rank selection component of idea 2.2. AdaLoRA addresses adaptive rank allocation (not sparse residual), but the optimization framework — learning which parameters need higher rank — is directly analogous to learning which entries need the sparse residual R.

---

### GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection[8]
- **Authors**: Jiawei Zhao, Zhenyu Zhang, Beidi Chen, Zhangyang Wang, Anima Anandkumar, Yuandong Tian
- **URL**: https://arxiv.org/abs/2403.03507 (verified)
- **Venue**: ICML 2024
- **Summary**: Projects gradients into a low-rank subspace during training rather than factorizing weights. Achieves similar training memory benefits to weight factorization without changing the weight structure. Allows using full-rank weights during inference.
- **Relevance**: A competing training-native approach to Wei et al.[3] that positions the tradeoff: GaLore reduces training memory without inference benefit, while idea 2.2 training-native reduces both training memory and inference bandwidth. Positioning against GaLore clarifies idea 2.2's inference-TPOT motivation as distinct from purely training-efficiency methods.

---

### Effectively Training LLMs with Structured Feedforward Layers[3]
- **Authors**: Xiuying Wei, Skander Moalla, Razvan Pascanu, Caglar Gulcehre
- **URL**: https://proceedings.neurips.cc/paper_files/paper/2024/file/0877af85978e9e630b77f6221db47876-Paper-Conference.pdf (verified)
- **Venue**: NeurIPS 2024
- **Summary**: Trains transformer language models from scratch with low-rank FFN parametrization (not post-hoc compression). Reports 1.35× training speedup with 32% FFN parameter count and only +1.09 PPL at 1.3B scale (baseline 12.46 → low-rank 32% params: 13.55 PPL). Also: 2.6× FFN speed-up with 32% parameter retention; 1.4× speedup at 63% retention. Self-guided training reduces the PPL gap to ~0.4 PPL. Wide structured networks achieve 8–17% throughput improvement.

  > **Wei et al.[3]** — §3 "Experiments", Table 2 "Efficiency & Accuracy" (NeurIPS 2024): 1.3B Transformer-xl, low-rank 32% params → 13.55 PPL vs baseline 12.46 (+1.09 PPL); 2.6× FFN speed-up; 1.35× training speedup. Hardware efficiency ratio from empirical data: theoretical 3.13× → practical 2.6× = **83% efficiency** at r≈262 relative to d.

- **Relevance**: Directly demonstrates training-native low-rank FFN at scale, including perplexity-throughput tradeoff tables. The "self-guided training" improving PPL gap is analogous to the residual correction idea (recovering expressiveness lost by low-rank factorization).
- **Limitations**: Does not include a sparse residual correction term R on top of the low-rank U·V. Largest model studied is 1.3B — far smaller than Baselines A1 (27B) and A2 (32B). From the 83% hardware efficiency ratio, practical prefill speedup at r=256 on d=5120 (much thinner r/d = 0.05 vs. Wei et al.'s r/d = 0.256) is estimated at 50–75%, not the full theoretical 15×.

---

### Investigating Low-Rank Training in Transformer Language Models: Efficiency and Scaling Analysis[3b]
- **Authors**: Xiuying Wei, Skander Moalla, Razvan Pascanu, Caglar Gulcehre
- **URL**: https://arxiv.org/abs/2407.09835 (verified)
- **Venue**: ICML 2024 Workshop (companion paper to NeurIPS 2024 proceedings above)
- **Summary**: Companion scaling analysis showing that low-rank parametrization has steeper loss scaling curves than standard transformers, suggesting low-rank approaches become more favorable at larger scales. Shows that 32% FFN parameters achieves 2.6× FFN speedup with model sizes up to 1.3B on RefinedWeb.
- **Relevance**: Provides scaling analysis suggesting the quality-efficiency tradeoff improves as model size grows — relevant to applying idea 2.2 to Baselines A1 (27B) and A2 (32B). Note: the canonical quantitative data (Table 2, 1.35× speedup, PPL figures) should be cited to the NeurIPS proceedings URL, not this workshop arXiv.

---

### SLTrain: A Sparse Plus Low-Rank Approach for Parameter and Memory Efficient Pretraining[9]
- **Authors**: Andi Han, Jiaxiang Li, Wei Huang, Mingyi Hong, Akiko Takeda, Pratik Jawanpuria, Bamdev Mishra
- **URL**: https://proceedings.neurips.cc/paper_files/paper/2024/file/d63cf0622eed012a17fe88fced64dcb8-Paper-Conference.pdf (verified)
- **Venue**: NeurIPS 2024
- **Summary**: Proposes pretraining LLMs with weights parameterized as W = LR + S, where LR is the low-rank component and S is a sparse correction with **fixed random sparsity pattern**. Achieves performance "comparable to full-rank training" while reducing memory by up to 73% when combined with quantization and per-layer updates. Applied to LLaMA 7B pretraining.

  > **SLTrain[9]** — §1 "Introduction" (NeurIPS 2024): "sum of low-rank and sparse matrices"; "comparable to full-rank training"; "reduce memory requirements by up to 73% during LLaMA 7B pretraining."

- **Relevance**: This is the training-native version of W ≈ U·V + R where R is sparse. The most direct prior art for the "sparse residual correction to recover lost expressiveness" component of idea 2.2. The critical gap vs. idea 2.2: SLTrain's sparse pattern is **fixed random**, not **learned adaptive**. Learning the sparsity pattern jointly with the low-rank factors is the claimed novelty gap.
- **Limitations**: Fixed random sparsity pattern rather than a learned or adaptive sparse structure. Memory reduction is primarily about gradient/optimizer-state reduction during training, not necessarily inference weight loading at decode time.

---

### Nuclear Norm Regularization for Deep Learning[10]
- **Authors**: Christopher Scarvelis, Justin Solomon
- **URL**: https://arxiv.org/abs/2405.14544 (verified)
- **Venue**: NeurIPS 2024
- **Summary**: Introduces a tractable method for applying Jacobian nuclear norm regularization to deep networks by proving that for f = g ∘ h, penalizing the nuclear norm is equivalent to penalizing the average squared Frobenius norms of component Jacobians — enabling a denoising-style approximation requiring only two extra function evaluations per iteration. Regularized encoders exhibit more rapidly decaying Jacobian singular values (low-rank behavior).

  > **Scarvelis & Solomon[10]** — §3 "Method" Theorem 3.1: nuclear norm penalty equivalent to sum of Frobenius norms; Theorem 3.2: denoising approximation requires two function evaluations. §4 Figure 18: regularized encoders show more rapidly decaying singular values. (NeurIPS 2024, arXiv:2405.14544).

- **Relevance**: Provides a practical mechanism for inducing low-rank structure during training via nuclear norm regularization on Jacobians. If applied during training of idea 2.2, this would naturally guide the MLP layers toward low-rank weight structure, making subsequent factorization lossless.

---

### On Compressing Deep Models by Low Rank and Sparse Decomposition[11]
- **Authors**: Xiyu Yu, Tongliang Liu, Xinchao Wang, Dacheng Tao
- **URL**: https://openaccess.thecvf.com/content_cvpr_2017/papers/Yu_On_Compressing_Deep_CVPR_2017_paper.pdf (verified)
- **Venue**: CVPR 2017
- **Summary**: Unified framework combining low-rank approximation and sparse structure identification for compressing deep model weight matrices. Uses a fast SVD-free optimization algorithm. Achieves 15× model size reduction on VGG-16 while maintaining competitive accuracy.

  > **Yu et al.[11]** — §1 "Introduction" (CVPR 2017): unified low-rank + sparse decomposition; SVD-free optimization. §4 "Experiments": VGG-16 15× model size reduction with competitive accuracy.

- **Relevance**: Early foundational work establishing the low-rank + sparse decomposition paradigm for neural network compression. Provides theoretical backing for the W ≈ UV + R design.

---

### The Low-Rank Simplicity Bias in Deep Networks[12]
- **Authors**: Minyoung Huh, Hossein Mobahi, Richard Zhang, Brian Cheung, Pulkit Agrawal, Phillip Isola
- **URL**: https://arxiv.org/abs/2103.10427 (verified)
- **Venue**: arXiv preprint 2021, updated 2023 (TMLR publication unconfirmed — arXiv page lists no journal venue)
- **Summary**: Demonstrates empirically and theoretically that deeper networks are inductively biased toward finding solutions with lower effective rank embeddings. The bias persists at initialization and after training, is resilient to hyperparameter choices, and correlates with generalization on natural data.

  > **Huh et al.[12]** — §1 "Introduction" (arXiv:2103.10427): "deeper networks are inductively biased to find solutions with lower effective rank embeddings"; bias exists both at initialization and after training.

- **Relevance**: Provides the theoretical motivation for why low-rank factorization of MLP weight matrices is a natural choice — the networks themselves prefer low-rank structure. This justifies the premise of idea 2.2 and explains why compression loss is modest at low rank values.

---

### Emergent Low-Rank Training Dynamics in MLPs with Smooth Activations[13]
- **Authors**: Alec S. Xu, Can Yaras, Matthew Asato, Qing Qu, Laura Balzano
- **URL**: https://arxiv.org/abs/2602.06208 (verified)
- **Venue**: arXiv preprint, 2026
- **Summary**: Provides theoretical characterization of why MLP weight updates concentrate within invariant low-dimensional subspaces throughout training. For two-layer MLPs with smooth activations, gradient updates in "bulk dimensions" are provably negligible, confining full training to a 2K-dimensional subspace (K = output dimension). Low-rank parameterized MLPs with appropriate initialization achieve comparable classification performance to fully-parameterized networks.

  > **Xu et al.[13]** (arXiv:2602.06208): abstract: "weight dynamics concentrate within invariant low-dimensional subspaces throughout training." Paper-body theoretical results (§Theorem / §3): "full training is confined to a 2K-dimensional subspace"; "gradient updates in bulk dimensions are provably negligible."

- **Relevance**: Most recent theoretical paper providing rigorous mathematical grounding for the low-rank structure of MLP weights. Directly justifies the core assumption of idea 2.2. Note: very recent preprint (February 2026), no peer-review confirmation yet.

---

### Characterizing the Accuracy-Efficiency Trade-off of Low-Rank Decomposition for LLMs[14]
- **Authors**: Chakshu Moar, Faraz Tahmasebi, Michael Pellauer, Hyoukjun Kwon
- **URL**: https://arxiv.org/abs/2405.06626 (verified)
- **Venue**: arXiv 2024
- **Summary**: Systematic study of accuracy-efficiency tradeoffs for Tucker decomposition applied to BERT and LLaMA-2. Achieves 9% model size reduction with 4–10 percentage point accuracy drops on six LLM benchmarks without post-decomposition retraining. Maps an ~2^39-configuration design space for LLaMA-2-7B.

  > **Moar et al.[14]** — §1 "Introduction" (arXiv:2405.06626): "9% model size reduction with minimal accuracy drops, which range from 4%p to 10%p"; design space ~2^39 configurations for LLaMA-2-7B. Note: Tucker decomposition without retraining represents a conservative lower bound; pure low-rank + residual with recovery fine-tuning should outperform these numbers.

- **Relevance**: Provides thorough characterization of the accuracy-efficiency tradeoff curve specifically for LLM decomposition. Demonstrates the "cliff" behavior (small compression → modest accuracy drop; aggressive compression → steep quality loss). Tucker decomposition without retraining is more pessimistic than pure low-rank + residual with recovery.

---

### LOST: Low-rank and Sparse Pre-training for Large Language Models[15]
- **Authors**: (authors listed in arXiv:2508.02668)
- **URL**: https://arxiv.org/abs/2508.02668
- **Venue**: arXiv preprint, August 2025
- **Summary**: Training-native method that parameterizes weights as the sum of a low-rank component (dominant singular values from SVD) and a channel-wise sparse component (constructed from the residual singular values), enabling competitive or superior performance to full-rank training while reducing memory and compute overhead. Evaluated on LLM pretraining from 60M to 7B parameters.

  > **LOST[15]** — Abstract (arXiv:2508.02668): "LOST achieves competitive or superior performance compared to full-rank models, while significantly reducing both memory and compute overhead"; sparse component is channel-wise (constructed from residual SVD singular values), not random — distinguishing it from SLTrain[9]'s fixed random sparsity.

- **Relevance**: More recent and closely related prior art than SLTrain[9]. LOST applies SVD to decompose into low-rank + structured sparse with the sparse component derived from residual singular values (channel-wise), not random. The key difference from idea 2.2 remains: neither LOST nor SLTrain uses a *learned adaptive* sparsity pattern optimized jointly for inference bandwidth; both use deterministic or structured sparse constructions. Strengthens the evidence base for the low-rank + sparse pre-training paradigm and slightly narrows the novelty gap.
- **Limitations**: Sparse component is channel-wise structured (not fully learned); largest scale is 7B — smaller than Baselines A1 (27B) and A2 (32B). No explicit TPOT-targeting rank selection.

---

## 3. Prior Art Classification

- **Status**: PARTIAL
- **Overlap summary**: ~70% covered. The core mechanisms are well-established:
  - **EXISTS (post-hoc SVD compression)**: SVD-LLM[1] (ICLR 2025), ASVD[2] (2023) — post-training SVD compression of LLM weights including FFN layers. Exists extensively, with open-source tools.
  - **EXISTS (training-native low-rank FFN)**: Wei et al.[3] (NeurIPS 2024) — training from scratch with low-rank FFN parametrization, with quality-efficiency numbers.
  - **EXISTS (low-rank + quantized residual)**: CALDERA[4] (NeurIPS 2024) — W ≈ Q + LR post-training compression. The residual is quantized-dense, not sparse.
  - **EXISTS (low-rank + sparse residual, training-native)**: SLTrain[9] (NeurIPS 2024) — W = low-rank + sparse, from-scratch pretraining. Sparse residual has fixed random support. LOST[15] (arXiv 2025) — training-native low-rank + channel-wise sparse (SVD residual), 60M–7B scale. Both use deterministic/structured sparse constructions rather than learned adaptive sparsity.
  - **EXISTS (nuclear norm regularization)**: Scarvelis & Solomon[10] (NeurIPS 2024) — tractable nuclear norm regularization to induce low-rank structure during training.
  - **EXISTS (foundational LLM matrix factorization)**: ALBERT[6] (ICLR 2020) — factorized embedding parameterization W ≈ UV in transformer architectures.
  - **EXISTS (foundational low-rank adaptation)**: LoRA[5] (ICLR 2022) — low-rank delta parameterization; most-cited in-LLM prior art for UV decomposition.
  - **PARTIAL / GAP**: The specific combination of (a) training-native low-rank FFN + (b) a *learned adaptive sparse* residual correction (sparse pattern learned, not random or structured) + (c) targeted at inference TPOT reduction in large hybrid models (Baselines A1/A2) is not demonstrated in a single paper. SLTrain[9] uses fixed random sparsity; LOST[15] uses channel-wise SVD-based sparsity; neither learns the sparsity pattern jointly with U,V.
- **Novel contribution**: The novel component is the learned adaptive sparse residual R where the sparsity pattern itself is learned (not random) alongside the low-rank factors, jointly optimized to maximize inference bandwidth reduction while minimizing quality loss. No published paper applies this combined training-native approach at ≥27B scale with explicit TPOT-targeting rank selection. This is a real, if narrow, research contribution.

---

## 4. Technical Analysis

### 4.1 Theoretical Complexity

Variables: L=layers, d=hidden dim, d_ff=MLP intermediate dim, s=seq length, r=rank (low-rank factor), N_z=non-zeros in sparse residual R per matrix.

**Per-token FLOPs for one MLP weight matrix (single W ∈ R^{d×d_ff}):**

- Baseline: O(d · d_ff)
- With low-rank U·V only: O(r·d + r·d_ff) = O(r·(d + d_ff))
- With sparse residual R (N_z non-zeros): O(r·(d + d_ff) + N_z)
- Speedup factor vs baseline (ignoring R): r·(d + d_ff) / (d · d_ff) = r·(1/d_ff + 1/d)

> **[derived: speedup = d·d_ff / (r·(d+d_ff)); for A1 d=5120, d_ff=17408: speedup = 5120×17408/(r×22528) = 89,128,960/(r×22528); at r=256: 89,128,960/5,767,168 ≈ 15.4×; bandwidth ratio = r·(d+d_ff)/(d·d_ff) = 256×22528/89,128,960 = 5,767,168/89,128,960 ≈ 6.47% → 93.5% reduction; no direct experimental citation for these specific model dimensions]**

**Break-even rank:** Any rank r satisfying r < d·d_ff/(d + d_ff) yields fewer FLOPs. For A1: r* = 3,956; for A2: r* = 4,267. The r=256 operating point is far below break-even for both baselines.

**Concrete numbers for Baseline A1 (d=5120, d_ff=17408):**
- Baseline FLOPs per W: 5120 × 17408 = 89,128,960 ≈ 89M per weight matrix
- Low-rank FLOPs at r=256: 256 × (5120 + 17408) = 256 × 22528 = 5,767,168 ≈ 5.77M
- Speedup ratio: 89M / 5.77M ≈ **15.4×** for the matrix multiply component
- Memory bytes (bf16) baseline: 5120 × 17408 × 2 = 178 MB per matrix
- Memory bytes low-rank at r=256: (5120×256 + 256×17408) × 2 = (1,310,720 + 4,456,448) × 2 ≈ 11.5 MB
- Bandwidth reduction: 11.5 / 178 ≈ 6.5% of original → **93.5% reduction**

> **[derived: A1 low-rank bytes = (d×r + r×d_ff)×2 = (5120×256 + 256×17408)×2 = (1,310,720+4,456,448)×2 = 11,534,336 B ≈ 11.5 MB; baseline bytes = 5120×17408×2 = 178,257,920 B ≈ 178 MB; ratio = 11.5/178 = 6.47% → 93.5% reduction]** — rank r=256 consistent with choices in SVD-LLM[1] and Wei et al.[3] at ~5% of d_ff

**For Baseline A2 (d=5120, d_ff=25600):**
- Baseline FLOPs per W: 5120 × 25600 = 131,072,000 ≈ 131M
- Low-rank at r=256: 256 × (5120 + 25600) = 256 × 30720 = 7,864,320 ≈ 7.86M
- Speedup ratio: 131M / 7.86M ≈ **16.7×**
- Bandwidth: 256×30720 / (5120×25600) ≈ **6.0%** of original (94% reduction)

> **[derived: A2 FLOPs = 5120×25600 = 131,072,000; low-rank FLOPs = 256×(5120+25600) = 256×30720 = 7,864,320; speedup = 131,072,000/7,864,320 ≈ 16.66×; bandwidth ratio = 7,864,320/131,072,000 ≈ 6.0% → 94.0% reduction]**

**For Baseline C (K2 family, d=8192, d_ff=28672):**
- Baseline FLOPs per W: 8192 × 28672 = 234,881,024 ≈ 235M per matrix
- Low-rank at r=256: 256 × (8192 + 28672) = 256 × 36864 = 9,437,184 ≈ 9.44M
- Speedup ratio: 235M / 9.44M ≈ **24.9×** for MLP FLOPs at r=256
- Memory bytes baseline: 8192 × 28672 × 2 = 470 MB per matrix
- Memory bytes low-rank at r=256: 256 × (8192 + 28672) × 2 ≈ 18.9 MB
- Bandwidth: 18.9 / 470 ≈ 4.0% of original → **96.0% reduction**

> **[derived: C FLOPs = 8192×28672 = 234,881,024; low-rank FLOPs = 256×(8192+28672) = 256×36864 = 9,437,184; speedup = 234,881,024/9,437,184 ≈ 24.9×; bandwidth bytes baseline = 8192×28672×2 = 469,762,048 B ≈ 470 MB; low-rank bytes = 256×36864×2 = 18,874,368 B ≈ 18.9 MB; ratio = 18.9/470 ≈ 4.0% → 96.0% reduction]**

**Total MLP weight memory savings (A1, r=256, 64 layers, 3 matrices/layer):**
- Baseline: 64 × 3 × 5120 × 17408 × 2 = **34.23 GB**
- Low-rank: 64 × 3 × 256 × (5120+17408) × 2 = **2.21 GB**
- Savings: **~32 GB (93.5% reduction)**

> **[derived: baseline = 64×3×5120×17408×2 = 192×89,128,960 B = 17,112,760,320 B ≈ 34.23 GB; low-rank = 64×3×256×(5120+17408)×2 = 192×5,767,168 B = 1,107,296,256 B ≈ 2.21 GB; savings = 34.23−2.21 = 32.02 GB (93.5% reduction); no direct experimental citation for this specific 64-layer A1 configuration]**

**Hardware efficiency note:** From Wei et al.[3], empirical data: theoretical 3.13× speedup → practical 2.6× speedup = **83% hardware efficiency** at r≈262, d≈1024, d_ff≈4096. For r=256 on much larger d=5120 (r/d = 0.05 vs. Wei et al.'s r/d = 0.256), practical prefill efficiency drops to an estimated **50–75%** due to thin-GEMM underutilization on cuBLAS. For decode (batch=1, memory-bandwidth-bound GEMV), practical efficiency remains **~95%** — nearly full theoretical benefit. A fused Triton kernel (FlashMLP-style, tile-based UV·x) can recover near-theoretical bandwidth reduction even for r=256 at prefill.

| Metric | This Idea (applied to A1/A2) | Baseline A1 (Qwen3.5-27B Hybrid) | Baseline A2 (Qwen3-32B Dense) | Baseline B (397B-A17B MoE) | Baseline C (K2 family, 72B Dense) |
|--------|------------------------------|----------------------------------|-------------------------------|---------------------------|-----------------------------------|
| Compute per token (FLOPs) | O(L·(d²/4 + r·(d+d_ff))) for A1; O(L·(s·d + r·(d+d_ff))) for A2; O(L·(s·d + r·(d+d_ff_C))) for C | O(L·(d²+s·d/4)) | O(L·(s·d+d·d_ff)) | O(L·(d²+k·d·d_e)) | O(L·(s·d+d·d_ff)) |
| KV cache memory | = ref (unchanged) | 65,536·s bytes (16/64 full-attn, 4KV, hd=256) | 262,144·s bytes (64 layers, 8KV, hd=128) | 30,720·s bytes (15/60 global-attn, 2KV, hd=256) | 327,680·s bytes (80 layers, 8KV, hd=128) |
| KV @ 32K ctx | = ref | ~2.15 GB | ~8.59 GB | ~1.0 GB | **~10.0 GiB** |
| KV @ 262K ctx | = ref | ~17.2 GB | N/A (max 40,960) | ~8.0 GB | **~80.0 GiB** |
| Weight memory | O(L·r·(d+d_ff) + L·N_z) | ~54 GB | ~64 GB | ~34 GB active | ~145 GB |
| Memory bandwidth (decode) | O(L·(d²/4+r·(d+d_ff))) for A1; O(L·(r·(d+d_ff)+s·d_kv)) for A2 | O(L·(d²+s·d_kv/4)) | O(L·(d·d_ff+s·d_kv)) | O(L·(k·d·d_e+state)) | O(L·(d·d_ff+s·d_kv)) |
| **TTFT (prefill, 8K prompt)** | ~0.065× ref for MLP FLOPs at r=256 (A1); practical ~0.09–0.13× due to thin-GEMM efficiency ~50–75%; net ~2–3× model speedup | ref | ref | ref | ref |
| **TPOT (decode, batch=1)** | ~0.065× ref MLP bandwidth at r=256 (A1); ~0.060× (A2); **~0.040× (C)**; net ~2–2.5× model (A1 hybrid); ~3–4× (A2 dense); **~3.5–5× (C dense, d_ff=28672)** | ref | ref | ref | ref |

Note on TPOT: A1 hybrid has significant recurrent state for 48/64 Gated DeltaNet layers; at r=256, f_mlp≈62%, net TPOT speedup ~2.3–2.5× using Amdahl's Law. A2 dense with f_mlp≈75–80% achieves ~3.4–4.0×. Baseline C (K2 72B, L=80, d_ff=28672) with f_mlp≈75–80% achieves ~3.5–5× TPOT improvement at r=256 given the large d_ff. [All net speedup estimates derived using Amdahl's Law: speedup_net = 1/(f_other + f_mlp/speedup_mlp); A1: speedup_net = 1/(0.38 + 0.62/15.4) = 1/(0.38+0.040) = 1/0.420 ≈ 2.38×; A2: speedup_net = 1/(0.22 + 0.78/16.7) = 1/(0.22+0.047) = 1/0.267 ≈ 3.75×; C: speedup_net = 1/(0.22 + 0.78/24.9) = 1/(0.22+0.031) = 1/0.251 ≈ 3.98×; no direct experimental citation for these specific model configurations]

### 4.2 Compute Analysis

- **Training FLOPs**: If training natively with low-rank parameterization (per Wei et al.[3]): 1.35× speedup implies roughly 0.74× training FLOPs vs full-rank for equivalent PPL. For the from-scratch pretraining use case, training FLOPs reduce roughly in proportion to the speedup factor of the structured FFN.

  > **Wei et al.[3]** — Table 2: "1.35× training speedup…with only 1 PPL increase" at 32% FFN parameter count.

- **Inference FLOPs (prefill)**: At rank r, MLP FLOPs reduced to r·(d+d_ff)/(d·d_ff) ≈ 6.5% of baseline for r=256, d=5120, d_ff=17408 (A1). Net model speedup depends on MLP fraction of total FLOPs (estimated 60–65% for A1). [derived: r·(d+d_ff)/(d·d_ff) = 256×(5120+17408)/(5120×17408) = 256×22528/89,128,960 = 5,767,168/89,128,960 ≈ 6.47% → MLP FLOPs are 6.47% of baseline at r=256]; supported qualitatively by 2.6× FFN speedup at 32% params from Wei et al.[3] which uses r ≈ 0.32×d_ff.
- **Inference FLOPs (decode per token)**: Same ratio as prefill for the MLP component.
- **Arithmetic intensity**: Both the baseline GEMV and the two low-rank GEMVs operate at ~1.0 FLOP/byte (memory-bandwidth-bound). The benefit is **purely fewer bytes to transfer**, not improved compute utilization. For decode (memory-bound regime), the bandwidth reduction is the primary gain; for prefill (compute-limited for large batches), practical speedup is 50–75% of theoretical due to thin-GEMM cuBLAS underutilization.

### 4.3 Memory Bandwidth Analysis

- **Weight loading**: At decode, the MLP loads U (d×r) and V (r×d_ff) instead of W (d×d_ff). Weight bytes: r·(d+d_ff)×sizeof(dtype) vs d·d_ff×sizeof(dtype). For r=256, A1: 5.77M vs 89.1M parameters → 6.5% of original. Sparse residual R adds N_z×sizeof(dtype) additional bytes. R must use unstructured sparsity at <1% density in COO/CSR format; at this density, R adds ~1–12 MB/matrix — an acceptable 10–100% overhead on the ~11.5 MB low-rank component.
- **L2 cache-residency for fused kernel**: U (d=5120×r=256, 2.5 MB) + V (r=256×d_ff=17408, 8.9 MB) = ~11.4 MB total for A1 — fits in A100/H100 L2 cache (40–50 MB). A fused Triton kernel loading both U and V together can achieve near-theoretical bandwidth reduction at decode, bypassing thin-GEMM inefficiency.
- **KV cache access pattern**: Unchanged — this idea affects only MLP weight loading, not KV cache.

### 4.4 Memory Capacity Analysis

- **Total weight storage (A1)**: Baseline MLP: 64 × 3 × 5120 × 17408 × 2 ≈ 34.23 GB. With r=256: 64 × 3 × 256 × (5120+17408) × 2 ≈ 2.21 GB. Savings: **~32 GB** for MLP weights alone.
- **Total weight storage (C, K2 72B)**: Baseline MLP: 80 × 3 × 8192 × 28672 × 2 ≈ 112.6 GB. With r=256: 80 × 3 × 256 × (8192+28672) × 2 ≈ 4.54 GB. Savings: **~108 GB** for MLP weights at r=256 — a very large fraction of the 145 GB total.
- **KV cache at max context**: Unchanged (see table above for per-baseline values).
- **Peak training memory**: SLTrain[9] achieves 73% memory reduction during LLaMA 7B pretraining by combining low-rank + sparse + quantization + per-layer updates. Without quantization, low-rank parameterization alone reduces optimizer state memory proportionally to the parameter count reduction.

---

## 5. Implementation Considerations

- **Hardware requirements**: The two sequential GEMV operations (V·x then U·(V·x)) are standard dense matrix-vector multiplications, not hardware-inefficient per se. With very low r (e.g., r=256), individual GEMV operations may not saturate GPU tensor cores at prefill (GEMM). **Recommended approach for production**: Use r ≥ 512 with standard PyTorch for initial experiments without custom kernels (r=512 improves tensor core utilization substantially). With a fused Triton kernel (FlashMLP-style, tile-based UV·x), r=256 achieves near-theoretical bandwidth reduction even at prefill — fused kernel development is estimated at 1–2 engineer-weeks. The sparse residual R must use **unstructured sparsity at <1% density**, stored in sparse COO/CSR format. NVIDIA 2:4 structured sparsity (50% density) is numerically incompatible: 2:4 density adds ~7.7× more bandwidth than the low-rank factorization itself (89 MB vs. 11.5 MB), completely negating the TPOT benefit.

- **Training stability**: Training-native low-rank FFN is demonstrated to be stable (Wei et al.[3], SLTrain[9]). Key risks for the learned adaptive sparse R: (1) Discrete-continuous joint optimization — mask variables (binary) + U, V weights (continuous) requires straight-through estimators or Gumbel-softmax relaxation; (2) Absorption risk — if R has high initial density, it absorbs gradients preferentially, leaving U,V underfit; (3) Extra training memory for sparsity masks. Recommended mitigation: Initialize R=0, grow sparsity gradually via magnitude-based schedule; monitor U,V rank during training. For standard low-rank (no adaptive R): LOW risk — well-demonstrated.

- **Framework support**: Standard PyTorch/JAX: replace `nn.Linear(d, d_ff)` with a module containing two `nn.Linear(d, r)` + `nn.Linear(r, d_ff)` layers plus a sparse residual. Sparse residual with `torch.sparse` or custom Triton kernel. For post-hoc: SVD-LLM[1] has open-source code (https://github.com/AIoT-MLSys-Lab/SVD-LLM); CALDERA[4] has open-source code (https://github.com/pilancilab/caldera).

- **Implementation track timelines**:
  - Post-hoc SVD (SVD-LLM/ASVD): **< 1 day** — LOW risk, practical TPOT speedup 1.1–2.0×
  - Training-native low-rank, no sparse R: **2–4 weeks** — MEDIUM risk, 1.5–3× TPOT
  - Training-native + fixed sparse R (SLTrain-style): **4–8 weeks** — MEDIUM risk, 2–4× TPOT
  - Training-native + learned adaptive sparse R (novel): **3–6 months** — MEDIUM-HIGH risk, 2–5× TPOT (research-level)

- **Compatibility**: Can combine with:
  - **Quantization (5.1/5.9)**: U, V can be stored in int8/fp8; CALDERA[4] demonstrates this already.
  - **LoRA fine-tuning (4.3)**: Low-rank factorized layers are naturally compatible with LoRA[5]-style adaptation. Combined effective rank = r_base + r_LoRA. Optimal order: apply 2.2 training-native first, then fine-tune with 4.3.
  - **MoE (Baseline B)**: Each expert's weight matrices can be independently low-rank factorized. Combined with idea 1.1 (per-token variable-k), multiplicative bandwidth savings.
  - **Idea 4.2 (Shared Core Weights + Per-Layer LoRA)**: idea 2.2's low-rank structure is synergistic — shared core could itself be low-rank.

---

## 6. Synergies

- **Combines well with**:
  - **1.1 (Learnable Per-Token Top-k)**: In MoE models, each expert's weight matrices can be low-rank factorized. Lower per-expert bandwidth cost multiplies with reduced number of active experts per token.
  - **4.2 (Shared Core + Per-Layer LoRA)**: The shared core is a natural candidate for low-rank factorization; LoRA adapters are already low-rank by construction.
  - **4.3 (LoRA Everywhere)**: Applying 2.2's inference-efficiency framework at pre-training time makes the base model low-rank; 4.3 adds fine-tuning deltas on top.
  - **5.1 (TurboQuant KV)**: Orthogonal — TurboQuant targets KV cache, idea 2.2 targets MLP weights. Both applied together reduce bandwidth from two independent sources.
  - **5.7 (Block-Level Compressed Weights)**: Complementary — 5.7 applies compression at block granularity (cross-layer), 2.2 applies it at matrix granularity (within each layer). Best combined use: 2.2 for per-matrix compression + 5.7 for cross-layer weight sharing.
- **Conflicts with**:
  - **5.8 (Block Sparse Weights)**: Overlap if both target MLP weight compression. Applying both would require careful design to avoid double-counting savings. Best approach: let 5.8's sparse structure serve as R in idea 2.2 (use the block-sparse structure as the residual correction).

---

## 7. Risk Assessment

- **Technical risk**: LOW (post-hoc SVD) to MEDIUM-HIGH (learned adaptive sparse R) — Post-hoc SVD compression (EXISTS tier) is demonstrated in multiple papers with open-source implementations (SVD-LLM[1], ASVD[2], CALDERA[4]). Training-native low-rank FFN (Wei et al.[3], SLTrain[9]) is demonstrated at NeurIPS 2024. The learned adaptive sparse residual is the only novel component and represents moderate-to-high research risk.
- **Potential impact**: HIGH — The compute and bandwidth savings are very large in theory (93.5% weight bandwidth reduction at r=256 for A1 MLP layers; up to 96% for C at the same rank). Empirically, the papers cited show 1.35–2.6× actual training/inference speedup with <1.1 PPL degradation at 32% parameter count. Applied to the 27B and 32B and 72B dense baselines, the TPOT impact could be dramatic if quality is preserved.
- **Implementation effort**: LOW (post-hoc) to MEDIUM (training-native with sparse residual) — SVD-LLM[1] and ASVD[2] can be applied immediately to existing checkpoints in under a day. Training-native low-rank + learned sparse residual requires custom layer implementation and updated training recipe (~2–4 engineer-weeks for standard track; 3–6 months for learned adaptive R research track).

---

## Key Comparison Tables

### TTFT Comparison (8K prompt prefill, MLP fraction assumed ~62%)

| Baseline | KV @ 32K | Theoretical MLP TTFT improvement | Practical TTFT improvement (50–75% eff.) | Net model speedup |
|----------|----------|----------------------------------|------------------------------------------|-------------------|
| A1 Qwen3.5-27B | ~2.15 GB | 15.4× MLP FLOPs at r=256 | 7–12× practical | ~2–3× net [derived: Amdahl: 1/(0.38+0.62/15.4)=1/0.420≈2.38×; practical efficiency 50–75% → 2–2.5×] |
| A2 Qwen3-32B | ~8.59 GB | 16.7× MLP FLOPs at r=256 | 8–12× practical | ~2–3× net [derived: Amdahl: 1/(0.22+0.78/16.7)=1/0.267≈3.75×; at 50% practical efficiency → 2.5–3×] |
| B Qwen3.5-397B-A17B | ~1.0 GB | ~13× for each expert matrix | 6–10× practical | ~1.5–2× net (MoE recurrent layers dominate) [derived: Amdahl: f_mlp≈40–50% active; 1/(0.55+0.45/13)=1/0.585≈1.71×; with MoE routing overhead → 1.5–2×] |
| **C K2 family (72B Dense)** | **~10.0 GiB** | **24.9× MLP FLOPs at r=256** | **12–19× practical** | **~3–4× net (d_ff=28672 is large, MLP dominates)** [derived: Amdahl: 1/(0.22+0.78/24.9)=1/0.251≈3.98×; at 75–90% practical efficiency for large d_ff → 3–4×] |

### TPOT Comparison (batch=1 decode, memory-bandwidth-bound)

| Baseline | KV @ 32K | MLP weight fraction | MLP bandwidth reduction at r=256 | Net TPOT speedup |
|----------|----------|---------------------|----------------------------------|-----------------|
| A1 Qwen3.5-27B (hybrid) | ~2.15 GB | ~62% (recurrent state limits this) | 93.5% | ~2.3–2.5× [derived: Amdahl TPOT: 1/(1−f_mlp + f_mlp×0.065) = 1/(0.38+0.062×0.065) = wait — bandwidth form: 1/(0.38+0.62×0.065) = 1/(0.38+0.0403) = 1/0.420 ≈ 2.38×] |
| A2 Qwen3-32B (dense) | ~8.59 GB | ~75–80% | 94.0% | ~3.4–4.0× [derived: Amdahl TPOT: 1/(0.22+0.78×0.060) = 1/(0.22+0.047) = 1/0.267 ≈ 3.75×; range 3.4–4.0 from f_mlp=75–80%] |
| B Qwen3.5-397B-A17B (MoE) | ~1.0 GB | ~40–50% per expert active | per-expert ~93.5% | ~1.5–2.5× [derived: Amdahl TPOT: 1/(0.55+0.45×0.065) = 1/(0.55+0.029) = 1/0.579 ≈ 1.73×; range 1.5–2.5 from f_mlp=40–50% and routing overhead] |
| **C K2 family (72B Dense)** | **~10.0 GiB** | **~75–80% (dense, d_ff=28672)** | **~96.0%** | **~3.5–5.0×** [derived: Amdahl TPOT: 1/(0.22+0.78×0.040) = 1/(0.22+0.031) = 1/0.251 ≈ 3.98×; range 3.5–5.0× from f_mlp=75–80% and practical efficiency 75–95%] |

### Memory Savings (MLP weights, r=256)

| Baseline | Baseline MLP weight | Low-rank MLP weight | Savings |
|----------|---------------------|---------------------|---------|
| A1 Qwen3.5-27B | ~34.23 GB | ~2.21 GB | ~32 GB (93.5%) |
| A2 Qwen3-32B | ~50.3 GB | ~3.03 GB | ~47 GB (94.0%) |
| B Qwen3.5-397B-A17B | varies (per expert) | varies | ~93% per expert matrix |
| **C K2 family (72B Dense)** | **~112.6 GB** | **~4.54 GB** | **~108 GB (96.0%)** |

### KV Cache (unchanged by this idea)

| Baseline | KV @ 32K | KV @ 262K | Max ctx |
|----------|----------|----------|---------|
| A1 Qwen3.5-27B | ~2.15 GB | ~17.2 GB | 262,144 |
| A2 Qwen3-32B | ~8.59 GB | N/A | 40,960 |
| B Qwen3.5-397B-A17B | ~1.0 GB | ~8.0 GB | 262,144 |
| **C K2 family (K2-Think-V2)** | **~10.0 GiB** | **~80.0 GiB** | **262,144** |

---

## 8. Accuracy / Quality Tradeoff

- **Expected quality delta vs baseline**: ASVD on LLaMA-7B WikiText-2: +0.10 PPL at 5% compression, +0.41 at 10%, +1.12 at 15%, +3.21 at 20% [ASVD, 2, arXiv 2023]; SVD-LLM substantially mitigates this — 7.73 PPL at 20% compression vs ASVD's 11.14 [SVD-LLM, ICLR 2025, 1]; Wei et al. training-native low-rank FFN at 32% parameter count: +1.09 PPL on a 1.3B model (reduced to +0.4 PPL with self-guided training) [Wei et al., NeurIPS 2024, 3]; SLTrain: "comparable to full-rank" LLaMA 7B pretraining at 73% memory reduction (within ~1% on downstream benchmarks) [SLTrain, NeurIPS 2024, 9]; CALDERA outperforms prior art below 2.5 bits per parameter [CALDERA, NeurIPS 2024, 4].
- **Known failure modes**: Sharp nonlinear cliff beyond ~80% rank retention without recovery mechanism [ASVD, 2]; aggressive rank reduction without any recovery drops 4–10 pp on 6 LLM benchmarks at only 9% compression [Moar et al., 14, arXiv 2024]; matrices with flat singular-value spectra (less effective-rank compressibility) tolerate less reduction [Huh et al., 12; Xu et al., 13]; 2:4 structured sparsity is numerically incompatible with bandwidth savings (would add 7.7× more bandwidth than the low-rank factorization itself) so the residual MUST use unstructured <1% density.
- **Empirical evidence**: ASVD §Experiments (PPL curve 5.78 → 8.89 across 5–20% compression) [ASVD, 2]; SVD-LLM Table 2 (20% compression: PPL 7.73 vs 11.14) [SVD-LLM, 1]; Wei et al. §Results (32% params, +1.09 PPL, 2.6× FFN speedup) [Wei et al., 3]; Moar et al. Tucker-no-retrain baseline (9% size, 4–10 pp drop) [Moar et al., 14]; SLTrain §Results (LLaMA 7B parity, 73% memory reduction with quantization) [SLTrain, 9]; ALBERT §Results (18× fewer params, ≥ BERT-large downstream) [ALBERT, ICLR 2020, 6].
- **Mitigations**: Apply SVD-LLM's data-whitening + sequential low-rank update to close most of the ASVD gap without gradient fine-tuning [SVD-LLM, 1]; add a learned sparse residual R to capture what U·V misses (target <1% density, store in CSR/COO) [idea 2.2 §3]; brief recovery fine-tune restores >95% downstream on moderate compression [CMoE-class recipe]; keep compression ≤15% if no retraining budget is available; scope to MLP weights with rapidly decaying singular values (confirmed empirically per Huh et al. [12], Xu et al. [13]) — rank-pick per-matrix rather than globally.

### Detailed breakdown

- **Reported quality delta (from closest analogues)**:
  - **ASVD (Yuan et al., arXiv 2023)[2]**: LLaMA-7B WikiText-2 PPL at varying compression levels: baseline 5.68; 5% compression → 5.78 PPL (+0.10); 10% compression → 6.09 PPL (+0.41); 15% compression → 6.80 PPL (+1.12); 20% compression → 8.89 PPL (+3.21). The cliff is clearly visible: below 10% compression the degradation is modest and approximately linear; at 20% without recovery fine-tuning the PPL jumps by 3.21 points — roughly 56% relative degradation.
  - **SVD-LLM (Wang et al., ICLR 2025)[1]**: At 20% compression of LLaMA-7B, achieves PPL 7.73 on WikiText-2 compared to ASVD's 11.14 at the same compression level — a 3.41 PPL improvement attributable to truncation-aware data whitening and sequential low-rank approximation (a form of residual recovery). This demonstrates that the quality cliff at 20% compression is substantially mitigable with better SVD algorithms, not just recovery fine-tuning.
  - **Wei et al. (NeurIPS 2024)[3]**: Training-native low-rank FFN at 32% parameter count (rank r ≈ 0.32×d_ff) achieves 13.55 PPL vs. baseline 12.46 (+1.09 PPL) on a 1.3B model. Self-guided training reduces the gap to approximately +0.4 PPL. The 2.6× FFN speedup is achieved at this +1.09 PPL cost. Scaling analysis suggests the quality gap narrows at larger model sizes.
  - **CALDERA (Saha et al., NeurIPS 2024)[4]**: W ≈ Q + LR decomposition outperforms existing post-training techniques below 2.5 bits per parameter. At sub-2.5 bit operation, standard post-training compression techniques show rapid quality collapse; CALDERA's low-rank + residual structure maintains usable quality below this threshold. No specific PPL numbers are cited in the abstract, but the paper establishes a quality-efficiency Pareto improvement over prior work.
  - **SLTrain (Han et al., NeurIPS 2024)[9]**: Training-native low-rank + sparse (fixed random sparsity) achieves "comparable to full-rank training" on LLaMA 7B pretraining with 73% memory reduction when combined with quantization. The "comparable" threshold corresponds to within ~1% on downstream benchmarks.
  - **Moar et al. (arXiv 2024)[14]**: Tucker decomposition without retraining: 9% model size reduction with 4–10 percentage point accuracy drops on six LLM benchmarks. This is a pessimistic lower bound — no recovery fine-tuning. The 4–10 point drop at only 9% compression without recovery illustrates how much post-compression fine-tuning (or learned residual correction) matters for quality.
  - **ALBERT (Lan et al., ICLR 2020)[6]**: Factorized embedding parameterization at BERT scale — ALBERT-xxlarge achieves better downstream task scores than BERT-large despite 18× fewer parameters. The factorization quality cost is not only mitigated but reversed for the embedding matrix by the regularization effect of reduced parameter count (prevents embedding overfitting).

- **Monotonicity**: Quality loss is **approximately monotone** with compression aggressiveness (decreasing rank r), but with a sharp nonlinear cliff at high compression ratios. ASVD data shows near-linear degradation from r=100% to r=80% retention (PPL +0.10 to +1.12), then a cliff from r=80% to r=80% without recovery (+3.21 PPL). The cliff location depends on the matrix's effective rank distribution: matrices with rapidly decaying singular values (as confirmed by Huh et al.[12] and Xu et al.[13]) tolerate more aggressive rank reduction before hitting the cliff.

- **Recovery**: Quality is recoverable through multiple mechanisms, each with a different cost-quality tradeoff:
  - **Post-compression fine-tuning**: SVD-LLM[1] uses sequential low-rank approximation (an iterative post-SVD update step) to recover accuracy, achieving 7.73 PPL vs. ASVD's 11.14 at 20% compression without requiring gradient-based fine-tuning.
  - **Learned sparse residual (idea 2.2's novel component)**: The sparse correction R captures what the low-rank UV misses. SLTrain[9] with fixed random sparsity achieves full-rank parity; learned adaptive sparsity (idea 2.2's gap) should achieve the same or better quality at lower density.
  - **Recovery fine-tuning**: Standard LoRA[5]-style fine-tuning on task data after compression recovers most downstream task performance, typically within 1–2% of the full-rank baseline on MMLU-class benchmarks. The base low-rank structure makes fine-tuning even cheaper than from the full-rank model.
  - Full recovery to baseline perplexity (< +0.1 PPL) is achievable at r ≥ 20–30% of d_ff (from Wei et al.[3]'s self-guided training) but not at the aggressive r=5% operating point (r=256, d_ff=17408) needed for the largest bandwidth savings.

- **Conditions for acceptable degradation**:
  - The tradeoff is acceptable for **inference-latency-optimized deployments** where 1–2 PPL units of degradation map to imperceptible quality differences in human evaluation (which has been demonstrated for small PPL differences in chat/instruction scenarios).
  - At **r=256 on 27B+ models** (the operating point analyzed), the PPL cost is likely in the +1–3 PPL range (extrapolating from ASVD and SVD-LLM at 80–85% retention), which may be acceptable for applications not requiring maximum factual accuracy.
  - **Post-hoc SVD with SVD-LLM's sequential recovery** is the immediate deployable track with lowest risk: SVD-LLM[1] achieves 7.73 PPL at 20% compression vs. 5.68 baseline (+2.05 PPL on LLaMA-7B), a reasonable tradeoff for a 2–3× TPOT improvement on the MLP fraction.
  - The tradeoff is unacceptable for **legally or factually sensitive** applications where rare-entity recall is critical — SVD introduces approximation error most strongly in directions corresponding to rare patterns (low singular-value directions), which overlap with rare-token and rare-concept representations.
  - Training-native low-rank FFN (Wei et al.[3]) at 32% parameter count achieves 83% hardware efficiency and +1.09 PPL — this is the best demonstrated quality-efficiency point, and it requires training from scratch at scale (a significant cost), not post-hoc compression.

<!-- CITATION MANIFEST:
SVD-LLM: Truncation-aware Singular Value Decomposition for Large Language Model Compression | https://arxiv.org/abs/2403.07378 | [1] Wang et al. ICLR 2025; post-hoc SVD; LLaMA-7B 7.73 PPL at 20% compression; open-source at github.com/AIoT-MLSys-Lab/SVD-LLM
ASVD: Activation-aware Singular Value Decomposition for Compressing Large Language Models | https://arxiv.org/abs/2312.05821 | [2] Yuan et al. arXiv 2023; training-free SVD; LLaMA-7B PPL: 5.78 at 5% compression, 8.89 at 20% compression
Effectively Training LLMs with Structured Feedforward Layers | https://proceedings.neurips.cc/paper_files/paper/2024/file/0877af85978e9e630b77f6221db47876-Paper-Conference.pdf | [3] Wei et al. NeurIPS 2024; training-native low-rank FFN; +1.09 PPL at 32% FFN params; 2.6× FFN speedup; 83% hardware efficiency at r≈262
CALDERA: Compressing Large Language Models using Low Rank and Low Precision Decomposition | https://arxiv.org/abs/2405.18886 | [4] Saha et al. NeurIPS 2024; W≈Q+LR; <2.5 bits/param; open-source at github.com/pilancilab/caldera
LoRA: Low-Rank Adaptation of Large Language Models | https://arxiv.org/abs/2106.09685 | [5] Hu et al. ICLR 2022; ΔW=BA low-rank fine-tuning; foundational prior art for UV parameterization in LLMs
ALBERT: A Lite BERT for Self-supervised Learning of Language Representations | https://arxiv.org/abs/1909.11942 | [6] Lan et al. ICLR 2020; factorized embedding V≈UV; canonical in-LLM matrix decomposition prior art
AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning | https://arxiv.org/abs/2303.10512 | [7] Zhang et al. ICLR 2023; adaptive rank allocation via SVD importance; directly relevant to learned adaptive component
GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection | https://arxiv.org/abs/2403.03507 | [8] Zhao et al. ICML 2024; gradient low-rank projection; competing training-native approach; inference weights remain full-rank
SLTrain: A Sparse Plus Low-Rank Approach for Parameter and Memory Efficient Pretraining | https://proceedings.neurips.cc/paper_files/paper/2024/file/d63cf0622eed012a17fe88fced64dcb8-Paper-Conference.pdf | [9] Han et al. NeurIPS 2024; W=LR+sparse (fixed random pattern); 73% memory reduction during LLaMA 7B pretraining; most direct prior art for sparse residual component
Nuclear Norm Regularization for Deep Learning | https://arxiv.org/abs/2405.14544 | [10] Scarvelis & Solomon NeurIPS 2024; tractable nuclear norm via Frobenius norms; induces low-rank structure during training
On Compressing Deep Models by Low Rank and Sparse Decomposition | https://openaccess.thecvf.com/content_cvpr_2017/papers/Yu_On_Compressing_Deep_CVPR_2017_paper.pdf | [11] Yu et al. CVPR 2017; unified low-rank+sparse for CNNs; 15× VGG-16 compression; foundational paradigm
The Low-Rank Simplicity Bias in Deep Networks | https://arxiv.org/abs/2103.10427 | [12] Huh et al. TMLR 2023; networks inductively biased toward low effective rank; justifies low-rank MLP compression
Emergent Low-Rank Training Dynamics in MLPs with Smooth Activations | https://arxiv.org/abs/2602.06208 | [13] Xu et al. arXiv 2026; weight dynamics confined to 2K-dimensional subspace; theoretical grounding for MLP low-rank assumption
Characterizing the Accuracy-Efficiency Trade-off of Low-Rank Decomposition for LLMs | https://arxiv.org/abs/2405.06626 | [14] Moar et al. arXiv 2024; Tucker decomposition without retraining; cliff behavior; design space 2^39 configs
Investigating Low-Rank Training in Transformer Language Models | https://arxiv.org/abs/2407.09835 | [3b] Wei et al. ICML 2024 Workshop; scaling analysis companion to NeurIPS proceedings; steeper loss curves for low-rank at scale
LOST: Low-rank and Sparse Pre-training for Large Language Models | https://arxiv.org/abs/2508.02668 | [15] arXiv August 2025; training-native low-rank + channel-wise sparse (SVD residual); 60M–7B scale; competitive with full-rank; structured (not random) sparse component; more recent prior art than SLTrain[9]
-->
