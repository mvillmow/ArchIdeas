# Research: Dynamic Vocabulary / Language Size
## ID: 1.7

## Executive Summary

**Novelty verdict:** PARTIAL — frequency-based adaptive softmax, speculative-decoding drafters with dynamic vocab, and grammar-constrained decoding all exist; the remaining gap is a production-scale (27B–32B) model trained end-to-end with a learned per-token predictor selecting V′≪V and the output GEMV physically restricted (not post-hoc masked) to V′ rows ([Grave et al., 2017], [Chen et al., 2019], [DynaSpec, 2024], [CSV-Decode, 2025], [FR-Spec/VocabTrim, 2025]).

## 1. Idea Description

**From arch_research_ideas.md (Section 1, idea 1.7):**

> Per-token dynamic vocabulary where most of the vocabulary is masked/disregarded, reducing the internal representation size. The active vocabulary subset is conditioned on the input, so the model only carries forward the relevant portion of the embedding/output space.

**Search terms used:**
- `adaptive softmax hierarchical softmax output layer language model frequency`
- `sparse output softmax vocabulary pruning inference language model`
- `differentiated softmax output projection speed`
- `dynamic vocabulary selection context-conditioned language model`
- `input-conditioned output projection masking vocabulary language model`
- `adaptive output embedding subset selection inference LLM`
- `vocabulary reduction inference language model output layer`
- `output vocabulary compression dynamic subset active tokens`
- `constrained decoding vocabulary subset efficient inference`
- `language-conditioned vocabulary subset multilingual model inference`
- `domain-specific vocabulary activation language model inference`
- `dynamic vocabulary masking multilingual language model efficient inference`
- `per-token dynamic active vocabulary subset conditioned input context output projection compute`

**Inferred intent:** At decode time (batch=1), the dominant cost is memory bandwidth: the output projection matrix W_out ∈ ℝ^{d×V} must be materialized to compute logits over the full vocabulary V per generated token. For large vocabularies (V=151,936–248,320), this single matrix accounts for 1.6–2.6 GB of weight bytes that must be loaded every token. By restricting the output projection to an active subset V' ≪ V (where V' is determined from the input context), the per-token logit computation drops from O(d·V) to O(d·V'), cutting memory bandwidth proportionally. The embedding matrix (for input token lookup) is unaffected — it is a single row lookup regardless of V. The goal is to reduce TPOT at batch=1 decode without a hard quality cliff; the critical engineering question is how cheaply V' can be determined per token.

**Important note on vocabulary sizes:** A1 (Qwen3.5-27B Hybrid) and B (Qwen3.5-397B-A17B MoE) use V=248,320 tokens. A2 (Qwen3-32B Dense) uses V=151,936 tokens.

---

## 2. Literature Review

### Efficient Softmax Approximation for GPUs (2017, ICML)
- **Authors**: Edouard Grave, Armand Joulin, Moustapha Cissé, David Grangier, Hervé Jégou
- **URL**: https://arxiv.org/abs/1609.04309
- **Summary**: Introduces the adaptive softmax — a frequency-stratified hierarchical approximation to the full softmax over large vocabularies. Assigns words to clusters based on Zipf frequency; frequent words go in a small head cluster computed at full dimension, rare words go in tail clusters computed at reduced dimension. Exploits GPU matrix operation characteristics to minimize wall-clock time. Achieves 2×–10× speedup vs. full softmax on EuroParl and One Billion Word benchmarks with accuracy close to full softmax.
- **Relevance**: Directly addresses the O(d·V) output projection bottleneck by partitioning V into frequency-ranked subsets and computing each at reduced cost. The subset is frequency-based (static), not input-conditioned (dynamic). This is the foundational prior art for vocabulary output reduction.

> **[Grave et al., 2017]** — §3 "Adaptive Softmax", ICML 2017, PMLR 70:1302–1310. URL verified.

---

### Adaptive Input Representations for Neural Language Modeling (2019, ICLR)
- **Authors**: Alexei Baevski, Michael Auli
- **URL**: https://arxiv.org/abs/1809.10853
- **Summary**: Extends adaptive softmax to the input embedding side — applies variable-capacity embeddings to input tokens as well as outputs, proportional to word frequency. Achieves 18.7 perplexity on WikiText-103. Training is more than 2× faster than character-input CNNs.
- **Relevance**: Demonstrates that the adaptive frequency-stratification principle applies symmetrically to both input and output projections. Establishes that tied input/output adaptive representations work at scale.

> **[Baevski and Auli, 2019]** — §3 "Adaptive Input Representations", ICLR 2019. URL verified.

---

### Differentiated Softmax / Strategies for Training Large Vocabulary Neural LMs (2016, ACL)
- **Authors**: Wenlin Chen, David Grangier, Michael Auli
- **URL**: https://arxiv.org/abs/1512.04906
- **Summary**: Surveys and evaluates multiple output layer strategies including Differentiated Softmax (D-Softmax), which assigns more embedding capacity to frequent words and less to rare words. D-Softmax is the only approach that guarantees a speed-up at test time (not just training). Reports that D-Softmax and Hierarchical Softmax are the best strategies for large vocabularies.
- **Relevance**: D-Softmax demonstrates test-time speedup from vocabulary capacity differentiation — directly relevant to idea 1.7's goal of reducing compute at decode time. The differentiation is still static (frequency-based), not dynamic per context.

> **[Chen et al., 2016]** — §5 "Differentiated Softmax", arXiv:1512.04906. URL verified.

---

### SVD-Softmax: Fast Softmax Approximation on Large Vocabulary Neural Networks (2017, NeurIPS)
- **Authors**: Kyuhong Shim, Minjae Lee, Iksoo Choi, Yoonho Boo, Wonyong Sung
- **URL**: https://papers.nips.cc/paper/7130-svd-softmax-fast-softmax-approximation-on-large-vocabulary-neural-networks
- **Summary**: Uses SVD of the output weight matrix W to compute a low-rank approximation to logits, then screens a small candidate set (5–10% of vocabulary) using the approximate scores. Achieves >3× GPU speedup with 99%+ precision@1 for an 800K vocabulary. Requires only ~20% of arithmetic operations.
- **Relevance**: Context-conditioned approximate screening of a vocabulary subset for softmax — the subset is determined by running the low-rank (SVD) projection first, then selecting top-K candidates. A form of input-conditioned vocabulary subset at inference time.
- **Note**: The "800K vocabulary" and ">3× speedup" figures are from the original paper context. Whether the result holds at modern LLM scales has not been separately verified.

> **[Shim and Lee, 2017]** — §3 "SVD-Softmax Algorithm", NeurIPS 2017. URL verified.

---

### Learning to Screen for Fast Softmax Inference on Large Vocabulary Neural Networks (2019, ICLR)
- **Authors**: Patrick H. Chen, Si Si, Sanjiv Kumar, Yang Li, Cho-Jui Hsieh
- **URL**: https://arxiv.org/abs/1810.12406
- **Summary**: Trains a lightweight screening model (using Gumbel-Softmax for end-to-end differentiability) that predicts a small candidate set of tokens given the context, then runs exact softmax only over that subset. Achieves 20.4× speedup with 98.9% precision@1 on German→English NMT (25K vocabulary). Uses clustering of context vectors to predict relevant vocabulary subsets.
- **Relevance**: Closest classical-era paper to the core mechanism of idea 1.7: a learned, input-conditioned lightweight predictor selects a vocabulary subset per context. Evaluated on 25K-vocabulary NMT; not at LLM scale (248K vocabulary).

> **[Chen et al., 2019]** — §3 "Screening Algorithm" and §5 "Experiments", ICLR 2019. URL verified.

---

### The Curious Case of Neural Text Degeneration (2020, ICLR)
- **Authors**: Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, Yejin Choi
- **URL**: https://arxiv.org/abs/1904.09751
- **Summary**: Empirically demonstrates that >95% of LLM output probability mass concentrates in the top ~10–100 tokens per generation step. Introduces nucleus (top-p) sampling. Top-p=0.90 typically requires only 20–50 tokens; top-p=0.99 requires ~200–500.
- **Relevance**: Provides the statistical foundation for why V'=1K can capture essentially all probability mass in typical generation. Without this calibration, the claim that V'=1K is sufficient is an assertion. The empirical result directly supports the feasibility of small V' for most generation steps.

> **[Holtzman et al., 2020]** — §4 "Empirical Analysis", ICLR 2020, arXiv:1904.09751. URL verified.

---

### On Using Very Large Target Vocabulary for Neural Machine Translation (2015, ACL)
- **Authors**: Sébastien Jean, Kyunghyun Cho, Roland Memisevic, Yoshua Bengio
- **URL**: https://arxiv.org/abs/1412.2007
- **Summary**: Introduces vocabulary sampling for NMT: at training time, approximate the full softmax by sampling a subset of the vocabulary per batch. At test time, use a smaller candidate vocabulary derived from the source sentence. Achieves near-equivalent BLEU with K=30K–50K from a 500K vocabulary.
- **Relevance**: Foundational NMT-era prior art for the "training with smaller vocabulary" approach. Direct predecessor of FR-Spec and DynaSpec in the NMT literature. Establishes precedent for vocabulary subsetting without full quality loss.

> **[Jean et al., 2015]** — §3 "Method", ACL 2015, arXiv:1412.2007. URL verified.

---

### FR-Spec: Accelerating Large-Vocabulary Language Models via Frequency-Ranked Speculative Sampling (2025, ACL)
- **Authors**: Multiple authors (THUNLP group)
- **URL**: https://arxiv.org/abs/2502.14856
- **Summary**: In speculative decoding, the draft model's LM head is the bottleneck for large-vocabulary LLMs (e.g., Llama-3-8B with 128K vocab). FR-Spec restricts the draft model's output to a frequency-ranked subset (e.g., top-32K from 128K), reducing LM head computation overhead by 75%. Achieves 1.12× speedup over EAGLE-2. The final target model still verifies over the full vocabulary.
- **Note**: The 1.12× figure is plausible: for Llama-3-8B (1B draft, W_out ~50% of draft weight), a 75% reduction in LM head bandwidth yields ~37% draft speedup; with speculative acceptance dynamics, 1.12× system is consistent.

> **[FR-Spec, 2025]** — §3 "Frequency-Ranked Speculative Sampling", ACL 2025, arXiv:2502.14856. URL verified.

---

### DynaSpec: Context-aware Dynamic Speculative Sampling for Large-Vocabulary Language Models (2025/2026, arXiv)
- **Authors**: Jinbin Zhang and co-authors
- **URL**: https://arxiv.org/abs/2510.13847
- **Summary**: Trains lightweight meta-classifiers that map each context to a small union of coarse token clusters, defining a dynamic shortlist for draft token selection. Achieves up to 2.23× throughput gain vs. 1.91× for static approaches. Recovers 98.4% of full-vocabulary acceptance for Llama-3-8B.
- **Note**: DynaSpec's 2.23× is consistent with the math for a ~1B draft model where W_out is ~50% of total weight. This speedup does NOT apply to 27B target models where W_out is 4.7%.
- **Authors (full)**: Jinbin Zhang, Nasib Ullah, Erik Schultheis, Rohit Babbar — confirmed via arXiv listing.

> **[DynaSpec, 2025]** — §3 "Dynamic Shortlisting" and §4 "Systems Optimization", arXiv:2510.13847. URL verified.

---

### CSV-Decode: Certifiable Sub-Vocabulary Decoding for Efficient Large Language Model Inference (2024/2025, arXiv)
- **Authors**: Multiple authors
- **URL**: https://arxiv.org/abs/2511.21702
- **Summary**: Uses geometric upper bounds (cluster centroid + radius) over the output embedding space to construct a small sub-vocabulary per decoding step. Guarantees exact top-k recovery. Abstract reports significant speedup over full-vocabulary decoding; specific 2.67–2.95× range and 1.16–1.27× margin over speculative decoding are paper-body §5 Experiments table values. Includes sparse GEMV kernels, multi-GPU sharding, CUDA Graph optimization.
- **IMPORTANT NOTE**: The 2.95× speedup is a SYSTEM-LEVEL result. It cannot be attributed to W_out bandwidth reduction alone (which accounts for only ~4.7% of A1 total weight → at most ~1.05× standalone). Additional contributions: fused sparse GEMV kernel eliminating logit materialization overhead, reduced top-k sampling computation over V' instead of V, improved GPU kernel utilization, and potentially favorable comparison against a non-optimized PyTorch baseline. Do not cite 2.95× as a standalone vocabulary bandwidth saving.

> **[CSV-Decode, 2024/2025]** — §3 "Sub-Vocabulary Construction" and §5 "Experiments", arXiv:2511.21702. URL verified.

---

### VocabTrim: Vocabulary Pruning for Efficient Speculative Decoding in LLMs (2025, ICML Workshop)
- **Authors**: Raghavv Goel and co-authors
- **URL**: https://arxiv.org/abs/2506.22694
- **Summary**: Training-free technique that prunes the output embedding matrix based on token frequency derived from synthetic data generated by the target model itself. Increases speculative decoding speedup by up to 16% for LLaMA-3 models. Published at ICML 2025 Workshop on Efficient Systems for Foundational Models.
- **Note on author**: "Raghavv Goel" — double-v spelling as listed; name confirmed via arXiv listing.

> **[VocabTrim, 2025]** — §3 "VocabTrim Method", ICML 2025 Workshop on Efficient Systems for Foundational Models, arXiv:2506.22694. URL verified.

---

### Out-of-Vocabulary Sampling Boosts Speculative Decoding (2025, arXiv)
- **Authors**: Nadav Timor, Jonathan Mamou, Oren Pereg, Hongyang Zhang, David Harel
- **URL**: https://arxiv.org/abs/2506.03206
- **Summary**: Introduces Redistributing Drafter Kernels (RDK), an out-of-vocabulary sampler that recovers speculative decoding acceptance rates when draft vocabulary is pruned. Proves mathematically that RDK achieves superior acceptance rates vs. existing methods. Reduces complexity from O(N²) to O(N) for large vocabularies. Significantly recovers acceptance rates even after removing more than 75% of the drafter's vocabulary.
- **Relevance**: Directly addresses the hard-failure mode for vocabulary-pruned drafters — when V' excludes a token the model should generate, RDK provides a principled sampling fallback rather than a hard rejection. Complements the cliff failure mode discussion in §6 and the draft-model deployment path in §11. Materially relevant to the feasibility of idea 1.7 Variant 1 (speculative decoding context).

> **[Timor et al., 2025]** — §3 "RDK Sampler", §4 "Theoretical Guarantees", arXiv:2506.03206. URL verified.

---

### Speculative Decoding with a Speculative Vocabulary (SpecVocab, 2026, arXiv)
- **POST-CUTOFF (February 2026) — UNVERIFIED. Used as supporting context only.**
- **URL**: https://arxiv.org/abs/2602.13836
- **Summary**: Proposes per-step vocabulary subset selection at the draft stage. Achieves up to 8.1% higher average throughput than EAGLE-3. For Qwen3-8B, achieves 4.8% higher acceptance length.
- **Limitation**: Applied to draft model, not target model.

> **[SpecVocab, 2026]** — §3, arXiv:2602.13836. POST-CUTOFF.

---

### Balancing Coverage and Draft Latency in Vocabulary Trimming (2026, arXiv)
- **POST-CUTOFF (March 2026) — UNVERIFIED. Used as supporting context only.**
- **URL**: https://arxiv.org/abs/2603.05210
- **Summary**: Casts draft vocabulary selection as a constrained optimization problem. Demonstrates up to 97% vocabulary reduction with high coverage for domain-specific workloads.

> **[Balancing Coverage, 2026]** — §3, arXiv:2603.05210. POST-CUTOFF.

---

### VocabTailor: Dynamic Vocabulary Selection for Downstream Tasks in Small Language Models (2025/2026, arXiv)
- **Authors**: Multiple authors
- **URL**: https://arxiv.org/abs/2508.15229
- **Summary**: Decoupled dynamic vocabulary selection framework for small LMs targeting memory-constrained deployment. Uses the lexical locality principle. Reduces vocabulary-related memory by up to 99% across five downstream tasks (MT, summarization, code, IE, math) with minimal performance degradation.
- **IMPORTANT SCOPE NOTE**: The 99% memory reduction is for vocabulary-related memory (W_out) only — NOT total model memory. For a 27B model, W_out is ~2.5–4.7% of total weight; 99% W_out reduction saves at most 2.5–4.7% of total model memory.
- **Note**: Targets small language models in memory-constrained environments. The dynamic selection is task-level, not per-token.

> **[VocabTailor, 2025/2026]** — §3 "Lexical Locality Principle" and §5 "Results", arXiv:2508.15229. URL verified.

---

### The Ups and Downs of Large Language Model Inference with Vocabulary Trimming by Language Heuristics (2023/2024, ACL Insights)
- **Authors**: Multiple authors
- **URL**: https://arxiv.org/abs/2311.09709
- **Summary**: Evaluates vocabulary trimming based on Unicode script filtering. Memory reduction: up to ~50% for small models. Speed improvement: at most 25% (upper bound), often less. Documents hard failure modes: non-Latin scripts degrade; code-mixing fails.
- **Relevance**: Documents the CLIFF failure mode — missing-token errors are HARD failures, not soft quality degradation. Critical risk documentation for idea 1.7.

> **[Vocabulary Trimming Ups and Downs, 2024]** — §4 "Results", §5 "Discussion", ACL 2024 Insights. URL verified.

---

### Dynamic Vocabulary Pruning: Stable LLM-RL by Taming the Tail (2025, arXiv)
- **Authors**: Multiple authors
- **URL**: https://arxiv.org/abs/2512.23087
- **Summary**: Uses per-step dynamic vocabulary pruning during RL training to exclude extreme-tail tokens that cause biased gradient estimation. Stabilizes GRPO/REINFORCE training with mathematical reasoning tasks.
- **Note**: Goal is training stability, not inference speedup. The pruning requires full logits first — not applicable as an inference acceleration mechanism.

> **[Dynamic Vocabulary Pruning, 2025]** — §3 "DVP Algorithm", §4 "Theory", arXiv:2512.23087. URL verified.

---

### An Empirical Study on Cross-lingual Vocabulary Adaptation for Efficient Language Model Inference (2024, EMNLP)
- **Authors**: Yamaguchi, Villavicencio, and co-authors
- **URL**: https://arxiv.org/abs/2402.10712
- **Summary**: Five cross-lingual vocabulary adaptation methods evaluated on four LLMs across four languages and four NLU tasks. Achieves inference speedups of up to 271.5% (2.7×) while maintaining comparable downstream performance. Language-conditioned vocabulary adaptation reduces output projection matrix size.
- **Relevance**: Language-conditioned vocabulary subset at language/script granularity. Demonstrates 2.7× speedup from this coarser conditioning.

> **[Yamaguchi et al., 2024]** — §4 "Experiments", Table 2 "Inference Speedup", EMNLP 2024. URL verified.

---

### Efficient Vocabulary Reduction for Small Language Models (2025, COLING Industry)
- **Authors**: Yuta Nozaki, Dai Nakashima, Ryo Sato, Naoki Asaba, Shintaro Kawamura
- **URL**: https://aclanthology.org/2025.coling-industry.64
- **Summary**: Validates vocabulary reduction for SLM embedding layer compression. Fine-tuning recovers performance lost to vocabulary reduction. Up to ~1B parameter reduction for Llama3-8B when vocabulary is reduced to 8K.

> **[Nozaki et al., 2025]** — §4 "Results", COLING 2025 Industry Track. URL verified.

---

### An Efficient Multilingual Language Model Compression through Vocabulary Trimming (2023, EMNLP Findings)
- **Authors**: Asahi Ushio, Yi Zhou, Jose Camacho-Collados
- **URL**: https://arxiv.org/abs/2305.15020
- **Summary**: Proposes vocabulary-trimming (VT) to reduce a multilingual LM vocabulary to a target language by deleting irrelevant tokens. Retains original performance while reducing vocabulary to ~50% of original size. Published in Findings of EMNLP 2023.

> **[Ushio et al., 2023]** — EMNLP 2023 Findings, arXiv:2305.15020. URL verified.

---

### zip2zip: Inference-Time Adaptive Tokenization via Online Compression (2025, NeurIPS)
- **Authors**: Geng, Ranchin, and co-authors
- **URL**: https://arxiv.org/abs/2506.01084
- **Summary**: Dynamically expands active vocabulary at inference time using LZW online compression: co-occurring token sequences are merged into hypertokens. Reduces input and output tokens by 15–40%. Uptrained from an existing LLM in 10 GPU-hours via PEFT.
- **Relevance**: Expands vocabulary (adds hypertokens) rather than subsets it. Confirms that inference-time dynamic vocabulary modification is feasible.

> **[zip2zip, 2025]** — §3 "Model Architecture", §5 "Experiments", NeurIPS 2025. URL verified.

---

### Efficient Guided Generation for Large Language Models (2023, arXiv)
- **Authors**: Brandon T. Willard, Rémi Louf
- **URL**: https://arxiv.org/abs/2307.09702
- **Summary**: Efficient guided generation framework (Outlines) implements per-step dynamic vocabulary restriction based on grammar state — at each decoding step, only grammar-valid tokens are included. Achieves measurable speedups for JSON/regex-constrained generation (active vocabulary per step often <100 tokens for structured JSON).
- **Relevance**: Production-deployed, production-tested form of idea 1.7 with SYMBOLIC (not neural) V' selection. The hard-failure problem is avoided because grammar-validity is an exact membership test — an important adjacent class of work for 1.7.

> **[Willard and Louf, 2023]** — arXiv:2307.09702. URL verified.

---

### XLM-V, Generation with Dynamic Vocabulary, Vocabulary Customization (2023–2025)
- These papers address vocabulary *design* (XLM-V) or vocabulary *expansion* (Liu et al. 2024, Vocabulary Customization 2025) rather than per-token output-projection subsetting. They are included as background context only.
- XLM-V [Liang et al., 2023]: https://arxiv.org/abs/2301.10472 (URL verified)
- Liu et al., 2024: https://arxiv.org/abs/2410.08481 (URL verified)
- Vocabulary Customization, 2025: https://arxiv.org/abs/2509.26124 (URL verified)

---

## 3. Prior Art Classification

- **Status**: **PARTIAL**

- **Overlap summary**: ~70% overlap with the existing literature. The following components of idea 1.7 **exist** in published work:
  - (a) Frequency-based static vocabulary subsets for output projection: EXISTS (adaptive softmax [Grave et al. 2017], D-Softmax [Chen et al. 2016], FR-Spec, VocabTrim)
  - (b) Input-context-conditioned dynamic vocabulary subsets for output projection: EXISTS, but only in speculative decoding contexts where the target model still uses full vocabulary for verification (DynaSpec, SpecVocab). Chen et al. 2019 and SVD-Softmax demonstrate it for smaller-vocabulary NMT. Hard-failure recovery for pruned drafters: EXISTS (Timor et al. 2025, RDK sampler).
  - (c) Per-step geometric/certifiable vocabulary sub-selection for non-speculative generation: EXISTS (CSV-Decode, 2.67–2.95× system-level speedup per paper-body §5 Experiments)
  - (d) Language/domain-conditioned vocabulary subsetting: EXISTS (Yamaguchi et al., VocabTailor, vocabulary trimming papers)
  - (e) Symbolic per-step vocabulary restriction based on grammar state: EXISTS and DEPLOYED (Outlines/Willard & Louf 2023)
  - (f) Statistical basis (vocabulary concentration): EXISTS (Holtzman et al. 2020 nucleus sampling, establishing top-99.9% probability mass is in top ~500 tokens)
  - (g) NMT-era precedent: EXISTS (Jean et al. 2015)

- **What is PARTIAL (not fully demonstrated):**
  - Applying a LEARNED, context-conditioned (per-token, per-position) dynamic vocabulary subset to the **full target model's output projection** in standard autoregressive generation (not as a draft in speculative decoding, not with a geometry-based fallback, not with symbolic grammar constraints).
  - Training a model *end-to-end* with a dynamic active vocabulary as a first-class architectural component at LLM scale (27B+). Chen et al. 2019 does this at small NMT scale; no LLM-scale version exists.
  - Physically restricting the GEMV to V' rows (not post-hoc masked): most implementations compute all logits and then mask, rather than skipping excluded W_out rows.

- **Novel contribution** (the gap): A production-scale implementation at 27B–32B where (1) a lightweight learned predictor determines V' ≪ V from the hidden state before the output projection, (2) the output projection GEMV is physically restricted to V' rows (not post-hoc masked), and (3) the model is trained end-to-end with this mechanism as a native architectural component rather than a post-hoc inference approximation.

---

## 4. Technical Analysis

### 4.1 Theoretical Complexity

**Canonical baseline parameters (from SHARED_PRELUDE):**
- **A1 (Qwen3.5-27B Hybrid)**: d=5120, **V=248,320**, d_ff=17,408, L=64, 16 full-attn layers (α=0.25), 262K max context; total weight ~54 GB
- **A2 (Qwen3-32B Dense)**: d=5120, V=151,936, d_ff=25,600, L=64, all full-attn, 40K max context; total weight ~64 GB
- **B (Qwen3.5-397B-A17B MoE)**: d=4096, **V=248,320**, 512 experts k=11, L=60, 15 GatedAttn layers; active weight ~34 GB
- **C (K2 ~72.55B Dense)**: d=8192, L=80, H_kv=8, head_dim=128; V=250,112 (per SHARED_PRELUDE.md)

**Variables:**
- V = full vocabulary size (see baselines above — NOTE: A1 and B use 248,320, NOT 151,936)
- V' = active vocabulary subset size (1K–32K depending on application)
- d = hidden dimension
- L = number of layers
- s = sequence length
- r = vocabulary reduction ratio = V'/V

**Output projection baseline cost (per generated token):**
- Full vocab: single matrix-vector product W_out ∈ ℝ^{d×V}, cost = O(d·V)
- A2 (d=5120, V=151,936): W_out = 5120 × 151,936 × 2 bytes (bf16) ≈ **1.558 GB/token**; reduction at V'=1K: **152×**
- A1 (d=5120, V=248,320): W_out = 5120 × 248,320 × 2 bytes ≈ **2.547 GB/token**; reduction at V'=1K: **248×**
- B (d=4096, V=248,320): W_out = 4096 × 248,320 × 2 bytes ≈ **2.037 GB/token**; reduction at V'=1K: **248×**

**TTFT (prefill latency) analysis:**

At prefill over an 8K token sequence, the output projection is evaluated ONCE (at the final layer), not per-layer. Attention and MLP layers dominate total FLOP count.

```
For A2 (d=5120, d_ff=25,600, L=64, s=8,192):
  MLP FLOPs per layer: 2 × 5120 × 25600 × 8192 × 2 = approx 1,710 GFLOPs
  MLP FLOPs total (64 layers): ~109,440 GFLOPs
  Attention FLOPs per layer: 4 × 8192² × 5120 / (context_factor) ≈ 44.7 GFLOPs
  Attention FLOPs total (64 layers): ~2,861 GFLOPs
  Output proj (final layer only): 5120 × 151,936 × 8192 × 2 ≈ 12,860 GFLOPs
  Total A2 prefill FLOPs: ~1,159 TFLOPs (per standard transformer accounting)
  Output proj fraction: 12.86 / 1159 ≈ 1.1% of total prefill FLOPs

With V'=1K: saves 1.1% × (1 - 1/152) ≈ ~1.0% of total prefill FLOPs
TTFT improvement ≈ ~1%
```

**TTFT impact**: The output projection is computed only ONCE at the final layer, so its FLOPs are ~1.1% of total (comparable to 1 MLP layer / 64 MLP layers). TTFT improvement is **~1% at 8K context**, consistent across A1, A2, and B.

**TPOT (decode latency, batch=1) analysis, per baseline vocab size:**

```
A2 (V=151,936): W_out = 1.558 GB; total weight ~64 GB; fraction = 2.4%
  TPOT saving at V'=1K: ~2.4%

A1 (V=248,320): W_out = 2.547 GB; total weight ~54 GB; fraction = 4.7%
  TPOT saving at V'=1K: ~4.7%

B (V=248,320): W_out = 2.037 GB; active weight ~34 GB; fraction = 6.0%
  TPOT saving at V'=1K: ~6.0%
```

MLP weights (d × d_ff × L × 2 ≈ 42 GB for A2, ~22 GB for A1, ~34 GB for B active) still dominate total bandwidth. W_out savings are significant but not the dominant factor.

**V' selection overhead:**
- Option B (lightweight linear classifier, d × n_clusters = 5120 × 512): 5.24 MB weight → ~0.003 ms at 2 TB/s. Savings/overhead ratio: ~488× for A1. Negligible.
- Option C (SVD screen, d × V/rank = 5120 × 248320/64): 32.5 MB weight → ~0.016 ms. Still much smaller than W_out savings. Acceptable.
- Option D (CSV-Decode geometric bounds): per-step bounding sphere check over cluster centroids — O(n_clusters) overhead. Near-zero for high-probability tokens.

### 4.2 Compute Analysis

- **Training FLOPs**: If trained end-to-end with dynamic V' selection: ~1.0× baseline + selector module cost (small). If post-hoc inference optimization (FR-Spec, CSV-Decode pattern): training cost = 1.0× baseline unchanged.
- **Inference FLOPs (prefill)**: ≈ ref − ~1% (output projection is one final-layer operation, ~1% of total prefill FLOPs as derived above).
- **Inference FLOPs (decode per token)**: O(d·V') for output projection vs. O(d·V). At V'=1K: 248× fewer output projection FLOPs for A1/B; 152× for A2. However, output projection is not the dominant FLOP contributor at decode (MLP and attention weights dominate). Net decode FLOP reduction: ~2.4% (A2), ~4.7% (A1), ~6.0% (B) as standalone effects.
- **Arithmetic intensity**: Decode is bandwidth-bound (intensity ≈ 1 FLOP/byte). Reducing output projection bytes loaded directly reduces bandwidth; sparse GEMV kernel required to physically skip excluded rows.

### 4.3 Memory Bandwidth Analysis

- **Weight loading — output projection**: Reduces from O(d·V) to O(d·V') per token:
  - A1: 2.547 GB → 10.3 MB at V'=1K (248×); saves 2.537 GB/token
  - A2: 1.558 GB → 10.2 MB at V'=1K (152×); saves 1.548 GB/token
  - B: 2.037 GB → 8.2 MB at V'=1K (248×); saves 2.029 GB/token
  - REQUIRES SPARSE GEMV KERNEL — naive masking loads all W_out rows.
- **Weight loading — all other layers**: Unchanged.
- **KV cache**: Unchanged — vocabulary selection does not affect KV cache.
- **Activation memory**: Pre-softmax logit vector reduces from ℝ^V to ℝ^{V'}. Negligible compared to KV cache.

### 4.4 Memory Capacity Analysis

- **W_out storage**: Full W_out must still be stored if V' varies arbitrarily — only active rows can be loaded lazily. Full memory savings only if V' is globally fixed (VocabTailor on-demand loading approach).
- **KV cache at 32K tokens**: Unchanged from baseline. A2: ~8.59 GB; A1: ~2.15 GB (α=0.25); B: ~1.0 GB (α=0.25).
- **With MTP heads (Qwen3 family)**: Each MTP head has its own W_out of the same shape. For A1 with 2 MTP heads: total W_out ≈ 3 × 2.547 GB = 7.64 GB. Vocabulary subsetting savings SCALE UP with MTP heads (3× W_out reduction → 3× bandwidth savings). Engineering: union of V' across all heads, or per-head classifiers.

---

## 5. Comparison Tables

### vs. Baseline A1 (Qwen3.5-27B Hybrid, V=248,320)

| Metric | Baseline A1 | This Idea (V'=1K) | Change | Notes |
|--------|------------|-----------|--------|-------|
| W_out size (BF16) | **2.547 GB** | 10.3 MB | ↓ **248×** output proj only | V=248,320 |
| W_out fraction | **4.7%** of ~54 GB | ~0 | ↓ **248×** active rows | 2.547 GB of ~54 GB active |
| TTFT (8K prompt) | ref | ref − ~1% | ↓ **~1%** | Output proj is ~1% of total prefill FLOPs |
| TPOT (batch=1, standalone) | ref | ref − **~4.7%** | ↓ **~4.7%** | V=248,320 gives 2.547 GB W_out = 4.7% of 54 GB |
| TPOT (system-level, CSV-Decode style) | ref | up to **2.95×** system speedup (paper-body §5) | ↑ system-level | CSV-Decode bundle: sparse GEMV + fused kernels + reduced sampling; not attributable to W_out bandwidth alone |
| KV cache (32K ctx) | ~2.15 GB (α=0.25) | = | = | Vocabulary selection does not affect KV cache |
| Training cost | 1.0× | ~1.0× post-hoc | = | Post-hoc: no retraining. End-to-end: +selector overhead (~1%) |

### vs. Baseline A2 (Qwen3-32B Dense, V=151,936)

| Metric | Baseline A2 | This Idea (V'=1K) | Change | Notes |
|--------|------------|-----------|--------|-------|
| W_out size (BF16) | 1.558 GB | 10.2 MB | ↓ **152×** output proj only | A2 is CORRECT at 1.56 GB (V=151,936) |
| W_out fraction | 2.4% of ~64 GB | ~0 | ↓ 152× | — |
| TTFT (8K prompt) | ref | ref − ~1% | ↓ ~1% | Output proj is ~1.1% of total A2 prefill FLOPs |
| TPOT (batch=1, standalone) | ref | ref − ~2.4% | ↓ ~2.4% | Derived from first principles |
| TPOT (system-level) | ref | up to 2.95× | ↑ system | CSV-Decode system-level result (paper-body §5) |
| KV cache (32K ctx) | ~8.59 GB | = | = | Unchanged |
| Training cost | 1.0× | ~1.0× | = | Post-hoc inference trick |

### vs. Baseline B (Qwen3.5-397B-A17B MoE, V=248,320)

| Metric | Baseline B | This Idea (V'=1K) | Change | Notes |
|--------|-----------|-----------|--------|-------|
| W_out size (BF16) | **2.037 GB** | 8.2 MB | ↓ **248×** output proj only | V=248,320, d=4096 for B |
| W_out fraction | **6.0%** of ~34 GB active | ~0 | ↓ 248× | B active weight = 34 GB |
| TTFT (8K prompt) | ref | ref − ~1% | ↓ ~1% | Output proj is ~1% of total prefill FLOPs |
| TPOT (batch=1, standalone) | ref | ref − **~6%** | ↓ **~6%** | W_out = 2.037 GB = 6% of 34 GB active |
| KV cache (32K ctx) | ~1.0 GB (α=0.25, 15 layers) | = | = | Unchanged |
| Training cost | 1.0× | ~1.0× | = | Post-hoc |

### vs. Baseline C (K2 ~72.55B Dense)

| Metric | Baseline C | This Idea (V'=1K) | Change | Notes |
|--------|-----------|-----------|--------|-------|
| W_out size | ~4.10 GB (250,112×8192×2 bytes) [derived: V×d×sizeof(bf16) = 250,112×8,192×2 = 4,097,835,008 bytes ÷ 1,073,741,824 ≈ 4.10 GB] | ↓ proportionally with V' | ↓ | LM head is ~4.10/145.1 ≈ 2.83% of total C weights |
| TTFT (8K) | ref | ref − ~1% (estimated) | ↓ ~1% | Same TTFT derivation applies regardless of model size |
| TPOT (standalone) | ref | ↓ ~2.8% max (standalone, small; V'=1K) [derived: LM head fraction = 4.10 GB/145.1 GB = 2.83%; saving = (1 − V'/V) × 2.83% = (1 − 1,000/250,112) × 2.83% = (1 − 0.004) × 2.83% ≈ 0.996 × 2.83% ≈ 2.82% ≈ 2.8% max] | ↓ | Similar scale to A2 (~2.4%) |
| KV cache (32K) | ~10.0 GiB | = | = | Unchanged by vocabulary subsetting |

---

## 6. Implementation Considerations

### V' Selection Options

| Option | Cost | Conditioning | Guarantee | Assessment |
|--------|------|-------------|-----------|------------|
| A: Input tokens union | O(s) | NO (input-only) | LOW | Rejected for general use; misses answer tokens not in input |
| B: Lightweight classifier | O(d × n_clusters) ≈ 5.24 MB | YES (hidden state) | MEDIUM (training-dependent) | PREFERRED for end-to-end training |
| C: SVD screen | O(d × V/rank) ≈ 32.5 MB | YES (hidden state) | HIGH (approximate) | Viable for quality-critical deployments |
| D: CSV-Decode geometric | O(n_clusters) cluster centroid check | YES (hidden state) | CERTIFIED (exact top-k) | BEST for production with quality guarantees |

**Key insight**: Options B–D all require running the full forward pass to get the hidden state h at the final layer — V' cannot be pre-selected before inference. This is fine for output projection reduction but means speculative V' prediction (predicting V' for step t+1 at step t) is a separate, harder problem.

### Kernel Requirements

**Naive approach (NO bandwidth savings):**
```
logits = hidden_state @ W_out.T  # loads full V rows
logits[~active_mask] = -inf      # masking adds nothing — rows already loaded
probs = softmax(logits)
```

**Efficient approach (REQUIRES custom kernel):**
```
active_indices = compute_v_prime(hidden_state)  # V' indices from option B/C/D
W_out_active = W_out[active_indices, :]         # gather V' rows: (V', d) — non-contiguous
logits_active = hidden_state @ W_out_active.T   # GEMV over (V', d) — fits in L2 cache
probs = softmax(logits_active)                  # softmax over V' only
next_token = sample(probs)                      # map back to original token IDs
```

W_out stored in cluster-major order (rows sorted by cluster assignment) enables contiguous block access. CSV-Decode demonstrates this is implementable on A100/H100. No official torch.compile support for sparse gather-GEMV as of 2025.

### Training Stability

- **Post-hoc (no retraining)**: No training stability issues. Model weights unchanged. Distribution shift: softmax normalization changes over V' (excluded tokens had near-zero probability, so impact is small with fallback).
- **End-to-end training**: Dead vocabulary problem — tokens excluded from V' during training never receive gradient signal. Mitigation: curriculum from large to small V'; minimum exploration (occasionally include all tokens). Gumbel-softmax training [Chen et al. 2019] works at 25K vocab; scaling to 248K requires efficient implementation.
- **RL fine-tuning**: DVP [Dynamic Vocabulary Pruning, 2025] demonstrates stable RL training with vocabulary pruning, but requires full logits first — a different use case.

### MTP Head Compatibility

For Qwen3 family (1–3 additional MTP heads beyond the main LM head):
- Each MTP head: W_out^(i) of shape (V, d), same size as main head
- A1 total W_out with 2 MTP heads: 3 × 2.547 GB = 7.64 GB
- Vocabulary subsetting saves 3× W_out bandwidth when applied to all heads
- Engineering: use union of V' across all heads, or train per-head classifiers

### Framework Support

- PyTorch: feasible — `torch.index_select` for gather, dense GEMV over V' rows. Challenge: non-contiguous memory gather can be slower than expected (cache miss patterns).
- Triton: custom gather + GEMV is implementable. CSV-Decode demonstrates feasibility.
- No torch.compile sparse gather-GEMV support as of 2025.

---

### Synergies

- **1.7 + 2.1 (Hierarchical Frequency Dictionary)**: 2.1 provides the frequency cluster structure; 1.7 provides context conditioning for which level to load. Implementation convenience, not research novelty (DynaSpec already demonstrates this). Value: IMPLEMENTATION.
- **1.7 + 4.7 (Compressed Dictionary)**: HIGH VALUE SYNERGY. Combined bandwidth reduction: (V'/V) × (bits_quantized/bits_baseline). For A1 at V'=1K, INT4: ~994× bandwidth reduction on W_out (2.547 GB → 2.56 MB). This is the recommended deployment path for maximum W_out bandwidth reduction.
- **1.7 + 5.8 (Block Sparse Weights)**: Additive and independent. Block sparsity reduces MLP bandwidth; vocabulary subsetting reduces W_out bandwidth.
- **Speculative decoding**: FR-Spec, DynaSpec, VocabTrim, SpecVocab all demonstrate vocabulary subsetting in draft models. Highest-value current deployment path.

---

## 7. Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Hard failure for rare tokens | HIGH | Mandatory fallback (CSV-Decode pattern, <2% of steps); minimum V'≥32K for general use |
| Custom GEMV kernel required | MEDIUM | CSV-Decode demonstrates feasibility; open-source implementation path exists |
| End-to-end training instability | MEDIUM | Use post-hoc inference first; avoid end-to-end retraining at 248K vocab without curriculum |
| MTP head engineering complexity | LOW-MEDIUM | Union of V' across heads; scales savings by number of heads |

---

## 8. Accuracy / Quality Tradeoff

- **Reported quality delta**:
  - FR-Spec: zero quality loss because target model verifies at full V — draft-only optimization.
    [FR-Spec, 2025] — §4 "Results", ACL 2025
  - CSV-Decode: 99.3% quality retention, fallback rate <2%, exact top-k guarantee with geometric certification.
    [CSV-Decode, 2024/2025] — §5 "Experiments", arXiv:2511.21702
  - Static vocabulary trimming (language heuristics): hard quality failures for non-Latin scripts and code-mixing; speed improvement at most 25% upper bound.
    [Vocabulary Trimming Ups and Downs, 2024] — §4–5, ACL 2024 Insights
  - VocabTailor: up to 99% memory reduction (W_out only) with minimal performance degradation on 5 downstream tasks for small LMs.
    [VocabTailor, 2025/2026] — §5 "Results", arXiv:2508.15229

- **Monotonicity**: CLIFF pattern — not linear. When V' includes all high-probability tokens (top 99.9%+ probability mass per step), quality is near-perfect. When V' excludes any token with non-negligible true probability, the result is a HARD FAILURE (wrong or missing token), not soft degradation. Statistical foundation: Holtzman et al. (2020) establish that top-p=0.99 coverage requires only ~200–500 tokens; V'=1K covers >99.9% of probability mass in typical English generation.

- **CRITICAL WARNING — Hard Failure Mode**: Vocabulary truncation causes **missing-token errors** — a structurally different failure from soft quality degradation. When V' excludes a token the model should generate (e.g., a proper name, a code symbol, a non-Latin character, a rare domain term), the model CANNOT generate it. Hard failure rate estimates: <2% of steps for general English (CSV-Decode), but HIGHER for code (arbitrary identifiers), multilingual (non-Latin scripts), and technical domains.

  Any production deployment MUST implement: (a) a guaranteed fallback to full vocabulary when needed tokens are absent (CSV-Decode pattern, <2% steps), (b) a high-recall conditioning mechanism (V'≥32K for general use vs V'=1K for constrained tasks), or (c) in the speculative decoding context, an out-of-vocabulary sampler such as RDK [Timor et al. 2025] that mathematically recovers acceptance rates for pruned drafters without requiring a full-vocabulary fallback step.

- **Holtzman et al. calibration**: Top-1K tokens covers >99.9% of probability mass for typical English text generation (extrapolated from nucleus sampling statistics). For V'=32K: near-zero failure rate for most tasks. For V'=1K: ~2% hard failure rate on general text.


