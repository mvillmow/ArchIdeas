# Research: Hierarchical Frequency-Based Dictionary
## ID: 2.1

## 1. Idea Description

A cascade of dictionaries D1→D2→…→DL where D1 contains the highest-frequency tokens plus a special `<next_dict>` token. When the model needs a less common token, it triggers loading the next dictionary level. This reduces compute by handling common tokens cheaply.

**Inferred intent for inference speedup:** At decode time (batch=1), the output-projection step — a matrix-vector multiply over the full vocabulary V — is the single most bandwidth-intensive operation outside the transformer stack, costing O(d·V) FLOPs and loading O(d·V) bytes of embedding weights per token. For large vocabularies (e.g., V=248,320 for Qwen3.5-27B), this is non-trivial. Because token frequency follows a Zipf distribution, a small fraction of the vocabulary (the "head") accounts for the bulk of probability mass. By organizing the output layer into a cascade where D1 (head) covers ~90% of the probability mass with |D1| << V, the expected per-token output-projection cost drops dramatically. The `<next_dict>` token acts as a sentinel: when predicted, a second pass over D2 is executed; in ~90% of decode steps, only D1 is needed.

**Key distinction from 4.7 (Compressed Dictionary):** Idea 2.1 is a sequential cascade — a token is either predicted from D1, or triggers loading D2, D3, etc. in order. The dictionaries are separate weight matrices loaded one at a time. Idea 4.7 is a single dictionary with context-gated active subset — it does not cascade; it gates within a single structure.

---

## Executive Summary

**Novelty verdict:** EXISTS — ~90% covered by adaptive softmax (mechanically equivalent two-level cascade), hierarchical softmax, class-based RNN outputs, adaptive input representations, FR-Spec, and VOCABTRIM; residual "novel contribution" of a `<next_dict>` sentinel token is a minor implementation variant and strictly inferior to adaptive softmax's cluster-selector head on differentiability/renormalization/inference filtering ([Grave et al., ICML 2017, 3], [Zhao et al., ACL 2025, 1], [Baevski & Auli, ICLR 2019, 8]).

**One-line description:** A frequency-stratified cascade of output dictionaries D1→D2→…→DL where common tokens are resolved cheaply from the small head tier D1; rare tokens trigger evaluation of subsequent tiers — mechanically equivalent to adaptive softmax (Grave et al., ICML 2017), which has already been demonstrated and shipped.

**Value proposition:** Up to 76× bandwidth reduction on the LM head projection for 90%-covered tokens, yielding ~2–5% net TPOT improvement for 27–32B models (where the LM head is ~2.5–4.7% of total weight) and ~30% draft-model TPOT improvement in speculative decoding contexts where the LM head constitutes ~30% of total weight.

---

## 2. Literature Review

### Efficient Softmax Approximation for GPUs (Adaptive Softmax)[3]
- **Authors**: Édouard Grave, Armand Joulin, Moustapha Cissé, David Grangier, Hervé Jégou
- **URL**: https://arxiv.org/abs/1609.04309 (verified)
- **Venue**: ICML 2017, PMLR vol. 70, pp. 1302–1310
- **Summary**: Proposes "adaptive softmax," which partitions vocabulary into a head cluster (most frequent words, small size) and multiple tail clusters (less frequent, larger sizes), exploiting Zipf's law to minimize expected GPU computation time. The head classifier is evaluated on every input; tail classifiers are only evaluated when the head classifier selects the corresponding cluster-entry token. Achieves 2×–10× speedup over full softmax with perplexity close to full softmax on EuroParl and One Billion Word benchmarks; on One Billion Word achieves 43.9 PPL (vs. 43.7 for a model 8× larger using 32 GPUs for 3 weeks).
- **Relevance**: This is the anchor paper for idea 2.1. The adaptive softmax is structurally equivalent to the proposed hierarchical frequency dictionary cascade: a head tier (D1) evaluated first, with tail tiers (D2…DL) evaluated conditionally only for less frequent tokens. The head/tail cascade triggered by a cluster-selector token is mechanically equivalent to the D1→D2 cascade triggered by `<next_dict>`. The key distinction is that adaptive softmax uses a cluster hierarchy at the output layer only, while idea 2.1 frames this as a general "dictionary cascade" that could extend to embeddings and KV lookups.
- **Limitations**: Designed for training-time speedup on GPUs, not specifically for autoregressive inference TPOT optimization. Does not address the specific inference-serving scenario (batch=1 decode, memory-bandwidth bound). Does not use a `<next_dict>` token in the output vocabulary; instead uses a separate cluster selector head — a strictly better design for training stability and end-to-end differentiability. The paper predates large subword vocabularies (V~250K); the Qwen3.5-27B vocabulary is 248,320 tokens.

> **Adaptive Softmax[3]** — §"Adaptive softmax" (ICML 2017, PMLR vol. 70, pp. 1302–1310) — demonstrates 2×–10× training speedup from the head/tail partition; cluster structure with head covering high-frequency tokens.

---

### Hierarchical Probabilistic Neural Network Language Model (Hierarchical Softmax)[4]
- **Authors**: Frédéric Morin, Yoshua Bengio
- **URL**: https://proceedings.mlr.press/r5/morin05a.html (verified)
- **Venue**: AISTATS 2005, PMLR R5:246–252
- **Summary**: The foundational paper on hierarchical softmax for neural language models. Represents vocabulary as leaves of a binary tree; each token's probability is the product of binary decisions along the root-to-leaf path, reducing output-layer cost from O(V) to O(log V). Uses WordNet-derived hierarchy for tree construction. Demonstrates substantially reduced training cost without major perplexity loss.
- **Relevance**: Establishes the foundational concept of hierarchical output layers for language models. The D1→D2→…→DL cascade in idea 2.1 is a flat generalization of this tree structure — instead of a binary tree path, it is a linear sequence of tiers triggered by special sentinel tokens.
- **Limitations**: Tree structure is O(log V) worst-case but the WordNet hierarchy does not optimize for Zipf-law coverage. Unlike idea 2.1, there is no explicit high-frequency "head" tier covering bulk probability mass; the tree is balanced by semantic structure, not frequency.

---

### A Scalable Hierarchical Distributed Language Model[5]
- **Authors**: Andriy Mnih, Geoffrey E. Hinton
- **URL**: https://papers.nips.cc/paper/2008/hash/1e056d2b0ebd5c878c550da6ac5d3724-Abstract.html (verified)
- **Venue**: NeurIPS 2008, pp. 1081–1088
- **Summary**: Extends hierarchical softmax with a log-bilinear language model at each tree node, and introduces an automatic data-driven algorithm for word-tree construction. Achieves O(log V) output cost and demonstrates that hierarchical models can outperform flat non-hierarchical models.
- **Relevance**: Demonstrates that automatic (data-driven) construction of the output hierarchy achieves better results than manually-constructed (WordNet-based) hierarchies. Applicable to idea 2.1: the optimal frequency cutoffs for D1, D2, etc. should be learned from data rather than fixed.

---

### Extensions of Recurrent Neural Network Language Model (Class-Based Output)[6]
- **Authors**: Tomas Mikolov, Anoop Deoras, Daniel Povey, Lukáš Burget, Jan Černocký
- **URL**: https://www.fit.vut.cz/research/group/speech/public/publi/2011/mikolov_icassp2011_5528.pdf (URL not verified); DOI: 10.1109/ICASSP.2011.5528
- **Venue**: ICASSP 2011
- **Summary**: Introduces a two-level class-based output layer for RNN language models. Vocabulary is partitioned into frequency-based classes; the model first predicts a class, then predicts the word within that class, reducing output complexity from O(V) to approximately O(sqrt(V)) for balanced classes.
- **Relevance**: Directly implements the core of idea 2.1: a frequency-based two-level hierarchy where the output first selects a class (analogous to predicting from D1 or the `<next_dict>` token), then selects a word within that class (analogous to D2).

---

### Strategies for Training Large Vocabulary Neural Language Models (Differentiated Softmax)[7]
- **Authors**: Wenlin Chen, David Grangier, Michael Auli
- **URL**: https://arxiv.org/abs/1512.04906 (verified)
- **Venue**: ACL 2016
- **Summary**: Introduces "Differentiated Softmax" (D-Softmax), which assigns different output embedding dimensionalities to words based on frequency. The output projection matrix is block-sparse by construction. Confirms the principle that frequency-stratified output representations are beneficial.
- **Relevance**: Addresses the same problem as idea 2.1 (efficient large-vocabulary output layer) from a capacity-differentiation angle. The block-sparse output projection in D-Softmax is related to the cascade idea.

---

### Adaptive Input Representations for Neural Language Modeling[8]
- **Authors**: Alexei Baevski, Michael Auli
- **URL**: https://arxiv.org/abs/1809.10853 (verified)
- **Venue**: ICLR 2019
- **Summary**: Extends adaptive softmax to the input embedding layer as well, with variable-capacity embeddings per frequency tier (frequent tokens get larger embedding dimension, rare tokens get smaller). Uses tied input/output weights. Achieves state-of-the-art on WikiText-103 (18.7 PPL) and One Billion Word (23.02 PPL). Trains >2× faster than character CNN approaches.
- **Relevance**: The tied input/output weight regime means the frequency-stratified output structure is mirrored in the input. For idea 2.1 applied to a full transformer (not just the output layer), this approach shows the value of frequency-stratified representations end-to-end. If idea 2.1 is extended to input embeddings, this paper is the direct prior art. Also the standard reference for "partial tying" — the workaround for the weight-tying conflict mentioned in §6.

> **Adaptive Input Representations[8]** — §"Adaptive Softmax for Output Layers" and §"Adaptive Input Representations" (ICLR 2019, arXiv:1809.10853) — WikiText-103 PPL 18.7, One Billion Word 23.02; tied adaptive embeddings; partial tying approach directly applicable to §6 weight-tying conflict.

---

### Learning to Screen for Fast Softmax Inference on Large Vocabulary Neural Networks[9]
- **Authors**: Patrick H. Chen, Si Si, Sanjiv Kumar, Yang Li, Cho-Jui Hsieh
- **URL**: https://arxiv.org/abs/1810.12406 (verified)
- **Venue**: ICLR 2019
- **Summary**: Proposes a lightweight "screening" model that uses the clustering structure of context vectors to predict a small candidate set of words, then runs exact softmax only within that set. Achieves 20.4× inference speedup over full softmax on German→English translation (~25K vocab) with 98.9% precision@1 and 99.3% precision@5.
- **Relevance**: The screening model is a learned two-stage approach to output-layer efficiency. D1 acts as the "screener" that handles common tokens cheaply; only D2 is evaluated when D1's `<next_dict>` is predicted. The screening model uses context clustering rather than frequency.

---

### Breaking the Softmax Bottleneck: A High-Rank RNN Language Model (Mixture of Softmaxes)[10]
- **Authors**: Zhilin Yang, Zihang Dai, Ruslan Salakhutdinov, William W. Cohen
- **URL**: https://arxiv.org/abs/1711.03953 (verified)
- **Venue**: ICLR 2018
- **Summary**: Identifies a rank bottleneck in the output softmax and proposes Mixture of Softmaxes (MoS) — multiple output distributions whose weighted mixture avoids the low-rank constraint. The fundamental difference from idea 2.1: MoS uses a parallel mixture (all evaluated simultaneously), while idea 2.1 uses a sequential cascade (conditional evaluation).
- **Relevance**: Theoretical contrast to idea 2.1 — parallel mixture vs. sequential cascade. Confirms that multiple output matrices are beneficial; the design choice of parallel vs. sequential affects both quality and efficiency differently.
- **Limitations**: MoS does not implement the frequency-based cascade; it evaluates all mixture components simultaneously. No inference bandwidth reduction. Different mechanism from idea 2.1.

---

### Using the Output Embedding to Improve Language Models (Output Embedding Tying)[11]
- **Authors**: Ofir Press, Lior Wolf
- **URL**: https://arxiv.org/abs/1608.05859 (verified)
- **Venue**: EACL 2017
- **Summary**: Demonstrates that tying input and output embeddings (using the same weight matrix for both) improves perplexity and reduces parameter count. This is the standard reference for weight-tied output layers.
- **Relevance**: Directly relevant to idea 2.1's §6 Synergies note about "conflicts with standard weight-tied embeddings." The incompatibility of the cascade structure with standard weight tying is a real deployment concern for existing models. Modern LLMs including the Qwen3 family use tied or near-tied embeddings; the cascade requires partial tying per Baevski & Auli (2019) as mitigation.

---

### FR-Spec: Accelerating Large-Vocabulary Language Models via Frequency-Ranked Speculative Sampling[1]
- **Authors**: Weilin Zhao, Tengyu Pan, Xu Han, Yudi Zhang, Ao Sun, Yuxiang Huang, Kaihuo Zhang, Weilun Zhao, Yuxuan Li, Jie Zhou, Hao Zhou, Jianyong Wang, Zhiyuan Liu, Maosong Sun
- **URL**: https://arxiv.org/abs/2502.14856 (verified)
- **Venue**: ACL 2025
- **Summary**: Addresses the bottleneck of computing the draft model's LM head over large vocabularies (e.g., Llama-3-8B with 128K tokens) in speculative decoding. Prunes the draft model's output embeddings to retain only the most frequency-ranked tokens from large pre-training corpora, reducing LM Head compute overhead by 75% while achieving 1.12× end-to-end speedup over EAGLE-2. Ensures equivalence of the final output distribution.
- **Relevance**: This is the closest modern (2025) paper to idea 2.1's core motivation: frequency-ranked vocabulary reduction for output-layer speedup. FR-Spec uses a static frequency-based pruning of the output vocabulary (retaining the head), which is exactly what D1 in idea 2.1 does — but FR-Spec applies this to a draft model in speculative decoding rather than as a cascade.

> **FR-Spec[1]** — §"Abstract" (ACL 2025, arXiv:2502.14856) — "75% reduction in LM Head computation overhead" and 1.12× end-to-end speedup over EAGLE-2 for large-vocabulary (128K) draft models.

---

### VOCABTRIM: Vocabulary Pruning for Efficient Speculative Decoding in LLMs[2]
- **Authors**: Raghavv Goel, Sudhanshu Agrawal, Mukul Gagrani, Junyoung Park, Yifan Zao, He Zhang, Tian Liu, Yiping Yang, Xin Yuan, Jiuyan Lu, Chris Lott, Mingu Lee
- **URL**: https://arxiv.org/abs/2506.22694 (verified)
- **Venue**: ICML 2025 Workshop on Efficient Systems for Foundational Models
- **Summary**: Training-free technique that reconstructs the drafter LM head to include only the most frequently sampled tokens from the target model's vocabulary. Achieves 16% memory-bound speedup for Llama-3.2-3B-Instruct on Spec-Bench. Identifies a trade-off: limiting vocabulary slightly degrades acceptance rate.
- **Relevance**: Further corroborates the frequency-based vocabulary reduction approach for inference efficiency. The acceptance-rate degradation observed in VOCABTRIM quantifies the accuracy/quality tradeoff that idea 2.1 must also face.

> **VOCABTRIM[2]** — §"Abstract" (ICML 2025 Workshop, arXiv:2506.22694) — 16% memory-bound speedup; acceptance-rate tradeoff quantified.

---

### Exploiting Vocabulary Frequency Imbalance in Language Model Pre-training[12]
- **Authors**: Woojin Chung, Jeonghoon Kim
- **URL**: https://arxiv.org/abs/2508.15390 (verified)
- **Venue**: NeurIPS 2025
- **Summary**: Controlled study scaling vocabulary from 24K to 196K tokens with fixed compute/data. Finds that larger vocabularies reduce cross-entropy loss almost exclusively for the ~2,500 most frequent words (comprising ~75% of downstream benchmark tokens), while loss on rare tokens rises.
- **Relevance**: Provides empirical grounding for idea 2.1's frequency-stratification premise. Confirms that ~75% of downstream benchmark tokens come from the ~2,500 most frequent words — establishing a principled D1 coverage target.

> **Chung & Kim[12]** — §"Abstract" (NeurIPS 2025, arXiv:2508.15390) — confirms ~75% of downstream benchmark tokens come from ~2,500 most frequent words, providing empirical grounding for D1 sizing.

---

### Balancing Coverage and Draft Latency in Vocabulary Trimming for Faster Speculative Decoding[13]
- **Authors**: Ofir Ben Shoham
- **URL**: https://arxiv.org/abs/2603.05210 (verified)
- **Venue**: arXiv preprint, 2026
- **Summary**: Formalizes draft vocabulary selection as a constrained optimization problem over the coverage–latency Pareto frontier. Uses a Tree-structured Parzen Estimator to find optimal vocabulary subsets. Reports up to 16% latency reduction and 20% throughput gains on domain-specific tasks, and up to 97% vocabulary size reduction with high coverage maintenance.
- **Relevance**: Provides the most detailed analysis of the coverage-vs-latency tradeoff in frequency-based vocabulary reduction, directly quantifying the trade-off space that idea 2.1 must navigate. The 97% vocabulary reduction result (from ~128K to ~3,840 tokens maintaining high coverage) is highly relevant to sizing D1 for idea 2.1.

> **Ben Shoham[13]** — §"Abstract" (arXiv:2603.05210, 2026) — reports vocabularies shrunk by 97% while maintaining high coverage, validating |D1| ≈ 0.03×V as a practical design point.

---

### LLM Vocabulary Compression for Low-Compute Environments[14]
- **Authors**: Sreeram Vennam, Anish Joishy, Ponnurangam Kumaraguru
- **URL**: https://arxiv.org/abs/2411.06371
- **Venue**: Machine Learning and Compression Workshop @ NeurIPS 2024
- **Summary**: Compresses the final linear (LM head) layer of language models by grouping tokens based on BPE merge history, preventing materialization of the full logits tensor. Achieves up to 3.4× memory compression of the final linear layer and up to 3× throughput improvement with performance on par with GPT-Neo and GPT-2 on TinyStories.
- **Relevance**: Direct modern (2024) prior art for vocabulary-based output-layer compression. The BPE-grouping approach is complementary to frequency-stratified cascades: rather than routing uncommon tokens to later tiers, it groups them into coarser output nodes. Provides empirical evidence that lossy grouping of the LM head is practical at scale, and establishes the 3× throughput ceiling achievable via static grouping — which idea 2.1's cascade can exceed only by adding conditional routing overhead.

> **Vennam et al.[14]** — §"Abstract" (NeurIPS 2024 Workshop, arXiv:2411.06371) — 3.4× LM head memory compression and 3× throughput via BPE-based grouping; practical ceiling for static vocabulary compression without cascading.

---

## 3. Prior Art Classification

- **Status**: EXISTS
- **Overlap summary**: ~90% covered. The core mechanism of idea 2.1 — a multi-tier, frequency-stratified output layer where common tokens are resolved cheaply in a head tier and rare tokens trigger evaluation of subsequent tiers — is substantially covered by:
  1. **Adaptive softmax**[3] (Grave et al., ICML 2017): mechanically equivalent two-level (head+tail) cascade at the output layer, frequency-based cluster assignment, GPU-optimized implementation. This alone covers the core.
  2. **Hierarchical softmax**[4][5] (Morin & Bengio, 2005; Mnih & Hinton, 2008): tree-based cascade, foundational prior art.
  3. **Class-based RNN output**[6] (Mikolov et al., 2011): flat two-level frequency-based cascade nearly identical to D1→D2.
  4. **Adaptive input representations**[8] (Baevski & Auli, 2019): extends the same frequency-stratification to input embeddings with tied weights; provides the partial-tying solution for the weight-tying conflict.
  5. **FR-Spec**[1] (Zhao et al., 2025): frequency-ranked vocabulary head for inference speedup with 75% LM head reduction, validated at production scale (128K vocab).
  6. **VOCABTRIM / Ben Shoham**[2][13] (2025–2026): extends vocabulary pruning analysis to cover the Pareto frontier.

- **Novel contribution (marginal)**: The specific framing of a `<next_dict>` sentinel *token in the output vocabulary* — as opposed to a *cluster selector head* in adaptive softmax — is a minor implementation variant. In adaptive softmax, the cluster selector is a separate (small) head matrix; in idea 2.1, it is an explicit token in D1. This difference does not change the computational structure. Moreover, the `<next_dict>` approach is strictly inferior to the adaptive softmax cluster-selector head on three counts: (1) the `<next_dict>` decision is discrete, blocking gradient flow without explicit straight-through estimation; (2) renormalization of D2 scores requires an extra pass that adaptive softmax avoids through its joint normalization; (3) `<next_dict>` must be filtered at inference time to avoid being emitted to the user.

- **Comparison to adaptive softmax[3] — section 3 focus:**

  | Dimension | Adaptive Softmax[3] | Idea 2.1 (Hierarchical Dict.) |
  |-----------|------------------------------|-------------------------------|
  | Mechanism | Head cluster + tail clusters; separate cluster-selector head | D1 (head) + D2…DL; `<next_dict>` token in D1 |
  | Trigger | Cluster-selector head scores over J+1 options | `<next_dict>` predicted as top-1 token |
  | Frequency basis | Yes — by Zipf distribution | Yes — by token frequency |
  | Tiers | 2–5 clusters typical | L tiers |
  | Structural relationship | Reference baseline | Mechanically equivalent |
  | Training speedup | Yes, demonstrated 2–10× | Yes (equivalent) |
  | Inference TPOT focus | Implicit (primarily training) | Explicit design goal |
  | End-to-end differentiability | Yes — cluster selector is differentiable | Partial — `<next_dict>` is discrete |
  | Modern LLM scale | V ≈ 10K–65K typical | V = 248,320 (A1/B); 151,936 (A2) |
  | Tied input/output | No | Not specified |

  The structural equivalence is accurate for the core compute structure. The claim "mechanically equivalent" is more precise than "mechanically identical" given the trigger-mechanism and differentiability differences, both of which favor the existing adaptive softmax design.

---

## 4. Technical Analysis

### 4.1 Theoretical Complexity

**Setup for this idea applied to Baseline A2 (Qwen3-32B dense, V=151,936, d=5,120):**

Let:
- V = 151,936 (full vocabulary, A2)
- |D1| = vocabulary head (top-f fraction of probability mass)
- f = fraction of tokens resolved by D1 (default analysis: f = 0.90)
- |D1| = 1,000 (top-1000 tokens cover ~90% of probability mass; consistent with Grave et al. 2017's observation that top-20% vocabulary covers ~87% of PTB mass, and Ben Shoham 2026's finding that 97% reduction is achievable at high coverage per[13])
- |D2| = next tier, e.g., 10,000 additional tokens (rare tokens)
- For f = 0.90: expected output-projection cost per token = f · O(d·|D1|) + (1-f) · O(d·(|D1|+|D2|))

**Expected softmax cost per token:**
- With D1 covering 90% of probability mass and |D1| = 1,000:
  - E[cost] = 0.90 × (d·1,000) + 0.10 × (d·(1,000 + |D2|))
  - If |D2| = 10,000: E[cost] = d·(900 + 0.10×11,000) = d·(900 + 1,100) = d·2,000
  - Baseline full softmax cost: d·V = d·151,936
  - **Expected speedup on output projection: 151,936 / 2,000 ≈ 76×** [derived: E[cost] = f×(d×|D1|) + (1−f)×(d×(|D1|+|D2|)) = 0.90×(d×1,000) + 0.10×(d×11,000) = d×(900+1,100) = d×2,000; baseline = d×V = d×151,936; speedup = 151,936/2,000 = 75.97 ≈ 76×]

- For a three-tier cascade (D1=1K, D2=10K, D3=140K), f1=0.90, f2=0.09, f3=0.01:
  - E[cost] = 0.90·(d·1K) + 0.09·(d·11K) + 0.01·(d·151K)
  - = d·(900 + 990 + 1,510) = d·3,400
  - **Speedup on output projection: ~45×** [derived: E[cost] = 0.90×(d×1K) + 0.09×(d×11K) + 0.01×(d×151K) = d×(900+990+1,510) = d×3,400; baseline = d×151,936; speedup = 151,936/3,400 ≈ 44.7 ≈ 45×]

**TPOT impact:** At batch=1, decode is memory-bandwidth-bound. The output projection weight matrix (d×V = 5,120×151,936) requires loading ~1.57 GB per token at bf16. With the cascade, expected bytes loaded ≈ (d·E[|tiers evaluated|]·sizeof(bf16)), reducing to ~20 MB per token for the D1=1K case — a **~76× reduction in output-projection bandwidth** [derived: full LM head BW = d×V×2 bytes = 5,120×151,936×2 = 1,557,olean,832 bytes ≈ 1.57 GB; cascade D1 BW = d×|D1|×2 = 5,120×1,000×2 = 10,240,000 bytes ≈ 9.77 MB; 1,557 MB/9.77 MB ≈ 159×; with D2 triggered 10%: E[BW] = 0.90×9.77 MB + 0.10×(d×11,000×2/1e6) MB = 0.90×9.77 + 0.10×107.4 ≈ 8.79+10.74 = 19.5 MB; 1,557 MB/19.5 MB ≈ 80×; headline "76×" matches the speedup ratio from the E[cost] computation above]. However, the output projection is only a fraction of total decode bandwidth (the transformer layers dominate at 32B params). The output projection weight alone is 5,120×151,936×2 bytes ≈ 1.57 GB out of total ~64 GB model weight — about 2.4% of total decode bandwidth.

**Net TPOT impact for Baseline A2: ~2–2.5% reduction** for the output projection portion. [derived: LM head weight = V×d×2 = 151,936×5,120×2 = 1,555,793,920 bytes ≈ 1.57 GB; total model weight ≈ 64 GB; LM head fraction = 1.57/64 = 2.45%; cascade reduces LM head BW by ~76×, so saving on LM head ≈ (1−1/76)×2.45% ≈ 0.987×2.45% ≈ 2.42%; net TPOT reduction ≈ 2.4%]

**L2 cache-residency upside:** For |D1| = 1K tokens: 1,000 × 5,120 × 2 bytes ≈ 10 MB, which fits in H100 L2 cache (50 MB). If D1 is cache-resident (achievable with explicit prefetching/pinning), the 90% of decode steps that only evaluate D1 may achieve near-zero HBM bandwidth cost for the LM head — potentially exceeding the 76× bandwidth reduction for D1-resolved tokens. This is only realizable with careful D1 weight pinning; not automatic. [derived: D1 weight size = |D1|×d×2 bytes = 1,000×5,120×2 = 10,240,000 bytes ≈ 9.8 MB < H100 L2 cache (50 MB); if D1 is L2-resident, HBM BW for 90% of steps ≈ 0; remaining 10% load D2 at 5,120×10,000×2 ≈ 98 MB from HBM; E[HBM BW] ≈ 0.10×98 MB = 9.8 MB vs baseline 1,557 MB → >150× effective reduction for cache-resident D1]

**Output projection bandwidth as fraction of total:**

| Baseline | Model weight (bf16) | LM head (V×d) | LM head fraction | Net TPOT improvement |
|----------|---------------------|---------------|-----------------|----------------------|
| A1 Qwen3.5-27B (V=248,320, d=5,120) | ~54 GB | ~2.55 GB | ~4.7% | ~4.5% |
| A2 Qwen3-32B (V=151,936, d=5,120) | ~64 GB | ~1.57 GB | ~2.4% | ~2.4% |
| B Qwen3.5-397B (V=248,320, d=4,096) | ~34 GB active | ~2.04 GB | ~6.0% | ~5.8% |
| C K2 family (V=250,112, d=8,192) | ~145 GB | ~4.10 GB | ~2.8% | ~2.7% |

Note: For A1 and B, vocab is 248,320 (not 151,936); for A2, vocab is 151,936; for C (K2 family), vocab is 250,112.

| Metric | This Idea (2.1, applied to A2) | Baseline A1 (Qwen3.5-27B Hybrid) | Baseline A2 (Qwen3-32B Dense) | Baseline B (397B-A17B MoE) | Baseline C (K2 family, 72B Dense) |
|--------|----------------------------------------|----------------------------------|-------------------------------|---------------------------|-----------------------------------|
| Compute per token (FLOPs) | O(L·(s·d+d·d_ff)) + O(d·E[cascade_size]) vs O(d·V) | O(L·(d²+s·d/4)) | O(L·(s·d+d·d_ff)) | O(L·(d²+k·d·d_e)) | O(L·(s·d+d·d_ff)) |
| KV cache memory | O(L·s·d_kv) — unchanged | 65,536·s bytes total (16 full-attn layers, 4KV, hd=256) | 262,144·s bytes total (64 layers, 8KV, hd=128) | 30,720·s bytes total (15 global-attn layers, 2KV, hd=256) | 327,680·s bytes total (80 layers, 8KV, hd=128) |
| KV cache @ 32K ctx | = ref | ~2.15 GB | ~8.59 GB | ~1.0 GB | ~10.0 GiB |
| KV cache @ 262K ctx | = ref | ~17.2 GB | N/A (native max 40,960) | ~8.0 GB | ~80.0 GiB |
| Weight memory | O(L·d·d_ff) + O(d·V) ≈ same | ~54 GB | ~64 GB | ~34 GB active | ~145 GB |
| Memory bandwidth (decode) | O(L·(d·d_ff+s·d_kv)) + O(d·E[cascade]) | O(L·(d²+s·d_kv/4)) | O(L·(d·d_ff+s·d_kv)) | O(L·(k·d·d_e+state)) | O(L·(d·d_ff+s·d_kv)) |
| **TTFT (prefill, 8K prompt)** | ≈ ref — output projection negligible vs. 8K × L layers | ref | ref | ref | ref |
| **TPOT (decode, batch=1)** | ~0.97× ref (A2) / ~0.954× ref (A1) / ~0.94× ref (B) / ~0.973× ref (C) | ref | ref | ref | ref |

**Note on TPOT:** LM head fractions and net improvement estimates for each baseline [all derived: LM head BW = V×d×2 bytes; total model weight at bf16; fraction = LM head / total; net improvement = (1−1/76) × fraction ≈ 0.987 × fraction]:
- A1 (27B, V=248,320): LM head ≈ 2.55 GB / 54 GB ≈ 4.7%; 76× on 4.7% → ~4.7% net improvement
- A2 (32B, V=151,936): LM head ≈ 1.57 GB / 64 GB ≈ 2.4%; 76× on 2.4% → ~2.4% net improvement
- B (397B-A17B, V=248,320, d=4,096): LM head ≈ 2.04 GB active weight portion; active weight ~34 GB; ~6%; 76× on 6% → ~5.9% net improvement
- C (K2 72B, V=250,112, d=8,192): LM head ≈ 4.10 GB / 145 GB ≈ 2.8%; 76× on 2.8% → ~2.7% net improvement

For a small draft model (300M params, 128K vocab), the LM head is ~30% of total weight; 76× reduction on 30% gives ~1.43× net improvement — explaining why FR-Spec[1] targets draft models.

### 4.2 Compute Analysis

- **Training FLOPs**: Compared to Baseline A2 dense: reduction in output-projection FLOPs proportional to E[cascade_size]/V ≈ 2,000/151,936 ≈ 1.3% of output-step FLOPs. Output projection is a small fraction of total training FLOPs. Net training FLOPs reduction: <1% of total. [derived: output projection FLOPs per token = 2×d×V = 2×5,120×151,936 ≈ 1,557M; total decode FLOPs per token (A2, 64 layers) ≈ 2×L×(4×d²+2×d×d_ff) = 2×64×(4×5120²+2×5120×25600) ≈ 2×64×(524M+262M) = 100,663M; LM head fraction = 1,557M/100,663M ≈ 1.5%; cascade reduces to 2×d×2,000 ≈ 20.5M FLOPs; net saving on total ≈ (1,557−20.5)/100,663 ≈ 1.5% reduction]
- **Inference FLOPs (prefill)**: Output projection is computed once at the end of prefill (the last token's logits). For an 8K prompt, this is a negligible fraction of total prefill FLOPs. **= ref** — the cascade saves only the final output-projection step, which is tiny relative to 8K × L layers of attention+MLP.
- **Inference FLOPs (decode per token)**: Output projection reduced from O(d·V) to O(d·E[cascade_size]). For the numbers above: ~76× reduction on the LM head FLOPs. Since LM head FLOPs = d·V = 5,120×151,936 ≈ 780M FLOPs per token vs. total decode FLOPs (much larger) — the LM head is a modest fraction. Reduction is **measurable but not dominant** for 32B+ models.
- **Control flow overhead**: Conditional dispatch for batch training (different samples triggering different cascade tiers) adds an estimated +1–3% output-step overhead beyond the Big-O reduction, from host-device sync and masked/grouped kernel launches. [derived: host-device sync latency ≈ 1–5 μs per conditional; for B=1 decode this is negligible vs. HBM BW latency (~10–100 μs per 10 MB load); for large-batch training (B=1024), each batch has ~10% of samples (≈102) triggering D2, requiring a masked grouped GEMM — overhead relative to a single fused GEMM ≈ 2–5% from kernel launch + index gather; 1–3% is a conservative mid-range estimate]

### 4.3 Memory Bandwidth Analysis

- **Weight loading**: At decode (batch=1), the LM head is loaded O(d·|tiers_evaluated|) bytes per token. Expected reduction: ~76× for the output projection weights (D1=1K vs. V=151K). Actual bytes saved per token: ~1.52 GB/token (at bf16, d=5120, V=151K → 1.57 GB; cascade → ~21 MB). But this is only the LM head; full model decode loads ~64 GB per token for all layers.
- **L2 cache-residency upside**: |D1|=1K tokens × d=5120 × 2 bytes ≈ 10 MB, which fits in H100 L2 cache (50 MB). If D1 is cache-resident, the 90% of decode steps that only evaluate D1 may achieve near-zero HBM bandwidth cost for the LM head, potentially exceeding the 76× bandwidth reduction for D1-resolved tokens. Requires careful prefetching/pinning of D1 weights. [derived: D1 footprint = 1,000×5,120×2 = 9.77 MB; H100 L2 = 50 MB; D1 fits with 40 MB to spare; if L2-resident, 90% of steps pay 0 HBM BW for LM head; 10% pay D2 load = 10,000×5,120×2 = 97.7 MB HBM; E[HBM] = 0×0.9 + 97.7×0.1 = 9.77 MB vs baseline 1,557 MB → 159× effective reduction]
- **KV cache access pattern**: Unchanged — the cascade only affects the output projection, not attention or MLP weights.
- **Activation memory**: During training, the cascade introduces a branching computation graph. The `<next_dict>` prediction path requires storing activations for both D1 and D2 steps. Peak training activation memory may increase by O(|D2|) per sample that triggers D2 (10% of samples). Net training activation overhead: ~10% × |D2|/|D1| ≈ 10% × 10 = 100% overhead for D2 evaluations — but only 10% of samples trigger D2, so average overhead ≈ 10% of total output-projection activation. [derived: D1 activation per sample = d = 5,120 floats ≈ 10 KB (bf16); D2 activation per sample = d×|D2| = 5,120×10,000 = 51.2M floats ≈ 97.7 MB; 10% of batch triggers D2; average activation overhead = 0.10×97.7 MB = 9.77 MB per batch sample at output step; vs baseline output-step activation = d×V×2 = 1,557 MB per sample if stored; overhead is actually a reduction for 90% of samples and a moderate increase only when D2 is triggered]

### 4.4 Memory Capacity Analysis

- **Total weight storage**: The cascade weight matrices total |D1|·d + |D2|·d + … instead of V·d. For D1=1K, D2=10K, D3=~141K (covering full vocab), total = V·d — essentially the same. If D3 is dropped (covering only 99% of vocab), weight savings = 1%·V·d ≈ 1.5 GB. Negligible.
- **KV cache at max context (32K tokens)**: Unchanged (see table above for per-baseline values).
- **Peak training memory**: Modest increase from branching output computation. For most tokens (90%), training memory is reduced (only D1 scores computed). For 10%, D1+D2 scores are computed. Net: ~10% overhead vs. single-tier full softmax for the output step.

---

## 5. Implementation Considerations

- **Hardware requirements**: The cascade structure requires conditional execution — a CUDA kernel must branch on the predicted token from D1 being `<next_dict>`. This breaks the regular parallelism of a single large GEMM and introduces control flow divergence. For batch=1 inference, this is straightforward (a single `torch.argmax()` + Python conditional + second `nn.Linear` call suffices; host-device sync overhead ~1 μs, negligible). For large-batch training, different samples in a batch may trigger different cascade depths, requiring masked/grouped computation (similar to MoE dispatch). A Triton kernel implementing the conditional cascade is feasible but non-trivial. The adaptive-softmax implementation (https://github.com/facebookresearch/adaptive-softmax) demonstrates this is implementable in Torch.
- **Training stability**: The `<next_dict>` token must not be overloaded — it must remain a reliable sentinel without being confused with a real token. This requires careful initialization of D1 output weights. Three design flaws specific to `<next_dict>` (vs. adaptive softmax cluster-selector): (1) the discrete argmax selection blocks gradient flow to the D2 path from D1 without explicit straight-through estimation or Gumbel-softmax relaxation; (2) D1 probabilities assigned to `<next_dict>` must be redistributed over D2 tokens via a separate renormalization step that adaptive softmax avoids through its joint normalization; (3) `<next_dict>` must be filtered at inference time, adding serving complexity. All three issues are avoided by using the adaptive softmax cluster-selector head architecture instead.
- **Framework support**: PyTorch feasible; the Facebook Research adaptive-softmax GitHub repository provides a reference implementation (https://github.com/facebookresearch/adaptive-softmax). The conditional cascade can be implemented as a standard Python conditional with separate `nn.Linear` modules for each tier. JAX feasible but requires explicit lax.cond for the branching.
- **Compatibility**: Can combine with idea 1.7 (Dynamic Vocabulary — context-conditioned masking within D1 for further reduction). Can combine with idea 4.7 (Compressed Dictionary — D1 itself could be a compressed representation). Does not interact with attention, MLP, or KV cache ideas.

---

## 6. Synergies

- **Combines well with**:
  - **1.7 (Dynamic Vocabulary)**: D1 could be context-conditioned — not just the top-f frequency tokens globally, but the top-f tokens given the current context. Reduces D1 further.
  - **4.7 (Compressed Dictionary)**: D1 could itself be a compressed representation (fewer bits per entry), combining cascade-based and compression-based savings.
  - **2.2 (Matrix Decomposition)**: The D2 and D3 weight matrices could be low-rank decomposed, reducing their cost when they are evaluated.
  - **5.1 (TurboQuant)**: The LM head weight matrices (D1, D2) can be quantized aggressively since the LM head is not as precision-sensitive as attention.
- **Conflicts with**:
  - **Standard weight-tied embeddings**: If input embeddings are tied to the output projection (common practice, standard per Press & Wolf 2017[11]), the cascade structure requires rethinking the tying — either use partially-tied weights (Baevski & Auli 2019[8]) or untied weights.
  - **Speculative decoding**: In speculative decoding, the draft model already uses reduced vocabulary (FR-Spec[1]); applying idea 2.1 on top may be redundant.

---

## 7. Risk Assessment

- **Technical risk**: LOW — The mechanism is essentially adaptive softmax[3], which has been demonstrated at scale, implemented in open-source (Facebook Research), and validated in both training and inference contexts. The main risk is the `<next_dict>` token approach being less stable than a separate cluster-selector head (as in adaptive softmax); the correct mitigation is to adopt the adaptive softmax architecture directly — which means the `<next_dict>` novelty is abandoned.
- **Potential impact**: LOW–MEDIUM for large models (32B+, where LM head is ~2.5–4.7% of total weight), HIGH for small draft models (300M–1B, where LM head is 30%+ of total weight). The most impactful deployment is as a component of speculative decoding draft models, consistent with FR-Spec's[1] approach. For Baseline C (K2 family, 72B dense), the LM head is ~2.8% of total weight — same LOW–MEDIUM regime.
- **Implementation effort**: LOW — Reference implementation exists (https://github.com/facebookresearch/adaptive-softmax). Adapting to a modern Transformer with 248K vocab involves updating the cluster sizes and ensuring compatibility with modern training pipelines. Estimated effort: 1–2 engineer-weeks to integrate into an existing training codebase.

---

## Key Comparison Tables

### TTFT Comparison (8K prompt prefill)

| Baseline | TTFT | Change vs. ref | Notes |
|----------|------|----------------|-------|
| A1 Qwen3.5-27B | ref | — | — |
| A2 Qwen3-32B | ref | — | — |
| B Qwen3.5-397B-A17B | ref | — | — |
| **C K2 family (72B)** | ref | — | — |
| **This idea (all baselines)** | ≈ ref | ≈ 0% | Output projection computed only once at final prefill step; LM head FLOPs < 0.001% of 8K-prompt prefill FLOPs; negligible improvement [derived: prefill FLOPs (A2, 8K tokens, 64 layers) ≈ 2×8,192×64×(4×5120²+2×5120×25600) ≈ 2×8,192×64×786M ≈ 824×10¹² FLOPs; LM head at final step = 2×5,120×151,936 ≈ 1,557M FLOPs; fraction = 1,557M/(824×10¹²) ≈ 0.0000019 = 0.00019% of prefill; negligible] |

### TPOT Comparison (batch=1 decode)

| Baseline | TPOT | LM head fraction | LM head cascade savings | Net TPOT |
|----------|------|-----------------|------------------------|----------|
| A1 Qwen3.5-27B (V=248,320) | ref | ~4.7% [derived: 248,320×5,120×2=2,546,237,440 bytes≈2.55 GB; total≈54 GB; 2.55/54=4.72%] | 76× reduction → 98.7% LM head savings [derived: (1−1/76)×4.72%=0.9868×4.72%≈4.66%] | ~0.954× ref (~4.7% improvement) |
| A2 Qwen3-32B (V=151,936) | ref | ~2.4% [derived: 151,936×5,120×2=1,555,825,664 bytes≈1.57 GB; total≈64 GB; 1.57/64=2.45%] | 76× reduction → 98.7% LM head savings [derived: (1−1/76)×2.45%≈2.42%] | ~0.976× ref (~2.4% improvement) |
| B Qwen3.5-397B-A17B (V=248,320) | ref | ~6.0% [derived: 248,320×4,096×2=2,036,989,952 bytes≈2.04 GB; active weight≈34 GB; 2.04/34=6.0%] | 76× reduction → 98.7% LM head savings [derived: (1−1/76)×6.0%≈5.92%] | ~0.941× ref (~5.9% improvement) |
| **C K2 family (V=250,112, ~72B dense)** | ref | ~2.8% [derived: 250,112×8,192×2=4,097,835,008 bytes≈4.10 GB; total≈145.1 GB; 4.10/145.1=2.83%] | 76× reduction → 98.7% LM head savings [derived: (1−1/76)×2.83%≈2.79%] | **~0.973× ref (~2.7% improvement)** |

### KV Cache Comparison (all baselines, unchanged by this idea)

| Baseline | KV @ 32K | KV @ 262K | Max native ctx |
|----------|----------|----------|----------------|
| A1 Qwen3.5-27B | ~2.15 GB | ~17.2 GB | 262,144 |
| A2 Qwen3-32B | ~8.59 GB | N/A (max 40,960) | 40,960 |
| B Qwen3.5-397B-A17B | ~1.0 GB | ~8.0 GB | 262,144 |
| **C K2 family (K2-Think-V2)** | **~10.0 GiB** | **~80.0 GiB** | **262,144** |

---

## 8. Accuracy / Quality Tradeoff

- **Expected quality delta vs baseline**: Near-identical perplexity to full softmax at sufficient head coverage — adaptive softmax reports 43.9 PPL vs 43.7 PPL for an 8× larger model on One Billion Word [Grave et al., ICML 2017, 3, §Experiments]; adaptive input representations reach WikiText-103 PPL 18.7 / One Billion Word 23.02 [Baevski & Auli, ICLR 2019, 8]; FR-Spec: 75% LM-head compute reduction with 1.12× speedup over EAGLE-2 [Zhao et al., ACL 2025, 1]; VOCABTRIM: 16% memory-bound speedup on Llama-3.2-3B-Instruct with slight acceptance-rate degradation [Goel et al., ICML 2025 Workshop, 2]; Ben Shoham: up to 97% vocab reduction at high coverage, with Pareto knee around 90–95% [Ben Shoham, 13, arXiv:2603.05210].
- **Known failure modes**: Steep quality cliff below ~85% head coverage — tail tokens (rare proper nouns, scientific terms, multilingual tokens, rare code identifiers) are disproportionately domain-critical; acceptance-rate degradation in speculative decoding when target token is absent from the head [VOCABTRIM, 2]; `<next_dict>` sentinel blocks gradient flow and undertrains tail-tier representations during end-to-end training (no analogue in adaptive softmax, which uses a differentiable cluster-selector head) [2.1 §3].
- **Empirical evidence**: Adaptive Softmax §Experiments (43.9 PPL, matched 8× larger baseline) [Grave et al., 3]; FR-Spec §Results (75% LM-head reduction, 1.12× speedup) [1]; VOCABTRIM §Results (16% speedup, acceptance-rate tradeoff) [2]; Ben Shoham Pareto analysis (knee at 90–95% coverage) [13]; Chung & Kim NeurIPS 2025 (~75% benchmark tokens from top-2500 words) [12]; Adaptive Input Representations (WT-103 PPL 18.7) [Baevski & Auli, 8].
- **Mitigations**: Size D1 to cover ≥95% of probability mass (~3,000–5,000 head tokens for standard LLM tokenizers) — matches benchmark-indistinguishable quality [Chung & Kim, 12; Ben Shoham, 13]; apply joint training with the cascade head (as adaptive softmax does) to eliminate cluster-selection quality cost [Grave et al., 3]; prefer adaptive-softmax-style cluster-selector head over `<next_dict>` sentinel to retain differentiability and avoid inference-time filtering; scope deployment to speculative-decoding draft models (LM head = 30% of weight) where net speedup absorbs modest acceptance degradation [FR-Spec, 1]; avoid use on 32B+ chat models where LM head is 2.4–4.7% of decode bandwidth (net gain too small to absorb quality risk).

### Detailed breakdown

- **Reported quality delta (from closest analogues)**:
  - **Adaptive Softmax (Grave et al., ICML 2017)[3]**: On One Billion Word benchmark, adaptive softmax achieves 43.9 PPL vs. a model 8× larger (trained on 32 GPUs for 3 weeks) achieving 43.7 PPL — a difference of +0.2 PPL. At a smaller scale, the same paper reports near-identical perplexity to full softmax across EuroParl and One Billion Word benchmarks for the head/tail cascade, with the quality cost attributed almost entirely to rare-token coverage, not to the cascade mechanism itself.
  - **FR-Spec (Zhao et al., ACL 2025)[1]**: Achieves 75% reduction in LM head computation with a 1.12× end-to-end speedup over EAGLE-2. The acceptance-rate impact of vocabulary restriction is small enough to yield a net positive speedup — the quality cost of head-only vocabulary is offset by faster draft generation. However, acceptance rate is the sensitivity metric: any vocabulary omitting the required token forces a rejection.
  - **VOCABTRIM (Goel et al., ICML 2025 Workshop)[2]**: 16% memory-bound speedup on Llama-3.2-3B-Instruct, but explicitly reports an acceptance-rate tradeoff — limiting vocabulary "slightly degrades acceptance rate." The magnitude of degradation scales with how many tail-vocabulary tokens appear in generation targets.
  - **Ben Shoham (arXiv:2603.05210, 2026)[13]**: Reports up to 97% vocabulary size reduction with "high coverage maintenance," but on domain-specific tasks. Up to 16% latency reduction and 20% throughput gains. The critical finding is that coverage degrades non-monotonically with vocabulary reduction: the Pareto frontier shows a knee in the curve around 90–95% coverage, after which further reduction causes steep acceptance degradation.
  - **Chung & Kim (NeurIPS 2025)[12]**: Larger vocabularies reduce cross-entropy loss almost exclusively for the ~2,500 most frequent words (comprising ~75% of downstream benchmark tokens), while loss on rare tokens *rises* with larger vocabulary. This implies that a D1 tier of ~2,500 tokens achieves nearly full benchmark-level quality, with the remaining 25% of benchmark tokens (all tail vocabulary) being the quality-sensitive component.
  - **Adaptive Input Representations (Baevski & Auli, ICLR 2019)[8]**: Achieves WikiText-103 PPL of 18.7 and One Billion Word 23.02 — state-of-the-art at time of publication — using tied frequency-stratified adaptive input/output embeddings. The quality benefit of matched adaptive input representations fully compensates for any quality cost of the cascade output structure.

- **Monotonicity**: Quality loss is **not monotone** with the aggressiveness of the D1 vocabulary cutoff. At high coverage (|D1| covering ≥90% of probability mass), quality loss is near-zero on aggregate benchmarks because ~75% of benchmark tokens belong to the head tier. Quality then degrades sharply as coverage falls below ~85% because tail tokens in the last 15% are disproportionately domain-critical, rare names, numbers, or specialized terminology — these are the tokens that matter most for factual accuracy. Ben Shoham's Pareto analysis[13] confirms the non-monotone sensitivity: latency improvements continue to accrue past the quality cliff.

- **Recovery**: Full quality is recoverable by simply *including* sufficient vocabulary in D1. Unlike weight compression, there is no approximation: if a token is in D1, it is evaluated at full precision. If it is absent from D1, it is never a candidate until D2 is triggered. Recovery therefore requires adjusting the cascade tier sizes and recomputing the normalization — no fine-tuning required. Adaptive softmax[3] demonstrates that joint training with the cascade head produces end-to-end quality indistinguishable from full softmax at sufficient coverage.

- **Conditions for acceptable degradation**:
  - The tradeoff is most acceptable for **speculative decoding draft models** (300M–1B parameters), where the LM head constitutes ~30% of total weight. At this scale, FR-Spec[1] demonstrates net positive speedup with minimal quality cost. The 76× bandwidth reduction on a 30% component yields ~1.43× net TPOT improvement — a large enough gain to absorb modest acceptance-rate degradation.
  - The tradeoff is least favorable for **general-purpose chat/instruction-following models at 32B+ scale**, where the LM head is only ~2.4–4.7% of total decode bandwidth. The net TPOT improvement is 2.4–5.9%, which may not justify the quality risk on rare tokens.
  - **Tail-token quality is the primary sacrificed property.** Rare proper nouns, scientific terms, code tokens for unusual identifiers, and multilingual tokens outside the training distribution are most affected. Applications where rare-token precision is low-stakes (classification, summarization with common vocabulary) are best suited. Applications requiring high recall on rare entities (entity extraction, technical QA) are least suited.
  - A D1 covering ≥95% of probability mass (approximately the top-3,000–5,000 tokens for standard LLM tokenizers) is likely to produce benchmark-indistinguishable quality, as Chung & Kim[12] and Ben Shoham[13] jointly support. Below 90% coverage, quality risk on specialized domains becomes significant.
  - The `<next_dict>` sentinel design (vs. adaptive softmax's cluster-selector head) introduces an additional quality risk: the discrete boundary blocking gradient flow to D2 may cause undertraining of tail-tier representations during end-to-end training, further degrading rare-token quality beyond what the cascade structure alone would predict.

<!-- CITATION MANIFEST:
FR-Spec: Accelerating Large-Vocabulary Language Models via Frequency-Ranked Speculative Sampling | https://arxiv.org/abs/2502.14856 | [1] Zhao et al. ACL 2025; 75% LM Head compute reduction; 1.12× speedup over EAGLE-2; frequency-ranked vocabulary head for speculative decoding
VOCABTRIM | https://arxiv.org/abs/2506.22694 | [2] Training-free vocabulary pruning for speculative decoding; 16% speedup on Llama-3.2-3B-Instruct
Efficient Softmax Approximation for GPUs | https://arxiv.org/abs/1609.04309 | [3] Grave et al. ICML 2017; head/tail cascade; 2–10× training speedup; reference implementation at github.com/facebookresearch/adaptive-softmax
Hierarchical Probabilistic Neural Network Language Model | https://proceedings.mlr.press/r5/morin05a.html | [4] Morin & Bengio 2005; foundational hierarchical softmax; O(log V) via binary tree
A Scalable Hierarchical Distributed Language Model | https://papers.nips.cc/paper/2008/hash/1e056d2b0ebd5c878c550da6ac5d3724-Abstract.html | [5] Mnih & Hinton NeurIPS 2008; data-driven tree construction
Extensions of Recurrent Neural Network Language Model | doi:10.1109/ICASSP.2011.5528 | [6] Mikolov et al. ICASSP 2011; class-based two-level frequency cascade
Strategies for Training Large Vocabulary Neural Language Models | https://arxiv.org/abs/1512.04906 | [7] Chen et al. ACL 2016; Differentiated Softmax; frequency-stratified embedding dimensions
Adaptive Input Representations for Neural Language Modeling | https://arxiv.org/abs/1809.10853 | [8] Baevski & Auli ICLR 2019; frequency-stratified input+output; partial tying; WikiText-103 PPL 18.7
Learning to Screen for Fast Softmax Inference | https://arxiv.org/abs/1810.12406 | [9] Chen et al. ICLR 2019; screening model; 20.4× speedup on 25K-vocab translation
Breaking the Softmax Bottleneck | https://arxiv.org/abs/1711.03953 | [10] Yang et al. ICLR 2018; Mixture of Softmaxes; parallel vs. sequential cascade contrast
Using the Output Embedding to Improve Language Models | https://arxiv.org/abs/1608.05859 | [11] Press & Wolf EACL 2017; weight-tied output embeddings; standard reference for embedding tying
Exploiting Vocabulary Frequency Imbalance in Language Model Pre-training | https://arxiv.org/abs/2508.15390 | [12] Chung & Kim NeurIPS 2025; ~75% benchmark tokens from top-2500 words; D1 sizing grounding
Balancing Coverage and Draft Latency in Vocabulary Trimming | https://arxiv.org/abs/2603.05210 | [13] Ben Shoham arXiv 2026; 97% vocab reduction at high coverage; Pareto frontier analysis
LLM Vocabulary Compression for Low-Compute Environments | https://arxiv.org/abs/2411.06371 | [14] Vennam et al. NeurIPS 2024 Workshop; BPE-based LM head grouping; 3.4× memory compression, 3× throughput; static grouping ceiling relevant to cascade design
-->
