# Research: Swappable / Hot-Swappable Experts
## ID: 3.2

## 1. Idea Description

**From arch_research_ideas.md (Section 3, idea 3.2):**

> Modular experts that can be added, removed, or replaced post-training without retraining the whole model. Enables incremental knowledge updates and domain specialization.

**Inferred intent:** The idea targets operational flexibility and knowledge lifecycle management rather than inference speedup. A deployed MoE model should accept a new expert weight blob (a single FFN block) without requiring a full model reload, retraining, or router retraining. Primary value propositions: (a) domain specialization post-deployment by swapping in purpose-trained expert weights; (b) knowledge updates replacing stale domain experts; (c) capacity management by removing low-utility experts to reduce memory footprint. The inference-time compute profile is strictly unchanged from standard MoE because the same k experts are active per token regardless of which expert weights occupy which slots.

**Prerequisite:** Requires a MoE backbone with addressable expert slots. Baseline A1 (dense hybrid) and Baseline A2 (dense) do not have MoE expert structure. Idea 3.2 is directly applicable to architectures like Baseline B (Qwen3.5-397B-A17B, 512 experts) or any MoE built via sparse upcycling from a dense checkpoint.

---

## 2. Executive Summary

**Novelty verdict:** EXISTS — the remaining gap is narrow: zero-shot production hot-swap of full FFN expert tensors in a live LLM-scale inference server with no router retraining or restart; algorithmic components (expert addition, removal, zero-shot routing after swap) are all published ([BTX, 2024], [Nexus, 2024], [DES-MoE, 2024], [LoRAMoE, 2024], [PHATGOOSE, 2024]).

Swappable experts (idea 3.2) is an **operational architecture** modification, not a compute-efficiency one. TTFT and TPOT are strictly identical to the reference MoE baseline at all times during inference — the benefit is captured entirely in the update path (cost of adding new domain knowledge) and in deployment flexibility (no full model reload for domain updates).

Prior art coverage is approximately 80%. Individual mechanisms — expert addition (BTX, Nexus), expert removal (Lu et al.), forgetting prevention (DES-MoE), tuning-free steering (Free-MoE, DSMoE), and MoE creation from dense checkpoints (Sparse Upcycling, Drop-Upcycling) — are all published at small-to-medium scale. Two previously uncited papers substantially narrow the stated novelty gap: LoRAMoE[22] provides a practical PEFT realization of swappable LoRA experts, and PHATGOOSE[23] provides zero-shot gate vectors for routing after a swap, eliminating the router-retraining requirement algorithmically.

The true remaining gap is **zero-shot production hot-swap at LLM scale in a live inference server** — specifically, replacing a single expert weight tensor in a running MoE inference server with no router retraining, no continued training, and no server restart, while maintaining routing quality. All published work either requires some router fine-tuning after insertion or applies to adapter-scale modules, not full FFN expert blocks. The systems engineering of live expert replacement in vLLM-scale serving infrastructure has not been peer-reviewed.

The expert swap bandwidth per layer-stack is approximately **~960 MB** per layer-stack, well within H100 VRAM; the feasibility conclusion is unchanged.

---

## 3. Literature Review

### Modular Deep Learning
Pfeiffer et al., 2023 — TMLR survey, arXiv:2302.11529, §2 "Taxonomy", §4 "Module Composition", §6 "Training Modules"[1]

Comprehensive survey defining the four modularity dimensions: computation, routing, aggregation, and training. Explicitly discusses post-hoc module addition, removal, and composition as first-class modularity properties. Establishes the conceptual case for hot-swappable experts as a natural extension of modular architectures.

### Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints
Komatsuzaki et al., 2023 — ICLR, arXiv:2212.05055, §3 "Sparse Upcycling", Table 1[2]

Initializes a sparsely activated MoE model from a dense checkpoint by copying FFN weights into all expert slots, then training with sparse routing. T5 upcycled models significantly outperform dense counterparts using only ~50% of initial pretraining compute. The post-training MoE creation pattern directly enables expert-level modularity without training from scratch. The upcycling step requires ~50% of pretraining compute for continued training after injection.

### Branch-Train-Merge: Embarrassingly Parallel Training of Expert Language Models
Li et al., 2022 — arXiv:2208.03306, §2 "Branch-Train-Merge", Table 2[3]

Trains domain-specialized Expert Language Models independently in parallel, then combines at inference via ensembling or merging. New ELMs can be added (new domain) or removed (obsolete domain) without retraining the whole ensemble. Scaling to 64 domains (192B tokens, 22.4B total) performs as well as a dense LM trained with 2.5× more compute. Uses full-model ELMs (not FFN sub-experts in a shared backbone); no learned sparse router.

### Branch-Train-MiX: Mixing Expert LLMs into a Mixture-of-Experts LLM
Sukhbaatar et al., 2024 — arXiv:2403.07816 (Meta), §3 "Branch-Train-MiX", Table 1[4]

Extends BTM by inserting independently trained expert LLMs' FFN weights as experts in a unified MoE, then fine-tuning a learned router. Starting from Llama-2 7B: +18.8 points on math, +13.2 points on coding, +3.6 points on world knowledge. The clearest published example of "swapping in" a domain-trained FFN block into an existing shared backbone. Router fine-tuning step costs ~2% of original pretraining compute.

### Nexus: Specialization meets Adaptability for Efficiently Training Mixture of Experts
Gritsch et al., 2024 — arXiv:2408.15901, §3 "Nexus", §4 "Extending MoE with New Experts", Table 1[5]

Domain-conditioned MoE router where expert embeddings project from pre-computed domain embeddings. New experts can be added by training a new dense domain model and inserting its weights — the router projection scales automatically without full MoE retraining. Achieves 2.1% relative gain over baseline upcycling and 18.8% relative gain when extending with a new expert. Architecturally closest to true hot-swap: the routing mechanism is extensible by construction.

### Mod-Squad: Designing Mixture of Experts As Modular Multi-Task Learners
Chen et al., 2023 — CVPR, arXiv:2212.08066, §3.3 "Expert Specialization Loss", Table 2[6]

Mutual-dependence loss that encourages task-to-expert assignment specialization in MoE. Task-specific expert subsets can be extracted standalone; removing the majority of extra experts incurs <0.3% performance loss. Validates the "remove" direction of hot-swapping — expert removal incurs negligible quality loss if the right experts are retained.

### Not All Experts are Equal: Efficient Expert Pruning and Skipping for MoE LLMs
Lu et al., 2024 — ACL, arXiv:2402.14800, §3 "Expert Pruning", §4 "Dynamic Expert Skipping", Tables 2–3[7]

Post-training expert pruning and dynamic expert skipping. Expert importance scores computed from small calibration sets (no full retraining). ~1.33× inference speedup with ~2.9 points performance drop for 2-expert removal. Domain-specific pruning retains half the experts while maintaining domain performance. Addresses the "remove" half of hot-swap.

### Editing Models with Task Arithmetic
Ilharco et al., 2023 — ICLR, arXiv:2212.04089, §3 "Task Arithmetic", §4.2 "Adding task vectors", Table 1[8]

Task vectors (pretrained − fine-tuned weight deltas) can be added/negated/combined arithmetically. A "swap" is mathematically equivalent to negating the old expert's task vector and adding the new one — works without additional training. Performance degrades as more task vectors are combined (weight interference). Applied to full model, not individual expert slots.

### Drop-Upcycling: Training Sparse Mixture of Experts with Partial Re-initialization
Nakamura et al., 2025 — ICLR, arXiv:2502.19261, §3 "Drop-Upcycling", Table 1[9]

Combines dense checkpoint initialization with partial random re-initialization of expert weights to promote specialization divergence. Significantly improves expert specialization over standard upcycling. Practical recipe for initializing replacement experts with sufficient diversity to specialize productively. Partially discards pretrained knowledge; undesirable for replacement experts meant to preserve prior capability.

### Boosting Continual Learning of Vision-Language Models via Mixture-of-Experts Adapters
Yu et al., 2024 — CVPR, arXiv:2403.11549, §3 "Incremental MoE-Adapters", Table 1[10]

Incremental MoE-Adapters for continual learning with CLIP. Each new task adds a new adapter expert; the pool grows incrementally. Demonstrates the "add expert" use case for continual learning without retraining existing experts. Vision-language domain; experts are lightweight adapters rather than full FFN blocks.

### LoRA-Switch: Boosting the Efficiency of Dynamic LLM Adapters via System-Algorithm Co-design
Kong et al., 2024 — arXiv:2405.17741, §3 "LoRA-Switch", §4 "System Design", Table 2[11]

Token-wise routing across multiple LoRA adapters fused into a single CUDA kernel. Recovers 2.5× slowdown from dynamic adapters to only ~1.04× latency vs static. The closest published engineering solution to hardware-efficient hot-swap, though at LoRA adapter scale rather than full FFN expert blocks.

### Dynamic Expert Specialization: Towards Catastrophic Forgetting-Free Multi-Domain MoE Adaptation (DES-MoE)
Li et al., 2025 — EMNLP, aclanthology 2025.emnlp-main.932, §3 "DES-MoE Framework", Tables 1, 3[12]

Adaptive router + distillation + domain-gradient isolation framework. Reduces catastrophic forgetting by 89% vs full fine-tuning across 2–6 domains, with 68% faster convergence. Gradient isolation (routing domain-specific gradients only to domain-specific experts) is architecturally equivalent to what a hot-swap system needs. Requires fine-tuning (not zero-shot); evaluated up to ~7B parameters.

### Do Domain-Specific Experts Exist in MoE-based LLMs? (DSMoE)
Deakin A2I2 researchers, 2026 — arXiv:2604.05267 (very recent preprint, not yet peer-reviewed), §3 "Empirical Analysis", §4 "DSMoE Framework", Table 2[13]

Evaluates 10 MoE LLMs (3.8B–120B) for domain-specific expert existence. Proposes DSMoE: a training-free framework amplifying domain-specific experts without retraining. Provides empirical validation that domain-specific experts genuinely exist in current MoE LLMs (Mixtral, DeepSeek-MoE), validating the premise of hot-swappable experts.

### What Gets Activated: Uncovering Domain and Driver Experts in MoE Language Models
[Author et al.], 2026 — arXiv:2601.10159 (January 2026 preprint), §3 "Expert Taxonomy", §4 "Interventional Analysis"[14]

Finds that MoE experts fall into two categories: domain experts (consistently activated for specific content) and driver experts (causally influential on output quality regardless of domain). Informs which experts are safe to replace (domain) vs. which would degrade global performance (driver). Observational study; no expert replacement experiments.

### Free-MoE: Tuning-Free Mixture-of-Experts Purifying LLMs to Thrive across Any Field
[Author et al.], 2025 — ICLR 2025, OpenReview J8LYjgi7nH, §3 "DOWP Algorithm", Table 1[15]

Tuning-free domain-oriented weight purification achieving 2–6.8% gains on MMLU/HumanEval/GSM8K without fine-tuning. Demonstrates that expert-level weight manipulation at inference time yields measurable quality improvements. Constrained to capabilities already in the base model.

### Model Merging in LLMs, MLLMs, and Beyond (Survey)
[Author et al.], 2024 — ACM Computing Surveys, arXiv:2408.07666, §3 "Component-Level Merging", §5 "Expert Composition", §7 "Continual Learning"[16]

Comprehensive survey of model merging techniques including task arithmetic, TIES-merging, DARE, and expert merging. Model merging provides the theoretical basis for "replace an expert" via composition with residual traces of the old expert.

### Theory on Mixture-of-Experts in Continual Learning
Li et al., 2024 — ICLR 2025 (Spotlight), arXiv:2406.16437, §3 "Theoretical Framework", Theorem 1[17]

Proves that MoE architectures mitigate catastrophic forgetting in continual learning by routing to specialized experts, preventing gradient interference. Provides theoretical grounding for hot-swap viability as a continual learning mechanism under distributional assumptions.

### Little By Little: Continual Learning via Incremental Mixture of Rank-1 Associative Memory Experts (MoRAM)
[Author et al.], 2025 — arXiv:2506.21035, §3 "MoRAM Framework", §4 "Self-Activation Mechanism"[18]

Rank-1 expert key-value pairs with self-activation (content-addressable routing). New expert slots activate via intrinsic key, eliminating need for a separate router. Old experts frozen; new knowledge added without touching existing parameters. Represents the most granular implementation of hot-swappable experts; behavior at LLM scale with many accumulated experts not yet validated.

### LoRAMoE: Alleviating World Knowledge Forgetting in Large Language Models via MoE-Style Plugin
Dou et al., 2024 — ACL, arXiv:2312.09979, §3 "LoRAMoE", Table 1[22]

Freezes the LLM backbone and adds multiple LoRA modules as MoE experts with token-wise routing. The de-facto practical realization of swappable experts using PEFT — directly implements "add a new task expert without touching others." Each LoRA expert can be independently added or replaced. Widely deployed in the PEFT community.

### PHATGOOSE: Learning to Route Among Specialized Experts for Zero-Shot Generalization
Muqeeth et al., 2024 — arXiv:2402.05859, §3 "PHATGOOSE", §4 "Zero-shot Routing", Table 2[23]

Provides zero-shot gate vectors for routing to merged/swapped experts, requiring no router retraining. Trained-once gate tokens that activate when the corresponding expert is relevant, without requiring router fine-tuning after expert insertion. Narrows the prior art gap: zero-shot routing for swapped experts is algorithmically demonstrated.

### Fast Inference of Mixture-of-Experts Language Models with Offloading
Eliseev & Mazur, 2023 — arXiv:2312.17238[24]

Proposes expert parameter offloading to CPU/NVMe with prefetch for consumer-hardware MoE inference. Demonstrates that Mixtral-8×7B runs on a single commodity GPU by treating expert weights as a CPU-resident cache loaded on demand. Establishes the empirical bandwidth and latency bounds for expert weight transfer across PCIe — the direct systems-level prior art for hot-swap I/O cost modeling.

### Fiddler: CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts Models
Kamahori et al., 2024 — ICLR 2025, arXiv:2402.07033[25]

Profiles arithmetic intensity of attention vs. expert layers to assign high-intensity work to GPU and expert computation to CPU, reducing VRAM requirements without throughput collapse. The CPU-GPU scheduling approach is directly applicable to designing a hot-swap buffer that stages incoming expert weights on CPU while the GPU completes in-flight requests.

---

## 4. Prior Art Classification

**Status: EXISTS (~80% overlap).** Individual mechanisms are well-published:
- Expert addition post-training: BTX[4], Nexus[5], MoE-Adapters[10], MoRAM[18]
- Expert removal: Expert Pruning[7], Mod-Squad[6]
- Expert replacement via weight composition: Task Arithmetic[8], Model Merging Survey[16]
- Domain-expert steering without replacement: DSMoE[13], Free-MoE[15]
- MoE creation from dense: Sparse Upcycling[2], Drop-Upcycling[9]
- Zero-shot routing for swapped experts: PHATGOOSE[23] — closes router-retraining gap algorithmically
- Practical swappable PEFT experts: LoRAMoE[22]
- Catastrophic-forgetting-free multi-domain MoE: DES-MoE[12]

**Restated novel contribution (narrowed):** The remaining gap is **zero-shot production hot-swap at LLM scale within a live inference server** — specifically, replacing a full FFN expert weight tensor in a running MoE inference server with no router retraining, no continued training, no server restart, while maintaining routing quality and supporting concurrent requests. The algorithmic components individually exist; the systems engineering integration at production LLM scale (Baseline B-class, 512 experts, vLLM serving) has not been peer-reviewed.

---

## 5. Technical Analysis

### 5.1 Theoretical Complexity

**Notation:** L = layers; E_total = expert slots per MoE layer; E_new = experts being swapped (typically 1); k = active experts per token; d = hidden dim; d_e = per-expert FFN intermediate dim (≈ 1024 for Baseline B); s = sequence length.

**Key insight: swapping experts does NOT change inference FLOPs or memory bandwidth per token.** The number of active experts k and architecture of each expert (d × d_e) are unchanged. TTFT and TPOT are strictly equal to the reference MoE baseline during inference.

**Expert swap I/O bandwidth:**
For a single-expert swap across all L layers in Baseline B config (L=60, d=4096, d_e≈1024):
- Per-layer expert: 2 × d × d_e × 2 bytes (up+down, bf16) = 2 × 4096 × 1024 × 2 ≈ 16 MB
- Across all 60 layers: ~960 MB per expert — approximately **~960 MB** per expert swap
- At NVSwitch bandwidth 900 GB/s: ~1.6 ms for single-expert hot-swap — negligible vs inference latency

**Expert storage library:** For M=10 domain variants across 512 experts: 10 × 512 × ~960 MB ≈ **~4.9 TB** expert storage on disk/object store.

**Router retraining cost:** ~2% of pretraining compute if router retraining required (BTX[4]). Zero with PHATGOOSE zero-shot routing[23] or DSMoE tuning-free steering[13].

### 5.2 Key Comparison Tables

**vs. Baseline A1 (Qwen3.5-27B Hybrid)**

| Metric | Baseline A1 | Idea 3.2 | Change | Notes |
|--------|------------|----------|--------|-------|
| Compute (FLOPs/token) | O(L·(d²+s·d/4)) | N/A — requires MoE backbone | N/A | A1 is dense hybrid; hot-swap not applicable without upcycling to MoE first |
| KV cache (32K ctx) | ~2.15 GB | N/A (same if upcycled) | N/A | No change to KV if MoE architecture preserved |
| TTFT | ref | = ref (MoE) | = | Swapping experts does not change inference computation |
| TPOT | ref | = ref (MoE) | = | Swapping experts does not change inference computation |
| Expert swap bandwidth | N/A | O(L·E_new·d·d_e) bytes one-time | New metric | One-time I/O, not per-token cost |

**vs. Baseline A2 (Qwen3-32B Dense)**

| Metric | Baseline A2 | Idea 3.2 | Change | Notes |
|--------|------------|----------|--------|-------|
| Compute (FLOPs/token) | O(L·(s·d+d·d_ff)) | O(L·(d²+k·d·d_e)) | ↓ ~46.5× at k=11,E=512 | This is the MoE architecture benefit, not specific to hot-swap |
| KV cache (262K ctx) | ~68.7 GB | = ref MoE | = | KV unchanged by hot-swap |
| Weight memory | ~64 GB bf16 | O(L·E·d·d_e) | ↑ significantly | More total stored weights (MoE overhead) vs dense |
| TTFT | ref | = ref | = | Swapping experts does not change inference computation |
| TPOT | ref | = ref (vs same-k MoE) | = | Hot-swap adds no TPOT improvement beyond MoE baseline |
| Training cost (subsequent domain updates) | 1.0× full retrain | ~2% via router fine-tuning; ~0% via PHATGOOSE/DSMoE | ↓ dramatically | Key operational benefit |

**vs. Baseline B (Qwen3.5-397B-A17B MoE)**

| Metric | Baseline B | Idea 3.2 | Change | Notes |
|--------|-----------|----------|--------|-------|
| Compute (FLOPs/token) | O(L·(d²+k·d·d_e)) | = ref | = | Identical during inference |
| KV cache (262K ctx) | ~8.0 GB | = ref | = | Identical |
| Weight memory | O(L·E·d·d_e) | = ref | = | Same VRAM; + expert library ~4.9 TB on disk |
| TTFT | ref | = ref | = | Identical |
| TPOT | ref | = ref | = | Identical |
| Knowledge update cost | 1.0× full retrain | ~1/512 of expert weights loaded one-time (~960 MB) | ↓ 512× I/O vs full reload | Primary benefit — update a single domain without model reload |
| Expert swap bandwidth | N/A | ~960 MB one-time I/O | New metric | Well within H100 80 GB VRAM |

**vs. Baseline C (K2 family, 72.55B dense Llama-arch)**

| Metric | Baseline C (K2) | Idea 3.2 | Change | Notes |
|--------|----------------|----------|--------|-------|
| Compute (FLOPs/token) | O(80·(s·d+d·d_ff)), d=8192, d_ff=28672 | = ref MoE baseline | N/A direct comparison | K2 is dense; idea 3.2 applies to MoE architectures only |
| KV cache (32K ctx) | ~10.0 GiB | = ref MoE | N/A | MoE KV depends on MoE backbone architecture |
| Knowledge update bandwidth | ~145.1 GB (full bf16 reload) | ~960 MB per expert swap (if MoE-based) | ↓ dramatically | If K2-family were upcycled to MoE with 512 experts, per-expert swap is ~1/512 of model |
| TTFT, TPOT | ref | = ref | = | No change at inference time |
| MLP FLOPs per token/layer | ~4.70 × 10⁸ | ≈ k/E_total × d_ff_total | ↓ at k<<E | MoE routing benefit over dense K2 |

---

## 6. Implementation Considerations

**Hardware requirements:** No specialized hardware required for inference after swap. The swap operation is a weight tensor copy from storage to VRAM. For NVLink-connected multi-GPU setups (tensor-parallel), a swap requires updating the relevant shard on each GPU — ~188 MB per GPU shard on a DGX H100, ~0.2 ms via NVSwitch.

**Framework support:**
- PyTorch: `nn.Parameter.copy_()` or `torch.load()` + parameter assignment — full support
- vLLM v0.6+: Weight update API exists; expert-level granularity requires minor customization — NEAR-TERM FEASIBLE
- TensorRT-LLM: Engine recompilation required for weight changes — BARRIER for hot-swap
- JAX: Parameter replacement is first-class (`tree_unflatten`) — full support

**Router stability after expert insertion:**

| Method | Retraining Cost | Quality | Feasibility |
|--------|----------------|---------|-------------|
| BTX router fine-tuning[4] | ~2% pretraining compute | High | FEASIBLE at 7B; unknown at 397B |
| Nexus domain-conditioned router[5] | Minimal (projection update) | High | FEASIBLE |
| PHATGOOSE zero-shot routing[23] | None | Good | FEASIBLE — no retraining |
| DSMoE routing bias[13] | None (training-free) | Limited to existing capabilities | FEASIBLE |
| MoRAM self-activation[18] | None (content-addressable) | Good for rank-1 granularity | FEASIBLE at small scale |

**Critical missing risk — hidden state compatibility:** Replacement experts trained from a different base model will have incompatible residual stream statistics, causing routing degradation and quality loss. All replacement experts must be trained from the same base model backbone as the MoE they are inserted into. Cross-model expert transplantation requires explicit alignment.

**Rollback capability:** Hot-swap enables sub-millisecond rollback to old expert weights if quality degrades after a swap — a significant production advantage not available with full model retraining. This is enabled by maintaining the old expert weights in VRAM or fast NVMe until the new expert's routing quality is confirmed.

**A/B traffic splitting:** With 3.3 (Dynamic Expert Router), new and old experts can be routed to probabilistically in a blend during a burn-in period, enabling gradual rollout before full commit.

---

## 7. Synergies

- **3.3 (Dynamic Expert Router):** 3.2 and 3.3 are complementary — 3.2 handles the weight-management problem (which weights occupy which slots), 3.3 handles the routing-update problem (how the router adapts to new experts). Together they form a complete modular MoE lifecycle system.
- **4.2 (Shared Core + Per-Layer LoRA):** Expert weights decomposed as shared core + per-expert LoRA delta; swapping an expert means only replacing the small LoRA delta, dramatically reducing swap bandwidth.
- **4.3 (LoRA Everywhere):** If all MoE experts are parameterized as base + LoRA, hot-swap becomes swapping only the LoRA component (<<1% of weight bytes). LoRA-Switch[11] demonstrates this at inference time.
- **1.1 (Learnable Top-k):** Freshly inserted expert can be ramped up by temporarily increasing k during a burn-in period, then returned to normal k.

**Conflicts:**
- Static compilation (TensorRT-LLM): Engine compilation bakes in specific weight values; hot-swap requires either dynamic-engine support or a recompile step.
- Full model quantization: Expert-specific quantization parameters computed jointly across all experts are invalidated by replacing one expert.

---

## Risk Assessment

**Technical risk: LOW-MEDIUM** — Individual components (expert insertion, removal, routing adaptation) are all published. The main unresolved question is router quality after zero-shot insertion at 397B scale without router retraining. PHATGOOSE[23] and DSMoE[13] address this algorithmically; no production-scale demonstration exists.

**Potential impact: MEDIUM-HIGH** — The value is primarily operational. Hot-swappable experts enable: (a) post-deployment knowledge updates without full model retraining; (b) domain specialization for enterprise/vertical deployments; (c) reduced knowledge staleness in long-lived deployments. These are real production pain points.

**Implementation effort: MEDIUM** — Proof-of-concept feasible in weeks using vLLM's weight update API with a smaller MoE (e.g., Mixtral 8×7B). Production-grade system with API endpoint, router calibration, concurrent-request safety, and rollback capability requires moderate engineering effort (6–12 months) but no novel algorithms.

**Recommended validation path:**
1. Implement proof-of-concept hot-swap on Mixtral 8×7B using vLLM weight update API with PHATGOOSE zero-shot routing
2. Evaluate routing quality before/after swap on domain-specific benchmarks (no router retraining)
3. Validate sub-millisecond rollback capability
4. Scale to Baseline B (397B) scale with Nexus domain-conditioned routing

---

<!-- CITATION MANIFEST -->
[1]: Modular Deep Learning — Pfeiffer et al., 2023 (TMLR, arXiv:2302.11529). Survey of modularity taxonomy; establishes post-hoc module add/remove/compose as first-class properties.
[2]: Sparse Upcycling — Komatsuzaki et al., 2023 (ICLR, arXiv:2212.05055). MoE creation from dense checkpoints via expert copy + routing training; ~50% of pretraining compute.
[3]: BTM — M. Li et al., 2022 (arXiv:2208.03306). Embarrassingly parallel domain expert training + merge; 64-domain scaling at 2.5× compute efficiency vs dense LM.
[4]: BTX — Sukhbaatar et al., 2024 (arXiv:2403.07816, Meta). Domain-expert FFN insertion into MoE + router fine-tuning; +18.8 math, +13.2 coding, +3.6 world knowledge vs Llama-2 7B.
[5]: Nexus — Gritsch et al., 2024 (arXiv:2408.15901). Domain-conditioned router extensible by design; 18.8% relative gain when extending with a new expert.
[6]: Mod-Squad — Chen et al., 2022 (CVPR 2023, arXiv:2212.08066). Task-specific expert specialization loss; expert subset extraction with <0.3% performance loss on 13 vision tasks.
[7]: Expert Pruning/Skipping — Lu et al., 2024 (ACL, arXiv:2402.14800). Post-training expert removal; ~1.33× speedup with ~2.9pt drop for 2-expert removal; domain-specific pruning retains 100% domain performance.
[8]: Task Arithmetic — Ilharco et al., 2023 (ICLR, arXiv:2212.04089). Weight delta arithmetic for post-training capability addition/removal; "swap = negate old + add new" without retraining.
[9]: Drop-Upcycling — Nakamura et al., 2025 (ICLR 2025, arXiv:2502.19261). Partial re-initialization of upcycled experts to promote specialization divergence.
[10]: MoE-Adapters — Yu et al., 2024 (CVPR, arXiv:2403.11549). Incremental adapter expert addition for continual learning with CLIP.
[11]: LoRA-Switch — Kong et al., 2024 (arXiv:2405.17741). Fused CUDA kernel for token-wise routing across LoRA adapters; ~1.04× latency overhead vs static adapters.
[12]: DES-MoE — J. Li et al., 2025 (EMNLP, aclanthology). Domain-gradient isolation + distillation; 89% forgetting reduction across 2–6 domains.
[13]: DSMoE — Deakin A2I2 researchers, 2026 (arXiv:2604.05267, very recent preprint). Empirical validation of domain-specific experts in 10 MoE LLMs; training-free domain amplification.
[14]: Domain/Driver Expert Taxonomy — [Author et al.], 2026 (arXiv:2601.10159). Expert taxonomy into domain experts (safe to replace) and driver experts (high causal impact, risky to replace).
[15]: Free-MoE — [Author et al.], 2025 (ICLR 2025, OpenReview J8LYjgi7nH). Tuning-free domain weight purification; 2–6.8% gains on MMLU/HumanEval/GSM8K without fine-tuning.
[16]: Model Merging Survey — [Author et al.], 2024 (ACM Computing Surveys, arXiv:2408.07666). Comprehensive survey of model merging including expert composition methods.
[17]: Theory of MoE in Continual Learning — Li et al., 2024 (ICLR 2025 Spotlight, arXiv:2406.16437). Proof that MoE routing prevents gradient interference; provides theoretical grounding for hot-swap viability.
[18]: MoRAM — [Author et al.], 2025 (arXiv:2506.21035). Rank-1 associative memory experts with self-activation; zero-shot expert insertion via content-addressable routing.
[22]: LoRAMoE — Dou et al., 2024 (ACL 2024, arXiv:2312.09979). LoRA modules as MoE experts with token-wise routing; de-facto practical realization of swappable PEFT experts.
[23]: PHATGOOSE — Muqeeth et al., 2024 (arXiv:2402.05859). Zero-shot gate vectors for routing to swapped experts; eliminates router retraining requirement algorithmically.
[24]: Expert Offloading (MoE) — Eliseev & Mazur, 2023 (arXiv:2312.17238). Expert weight offloading to CPU/NVMe with prefetch; establishes PCIe bandwidth and latency bounds for per-expert load operations — primary systems-level prior art for swap I/O cost modeling.
[25]: Fiddler — Kamahori et al., 2024 (ICLR 2025, arXiv:2402.07033). CPU-GPU orchestration for MoE inference via arithmetic-intensity-based partitioning; staging incoming expert weights on CPU while GPU completes in-flight requests directly informs hot-swap buffer design.

## 8. Accuracy / Quality Tradeoff

- **Reported quality delta (from closest analogues)**:
  - **BTX / Branch-Train-MiX (Sukhbaatar et al., arXiv:2403.07816, Meta)[4]**: After inserting independently trained domain-expert FFN weights into a Llama-2 7B MoE backbone and fine-tuning only the router (~2% of pretraining compute), the model achieves: +18.8 points on math benchmarks, +13.2 points on coding benchmarks, and +3.6 points on world knowledge benchmarks vs. the dense Llama-2 7B baseline. This is not a quality *loss* — expert insertion with router fine-tuning *improves* quality in the target domain. The quality cost only appears when the expert slot holds a mismatched or randomly initialized expert.
  - **Nexus (Gritsch et al., arXiv:2408.15901)[5]**: A 2.1% relative gain over standard upcycling baseline and an 18.8% relative gain when extending with a new domain expert. The domain-conditioned router architecture means that new expert insertion produces near-immediate quality gain without router retraining — the routing projection scales automatically. Quality delta on non-target domains during new expert insertion is approximately 0% due to routing isolation.
  - **Expert Pruning (Lu et al., ACL 2024)[7]**: Post-training expert removal of 2 experts from a production MoE causes ~2.9 percentage point performance drop with ~1.33× inference speedup. This quantifies the quality cost of the removal direction of hot-swap: losing ~2 of ~8 experts (25% of capacity) costs ~2.9 points. Domain-specific pruning retains half the experts while maintaining domain performance — routing isolation means per-domain quality is preserved even when 50% of experts are removed.
  - **Mod-Squad (Chen et al., CVPR 2023)[6]**: Expert subset extraction (removing non-specialist experts) incurs <0.3% performance loss on 13 vision tasks. The quality contrast between Mod-Squad (<0.3% loss) and Lu et al. (2.9 point loss) reflects the importance of training protocol — experts trained with specialization-inducing loss can be removed cleanly; experts not trained to specialize cannot.
  - **Free-MoE (ICLR 2025)[15]**: Tuning-free domain-oriented weight purification achieves 2–6.8% gains on MMLU, HumanEval, and GSM8K without any fine-tuning. Expert-level weight manipulation at inference time can produce measurable quality *improvements* on targeted evaluations, though constrained to capabilities already present in the base model.
  - **DES-MoE (Li et al., EMNLP 2025)[12]**: Domain-gradient isolation + distillation reduces catastrophic forgetting by 89% vs. full fine-tuning across 2–6 domains, with 68% faster convergence. The 89% forgetting reduction quantifies how much quality is preserved on prior domains when inserting domain-specific expert updates — this is the quality-preservation bound for the replacement direction of hot-swap.
  - **LoRAMoE (Dou et al., ACL 2024)[22]**: Frozen backbone with multiple LoRA expert modules achieves strong task performance while explicitly preventing world knowledge forgetting, validated on instruction tuning and knowledge-intensive QA. Quality on prior tasks is preserved at near-baseline when new LoRA experts are added.

- **Monotonicity**: Quality degradation from expert swapping is **not monotone** with swap aggressiveness. The relevant dimensions are:
  - **Number of experts swapped**: Swapping 1 of 512 experts in Baseline B (0.2% of slots) should have negligible routing impact. Swapping 50% of experts simultaneously risks significant routing distribution shift.
  - **Expert training compatibility**: An expert trained on the *same base model backbone* with router fine-tuning (BTX pattern) shows positive quality delta. An expert from a different model family or with incompatible residual stream statistics causes routing degradation and quality *loss* proportional to the statistical mismatch.
  - **Router method**: PHATGOOSE[23] zero-shot routing quality is good but slightly below BTX router fine-tuning — the gap depends on how far the new expert's function deviates from the prior expert's routing distribution.

- **Recovery**: Quality is recoverable through:
  - **Router fine-tuning (BTX pattern)[4]**: ~2% of pretraining compute restores and typically *exceeds* baseline quality on the target domain. Full recovery of prior-domain quality is maintained by routing isolation.
  - **PHATGOOSE zero-shot routing[23]**: Achieves good routing quality without any router retraining. Quality is slightly below the fine-tuned router but substantially above randomly initialized routing.
  - **Rollback**: Hot-swap's primary quality-safety mechanism. Because old expert weights can be retained in VRAM or fast NVMe, any quality regression is immediately reversible in sub-millisecond time — the quality tradeoff is exploratory and non-destructive.
  - Expert re-training from the correct base model backbone resolves cross-model incompatibility. The "hidden state compatibility" requirement (§6) is a hard constraint: cross-model transplants without explicit residual stream alignment cannot recover quality through router fine-tuning alone.

- **Conditions for acceptable degradation**:
  - Quality loss is acceptable and often *absent* when: the replacement expert is trained from the same base model backbone, router fine-tuning is applied (~2% pretraining compute), and the swap targets a domain-specific expert rather than a "driver expert" (high causal impact on output quality regardless of domain, per the 2026 taxonomy[14]).
  - Quality loss is most risky when: a "driver expert" is replaced, or a cross-model expert transplant is attempted without residual stream alignment, or router calibration is skipped entirely for a newly inserted expert with zero-shot routing.
  - The A/B traffic splitting approach using idea 3.3 enables gradual quality validation before full expert commitment — routing a small fraction of tokens to the new expert while monitoring quality metrics provides an empirical quality gate without disrupting production.
  - For enterprise deployments targeting domain specialization post-deployment, the quality tradeoff is highly favorable: BTX[4] demonstrates +18.8 points on the target domain at ~2% training cost, far exceeding any quality regression on non-target domains.
