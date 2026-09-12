# AI Architecture Research Ideas

## Section 1: Move Static to Dynamic (Eliminating Magic Numbers)

### 1.1 — Learnable Per-Token Top-k
**Description:** Replace the static k in top-k expert routing with a learned, per-token dynamic value. Each token determines how many experts it needs rather than using a fixed k across all tokens.
**Search:** `learnable adaptive top-k expert routing per-token mixture of experts`

### 1.2 — Per-Token Adaptive Depth
**Description:** A model with static prefix and postfix layer sets, but a dynamic middle section where per-token routing decides how many intermediate layers to execute. Tokens that are easy to process exit early; harder tokens use more compute.
**Search:** `adaptive depth early exit per-token transformer dynamic layer skipping`

### 1.3 — Per-Layer Adaptive Expert Count
**Description:** Instead of fixing the number of active experts per layer, allow each layer to dynamically choose how many experts to activate at inference time. Focus on inference-time adaptation rather than architecture search.
**Search:** `adaptive number active experts per-layer inference mixture of experts dynamic sparsity`

### 1.4 — Learned Dense vs Sparse Layer Assignment
**Description:** The model learns during training which layers should be dense and which should be sparse (MoE), rather than this being a hand-designed architectural choice.
**Search:** `learned sparse dense layer selection neural architecture search mixture of experts`

### 1.5 — Learned Sparsity Type (Unstructured vs MoE)
**Description:** Beyond choosing dense vs sparse, the model also learns *what kind* of sparsity to apply per layer — unstructured weight pruning vs structured MoE routing.
**Search:** `unstructured sparsity vs mixture of experts learned sparsity type neural network`

### 1.6 — Learned Layer Type (Transformer / SSM / Conv / Mamba)
**Description:** A hybrid model that learns which computational primitive to use at each layer position — self-attention, state space model, convolution, or Mamba block — either via NAS or dynamic inference-time selection.
**Search:** `hybrid transformer SSM mamba convolution architecture learned layer type selection`

### 1.7 — Dynamic Vocabulary / Language Size
**Description:** Per-token dynamic vocabulary where most of the vocabulary is masked/disregarded, reducing the internal representation size. The active vocabulary subset is conditioned on the input, so the model only carries forward the relevant portion of the embedding/output space.
**Search:** `adaptive vocabulary size dynamic token embedding sparse softmax language model`
**Search:** `conditional computation output vocabulary reduction large language model`

---

## Section 2: Compute Reduction

### 2.1 — Hierarchical Frequency-Based Dictionary
**Description:** A cascade of dictionaries D1→D2→…→DL where D1 contains the highest-frequency tokens plus a special `<next_dict>` token. When the model needs a less common token, it triggers loading the next dictionary level. This reduces compute by handling common tokens cheaply.
**Search:** `hierarchical vocabulary frequency dictionary cascaded embedding language model`
**Search:** `adaptive softmax hierarchical softmax frequency-based vocabulary`

### 2.2 — Compressed Dense Layers via Matrix Decomposition
**Description:** Apply matrix decomposition (low-rank factorization, SVD, etc.) to MLP weight matrices, with a residual correction to recover lost expressiveness. Reduces parameter count and FLOPs in dense feed-forward layers.
**Search:** `low-rank matrix decomposition MLP transformer compression residual correction`

---

## Section 3: New Architectures

### 3.1 — Layer-Level MoE (Full Block Routing)
**Description:** The MoE router wraps entire transformer blocks (attention + FFN + norm) rather than just the feed-forward layer. The router selects which complete architectural block to execute per token.
**Search:** `layer-level mixture of experts full block routing transformer architecture`
**Search:** `mixture of experts entire transformer layer routing sparse model`

### 3.2 — Swappable / Hot-Swappable Experts
**Description:** Modular experts that can be added, removed, or replaced post-training without retraining the whole model. Enables incremental knowledge updates and domain specialization.
**Search:** `modular expert hot-swap post-training mixture of experts continual learning`
**Search:** `replaceable expert modules neural network modularity`

### 3.3 — Dynamic Expert Router (Add/Remove Experts Post-Training)
**Description:** A router architecture that can accommodate insertion or removal of expert modules after training, automatically learning to route to new experts or bypass removed ones. Enables post-hoc knowledge editing.
**Search:** `dynamic expert insertion removal post-training router mixture of experts`
**Search:** `expandable mixture of experts adding experts after training`

### 3.4 — Recursive Internal State (Internal CoT)
**Description:** Internal chain-of-thought on the latent state — a while loop within the forward pass that iteratively refines activations. Has a fixed maximum iteration count and a learned early-exit condition. The model does "thinking" in latent space without emitting tokens.
**Search:** `iterative refinement latent chain of thought internal recurrence transformer`
**Search:** `universal transformer adaptive computation time pondering`
**Search:** `looped transformer recurrent depth iteration latent reasoning`

### 3.5 — Gated Internal DAG (Learned If/Else Control Flow)
**Description:** Within a layer, the computation graph is a DAG with learned gates that implement if/else branching. The model learns conditional control flow as part of its architecture, selecting different computation paths per input.
**Search:** `conditional computation gated directed acyclic graph neural network learned control flow`
**Search:** `dynamic neural network conditional branching learned routing`

### 3.6 — Recursive Internal DAG (Turing-Complete Flow Control)
**Description:** Combines the DAG structure (3.5) with recursion (3.4) to form a Turing-complete flow control representation within the model's forward pass — loops, branches, and state.
**Search:** `turing complete neural network learned program flow control recurrent DAG`
**Search:** `neural program induction learned control flow graph`

### 3.7 — Learnable State Machine
**Description:** Extension of 3.6 where explicit state is associated with the control flow. The model maintains and updates a learned finite state machine alongside its activations, governing computation paths.
**Search:** `learned state machine neural network differentiable finite automaton`
**Search:** `neural finite state machine learned transitions deep learning`

### 3.8 — Trainable Activation Functions (Pointwise + Windowed)
**Description:** Replace fixed activations with learned per-neuron nonlinearities: `Act = min(max(f(x), α), β)` where `f(x) = ax + b` (pointwise) or a small convolution over neighboring activations (windowed). α, β, a, b are all learned per neuron. Two variants: pointwise learnable nonlinearities and window-based learnable nonlinearities.
**Search:** `learnable activation function per-neuron trainable nonlinearity neural network`
**Search:** `KAN kolmogorov arnold network learnable activation`
**Search:** `convolutional activation function neighborhood activation neural network`

---

## Section 4: State Machine / Shared Weights

### 4.1 — State Machine Core
**Description:** A learned state machine that the model explicitly modifies during the forward pass alongside activations. The FSM state influences computation and is updated as part of inference — not just implicit in hidden states.
**Search:** `explicit learned state machine transformer forward pass differentiable automaton`

### 4.2 — Shared Core Weights + Per-Layer LoRA
**Description:** A single set of shared core weights across N layers, with each layer distinguished by its own LoRA adapter. Combines the parameter efficiency of weight sharing with per-layer specialization via low-rank corrections.
**Search:** `shared weights across layers per-layer low-rank adaptation LoRA transformer`
**Search:** `universal transformer weight sharing layer-specific adaptation`

### 4.3 — LoRA Everywhere (Full-Model Core + Adapter Split)
**Description:** All weighted operations in the entire model are decomposed into shared core weights + LoRA adapter. This is a training-time parameterization, not just fine-tuning — the model is natively structured as base+adapter throughout.
**Search:** `low-rank parameterization entire model training LoRA all weight matrices`
**Search:** `intrinsic low-rank structure neural network weights factorization`

### 4.4 — Skip List Layers (Replacing Per-Layer Residuals)
**Description:** Inspired by the skip list data structure: remove the per-layer residual connection and instead create skip connections at exponentially spaced intervals (N/2, N/4, N/8, etc.). The residual signal enters at multiple hierarchical scales rather than every layer.
**Search:** `skip list residual connection multi-scale skip connections deep neural network`
**Search:** `hierarchical skip connections exponential residual deep network`
**Search:** `dense connections DenseNet multi-scale residual neural network`

### 4.5 — Learned Residual Flow Control
**Description:** The model learns during training whether to update the residual at each layer and *which* residual stream to update (if multiple exist). Static at inference time, but trained to discover optimal information flow patterns.
**Search:** `learned residual connection gating selective residual update neural network`
**Search:** `multiple residual streams learned routing deep network`

### 4.6 — Mixture of Models
**Description:** A multi-agent architecture where multiple independently specialized models collaborate. See separate momora.md document for details.
**Search:** `mixture of models multi-agent language model collaborative inference`
**Search:** `model ensemble routing specialized language models`

### 4.7 — Compressed Dictionary
**Description:** A single output vocabulary dictionary with a compressed representation — fewer stored bits per entry, or fewer active entries via context-conditioned masking — computed and stored in compressed form rather than fully materialized. At inference, only the context-relevant vocabulary subset is decompressed/activated, reducing output-projection compute and memory bandwidth for the common case. Distinct from 2.1 (Hierarchical Dictionary, which is a cascade of separate dictionaries loaded sequentially): here the dictionary is a single structure with context-gated active subset, not a multi-level cascade. Focus: output-projection compute reduction and KV/embedding memory bandwidth at inference.
**Search:** `compressed vocabulary embedding output projection language model inference`
**Search:** `adaptive vocabulary masking dynamic output projection context-conditioned`
**Search:** `adaptive softmax hierarchical softmax frequency-based output layer language model`
**Search:** `sparse softmax output layer vocabulary reduction inference language model`
**Search:** `vocabulary factorization frequency-based output projection compression`

---

## Section 5: Memory Reduction

### 5.1 — TurboQuant (Learned Quantization Baked into Model — KV Cache Interface)
**Description:** Aggressive quantization of KV cache entries at the KV cache interface using TurboQuant-style quantization, learned during training and baked into the model rather than applied post-hoc. Scope is limited to the KV cache interface (quantized storage and dequantization on retrieval) — not applied to all model weights.
**Search:** `TurboQuant learned quantization KV cache language model`
**Search:** `quantization-aware training KV cache compression transformer`

### 5.2 — LSTM-Gated Attention
**Description:** Replace the softmax in attention with LSTM-like gating mechanisms. The gate controls information flow from key-value pairs to queries, potentially enabling better compression and selective memory.
**Search:** `LSTM gated attention replacing softmax transformer linear attention`
**Search:** `gated linear attention recurrent gating mechanism transformer`

### 5.3 — Grammar/State-Machine Structured Attention (KV Compression)
**Description:** Use a learned grammar or FSM to structure attention patterns, enabling KV cache compression. The grammar constrains which tokens can attend to which, and only the grammar-relevant entries are stored.
**Search:** `grammar constrained attention structured sparse attention KV cache compression`
**Search:** `finite state machine attention pattern language model`

### 5.4 — Linked Attention (Sparse Relevant-Only)
**Description:** Sparse attention where only the most relevant attended-to tokens are stored and computed; all others are dropped. A form of dynamic pruning of the KV cache based on relevance scores.
**Search:** `sparse attention KV cache eviction relevant token selection dynamic pruning`
**Search:** `adaptive KV cache pruning attention score threshold`

### 5.5 — Ragged Window Attention (Per-Token Variable Lookback)
**Description:** Variable-size sliding window per token — some tokens look back at few positions, others at many. Sparse attention with a learned or heuristic per-token lookback count.
**Search:** `variable window size attention per-token adaptive sliding window sparse`
**Search:** `dynamic attention span per-token lookback adaptive span transformer`

### 5.6 — Double Attention
**Description:** Run the attention operation twice per layer — two sequential attention passes. The second pass can refine or correct the first, potentially catching patterns missed in a single pass.
**Search:** `double attention two-pass attention mechanism transformer layer`
**Search:** `multi-round attention iterative attention refinement`

### 5.7 — Block-Level Compressed Weights
**Description:** Matrix decomposition applied at the block level rather than entire weight matrices. Each block of weights gets its own compressed representation (low-rank factorization, quantization, etc.).
**Search:** `block-wise weight compression low-rank factorization neural network`
**Search:** `block diagonal weight matrix decomposition transformer`

### 5.8 — Block Sparse Weights
**Description:** Structured sparsity where entire blocks of weight matrices are zeroed out, enabling hardware-efficient sparse computation.
**Search:** `block sparse weights structured sparsity transformer efficient inference`

### 5.9 — Dynamic Per-Value Numeric Type
**Description:** Mixed-precision at the individual value level — each weight or activation can have its own numeric type (fp16, int8, int4, etc.) chosen dynamically to minimize memory without sacrificing accuracy.
**Search:** `per-value mixed precision dynamic numeric type neural network weights`
**Search:** `heterogeneous quantization per-element precision neural network`

### 5.10 — Block-Level Dynamic Type Compression
**Description:** Extension of 5.9 at the block level: a block of weights is compressed together, but each element within the block stores its own type metadata. Amortizes type-storage overhead across the block.
**Search:** `block-level mixed precision element-wise type metadata weight compression`
**Search:** `grouped quantization variable precision per-element neural network`
