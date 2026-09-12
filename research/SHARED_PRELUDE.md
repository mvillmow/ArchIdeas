# Shared Context Prelude — Architecture Research (CANONICAL BASELINES)
## Include this entire file verbatim in every lead-agent prompt.

Source: HuggingFace authoritative `config.json` for each model, NVIDIA NIM model card, and Alibaba release notes. These values are ground truth and override any prior prelude.

---

## Baseline A1: Qwen3.5-27B (Hybrid, Multimodal)

Source: `huggingface.co/Qwen/Qwen3.5-27B/raw/main/config.json`

- **Layers (L)**: 64
- **Hidden dim (d)**: 5,120
- **MLP intermediate (d_ff)**: 17,408
- **Hybrid attention layout**: every 4th layer is full attention; the other 3 are linear attention (Gated DeltaNet)
  - **Full-attention layers (16 of 64)**: 24 Q-heads / 4 KV-heads, head_dim = **256** (GQA)
  - **Gated DeltaNet layers (48 of 64)**: 48 V-heads / 16 QK-heads, head_dim = 128
- **Vocab**: **248,320**
- **Context (native)**: **262,144** tokens, extensible via RoPE scaling
- **RoPE**: base = 10,000,000; mRoPE sections [11, 11, 10]; partial_rotary_factor = 0.25
- **Activation**: SiLU
- **Norms**: RMSNorm
- **dtype**: bf16
- **Extra**: MTP head, vision tower (multimodal)
- **Total params**: ~27B (all active at inference)

### Baseline A1 per-token / per-layer complexity
- Gated DeltaNet layers (48 of 64): O(d²) compute per token, O(d_state) fixed recurrent state per head (independent of seq length)
- Full-attention layers (16 of 64): O(s·d) compute, KV memory = 2 · 4 KV × 256 head_dim × s × 2 bytes = 4,096 · s bytes per layer
- MLP: 2 · d · d_ff FLOPs per token per layer = 2 · 5,120 · 17,408 ≈ 1.78 × 10⁸ FLOPs per token per MLP
- Total KV cache (bf16, full-attn layers only) = 16 · 2 · 4 · 256 · s · 2 = 65,536 · s bytes
  - @ 32,768 tokens: ~2.15 GB
  - @ 262,144 tokens: ~17.2 GB

---

## Baseline A2: Qwen3-32B (Dense)

Source: `huggingface.co/Qwen/Qwen3-32B/raw/main/config.json`

- **Layers (L)**: 64
- **Hidden dim (d)**: 5,120
- **MLP intermediate (d_ff)**: 25,600
- **Attention**: 64 Q-heads / 8 KV-heads (GQA), head_dim = 128
- **Vocab**: 151,936
- **Context (native)**: **40,960** tokens
- **RoPE base**: 1,000,000
- **Activation**: SiLU
- **Norms**: RMSNorm
- **dtype**: bf16
- **Total params**: ~32B (all active)

### Baseline A2 per-token / per-layer complexity
- Attention: O(s·d) compute, KV memory = 2 · 8 · 128 · s · 2 bytes = 4,096 · s bytes per layer
- MLP: 2 · 5,120 · 25,600 ≈ 2.62 × 10⁸ FLOPs per token per MLP layer
- **Total KV cache (bf16, all 64 layers)** = 64 · 2 · 8 · 128 · s · 2 = 262,144 · s bytes
  - @ 32,768 tokens: **~8.59 GB** (not ~68 GB — KV cache formula must use H_kv=8, not H_q=64)
  - @ 40,960 tokens: ~10.74 GB

---

## Baseline C: K2 Family (LLM360, Dense Llama-arch 72.55B)

Source: `huggingface.co/LLM360/K2-V2/raw/main/config.json` and `huggingface.co/LLM360/K2-Think-V2/raw/main/config.json`

- **Model type**: `LlamaForCausalLM` (`model_type = "llama"`), dense transformer
- **Layers (L)**: 80
- **Hidden dim (d)**: 8,192
- **MLP intermediate (d_ff)**: 28,672
- **Attention**: 64 Q-heads / 8 KV-heads (GQA 8:1), head_dim = 128
- **Vocab**: 250,112
- **Context (native)**:
  - K2-V2: **524,288** tokens (`rope_scaling: type=llama3, factor=4, orig_max_pos=131,072`)
  - K2-Think-V2: **262,144** tokens (`rope_scaling: type=yarn, factor=2, orig_max_pos=131,072`)
- **rope_theta**: 10,000,000 (both)
- **Activation**: SiLU/SwiGLU
- **Norms**: RMSNorm
- **dtype**: fp32 on disk (K2-V2); bf16 (K2-Think-V2)
- **Total params**: ~72.55B (72,550,195,200 counted; card says ~70B)
- **tie_word_embeddings**: false
- **K2-Think-V2 delta only**: adds `<think>`, `<think_fast>`, `<think_faster>` tokens via chat template — **no weight or architecture change**; identical in all quantitative analysis.

### Baseline C per-token / per-layer complexity
- Attention: O(s·d) compute, KV memory = 2 · 8 · 128 · s · 2 bytes = 4,096 · s bytes per layer
- MLP: 2 · 8,192 · 28,672 ≈ 4.70 × 10⁸ FLOPs per token per MLP layer
- **Total KV cache (bf16, all 80 layers)** = 80 · 2 · 8 · 128 · s · 2 = 327,680 · s bytes
  - @ 32,768 tokens: **~10.0 GiB**
  - @ 40,960 tokens: ~12.5 GiB
  - @ 262,144 tokens: ~80.0 GiB  (K2-Think-V2 native max)
  - @ 524,288 tokens: ~160.0 GiB (K2-V2 native max)

> **Use "Baseline C (K2 family)"** in all comparison tables and discussion. Where context-window differences matter, note K2-V2 (524K) vs K2-Think-V2 (262K) in a footnote; for 32K and shorter evaluations they are numerically identical.

---

## Baseline B: Qwen3.5-397B-A17B (Hybrid MoE + Gated DeltaNet)

Source: Qwen3.5 HF model card, NVIDIA NIM model card, Alibaba release.

- **Layers (L)**: 60. Layout: 15 groups of 4 layers. Each group = 3 × (Gated DeltaNet → MoE) followed by 1 × (Gated Attention → MoE). So 15 layers are full attention, 45 are Gated DeltaNet.
- **Hidden dim (d)**: 4,096
- **Gated DeltaNet layers (45 of 60)**: 64 V-heads / 16 QK-heads, head_dim = 128
- **Gated Attention layers (15 of 60)**: **32 Q-heads / 2 KV-heads**, head_dim = **256**
- **MoE**: 512 total experts, 10 routed + 1 shared active (k=11), fine-grained
- **Vocab**: **248,320**
- **Context (native)**: **262,144** native, extensible to ~1,010,000 via YaRN
- **Active / total params**: 17B / 397B
- **Reported decoding speedup**: 8.6×–19× over Qwen3-Max

### Baseline B per-token / per-layer complexity
- Gated DeltaNet layers (45 of 60): O(d²), recurrent state per head
- Gated Attention layers (15 of 60): KV per layer = 2 · 2 · 256 · s · 2 bytes = 2,048 · s bytes
- Total KV cache (bf16) = 15 · 2 · 2 · 256 · s · 2 = 30,720 · s bytes
  - @ 32,768 tokens: ~1.0 GB
  - @ 262,144 tokens: ~8.0 GB
- MoE MLP: per token effective compute = k active experts · (per-expert FLOPs) + router O(d · E_total) where E_total = 512

---

### KV Cache at Reference Context Lengths (all baselines)

| Model | 32K ctx | 40K ctx | 262K ctx | Max native ctx |
|-------|---------|---------|---------|----------------|
| A1 Qwen3.5-27B (16/64 full-attn layers) | ~2.15 GB | ~2.63 GB | ~17.2 GB | 262,144 |
| A2 Qwen3-32B (full KV, all 64 layers) | ~8.59 GB | ~10.74 GB | ~68.7 GB | 40,960 |
| B Qwen3.5-397B (15/60 global-attn layers) | ~1.0 GB | ~1.22 GB | ~8.0 GB | 262,144 |
| **C K2 family (all 80 layers)** | **~10.0 GiB** | **~12.5 GiB** | **~80.0 GiB** | 524,288 (K2-V2) / 262,144 (K2-Think) |

---

### KV Cache Formula (canonical)

```
KV_bytes = L_full_attn × 2 × H_kv × head_dim × seq_len × 2  (bf16 = 2 bytes)
```

Always use `num_key_value_heads` (H_kv), never `num_attention_heads` (H_q). For hybrid models, multiply only by the fraction of layers that are full-attention.

---

## Appendix: Baseline Parameter Verification Checklist

Source: extracted from HuggingFace `config.json` for each model. Use this table to verify any derived calculation in the research documents.

### A1 (Qwen3.5-27B)

| Parameter | Value | Source |
|-----------|-------|--------|
| L | 64 | HF config.json |
| d | 5,120 | HF config.json |
| d_ff | 17,408 | HF config.json |
| Full-attn layers | 16 of 64 (every 4th) | HF config.json |
| Full-attn Q/KV heads | 24 / 4 | HF config.json |
| Full-attn head_dim | 256 | HF config.json |
| DeltaNet V/QK heads | 48 / 16 | HF config.json |
| DeltaNet head_dim | 128 | HF config.json |
| V | 248,320 | HF config.json |
| Max context | 262,144 | HF config.json |
| KV/layer | 2×4×256×s×2 = 4,096·s bytes | derived |
| Total KV | 16×4,096·s = 65,536·s bytes | derived |
| KV @32K | 65,536×32,768 = 2.15 GB | derived |
| MLP FLOPs/tok/layer | 2×5,120×17,408 = 1.78×10⁸ | derived |

### A2 (Qwen3-32B)

| Parameter | Value | Source |
|-----------|-------|--------|
| L | 64 | HF config.json |
| d | 5,120 | HF config.json |
| d_ff | 25,600 | HF config.json |
| Q/KV heads | 64 / 8 | HF config.json |
| head_dim | 128 | HF config.json |
| V | 151,936 | HF config.json |
| Max context | 40,960 | HF config.json |
| KV/layer | 2×8×128×s×2 = 4,096·s bytes | derived |
| Total KV | 64×4,096·s = 262,144·s bytes | derived |
| KV @32K | 262,144×32,768 = 8.59 GB | derived |
| MLP FLOPs/tok/layer | 2×5,120×25,600 = 2.62×10⁸ | derived |

### B (Qwen3.5-397B-A17B)

| Parameter | Value | Source |
|-----------|-------|--------|
| L | 60 | Model card |
| d | 4,096 | Model card |
| DeltaNet layers | 45 of 60 | Model card |
| Gated Attn layers | 15 of 60 | Model card |
| Gated Attn Q/KV heads | 32 / 2 | Model card |
| Gated Attn head_dim | 256 | Model card |
| DeltaNet V/QK heads | 64 / 16 | Model card |
| MoE | 512 total, k=11 (10 routed + 1 shared) | Model card |
| V | 248,320 | Model card |
| Max context | 262,144 | Model card |
| Active/Total params | 17B / 397B | Model card |
| KV/layer | 2×2×256×s×2 = 2,048·s bytes | derived |
| Total KV | 15×2,048·s = 30,720·s bytes | derived |
| KV @32K | 30,720×32,768 = 1.0 GB | derived |

### C (K2 Family, 72.55B)

| Parameter | Value | Source |
|-----------|-------|--------|
| L | 80 | HF config.json |
| d | 8,192 | HF config.json |
| d_ff | 28,672 | HF config.json |
| Q/KV heads | 64 / 8 | HF config.json |
| head_dim | 128 | HF config.json |
| V | 250,112 | HF config.json |
| Max context (K2-V2) | 524,288 | HF config.json |
| Max context (K2-Think) | 262,144 | HF config.json |
| KV/layer | 2×8×128×s×2 = 4,096·s bytes | derived |
| Total KV | 80×4,096·s = 327,680·s bytes | derived |
| KV @32K | 327,680×32,768 = 10.0 GiB | derived |
| MLP FLOPs/tok/layer | 2×8,192×28,672 = 4.70×10⁸ | derived |

---

## Schema Variants

This corpus uses two document templates:

**Standard template** (Groups 1-4, plus research_5_1 through 5_8):
- `## Executive Summary`
- `## 1. Idea Description`
- `## 2. Literature Review`
- `## 3. Prior Art Classification`
- `## 4. Technical Analysis`
- `## 5. Implementation Considerations`
- `## 6. Synergies`
- `## 7. Risk Assessment`
- `## 8. Accuracy / Quality Tradeoff`
- Per-baseline comparison tables nested under Executive Summary

**Thematic template** (Group 6 entirely, plus research_5_9 and research_5_10):
- `## Executive Summary`
- `## Idea Description` / `## Idea Overview`
- `## Literature Review` (per-paper `###` subsections) OR `## Prior Art` (narrative prose)
- `## Technical Analysis` / `## Mechanism` / `## Complexity Analysis`
- `## Feasibility Assessment` / `## Risk Matrix`
- `## Synergies with Other Ideas`
- `## Benefits vs Baseline A1` (separate top-level section per baseline)
- `## Benefits vs Baseline A2`
- `## Benefits vs Baseline B`
- `## Benefits vs Baseline C`
- `## Accuracy / Quality Tradeoff`
- `## Citations`

**Files using thematic template (7 total):**
research_5_9, research_5_10, research_6_1, research_6_2, research_6_3, research_6_4, research_6_5

Audit dimensions D1/D3/D4/D7 apply only to standard-template docs (32 files). Thematic-template docs (7 files) are verified against their own structural checklist: per-baseline Benefits sections present, Accuracy/Quality Tradeoff present, Citations section present.
