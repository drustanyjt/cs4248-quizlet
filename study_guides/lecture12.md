# Lecture 12 — LLMs: Improving Efficiency

## Outline
- Overview (scaling laws, why efficiency matters)
- Quantization (PTQ, asymmetric quant, QAT/STE, mixed precision)
- KV Caching (intuition, VRAM formula, attention variants)
- Mixture of Experts (dense vs sparse, routing, load balancing)
- Distillation (response, logits, feature)

## Key Concepts

- **Inference cost driver**: full model used each step; dominant factor is **#tokens generated**, then model size/type. LLM text gen ~150x a Google search.
- **Scaling laws**: power-law in #params with diminishing returns; for a fixed compute budget there is an *optimal* model size — bigger isn't always better.
- **Quantization**: reduce precision of weights/activations (float32 → int8). Justified because large nets are over-parameterized (cf. LoRA). Pros: smaller memory, faster compute. Con: accuracy loss.
- **PTQ vs QAT vs QT**: PTQ = post-training (most common, most degradation); QAT = quant simulated in forward pass, backprop in full precision; QT = fully quantized training (rare).
- **Asymmetric quantization**: maps `[min(X), max(X)]` to signed int range via scale `s` and zero-point `z`. Symmetric uses a range centered at 0.
- **Outliers**: extreme weights skew `s`/`z`, collapsing many distinct floats to the same int. Fix: **clipping** with calibration (static = pre-inference dataset; dynamic = per-layer activation stats).
- **Straight-Through Estimator (STE)**: in QAT, treat (de-)quant as identity in the clipping range — gradient = 1 (0 outside). Sidesteps `round()`'s zero/undefined gradient.
- **Mixed Precision Training**: master weights in fp32; fp16 working copies for forward/backward; gradients rescaled to fp32 before updates.
- **KV Cache**: K and V for previous tokens were computed already — cache them in VRAM, so each step only does new Q × cached K/V (Q shrinks from 6×512 to 1×512).
- **KV cache tradeoff**: less compute, more VRAM. Reduce via smaller model/batch/precision or attention variants below.
- **Sliding window attention**: attend only to last *k* tokens — cache size depends on *k* not sequence length *t*.
- **Multi-Query Attention (MQA)**: single K/V head shared across all Q heads — cache shrinks by `1/h`.
- **Grouped-Query Attention (GQA)**: middle ground — *m* shared K/V heads across *m* Q-groups.
- **Multi-Head Latent Attention (MLA)**: project K/V to low-dim latent (`W_K^down ∈ R^{d×r}`, `W_K^up ∈ R^{r×d}`); only the latent is cached (DeepSeek).
- **Mixture of Experts (MoE)**: layer replaced by *n* expert subnets (often FFNNs) + a **gate** giving `G(x)_i` and a **router** picking experts. Output = weighted sum.
- **Dense MoE**: all experts active — easy, accurate, no efficiency gain.
- **Sparse MoE (Top-k)**: only top-*k* (commonly Top-2) experts run; outputs aggregated with rescaled (softmax-renormalized) probs. Con: gate collapses to a few prominent experts.
- **Sparse MoE mitigations**: **stochastic routing** (tunable Gaussian noise on gate logits → passive load balancing); **load-balancing auxiliary loss** (`f_i · P_i`); entropy regularization; soft routing.
- **SoftPlus**: smooth differentiable ReLU approximation, always positive — scales the noise's std-dev.
- **Distillation**: transfer from large *teacher* → smaller *student*:
  - **Response**: teacher labels data → student trains on labels. Easy; loses nuance.
  - **Logits**: minimize KL between teacher/student logits — soft labels carry richer info; can combine with student's supervised loss.
  - **Feature**: student mimics teacher's internal reps (L1/L2/cosine/KL); usually layered on logits distillation. Trend: ↑ accuracy ↔ ↑ complexity.

## Important Formulas

Asymmetric quantization (b = #target bits):
$$s = \frac{2^b - 1}{\max(\mathbf{X}) - \min(\mathbf{X})}, \quad z = -\text{round}(\min(\mathbf{X}) \cdot s) - 2^{b-1}$$
$$X_{quant} = X \cdot s + z, \quad X_{dequant} = (X_{quant} - z)/s$$

KV cache VRAM (factor 2 for K and V):
$$\text{VRAM} = 2 \cdot p \cdot b \cdot n \cdot h \cdot d \cdot t$$

MoE output and sparse top-k gate:
$$y = \sum_{i=1}^{n} G(x)_i E_i(x), \quad G(x) = \text{Softmax}(\text{TopK}(\mathbf{W}_{gate}\mathbf{x}, k))$$

Stochastic routing (noisy gate):
$$H(x) = \mathbf{W}_{gate}\mathbf{x} + \mathcal{N}(0,1) \cdot \text{SoftPlus}(\mathbf{W}_{noise}\mathbf{x})$$

Load-balancing auxiliary loss:
$$L_{aux} = w_{aux} \cdot N \cdot \sum_{i=1}^{n} f_i \cdot P_i$$

## Worked Example / Canonical Trap

**Quantization (slide 12)**: Matrix with `min = -9.553`, `max = 7.929`, b = 8.
- `s = 255 / (7.929 − (−9.553)) = 14.586`
- `z = −round(−9.553 · 14.586) − 128 = 11`
- Outlier trap: replace −9.553 with −999.999 → `s ≈ 0.253`, almost every "normal" value now collapses to int 125 (huge information loss). Fix with `clip(X, −10, 10)`.

**KV cache memory (slide 30)**: LLaMA 7B, p=2 (fp16), b=32, n=32, h=32, d=128, t=500 →
`VRAM = 2·2·32·32·32·128·500 ≈ 7.82 GB` on top of the ~14–16 GB for weights themselves.

## Exam Traps & Misconceptions
- PTQ is *not* always safe: **outliers** dominate the scale, mass-collapsing many floats to the same int (slide 16).
- `round()` has 0/undefined gradient — that's *why* QAT needs STE; STE pretends the quantizer is identity inside the clipping range.
- KV caching saves *compute* but *costs memory*; reducing `h` (MQA) shrinks the cache but each head's `d = d_model / h` grows, so the per-head cost balances.
- Dense MoE gives **no efficiency gain** — only sparse MoE does. The price is undertrained "loser" experts unless mitigations are added.
- Sparse MoE gating is theoretically discontinuous, but the original paper notes this isn't a problem in practice.
- In response distillation the student learns *only* hard labels; logits/feature distillation transfers the teacher's uncertainty (richer signal).

## Cross-References
- **L9 (Transformers/Attention)**: KV cache directly exploits the `softmax(QK^T/√d_k)V` structure — cache K and V because they're recomputed identically each step. MQA/GQA/MLA are attention-architecture tweaks.
- **L10 (Pretrained models)**: scaling laws explain why pre-trained foundation models exist; quantization/distillation are how those huge models get *deployed*.
- **L11 (Fine-tuning, LoRA)**: "over-parameterized" justification for quantization is the same intuition as LoRA's low-rank assumption; MLA reuses the down/up-projection trick. Distillation is an alternative to fine-tuning when you need a smaller model rather than an adapted large one.
