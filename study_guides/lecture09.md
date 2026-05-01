# Lecture 9 — Transformers

## Outline
- **Contextual Word Embeddings** — Motivation + ELMo
- **Transformers** — Positional Encoding, Core Layers (Multi-Head Attention, Feed-Forward), Encoder & Decoder
- **Extended Concepts** — Masking, Restricted (sparse / linear / flash) Attention
- **Core Tasks** — Classification, Sequence Labeling, Text Generation

## Key Concepts
- **Contextual Word Embeddings**: word vectors that vary by context (whole sentence + word order). Word2Vec/GloVe give the same vector for every sense of "light"; we want different vectors per usage.
- **ELMo**: 2-layer **Bi-LSTM** language model. Final embedding = weighted sum $emb_t = \gamma \sum_{j=0}^{2} s_j h_t^{(j)}$, with $\sum s_j = 1$ ($\gamma$ scaling, $s_j$ task-dependent normalized weights).
- **RNN limitations** motivating Transformers: vanishing/exploding gradients, hidden-state **bottleneck**, sequential processing → no parallelization.
- **Transformer**: encoder–decoder **without recurrences**. Core concept = **Attention** (alignment scores between **all** word pairs). Adds **Positional Encoding** + **Masking**.
- **Positional Encoding (PE)**: injected by adding $p_i$ to word embedding $e_i$ because attention is permutation-invariant. Sinusoidal scheme gives unique encoding per position, values in $[-1,1]$, independent of sequence length $N$.
- **Self-Attention**: $Q=K=V$ (same sequence). Used in encoder and decoder.
- **Cross-Attention** (a.k.a. Source–Target): $Q \neq K=V$. Decoder attends to encoder output ("memory").
- **Attention Head**: linear projections $W_Q, W_K, W_V$ map $d_{model}$ → $d_q = d_k = d_v = d_{model}/h$.
- **Multi-Head Attention (MHA)**: $h$ parallel heads (each with own weights) → concatenate → linear $W_O$ to $d_{model}$. Captures multiple relation types per word.
- **Feed-Forward (FF) Layer**: 2-layer FC net with ReLU; hidden size 2048 in original paper; ~2/3 of parameters.
- **Encoder Layer**: MHA → Add&Norm → FF → Add&Norm. Adds **residual connections**, **dropout**, **layer normalization**.
- **Decoder Layer**: **Masked MHA** (self) → Add&Norm → **Cross-Attention MHA** (Q from decoder, K=V from encoder memory) → Add&Norm → FF → Add&Norm.
- **Masking**: zeros out invalid alignments by adding $-\infty$ before softmax. Used for padding, MLM ("hidden" words), and causal "do-not-look-ahead" generation.

## Important Formulas
- Scaled Dot-Product Attention: $\text{Attention}(Q, K, V) = \text{softmax}\!\left(\dfrac{Q K^\top}{\sqrt{d_k}}\right) V$
- Sinusoidal Positional Encoding:
  $PE_{(pos, 2i)} = \sin\!\left(\dfrac{pos}{10000^{2i/d_{model}}}\right)$,
  $PE_{(pos, 2i+1)} = \cos\!\left(\dfrac{pos}{10000^{2i/d_{model}}}\right)$
- Attention head per-head dim: $d_q = d_k = d_v = d_{model}/h$
- Mask application: $a_{ij} + 0 = a_{ij}$, $a_{ij} + (-\infty) = -\infty$ → 0 after softmax
- ELMo final embedding: $emb_t = \gamma \sum_{j=0}^{2} s_j h_t^{(j)}$, $\sum_{j=1}^{2} s_j = 1$
- Original paper defaults: `num_layers=6`, `model_size=512`, `num_heads=8`, `ff_hidden_size=2048`, `dropout=0.1`

## Worked Example / Canonical Trap
- **Why divide by $\sqrt{d_k}$?** Dot products grow with $d_k$; large values push softmax into saturated regions where gradients vanish. Scaling stabilizes gradients.
- **Why $d_{q,k,v} = d_{model}/h$?** Keeps total trainable parameters independent of the number of heads (lecture's quick-quiz answer). With $d_{model}=512, h=8$, each head has dim 64.
- **Naive PE failure modes** (lecture explicitly walks through these):
  - Approach 1 (use raw position $0,1,\dots,N-1$): magnitudes dominate word embeddings; depends on $N$.
  - Approach 2 ($pos/(N-1)$): bounded but the value of a fixed position changes with sequence length $N$.
  - Sinusoidal fix: unique, bounded in $[-1,1]$, length-independent.
- **Causal mask example** for "I study at NUS": lower-triangular matrix with $-\infty$ above the diagonal so position $t$ only sees positions $\le t$.

## Exam Traps & Misconceptions
- **Attention complexity is $O(N^2)$** in sequence length (the $QK^\top$ matrix is $N\times N$). Sparse attention (BigBird, Longformer) targets $O(N)$.
- **Transformers are not "easy/faster to train"** despite parallelization — the slide explicitly flags this. They need large datasets and large compute.
- **Self-attention vs cross-attention**: encoder uses self-attention only; decoder uses **both** masked self-attention **and** cross-attention to encoder memory.
- **Why positional encoding is needed**: transformers process all tokens at once and have **no in-built mechanism** for word order or distance; without PE, attention is permutation-invariant.
- **Masking matrix uses $-\infty$, not 0**: $-\infty$ becomes 0 after softmax; adding 0 leaves the score unchanged.
- **ELMo's contextual vectors come from the Bi-LSTM hidden states**, not a transformer — ELMo predates the contextual-transformer era.
- **Self-Attention condition $Q=K=V$** refers to the *input source*, not the projected matrices (each still goes through its own $W_Q, W_K, W_V$).
- **FF role is "under-explored"**: the lecture admits the original paper doesn't justify FF beyond capacity/complexity, yet FF is ~2/3 of parameters.

## Cross-References
- **L8 (RNN + Attention)**: the RNN bottleneck and lack of parallelism are the direct motivation for the transformer; RNN attention (steps: score → softmax → context vector) is generalized into scaled dot-product $\text{softmax}(QK^\top/\sqrt{d_k})V$.
- **L7 / Word Embeddings**: Word2Vec/GloVe limitations (BoW, fixed window, single sense per word) motivate contextual embeddings.
- **L10 (BERT/GPT/LLMs)**: BERT = encoder-only with **MLM masking** (~15% tokens); GPT = decoder-only with **causal masking**; BART/T5 = encoder–decoder. Pretrain → fine-tune (or freeze).
- **L11+**: scaling, sparse/linear/flash attention strategies for long contexts.
