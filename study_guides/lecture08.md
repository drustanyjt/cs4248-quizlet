# Lecture 8 — Encoder-Decoder (RNNs)

## Outline
- **Recurrent Neural Networks (RNNs)**: Recap LMs & motivation, basic NN architectures, training RNNs, RNNs for language modelling.
- **Conditional RNNs**: Motivation & applications, encoder-decoder architecture, attention mechanism, beam search decoding.

## Key Concepts
- **Long-distance dependency problem**: n-gram LMs (Markov assumption) cannot capture context beyond the last $k$ words; motivates RNNs.
- **Hidden state $h_t$**: Core RNN concept — vector carrying information from previous time steps. Size of hidden state = size of hidden layer; (randomly) initialised, tuned via backprop.
- **RNN sequence patterns**: One-to-One (feedforward), Many-to-One (sentiment/text classification), One-to-Many (image captioning), Many-to-Many sequence labelling (POS, NER), Many-to-Many encoder-decoder (MT, summarisation).
- **BPTT (Backpropagation Through Time)**: Forward pass computes per-step losses $L_t$, aggregate to total $L$, backpropagate through entire unrolled graph.
- **LSTM / GRU**: Gated architectures used in practice — alleviate vanilla RNN's struggle with very long-distance dependencies (details "beyond scope" in lecture).
- **Conditional LM**: Assigns $P(w_1,...,w_N \mid c)$ where $c$ is a context (source sentence, image, audio).
- **Encoder-Decoder**: (1) Encoder maps context to a fixed-size vector $c$ (CNN for images, RNN for text); (2) Decoder is an RNN-LM that uses $c$.
- **Information bottleneck**: A single encoder vector $h_T^{enc}$ must capture *all* source-sentence info — motivates attention.
- **Attention**: Decoder gets "direct access" to all encoder hidden states; selectively focuses on relevant source positions.
- **Greedy decoding**: argmax word at each step — cannot backtrack on early mistakes.
- **Beam search**: Keeps $k$ most-probable partial hypotheses (beam size, typically 5–10); not optimal but much better than greedy and far cheaper than exhaustive search.

## Important Formulas
- Vanilla RNN hidden update: $h_t = \tanh(\theta_{hh} h_{t-1} + \theta_{xh} x_t)$ (with bias $b_h$).
- Output: $y_t = g_y(\theta_{hy} h_t)$; for LM: $y_t = \text{softmax}(\theta_{hy} h_t)$, with $\theta_{hh}\in\mathbb{R}^{H\times H}$, $\theta_{xh}\in\mathbb{R}^{E\times H}$, $\theta_{hy}\in\mathbb{R}^{H\times V}$.
- Kalchbrenner & Blunsom (2013) decoder injection: $h_t = \sigma(\theta_{hh} h_{t-1} + \theta_{xh} x_t + s)$ with $s = \theta_{cs} c$.
- Sutskever et al. (2014) seq2seq: $h_0^{dec} = h_T^{enc}$ (decoder initialised with last encoder hidden state). Note: paper uses LSTM, not vanilla RNN.
- Attention scores (lecture lists three):
  - dot: $e_i = h_t^\top h_s^{(i)}$
  - general: $e_i = h_t^\top \theta_a h_s^{(i)}$
  - concat: $e_i = v_a^\top \tanh(\theta_a [h_t, h_s^{(i)}])$
- Attention weights: $a_i = \frac{\exp(e_i)}{\sum_j \exp(e_j)}$ (softmax).
- Context vector: $c_t = \sum_i a_i \cdot h_s^{(i)}$.
- Decoder output with attention: $y_t = \text{softmax}(\theta_{hy}[c_t, h_t])$, with $\theta_{hy} \in \mathbb{R}^{2H\times V}$.
- Generalised (scaled dot-product) attention (foreshadowing Transformers): $\text{Attention}(Q,K,V) = \text{softmax}\!\left(\tfrac{QK^\top}{\sqrt{d_k}}\right) V$.
- Beam-search hypothesis score: $\text{score}(y_1,...,y_t) = \sum_{i=1}^{t} \log P(y_i \mid x, y_1,...,y_{i-1})$ (log-probs avoid underflow).

## Worked Example / Canonical Trap
- **Attention in 4 steps** (decoder step $t$, encoder hidden states $h_s^{(1..N)}$):
  1. score each encoder state vs $h_t$ (e.g., dot product).
  2. softmax over scores → weights $a$ summing to 1 (e.g., 0.85, 0.02, 0.06, 0.07).
  3. $c_t = \sum_i a_i h_s^{(i)}$.
  4. $y_t = \text{softmax}(\theta_{hy}[c_t, h_t])$.
- **Beam search ($k=2$) on "Ich ging nach Hause"**: at each step expand each of $k$ hypotheses to its $k$ best successors, giving $k^2$ candidates, then prune to top-$k$ by cumulative log-prob. A hypothesis ending in `</s>` is set aside; continue until $T$ steps reached or $n$ completed hypotheses. Backtrack to recover the full best sequence.

## Exam Traps & Misconceptions
- **BPTT problems** (Slide 28 quiz): correct answers are **A (small/vanishing — also exploding — gradients)** and **D (time complexity / sequential bottleneck)**, *not* "huge losses". Memory complexity is not flagged in the lecture's solution.
- **BPTT mitigations** (Slide 29 quiz): **C (truncated BPTT — "bring back Markov" by limiting history)** and **D (skip connections — Attention, LSTM/GRU shortcuts)**.
- **Greedy ≠ optimal**: argmax at each step can lock in a wrong early word ("I went *to* …") with no recovery.
- **Beam search is *not* guaranteed optimal** — only "less greedy" than argmax; exhaustive ($O(V^t)$) is intractable.
- **Encoder-decoder ≠ same RNN**: Sutskever uses *separate* encoder and decoder RNNs; only the final encoder state $h_T^{enc}$ is passed in (as $h_0^{dec}$).
- **Teacher forcing** (handwritten note on Slide 51): during *training* the decoder is fed the ground-truth previous word, not its own (possibly wrong) prediction — addresses error compounding.
- **Attention as interpretability**: lecture explicitly notes the controversy ("Attention is not Explanation" vs "…not not Explanation") — weights are *suggestive*, not definitive.
- **"Use LSTM/GRU in practice"**: vanilla RNN is taught for clarity but not used in practice; gate equations are *not* on the slides (out of scope here).

## Cross-References
- **L3 (n-gram LMs)**: Markov assumption and its long-distance-dependency failure directly motivate RNNs (Slides 4–13).
- **L7 (Word embeddings)**: input $x_t$ to RNN-LM is a word embedding (Slide 35 adds `nn.Embedding` to vanilla model).
- **L9 (Transformers)**: bottleneck + sequential BPTT motivate self-attention; Slide 66's $\text{Attention}(Q,K,V)$ explicitly previews the Transformer formula.
- **Conditional LM applications** (Slides 41–43): MT, image captioning, speech recognition, summarisation, QA — all map to encoder-decoder paradigms revisited later.
