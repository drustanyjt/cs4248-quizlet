# Lecture 10 — LLMs (BERT, T5/BART, GPT/LLaMA, Training & Prompting)

## Outline
- **Transformer-based LLMs:** Encoder-only (BERT, RoBERTa); Encoder-Decoder (T5, BART); Decoder-only (GPT, LLaMA).
- **Training & Working with LLMs:** data collection challenges, GPT-2-style mini tutorial, cloud APIs vs local models, prompt engineering.
- **The Good, the Bad, the Ugly:** emergent abilities, cost (compute / energy / water), model alignment, disruptions.
- Big picture: **decoder-only dominates** post-2022 (simpler, cheaper, generative, good zero-shot).

## Key Concepts
- **BERT (Bidirectional Encoder Representations from Transformers):** encoder-only, self-supervised. Pretrained on **MLM + NSP**, then fine-tuned per task. Input format `[CLS] Sentence A [SEP] Sentence B [SEP]`.
- **MLM (Masked Language Model):** randomly mask ~**15%** of input tokens with `[MASK]`, predict the originals. Uses bidirectional context.
- **NSP (Next Sentence Prediction):** binary classification — does sentence B follow sentence A?
- **RoBERTa:** BERT scaled up — MLM only (no NSP), **dynamic masking** (re-masked each epoch; BERT uses static masking from preprocessing), more data, longer training.
- **T5 (Text-to-Text Transfer Transformer):** encoder-decoder; every task reformulated as text-to-text with task-specific prefixes (e.g., `translate English to German:`, `stsb sentence1: …`). Multi-task learning, denoising objective.
- **BART:** encoder-decoder ≈ BERT encoder + GPT decoder. Trained by **corrupting documents and reconstructing** (denoising). Corruptions: token masking, token deletion, sentence permutation, document rotation, text infilling.
- **GPT (Generative Pretrained Transformer):** decoder-only, **causal masking** (only leftward context). Auto-regressive next-token prediction. GPT-3+ adds **RLHF**.
- **LLaMA tweaks vs vanilla GPT:** **pre-normalization** (LayerNorm inside residual blocks → better gradients, faster training), **SwiGLU** activation, **RoPE** (rotary positional embeddings → relative positions via rotation, extrapolates to unseen lengths).
- **Causal masking:** required at both training **and** inference (generate one token at a time, can't peek ahead).
- **Sliding window data prep:** treat training data as one long stream with `[EOS]` separators; chunk into context windows with overlap.
- **In-Context Learning (ICL) / X-shot prompts:** zero-shot (no examples), one-shot, few-shot — model performs new task purely via demonstrations in the prompt; **no parameter updates**, an emergent ability.
- **RLHF:** humans rank multiple responses to a prompt; rankings used to fine-tune the pretrained LM.

## Important Formulas / Notation
- Causal LM (next-token): $\mathcal{L}_{\text{CLM}} = -\sum_{t} \log P(x_t \mid x_{<t})$
- MLM: $\mathcal{L}_{\text{MLM}} = -\sum_{i \in M} \log P(x_i \mid x_{\setminus M})$ where $M$ = masked positions (~15%).
- NSP: $\mathcal{L}_{\text{NSP}} = -\log P(\text{IsNext} \mid [\text{CLS}], A, [\text{SEP}], B, [\text{SEP}])$
- BART denoising: cross-entropy between decoder output and original document (reconstructed from corrupted input).
- Document stream: $\text{doc}_1 + [\text{EOS}] + \text{doc}_2 + [\text{EOS}] + \dots$
- RoPE 2D rotation: $\mathbf{R}_{\theta,m} = \begin{bmatrix}\cos m\theta & -\sin m\theta\\ \sin m\theta & \cos m\theta\end{bmatrix}$ (rotate query/key by position $m$); attention dot product depends only on relative distance.
- RoPE frequencies: $\Theta = \{\theta_i = B^{-2(i-1)/d}\}$, base $B \approx 10000$.
- SwiGLU: $\text{SwiGLU}(x) = (xW + b) \otimes \text{Swish}_\beta(xV + c)$, with $\text{Swish}(x) = x \cdot \sigma(\beta x)$.

## Worked Example / Canonical Trap
- **Sentence-entailment input format (Quick Quiz):** correct answer is `[CLS] Sentence A [SEP] Sentence B [SEP]`. `[CLS]`'s final embedding feeds the classifier; `[SEP]` separates the two sentences. Trap options swap `[CLS]`/`[SEP]` or insert `[MASK]`.
- **Picking the right architecture:**
  - Sentence classification, NER, extractive QA → **encoder-only (BERT)**.
  - Machine translation, summarization, generative QA → **encoder-decoder (T5/BART)**.
  - Open-ended generation, chatbots, few-shot tasks → **decoder-only (GPT/LLaMA)**.
- **Data prep for a toy GPT-2:** load 100k IMDB reviews → tokenize → join with `[EOS]` → sliding windows (e.g., size 6, 50% overlap); target = inputs shifted left by 1 token.

## Exam Traps & Misconceptions
- BERT is **encoder-only with MLM + NSP**, NOT auto-regressive. GPT is **decoder-only with causal LM**. BART/T5 are **encoder-decoder**.
- RoBERTa **drops NSP** and uses **dynamic** (not static) masking; ALBERT uses SOP (not NSP).
- "No recurrence" in transformers means easier parallelization — it does **not** mean training is faster/cheaper overall.
- ICL ("in-context learning") does **no** weight updates. Empirically: correctness of demo labels barely matters, but **label space**, **input distribution/relevance**, **number of demos**, and **order** do (recency bias, majority-label bias).
- Causal masking is needed during inference too (the prompt is processed all at once; future positions must still be hidden).
- Encoder-only models like BERT are **not suited for free text generation** (bidirectional, no autoregressive head).
- More memory needed for **training** than inference because of activations, gradients, and optimizer state — model size alone is **not** the differentiator (the model must be loaded for both).
- Pre-normalization (LLaMA) ≠ post-normalization (original Transformer); LayerNorm sits **inside** residual blocks in LLaMA.

## Cross-References
- **L9 (Transformers):** attention, multi-head, encoder/decoder blocks, original sinusoidal positional encoding — Lecture 10 reuses all of these and contrasts pre- vs post-normalization, and absolute vs relative (RoPE) position encodings.
- **L2 (Tokenization):** BPE/WordPiece tokenizers underpin BERT/GPT input pipelines; the toy-LLM tutorial reuses a pretrained tokenizer.
- **L11 (Fine-tuning / Adaptation):** picks up where pretraining ends — fine-tuning BERT for downstream tasks (MNLI, NER, SQuAD), and parameter-efficient methods that build on the pretrained checkpoints introduced here.
- **Cost/alignment thread:** training LLaMA-65B ≈ 449 MWh, 173 tCO₂eq; text-generation inference ~150× a Google search — connects to the ethics/alignment material later in the lecture.
