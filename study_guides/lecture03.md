# Lecture 3 — n-Gram Language Models

## Outline
- **Language Models**: motivation, sentence probabilities, evaluation (extrinsic vs intrinsic, perplexity).
- **n-Gram Language Models**: basic counting approach (MLE), the Markov assumption, smoothing methods.
- **Discussion**: limitations of n-gram LMs (long-distance dependencies) and a teaser for RNNs & Transformers.

## Key Concepts
- **Language Model (LM)**: assigns a probability to a sentence/phrase/word; equivalently predicts $w_n$ from history $w_1 \ldots w_{n-1}$. Used in speech recognition, spelling/grammar correction, MT, text generation.
- **Chain Rule (decomposition)**: turns joint sentence probability into product of conditionals (one per token).
- **Markov Assumption**: condition only on the last $k$ words (not full history). This is what makes n-grams tractable.
- **n-gram model**: approximates $P(w_n | w_{1:n-1})$ using only the last $n-1$ words. Unigram = no context, bigram = 1 prior word, trigram = 2 prior words. Larger n → more data needed.
- **Maximum Likelihood Estimation (MLE)**: relative frequency $\text{Count}(\text{n-gram}) / \text{Count}(\text{prefix})$.
- **Sentence boundary tokens**: `<s>` and `</s>` are added so the model can score sentence starts/ends.
- **Extrinsic vs Intrinsic Evaluation**: extrinsic = run downstream task and compare; intrinsic = score on a held-out test corpus (cheaper, uses perplexity). Standard split is **80/10/10** train/dev/test.
- **Perplexity**: inverse probability of test corpus, normalized by length $N$. **Lower is better**; minimizing perplexity ⇔ maximizing probability. Use **log-form** in practice to avoid underflow.
- **OOV handling**: closed vocab (no unknowns) vs open vocab; replace unknowns with `<UNK>` token, or use **subword tokenization** (e.g., BPE), or smoothing.
- **Smoothing / discounting**: shifts probability mass from seen n-grams to unseen ones so no probability is zero.
- **Laplace (Add-1) Smoothing**: add 1 to every count; denominator gets $+V$ (vocab size). Simple but can shift too much mass.
- **Add-k Smoothing**: generalization with $0 < k \le 1$; $k$ tuned on dev set.
- **Backoff**: if higher-order n-gram is unseen, fall back to (n−1)-gram, then (n−2)-gram, …
- **Linear Interpolation**: weighted mix of unigram, bigram, trigram probabilities; weights $\lambda_i$ sum to 1, learned (e.g., via EM) on a held-out corpus. Generally better than backoff.
- **Kneser-Ney Smoothing**: combines **absolute discounting** (subtract fixed $d$, typically $0.75$) with a **continuation probability** that asks "how likely is $w$ to appear as a *novel continuation*?" rather than "how frequent is $w$?".
- **Long-distance dependency limitation**: the Markov assumption breaks for sentences where the cue is far back (e.g., "All jokes totalled landed, resulting in a movie that is very ___"). Motivates RNNs/Transformers.

## Important Formulas
Sentence probability via chain rule:
$$P(w_1, \ldots, w_N) = \prod_{i=1}^{N} P(w_i \mid w_{1:i-1})$$

Markov approximation (n-gram):
$$P(w_n \mid w_{1:n-1}) \approx P(w_n \mid w_{n-k:n-1})$$

Bigram MLE:
$$P(w_n \mid w_{n-1}) = \frac{\text{Count}(w_{n-1} w_n)}{\text{Count}(w_{n-1})}$$

Perplexity (general and bigram):
$$PP(W) = P(w_1, \ldots, w_N)^{-1/N} = \sqrt[N]{\prod_{n=1}^{N} \frac{1}{P(w_n \mid w_{n-1})}}$$

Log-perplexity (numerical stability):
$$\ln PP(W) = -\frac{1}{N} \sum_{n=1}^{N} \ln P(w_n \mid w_1, \ldots, w_{n-1})$$

Laplace (Add-1) bigram:
$$P_{\text{Laplace}}(w_n \mid w_{n-1}) = \frac{\text{Count}(w_{n-1} w_n) + 1}{\text{Count}(w_{n-1}) + V}$$

Add-k bigram:
$$P_{\text{add-}k}(w_n \mid w_{n-1}) = \frac{\text{Count}(w_{n-1} w_n) + k}{\text{Count}(w_{n-1}) + kV}$$

Linear interpolation (trigram):
$$\hat{P}(w_n \mid w_{n-2}, w_{n-1}) = \lambda_1 P(w_n) + \lambda_2 P(w_n \mid w_{n-1}) + \lambda_3 P(w_n \mid w_{n-2}, w_{n-1}), \quad \sum_i \lambda_i = 1$$

Kneser-Ney (bigram form):
$$P_{KN}(w_n \mid w_{n-1}) = \frac{\max[\text{Count}(w_{n-1} w_n) - d, 0]}{\text{Count}(w_{n-1})} + \lambda(w_{n-1}) P_{KN}(w_n)$$

Continuation probability (novel-context unigram):
$$P_{KN}(w) = \frac{|\{w' : \text{Count}(w'w) > 0\}|}{|\{(u,v) : \text{Count}(uv) > 0\}|}$$

Normalizing factor:
$$\lambda(w_{n-1}) = \frac{d}{\text{Count}(w_{n-1})} \cdot |\{w' : \text{Count}(w_{n-1} w') > 0\}|$$

Variables: $V$ = vocabulary size; $N$ = number of tokens in test corpus; $d \approx 0.75$ = absolute discount; $\lambda_i$ = interpolation weights.

## Worked Example / Canonical Trap
**Bigram with `<s>`/`</s>`** (corpus = three sentences: "I am Sam", "Sam I am", "I do not like green eggs and ham"):
- $P(\text{I} \mid \langle s \rangle) = 2/3$, $P(\text{am} \mid \text{I}) = 2/3$, $P(\text{Sam} \mid \text{am}) = 1/2$, $P(\langle /s \rangle \mid \text{Sam}) = 1/2$.

**Kneser-Ney "Kong vs glasses"**: For "I can't see without my reading ___", a plain unigram says $P(\text{Kong}) > P(\text{glasses})$ because "Hong Kong" is frequent. But "Kong" almost only follows "Hong" — it is **not a novel continuation**. KN's continuation count for "glasses" is high (preceded by many distinct words), so $P_{KN}(\text{glasses}) > P_{KN}(\text{Kong})$, fixing the prediction.

## Exam Traps & Misconceptions
- **Don't confuse "n-gram" with "n-1 prior words"**: a bigram (n=2) conditions on only **1** previous word, trigram (n=3) on **2**.
- **Perplexity range is $(1, \infty)$, not $(0, V)$**: minimum is 1 (model is certain and right); maximum is $\infty$ (probabilities → 0). It is *not* upper-bounded by vocabulary size.
- **Lower perplexity = better LM** (because $PP \propto 1/P$).
- **Unigram LMs ignore word order**: $P(\text{"alice saw the accident"}) = P(\text{"the accident alice saw"})$ — they're permutations.
- **Laplace smoothing can over-discount**: it moves a *lot* of mass to unseen n-grams when $V$ is large; that's why Add-k and KN exist.
- **In-lecture trap**: you cannot read $\text{Count}(w_{n-1})$ from a bigram-counts table by summing the row labelled $w_{n-1}$ alone — it misses occurrences of $w_{n-1}$ as the *second* word of some bigram. Use unigram counts.
- **`P(<s>)` is not needed** when comparing two sentence probabilities — it's a constant 1 / shared factor.
- **Markov assumption is an approximation**, not truth: it fails for long-distance dependencies (the "jokes ... funny" example).
- **Kneser-Ney's lower-order term is a continuation probability, not a unigram frequency** — that is the whole point of the method.

## Cross-References
- **L2 (Preprocessing)**: tokenization, `<UNK>`, BPE feed into n-gram counts.
- **Word Embeddings (~L7)**: distributed representations replace count-based estimates; tackle sparsity differently.
- **RNNs**: hidden state $h_t$ replaces Markov window — captures long-range context but has vanishing/exploding gradients and no parallelism.
- **Transformers**: attention removes the bottleneck and parallelizes; causal/decoder-only LM is the modern next-token task n-grams started.
- 80/10/10 split and perplexity recur in every later LM lecture.

**Not in slides (skip for exam)**: Good-Turing, Witten-Bell, Stupid Backoff, class-based n-grams, cache LMs, modified KN with multiple discounts, cross-entropy in bits ($\log_2$), entropy-rate derivations.
