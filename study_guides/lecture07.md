# Lecture 7 — Word Embeddings & NLP Ethics

## Outline
- Motivation (why we need word vectors beyond one-hot)
- Sparse Word Embeddings: Co-occurrence Vectors, PPMI, limitations
- Dense Word Embeddings: Basic Idea, Word2Vec (CBOW & Skip-gram), Negative Sampling, Properties, Practical Considerations
- NLP Ethics: dual-use, biases in word embeddings, debiasing

## Key Concepts

**One-Hot Encoding**: vector of length |V|, all zeros except a 1 at the word's index. Different words get **orthogonal** vectors — no notion of similarity (cat ≠ kitty).

**Distributional Hypothesis**: "You shall know a word by the company it keeps" (Firth, 1957). Words appearing in similar contexts have similar meanings (Harris, 1954; Wittgenstein, 1953).

**Document-Term Matrix (DTM) word vectors**: rows = words, columns = documents. Context defined as "set of documents containing w" — but context is too large to be useful.

**Co-occurrence / Term-Context Matrix**: |V|×|V| matrix where c_ij counts how often w_j appears in a small window around w_i. Row of w_i is its word vector.

**PMI (Pointwise Mutual Information)**: measures whether two words co-occur more than chance. Can be negative — but negative PMI has no good interpretation for word vectors.

**PPMI (Positive PMI)**: clipped PMI, replaces negatives with 0. The standard sparse vector. Refinements: raise context probabilities or use Add-1 smoothing for rare words.

**Sparse vs Dense vectors**: sparse = |V|-dim, mostly 0; dense = 100–1,000 dims, mostly non-zero. Dense vectors generalize better, capture synonymy, fewer params.

**Word2Vec**: neural-net learned dense embeddings. Two architectures: **CBOW** and **Skip-gram**. Note: Word2Vec learns **two embeddings per word** — input matrix V ∈ ℝ^(|V|×d) and output matrix U ∈ ℝ^(d×|V|).

**CBOW (Continuous Bag of Words)**: Given **context → predict center word**. Sums/averages context word vectors, then softmax over vocab. Faster.

**Skip-gram**: Given **center word → predict context words**. Slower but better for infrequent words.

**Negative Sampling (SGNS)**: Reformulates word prediction as **binary classification** (real (c,m) pair vs sampled negative pair) to avoid full softmax over |V|. Negatives drawn via α-weighted unigram frequency P_α(w_i) = Count(w_i)^α / Σ Count(w)^α, 0 < α < 1.

**Linear substructures**: dense embeddings exhibit semantic regularities like v(king) − v(man) + v(woman) ≈ v(queen); also tense (walk→walked) and comparatives (fast→faster→fastest).

## Important Formulas

PMI: $PMI(w_i, w_j) = \log_2 \frac{P(w_i, w_j)}{P(w_i) P(w_j)}$

PPMI: $PPMI(w_i, w_j) = \max\!\left(\log_2 \frac{P(w_i, w_j)}{P(w_i) P(w_j)},\ 0\right)$

CBOW loss: $L = -\log P(w_c \mid w_{c-m},\dots,w_{c+m}) = -\log P(u_c \mid \tilde v)$, with $\tilde v = \sum_{-m \le j \le m,\ j\ne 0} v_{c+j}$ and $P(u_c \mid \tilde v) = \frac{\exp(u_c^T \tilde v)}{\sum_{j=1}^{|V|} \exp(u_j^T \tilde v)}$

Skip-gram loss: $L = -\sum_{-m \le j \le m,\ j\ne 0} \log P(u_{c+j} \mid v_c)$, with $P(u_{c+j} \mid v_c) = \frac{\exp(u_{c+j}^T v_c)}{\sum_{j=1}^{|V|} \exp(u_j^T v_c)}$

SGNS positive prob: $P(+ \mid c, m) = \frac{1}{1 + \exp(-u_m^T v_c)} = \sigma(u_m^T v_c)$

SGNS loss: $L = -\left[\sum_{(c,m)\in B_{pos}} \log \sigma(u_m^T v_c) + \sum_{(c,m)\in B_{neg}} \log \sigma(-u_m^T v_c)\right]$

Negative sampling distribution: $P_\alpha(w_i) = \frac{\mathrm{Count}(w_i)^\alpha}{\sum_{w \in V} \mathrm{Count}(w)^\alpha}$, $0 < \alpha < 1$

## Worked Example / Canonical Trap

**PPMI computation** (slide 16). With counts: $P(w=\text{movie}, c=\text{cast}) = 1/35 \approx 0.03$; $P(\text{movie}) = 7/35 = 0.20$; $P(\text{cast}) = 3/35 = 0.09$. Then $PPMI = \log_2 \frac{0.03}{0.20 \cdot 0.09} \approx 0.74$.

**Word2Vec parameter count** (slide 32 quiz). |V|=10k, d=300, no bias: total = 2 × |V| × d = **6,000,000** (one V matrix + one U matrix).

**Antonym trap** (slide 58). "scary" and "funny" end up close in Word2Vec — both appear in movie-description contexts. Distributional Hypothesis can't distinguish polarity. Bad for sentiment analysis.

## Exam Traps & Misconceptions
- **CBOW = predict center from context**; **Skip-gram = predict context from center**. Easy to flip.
- Word2Vec stores **two matrices** (V input, U output) — final embedding is U-only, V-only, or their average.
- **Negative PMI is dropped** in PPMI: presence of one word suggesting absence of another has no reliable interpretation.
- **Antonyms / opposite-polarity words have HIGH similarity** in distributional embeddings (scary ≈ funny, hate ≈ love), not low — they share contexts.
- **Negative sampling avoids full softmax** over |V|; converts to binary classification.
- Negative samples drawn ~ Count(w)^α (α<1) **flattens** the unigram distribution to give rare words a fairer shot.
- Word2Vec **cannot handle polysemy** — "bank", "light", "mean" get one vector regardless of sense (slide 61 quiz answer = D).
- Word2Vec **cannot represent phrases** ("New York", "hot dog").
- Embeddings depend on training corpus (Wikipedia "house" vs Google News "house" → different neighbors).
- **Stemming/lemmatization breaks linear analogies** — "walked"/"walking" collapse.
- DTM-based word vectors **violate the Distributional Hypothesis** (context = whole document, too coarse) — slide 12 quiz.
- Cosine similarity is fine **once vectors are normalized**, but raw dot product mixes direction + length.

## NLP Ethics (lecture is light on technical detail here — mostly conceptual)
- **Dual use**: authorship attribution, text generation, user analysis, censorship — each has useful and harmful uses.
- **Bias pipeline**: biased society → biased data → biased model. E.g., Hungarian "ő egy mérnök" → "he is an engineer".
- **Bias in embeddings**: v(programmer) − v(man) + v(woman) ≈ v(homemaker).
- **Debiasing pipeline (Bolukbasi et al.)**: (1) measure bias via she-he analogies with cosine + δ similarity constraint; (2) identify gender subspace via PCA over gendered word-pair differences; (3) split vocab into gender-neutral N vs gender-specific S using a binary classifier; (4) learn transformation T minimizing $\|(TW)^T(TW) - (W^T W)\|_F^2 + \lambda \|(TN)^T (TB)\|_F^2$ (preserve geometry; zero out gender component on neutral words).

## Cross-References
- **L5** (Vector Space Models): cosine similarity, sparse tf-idf — this lecture extends to dense vectors and shows why one-hot/DTM fail.
- **L9** (Transformers / contextual embeddings): solves Word2Vec's polysemy + phrase weaknesses — same word gets different vectors per context.
- **L6** (Neural nets / MLPs): Word2Vec is a 2-layer NN trained as a word-prediction task; embeddings are a by-product.
- **Sentiment-analysis lectures**: connect to slide 58/68 — distributional embeddings carry polarity bias and demographic bias (names).
