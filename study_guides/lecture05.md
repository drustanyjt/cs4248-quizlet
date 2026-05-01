# Lecture 5 — Text Classification

## Outline
- **Text Classification**: common applications + formal setup
- **Naive Bayes Classifier**: BoW representation, definition + practical considerations, complete runthrough, discussion & limitations
- **Evaluation of Classifiers**: error types, confusion matrix, metrics, multiclass averaging
- **Vector Space Model**: vector representation of documents (binary / tf / tf-idf), document similarity (dot product, cosine)

## Key Concepts

**Text classification (formal setup)**: $X$ = set of documents, $Y$ = set of classes; learn hypothesis $h: X \to Y$ such that $h(x) = y$. Approaches: rule-based (manual decision rules) vs supervised learning from $\langle x, y \rangle$ pairs. Multilabel classification = doc assigned to >1 class.

**Common applications**: language detection, spam detection, subject/genre classification, authorship attribution, sentiment analysis.

**Bag-of-Words (BoW)**: represent doc as multiset of words (keep counts, ignore order/grammar). Affected by tokenization + normalization choices.

**Naive Bayes Classifier**: probabilistic classifier from Bayes Rule. Predict $y_{NB} = \arg\max_{y_i \in Y} P(y_i \mid x)$. The "naive" part = conditional independence assumption: words $w_1, \dots, w_n$ are independent given class. Treats each class as a separate (unigram) language model. Pros: simple, fast, interpretable, not data-hungry. Con: independence assumption usually false but works well in practice.

**Confusion matrix terms** (binary): TP, TN, FP (Type I), FN (Type II). FP and FN are not always equally costly (e.g., suicide prediction → recall > precision; news category → precision > recall).

**Vector Space Model (VSM)**: $|V|$-dimensional space, words are axes, documents are vectors. Document-term matrix has weights $w_{t,d}$ depending on representation: binary (0/1), term-frequency $tf_{t,d}$, or tf-idf. Document vectors are typically very sparse.

**Term frequency $tf_{t,d}$**: count of term $t$ in document $d$. Sublinear scaling via log dampens "more frequent ≠ proportionally more important".

**Inverse document frequency $idf_t$**: penalises terms common across many documents ($df_t$ = #documents containing $t$). Log dampens its effect.

**Multiclass evaluation**: build one-vs-rest confusion matrices per class. **Micro-averaging**: average TP/FP/FN/TN counts across matrices → favours bigger classes. **Macro-averaging**: average the per-class metrics → treats all classes equally.

## Important Formulas

Bayes rule for NB:
$$P(y \mid w_1, \dots, w_n) = \frac{P(w_1, \dots, w_n \mid y) P(y)}{P(w_1, \dots, w_n)} \propto P(y) \prod_{i=1}^{n} P(w_i \mid y)$$

NB MLE estimates:
$$\hat{P}(y) = \frac{N_y}{N}, \qquad \hat{P}(w_i \mid y) = \frac{\text{Count}(w_i, y)}{\sum_{w \in V} \text{Count}(w, y)}$$

Add-k (Laplace if $k=1$) smoothing:
$$\hat{P}(w_i \mid y) = \frac{\text{Count}(w_i, y) + k}{\sum_{w \in V} \text{Count}(w, y) + k|V|}, \qquad \hat{P}(y) = \frac{N_y + k}{N + k|Y|}$$

Log-space (avoid underflow):
$$\log P(y \mid w_1, \dots, w_n) \propto \log P(y) + \sum_{i=1}^{n} \log P(w_i \mid y)$$

Evaluation metrics:
$$\text{Accuracy} = \frac{TP + TN}{TP + FP + TN + FN}$$
$$\text{Precision} = \frac{TP}{TP + FP}, \qquad \text{Recall} = \frac{TP}{TP + FN}$$
$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

Sublinear tf, idf, and tf-idf weight (lecture form):
$$w_{t,d} = (1 + \log_{10} tf_{t,d}) \cdot \log_{10} \frac{|D|}{df_t}$$

Dot product and cosine similarity:
$$\text{dot}(v, w) = \sum_{i=1}^{n} v_i w_i, \qquad \cos(v, w) = \frac{v \cdot w}{|v| \cdot |w|} = \frac{\sum v_i w_i}{\sqrt{\sum v_i^2} \cdot \sqrt{\sum w_i^2}}$$

## Worked Example / Canonical Trap

**NB runthrough (movie reviews, Laplace smoothing)**: $V = \{funny, boring, movie, cast, good\}$, $|V| = 5$, $N = 7$, $N_{pos} = 4$, $N_{neg} = 3$.
- Priors: $P(pos) = (4+1)/(7+2) = 5/9$, $P(neg) = (3+1)/(7+2) = 4/9$.
- Likelihood (e.g., $\hat{P}(funny \mid pos) = (3+1)/(11+5) = 4/16$).
- Predict for "a funny movie and cast": $P(pos \mid \cdot) \propto (5/9)(4/16)(4/16)(3/16) = 0.0065$; $P(neg \mid \cdot) \propto (4/9)(1/14)(3/14)(3/14) = 0.0015$ → label **pos**.

**TF-IDF example** ($d_4$ = "dog watch dog tv", $|D|=5$): $w_{dog, d_4} = (1 + \log_{10} 2) \cdot \log_{10}(5/3) = 1.3 \cdot 0.22 = 0.29$; $w_{watch, d_4} = (1 + 0) \cdot \log_{10}(5/1) = 0.7$. Rare terms dominate.

**Cosine trap**: two paraphrased sentences with no shared content words ("The movie was funny..." vs "The flick was hilarious...") yield cosine = **0** under classic VSM — no semantic similarity captured.

## Exam Traps & Misconceptions

- NB likelihood denominator is **total token count in class $y$**, not number of documents in $y$.
- Add-k denominator adds $k|V|$ for likelihood, $k|Y|$ for prior — different sizes.
- Dropping the marginal $P(x)$ is fine because it doesn't depend on $y$ (argmax unchanged), but the result is no longer a true probability — use $\propto$.
- Always compute in **log-space** to avoid underflow on long docs.
- Removing "not" as a stopword destroys negation handling; common heuristic: prefix every word from negation to next punctuation with `NOT_`.
- Accuracy is misleading on **imbalanced** datasets (COVID example: always predict negative → 98% accuracy on a useless test).
- F1 is the **harmonic mean** of precision/recall — punishes large gaps; arithmetic mean would let one metric mask the other.
- Dot product overly favours frequent words and long documents; cosine normalises by length.
- Cosine on non-negative document vectors lies in $[0, 1]$ (general cosine is in $[-1, 1]$).
- Micro-averaging hides poor performance on small classes; macro-averaging treats all classes equally.
- Comparing F1 across classifiers with different #classes is unfair — random baselines differ (50% for 2 classes, ~10% for 10 classes).

## Cross-References

- **Lecture 2**: error types (Type I/II), evaluation foundations.
- **Lecture 3 (n-gram language models)**: NB likelihoods are class-conditional unigram LMs; Laplace/Add-k smoothing identical formula. NB extendable to bigrams/trigrams.
- **Later lectures (word vectors / embeddings)**: address VSM's limitation of treating "cinema" and "theater" as orthogonal dimensions; introduce dense semantic similarity.
