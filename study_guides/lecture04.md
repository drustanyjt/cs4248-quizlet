# Lecture 4 — Structure in Language

## Outline
- Sequences: common sequence tasks, POS tagging as case study, overview of sequence models
- Trees / Syntactic Parsing: Constituency Parsing (CFGs, PCFGs, ambiguity)
- Trees / Syntactic Parsing: Dependency Parsing (heads, dependents, relations)
- Applications: text simplification, information extraction (knowledge-graph triples)

## Key Concepts

**Sequence task types** — Classification (N→1, sentiment), Labeling/Tagging (N→N, POS, NER), Translation (N→M, MT, summarization), Generation (1→N, captioning).

**Part of Speech (POS)** — Word class / syntactic category. English: 8 main classes; Penn Treebank base set = **36 tags** plus **12 punctuation tags**.

**Closed vs Open class** — Closed: small fixed function-word set (prepositions, pronouns, determiners, conjunctions). Open: nouns, verbs, adjectives, adverbs — new members constantly added.

**Anchor word** — Word with only one possible POS tag; used in unsupervised tagging to bootstrap clusters.

**Baseline POS tagger** — Assign each word its most frequent tag; tag unknowns as nouns. ~92% accuracy (SOTA ~97-98%). Problems: imbalanced errors and downstream error propagation.

**POS ambiguity stats** — ~85% of word *types* unambiguous, but ~55-65% of word *tokens* ambiguous (ambiguous words are common).

**Hidden Markov Model (HMM)** — Hidden states = POS tags, observables = words. Lecture covers only the conceptual setup; Viterbi math is **NOT in lecture proper** (optional notebook only).

**Constituent** — Group of words behaving as a single unit; tested via topicalization, proform substitution, fragment answers.

**Context-Free Grammar (CFG)** — 4-tuple ⟨N, Σ, R, S⟩: non-terminals, terminals, rules A → α with **A a single non-terminal**, start symbol. More powerful than regex (recursive). Captures constituency + ordering.

**Derivation / Parse Tree** — Sequence of rule applications producing the string; visualized as a tree.

**Structural Ambiguity** — Grammar yields >1 parse for one sentence. Types: **Attachment** (where PP attaches) and **Coordination** (scope of "and"/"or").

**PCFG** — CFG with rules annotated by probabilities; rules sharing an LHS sum to 1. Estimated by counts on annotated data.

**CYK** — Bottom-up DP parser ("which non-terminals could have generated this substring?"). **Earley** — Top-down. Both symbolic, rule-based; input = text + grammar; no training.

**Dependency parsing** — Express sentence as directed binary relations between tokens. Each edge: one **head** (governor), one **dependent** (modifier), labeled from a fixed set (Universal Dependencies).

**Head criteria (Zwicky/Hudson)** — H determines syntactic + semantic category of construction; H obligatory; H selects D; form/position of D depends on H.

**Dependency types** — Head-complement (D required), Head-modifier (D optional), Head-specifier (det-noun), Coordination (head ambiguous).

## Important Formulas

PCFG rule probability (estimated from a treebank):
$$P(A \to \alpha) = P(\alpha \mid A) = \frac{\text{Count}(A \to \alpha)}{\text{Count}(A)}$$

Probability of a parse tree T over sentence S (product of all rule probabilities used):
$$P(T, S) = \prod_{i=1}^{n} P(A_i \to \alpha_i)$$

In practice sum log-probabilities to avoid arithmetic underflow.

Normalization constraint: for every LHS A, $\sum_\alpha P(A \to \alpha) = 1$.

## Worked Example / Canonical Trap

**Attachment ambiguity — "I book the flight through Singapore"** (toy grammar from slides):
- Parse 1: "through Singapore" attaches to NP "the flight" (a flight that goes via Singapore). P(T,S) ≈ 0.00000071.
- Parse 2: "through Singapore" attaches to VP "book" (booking via a Singapore agent). P(T,S) ≈ 0.00000024.
- PCFG prefers Parse 1 because it has higher product of rule probabilities.

**CFG violation quiz (slide 40)** — Two issues make a grammar non-CFG: (1) `VP NP → Verb NP` has *two* non-terminals on the LHS (CFG requires single non-terminal); (2) `the → Det` has a *terminal* on the LHS.

**Constituent test (slide 34)** — In "All students learned about syntactic parsing this week," "learned about syntactic" is **not** a constituent (it slices through the VP arbitrarily). "All students," "learned about syntactic parsing," "this week" all are.

**Dependency arrow direction** — Arrow goes from head to dependent. In "Chris eats," **eats** is the head (VERB), **Chris** is the dependent (nsubj). Saying "Chris is the head" is a common student error.

## Exam Traps & Misconceptions

- **CFG LHS rule**: must be exactly one non-terminal. Anything else (a terminal, or two non-terminals) breaks context-freeness.
- **Token vs type ambiguity**: 85% of types unambiguous, but 55-65% of tokens ambiguous — different statistics, easy to swap.
- **Baseline 92% sounds good** — but the lecture stresses errors are imbalanced and propagate.
- **HMM Viterbi is NOT examinable from this lecture** — only the conceptual mapping (hidden states = tags, observations = words). Forward/Viterbi/Baum-Welch math live in the optional notebook.
- **Head vs dependent confusion**: head = governor, dependent = modifier. Arrow points head → dependent.
- **PCFG probabilities** sum to 1 *per LHS*, not over the whole grammar.
- **PCFG picks the highest-probability parse** by multiplying rule probs — not by counting nodes.
- **Constituency vs dependency** are *different formalisms* (CFG vs dependency grammar), not levels of the same parse.
- "Bag-of-words ignores order" — the motivation for this whole lecture; word order matters ("Bob kills mosquitoes ..." vs "Hamlet kills Bob ...").

## Concepts NOT in the slides (common textbook material to flag)

- **Viterbi algorithm, Forward-Backward, Baum-Welch** — only in optional notebook.
- **CYK pseudocode / Chomsky Normal Form / binarization details** — mentioned by name only.
- **Transition-based vs graph-based dependency parsers (arc-standard, MST)** — not in slides.
- **Specific accuracy numbers for neural taggers / UAS / LAS metrics** — not in slides.
- **CRF math, BiLSTM equations** — only diagrams shown.

## Cross-References

- **Lecture 3 (Language Models)** — n-gram LMs handle order via Markov assumption; this lecture goes beyond LMs to phrase/tree structure.
- **Lecture 5 (Text Classification)** — Naive Bayes / VSM use bag-of-words, contrasted here as motivation.
- **Lecture 7 (Word Embeddings)** — PPMI, Word2Vec use bag-of-context, again contrasted as motivation.
- **Later lectures** — RNNs/LSTMs/Transformers (used here only as black-box sequence taggers); CRF layers; BERT for tagging.
- **Applications downstream** — NER, IE (knowledge-graph triples via nsubj/dobj), question answering, text simplification (apposition removal, relative-clause splitting), grammar checking.
