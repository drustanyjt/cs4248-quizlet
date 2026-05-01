# Lecture 2 — Strings & Words

## Outline
- Regular Expressions: basic concepts, metacharacters, character classes, repetition, groups, backreferences, lookarounds.
- Relationship between RegEx and Finite State Automata (FSA) / Regular Languages in the Chomsky Hierarchy.
- RegEx error types: False Positives (Type I) vs False Negatives (Type II).
- Corpus preprocessing: tokenization (character/word/subword), normalization (case folding), stemming/lemmatization.
- Sentence segmentation via rules or binary classifier.

## Key Concepts

**Regular Expression** — Search pattern matching character combinations in strings. Default matches first occurrence; global matches all.

**Metacharacters** — `.` (any char except newline), `^` (start), `$` (end), `|` (logical OR), `\b` (word/non-word boundary).

**Character class** — `[...]` valid set, `[^...]` negates. Predefined: `\d`=`[0-9]`, `\D`, `\s`=`[ \n\r\t\f]`, `\S`, `\w`=`[a-zA-Z0-9_]`, `\W`.

**Repetition** — `+` (1+), `*` (0+), `?` (0 or 1), `{n}`, `{l,u}`.

**Groups** — `(...)` organize patterns; captured individually as a tuple. **Backreference** `\1`, `\2`, ... finds repeated patterns (words starting/ending same letter: `\b([a-zA-Z])\w*\1\b`).

**Lookarounds (assertions)** — Match like a group but don't capture. `(?=B)` positive lookahead, `(?!B)` negative, `(?<=B)` positive lookbehind, `(?<!B)` negative.

**Regular Language / FSA** — Strings accepted by an FSA; equivalent in power to RegEx. Lowest level of the Chomsky Hierarchy (regular ⊂ context-free ⊂ context-sensitive ⊂ recursively enumerable).

**False Positive (Type I)** — Matching what we shouldn't (e.g. `the` matching `other`, `bathe`). **False Negative (Type II)** — Missing what we should match (e.g. `THE`). Reducing one often increases the other.

**Tokenization** — Splitting a string into tokens; vocabulary = set of unique tokens. Three approaches: character-based (trivial, no semantics), word-based (RegEx, language-dependent, OOV), subword-based (learned, tokens often morphemes).

**Out-Of-Vocabulary (OOV)** — Token unseen by a model. Subword tokenization splits OOV/rare words into known frequent subtokens.

**Maximum Matching** — Baseline segmentation: place pointer, find longest dictionary word, advance, repeat. Good on Chinese, fails on English.

**BPE (Byte-Pair Encoding) Token Learner** — Repeatedly merge the most frequently adjacent token pair, k times. `_` marks end-of-word.

**WordPiece** — Like BPE but merges pair maximizing $P(t_1,t_2)/(P(t_1)P(t_2))$ (likelihood ratio) instead of raw frequency. `_` marks continuation of a word.

**Token Segmenter** — Applies learned merges in order of learning to tokenize new words.

**Normalization** — Convert text to canonical form, defining equivalence classes (`Germany`/`GERMANY`→`germany`). Alternative: asymmetric expansion (query `window` searches `window, windows`).

**Case folding** — Lowercasing. Good for IR; bad for NER / MT (`us` vs `US`).

**Stemming** — Rule-based affix chopping; fast, no lexicon, stem may not be a real word (`mice`→`mic(e)`). **Porter Stemmer** = cascade of rewrite rules for English (`sses→ss`, `ies→i`, `(*v*)ing→ε`, `(m>1)ement→ε`).

**Lemmatization** — Reduce to dictionary headword; needs lexicon and POS tags. Same word lemmatizes differently as N/V/A (`running`: N→running, V→run, A→running).

**Sentence Segmentation** — `?` and `!` largely unambiguous; `.` ambiguous (`1.25`, `U.S.A.`, `Dr.`). Approaches: complex RegEx, decision-tree rules (EOS vs N-EOS), or ML classifier with features (capitalization, abbreviation list, word length).

## Important Formulas

WordPiece merge criterion (pick pair maximizing this likelihood ratio):
$$\frac{P(t_1, t_2)}{P(t_1) P(t_2)} = \frac{P(t_2 \mid t_1)}{P(t_2)} \propto \frac{N \cdot \text{count}(t_1, t_2)}{\text{count}(t_1)\cdot \text{count}(t_2)}$$

Compared to BPE which simply uses $\text{count}(t_1, t_2)$. $N$ = total tokens in current corpus (dropped because it doesn't affect ranking).

Bigram / unigram probabilities used above:
$$P(t_2\mid t_1) = \frac{\text{count}(t_1, t_2)}{\text{count}(t_1)}, \qquad P(t_2) = \frac{\text{count}(t_2)}{N}$$

Classic non-regular language (cannot be matched by pure RegEx / FSA): $\{0^n 1^n \mid n \ge 0\}$.

## Worked Example / Canonical Trap

**BPE merge ordering on corpus** `low×5, lower×2, newest×6, widest×3, longer×1`:
1. Most frequent pair `e,s` (9) → merge to `es`.
2. `es,t` (9) → `est`.
3. `est,_` (9) → `est_`.
4. `l,o` (8) → `lo`.
5. `lo,w` (7) → `low`.
6. `n,e` (6) → `ne`.
7. `ne,w` → `new`. 8. `new,est_` → `newest_`.

**BPE Token Segmenter for "newer\_"**: apply learned merges in order: `n e w e r _` → `(n,e)` → `ne w e r _` → `(ne,w)` → `new e r _` → `(e,r)` → `new er _` → `(er,_)` → `new er_`. Result: tokens `new`, `er_`.

**Maximum Matching trap**: greedy longest-match works on Chinese but breaks on `thetabledownthere` (no spaces) → `theta bled own there` instead of `the table down there`.

## Exam Traps & Misconceptions

- **`\b` is a zero-width boundary**, not a character — it matches between `\w` and `\W`, doesn't consume a character.
- Don't confuse **BPE's `_`** (end-of-word) with **WordPiece's `_`** (continuation marker on inner pieces).
- BPE picks the **most frequent** adjacent pair; WordPiece picks the pair maximizing **likelihood ratio** (PMI-like). Same loop otherwise.
- Quick-quiz answer: BPE with `k=0` degenerates to character-based tokenization; `k=∞` degenerates to word-based.
- Don't confuse **stemming** (rule chop, may produce non-words, no POS) with **lemmatization** (lexicon + POS, returns dictionary word).
- Stop-word removal can hurt sentiment analysis: words like `not`, `n't`, `never` flip polarity.
- **Maximum Matching** ≠ universal segmenter — it's specifically a Chinese baseline; expect it to fail on English.
- RegEx ≡ Regular Languages ≡ FSA. Languages like $0^n 1^n$ are NOT regular (need unbounded counter). Modern engines with backreferences/recursion exceed regular, but theoretical RegEx does not.
- False Positive vs False Negative: reducing one often increases the other; they are not symmetrically costly (medical testing example).
- A finite language (e.g. $\{\epsilon, 01, 0011, 000111\}$) is always regular — quick-quiz answer "Yes".

## Cross-References

- **Lecture 3 (n-Gram Language Models)** — bigram/unigram probabilities used in WordPiece criterion are formalized here; the pre-lecture activity ("probability of a sentence") sets this up.
- Tokenization output (vocabulary) feeds every downstream model: classifiers, embeddings, LMs.
- Penn Treebank Tokenizer rules (separate clitics like `doesn't → does n't`, keep hyphenated words, separate punctuation) recur as the de-facto preprocessing baseline.

**Slide-light spots / not in slides**: minimum edit distance, Unicode normalization (NFC/NFKC), the full Porter Stemmer rule set (only 5 example rules shown), Unigram and SentencePiece algorithms (named only), and any quantitative evaluation of tokenizers. If a textbook concept like edit distance or BPE dropout shows up, it is NOT examinable from this lecture.
