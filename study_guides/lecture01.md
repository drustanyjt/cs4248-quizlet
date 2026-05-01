# Lecture 1 — What is NLP and Why is it so Hard?

## Outline
- **What is NLP?** — basic definition, prominent applications, core building blocks of (written) language, fundamental NLP tasks across the analysis hierarchy.
- **Why is NLP hard?** — characteristics of language (ambiguity, expressivity, variation, scale, sparsity, unmodeled representation) and practical challenges (data collection, extraction, quality).
- **The Big Picture** — NLP at the intersection of linguistics / CS / AI, ethical considerations, course meta-topics.

## Key Concepts

**NLP**: subfield of linguistics + CS + AI; how to program computers to process/analyze human language data.
- **R (representation)**: abstract internal model of language/world. **Analysis** = NL → R, **Generation** = R → NL.
- **Communication progression**: symbolic (punch cards, 50s–70s) → formal (programming, 80s) → natural language (chatbots).

**Applications**: machine translation, conversational agents (speech recognition, language analysis, dialogue processing, IR, TTS), question answering, summarization, text generation (image captioning), LLMs with **emergent capabilities** (tasks not explicitly trained for), spelling correction, document clustering/classification (spam, sentiment, authorship).

**LLM reality check**: hallucinations, cost, non-determinism, poor interpretability, poor domain performance. Course goal: build *next-gen* LLM foundations, not use existing ones.

**NLP analysis hierarchy** (shallow → deep) — **memorize this**:
- **Lexical** (chars/morphemes/words): tokenization, normalization, stemming, lemmatization.
- **Syntactic** (phrases/clauses/sentences): POS tagging, syntactic parsing (constituents, dependencies).
- **Semantic** (meaning of words/sentences): WSD, NER, SRL.
- **Discourse** (paragraphs/documents): coreference/anaphora resolution, ellipsis resolution, stance detection.
- **Pragmatic** (world knowledge/common sense): textual entailment, intent recognition.

**Building blocks**: character → morpheme → word → phrase → clause (phrase + subject/verb) → sentence → paragraph → document → corpus.

**Morphology**:
- **Morpheme**: smallest meaning-bearing unit. Word = 1..n stems + 0..n affixes.
- **Free** = stem (stands alone). **Bound** = must attach.
- **Derivational** (prefix or suffix): changes meaning/POS (*un-happy*, *de-frost-er*, *hope-less*).
- **Inflectional** (suffix only in English): grammatical property — tense, number, possession, comparison (*walk-ed*, *elephant-s*, *Bob-'s*, *fast-er*).

**Tokenization granularities**: character-based, subword-based, word-based.

**Tasks per layer**:
- **WSD**: pick correct sense of polysemous word (*bank*: financial vs. river).
- **NER**: identify entities (PERSON, ORG, LOCATION, MONEY).
- **SRL**: predicate-argument — Who did What to Whom, What, When.
- **Coreference**: link expressions referring to same entity.
- **Ellipsis**: reconstruct omitted words (*"his brother [studied] at NTU"*).
- **Textual entailment**: $t \Rightarrow h$ if reader of $t$ infers $h$ likely true.
- **Intent recognition**: classify utterance by speaker goal using context.

**Why NLP is hard — 5 main challenges**: Ambiguity, Expressivity, Variation, Scale, Sparsity (lecturer added "Unmodeled" representation).
- **Ambiguity**: word sense, POS, syntactic (PP-attachment: *"I see the man with a telescope"*), anaphoric, **Winograd Schema** (one-word swap flips reference; needs world knowledge), pragmatic.
- **Expressivity**: same meaning, many forms — paraphrase, formality, idioms (*raining cats and dogs*), neologisms (*selfie, chillax*), sarcasm/irony, **algospeak** (*unalived, sewer slide* — moderation evasion).
- **Variation**: ~6,500 languages, ~150 families; domain & cultural bias.
- **Sparsity** → **Zipf's Law**: frequency ∝ 1/rank; holds across all corpus sizes.
- **Scale**: ~470,000 English dictionary words; >1M in web corpora.
- **Unmodeled representation**: meaning depends on context + shared world knowledge (*"I killed all the children"* — sysadmin or murderer?).

**Practical challenges**: collection (public datasets, APIs, web scraping — legal grey area); extraction (PDF/HTML/DOCX → plain text/Markdown); quality (heuristic/statistical/model-based filtering; deduplication; toxicity & bias; **PII** control; **data decontamination**; **AI inbreeding** / model collapse).

**Big Picture**: NLP = Linguistics ∩ CS ∩ AI ≈ computational linguistics. Desiderata: sensitivity, generality, formal guarantees, accuracy, efficiency, explainability, ethics — *often conflicting*.

## Important Formulas

The lecture is largely qualitative. The one named statistical relationship is:

$$\text{frequency}(w) \propto \frac{1}{\text{rank}(w)} \quad \text{(Zipf's Law)}$$

A log-log plot of frequency vs. rank is approximately linear with negative slope, regardless of corpus size or domain.

Textual entailment notation: given text $t$ and hypothesis $h$, write $t \Rightarrow h$ if a reader of $t$ would infer $h$.

(No other formulas; the actual mathematical form of Zipf — $f(r) \propto 1/r^s$ with parameter $s$ — is **not** in the slides and is not examinable beyond the inverse-rank relationship.)

## Worked Example / Canonical Trap

**Quick Quiz (slide 43, answer C)**: *"After binge-eating cookies, I went cold turkey after Christmas."* → **C: stopped eating cookies completely**. Idiom trap; A and D are literal-reading distractors.

**Quick Quiz (slide 45, open)**: *"I wish I would be on this plane flying through the clouds!"* — sentiment genuinely undecidable without context. Lesson: even sentiment labels are context-dependent.

**Winograd canonical pair**: *"I poured water from the bottle into the cup until it was full"* vs. *"...until it was empty"*. Same syntax; *it* = **cup** in first, **bottle** in second. Requires physical world knowledge.

## Exam Traps & Misconceptions

- **Don't confuse derivational with inflectional morphemes.** Derivational can be prefix or suffix and changes meaning/POS (*hope* → *hopeless*); inflectional is suffix-only in English and only adds a grammatical feature (*walk* → *walked*).
- **Don't confuse free morpheme with stem.** A free morpheme **is** a stem that can stand alone; not all stems are free (some only appear bound).
- **Don't confuse coreference with ellipsis.** Coreference = different expressions referring to the same entity; ellipsis = words *omitted* and reconstructed from context.
- **Don't confuse WSD with NER.** WSD picks among senses of a known word; NER identifies real-world entity types.
- **Common trap: "NLP is solved by LLMs."** The lecturer explicitly rejects this — hallucinations, cost, determinism, interpretability, domain specificity remain open.
- **Trap on Zipf's Law**: it holds across *all* corpus sizes and domains (slide 47 shows three corpora). The takeaway is *infrequent words always exist*, not that small corpora behave differently.
- **Pragmatic ≠ Discourse.** Discourse stays inside the text; pragmatic requires *outside* world knowledge / context.

## Cross-References

- **Lecture 2** (next): text preprocessing — capturing strings/words, cleaning text. Pre-lecture activity: relationship between Finite State Machines and Regular Expressions.
- The full lexical-analysis row (tokenization / normalization / stemming / lemmatization) is **flagged in lecturer's annotation** as the L2 deep-dive — only conceptual coverage here.
- Statistical modeling methods (LSTM, CNN, Transformers) named in "Course Meta Topics" but not detailed — covered in later lectures.
- Recommended notebook *Data Preparation for Training LLMs* expands on the data-quality pipeline; **practical pipeline details from optional notebooks (Token Indexing, Data Batching, NumPy) are NOT examinable** unless re-introduced in lecture.
- Common textbook concepts **NOT in these slides** (so not examinable from L1 alone): formal Zipf parameter $s$, Heaps' Law, IPA / phonetics notation, formal grammar definitions, BPE/WordPiece algorithms (mentioned only as "subword-based" granularity).
