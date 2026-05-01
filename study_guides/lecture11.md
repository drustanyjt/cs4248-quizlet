# Lecture 11 — Augmenting LLMs

## Outline
- **Fine-Tuning LLMs**: Motivation & Challenges, Prompt Tuning, LoRA
- **LLM Augmentation**: Reasoning, RAG, Tool Use, Routing
- **Data Augmentation**: Teacher-generated data (Distillation), Self-instruction, Safety
- This lecture targets *effectiveness*; Lecture 12 targets *efficiency*

## Key Concepts
- **Why fine-tune**: domain adaptation, task adaptation (instruction tuning), knowledge update, style/tone, bias reduction & alignment.
- **Instruction fine-tuning**: samples are *instruction – input (optional) – response* triples; aligns the next-word predictor with user intents.
- **FT challenges**: data quality, compute cost, weakened guardrails, catastrophic forgetting, overfit/underfit, hyperparameter sensitivity, custom-benchmark eval.
- **PEFT**: train only a subset of params; categories — additive, adapters, soft prompts, selective, reparametrization-based.
- **Prompt Tuning**: prepend trainable *soft prompt* embeddings; LLM frozen. Soft prompts ≠ real words; easy to swap per task.
- **Prefix vs Prompt Tuning**: prompt tuning adds vectors only at input; prefix tuning adds trainable vectors to *every* transformer block.
- **LoRA**: adapter where pretrained $W$ is frozen and low-rank update $\Delta W = BA$ is trained. Init: $B=0$, $A \sim \mathcal{N}(0, \sigma^2)$ so $\Delta W = 0$ initially.
- **Why low-rank works**: large-model weight matrices live in approximately low-rank subspaces (PCA on GPT-2 XL weights captures variance with far fewer dims).
- **Adapters**: small trainable modules (FFN_down → ReLU → FFN_up + residual + LayerNorm) added in *parallel*, *sequential*, or *hybrid*. LoRA "only" adds low-rank matrices.
- **Reasoning LLM**: same architecture — distinguished by training (SFT/ICL with intermediate steps) and larger token compute budget.
- **Chain-of-Thought**: prompt with worked intermediate steps; works on arithmetic, commonsense, symbolic tasks.
- **Tree of Thought + Preference Learning**: decompose hierarchically, decode alternatives per step, use RL/MCTS for step-level partial credit.
- **RAG**: knowledge-based prompt engineering — retrieve relevant *chunks* and prepend to prompt. Benefits: grounding, recency, transparency, customisation.
- **Chunking**: fixed-size (±overlap), recursive, document-based, semantic, agentic.
- **Sparse vs Dense retrieval**: sparse = |V|-dim term vectors (TF-IDF, BM25); dense = neural embeddings (single-vector BERT [CLS], or multi-vector ColBERT late interaction).
- **Tool Use**: LLM calls calculators, search, code, custom functions for *complex but deterministic* problems. Examples: Toolformer (self-annotates API calls), PAL (Python interleaved with reasoning), TART (table tools), OpenAI function calling.
- **Routing**: choose which augmentation (direct / CoT / rewrite / retrieve / refuse) and/or which LLM; different LLMs vary in retrieval-utilization ability.
- **Response Distillation**: large *teacher* labels data, small *student* trains on labels — easy but expensive labelling, loses nuance ("AI inbreeding").
- **Self-Instruction**: seed tasks → generate instructions → generate responses → **filter/dedup (critical!)** → SFT → optionally iterate.
- **Confidence scoring**: native logits→softmax (avg/min token), proxy (self-consistency = fraction of N samples agreeing; verifier checks), or ask LLM directly (least reliable).
- **Temperature scaling**: post-hoc divide logits by learned $T$ before softmax; smooths ($T>1$) or sharpens ($T<1$).
- **LLM-as-Judge (LLMaaJ)**: scalable eval; needs rubric, calibration, blinding, randomisation; risks self-preference, prompt injection.

## Important Formulas
- LoRA forward pass: $h = xW + x\Delta W = xW + x\left(\dfrac{\alpha}{r} A B\right)$
- LoRA shapes: $W \in \mathbb{R}^{d \times k}$, $A \in \mathbb{R}^{d \times r}$, $B \in \mathbb{R}^{r \times k}$, with $r \ll \min(d, k)$
- LoRA init: $B = 0$, $A \sim \mathcal{N}(0, \sigma^2)$, $\alpha$ = scaling factor
- Domain-adapter loss (UDAPTER): $\Delta_t = \mathrm{div}(h^{src}, h^{tgt})$ (e.g., KL)
- Temperature-scaled softmax: $q_i = \dfrac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$
- "I-Know" probability: $P(IK)$ — model fine-tuned to predict its own correctness

## Worked Example / Canonical Trap
- **LoRA parameter count**: with $W \in \mathbb{R}^{8 \times 8}$ (64 weights) and rank $r=2$, $\Delta W = AB$ where $A^{8 \times 2}$ and $B^{2 \times 8}$ ⇒ only $8 \cdot 2 + 2 \cdot 8 = 32$ trainable weights (half of full fine-tune, and savings scale dramatically with $d$).
- **Why LoRA over adapters? (lecture quiz answer)**: LoRA weights can be **merged** with pretrained weights ($W' = W + BA$) for **zero inference latency**, unlike sequential adapters that add forward-pass depth.
- **Naive $\Delta W$ trap**: implementing $\Delta W$ as a full $d \times k$ matrix gives no parameter savings — defeats the point. Must factor as low-rank $BA$.
- **RAG diverse-data quiz**: under deadline pressure, *create chunks with sufficient overlap* to lower the risk of splitting related content.

## Exam Traps & Misconceptions
- **Prompt tuning ≠ prompt engineering**: soft prompts are *trained continuous embeddings*, not human-written words.
- **LoRA freezes $W$**: only $A, B$ are updated; the pretrained model is preserved → multiple task adapters can be swapped on the same base.
- **Full fine-tuning risks**: catastrophic forgetting, basically impossible to "unlearn", overfit on small data, weakens existing safety guardrails.
- **Reasoning LLMs are not a new architecture**: distinguished by training data (intermediate steps) and compute/token budget.
- **RAG is still better than hallucination** even when retrieval is imperfect (Streufdorf example).
- **Toolformer is self-supervised**: it filters API calls by whether the call *reduces loss* on the next tokens.
- **Self-instruction filtering is the critical step** — skipping it amplifies noise and causes model collapse.
- **LLMaaJ is unsuitable for high-stakes correctness** (medical, legal); it is susceptible to self-preference and prompt injection.
- **Data augmentation safety**: paraphrasing/back-translation can produce near-duplicates; augmentations can leak across train/test (evaluation contamination); PII patterns (NRIC, phone) must be validated.

## Cross-References
- **L10 (Pretrained Models / Transformers)**: provides the frozen $W$ that PEFT/LoRA/adapters modify; CoT and instruction fine-tuning build on the next-word prediction objective.
- **L12 (Efficiency)**: this lecture is *effectiveness*, L12 is *efficiency* — distillation extends here (logits / feature distillation noted as "more details in L12"); LoRA's mergeability ties into inference cost.
- **CS3245 (IR)**: sparse retrieval (TF-IDF, BM25, inverted index) and dense retrieval (DSSM, ColBERT) underpin RAG storage/retrieval.
