# Lecture 6 — Introduction into Connectionist Machine Learning

## Outline
- Generative vs. Discriminative Classifiers
- Logistic Regression: probabilistic setup, cross-entropy loss, gradient descent, overfitting & regularization, multiclass
- Towards Neural Networks: motivation (XOR problem), basic feedforward NN architecture

## Key Concepts
- **Generative classifier**: learns joint $P(x,y)$, applies Bayes' rule to get $P(y|x)$; e.g., Naive Bayes. Models "what each class looks like."
- **Discriminative classifier**: learns $P(y|x)$ directly; only models the decision boundary between classes.
- **Bias trick**: introduce constant feature $x_0 = 1$ so bias absorbs into $\theta$; then $h_\theta(x) = f(\theta^T x)$.
- **Logistic (sigmoid) function**: squashes any real number into a probability.
- **Logistic Regression**: linear model + sigmoid → real-valued probability $\hat{y} = P(y=1|x,\theta)$. Discriminative, linear decision boundary.
- **Cross-entropy loss** $L_{CE}$: derived from negative log-likelihood of Bernoulli; convex for LR → unique global minimum.
- **Gradient**: vector of partial derivatives, points in direction of steepest ascent. Descent moves opposite direction.
- **Learning rate $\eta$**: scaling factor on gradient (typical 0.01–0.0001). Too small → slow; too large → oscillates/diverges.
- **GD variants**: Batch (whole dataset, smooth), Mini-batch (some samples, in practice called "SGD"), Stochastic (per-sample, choppy).
- **Overfitting**: model "too powerful" → very large $\theta$ values → fits artifacts of training data (e.g., LR over-emphasizes pronoun feature if training set has artifact).
- **Regularization**: penalty added to loss to discourage large $\theta$. L2 = Ridge; L1 = Lasso.
- **Softmax**: generalizes sigmoid to $C$ classes; converts vector of scores into vector of probabilities summing to 1.
- **XOR problem**: not linearly separable → motivates stacking LR units.
- **Neural Network (Feedforward)**: stacked LR-like units, no loops. **Depth** = #layers, **Width** = #neurons per layer.
- **Activation function**: must be **non-linear** for hidden layers (else network collapses to single linear transform). Examples: Sigmoid, Tanh, ReLU, Leaky ReLU, Sign.

## Important Formulas
- Sigmoid: $\sigma(x) = \dfrac{1}{1+e^{-x}}$ (general logistic: $f(x) = \dfrac{L}{1+e^{-k(x-x_0)}}$, with $L=1$ to be a probability)
- Logistic regression hypothesis: $\hat{y} = h_\theta(x) = \sigma(\theta^T x) = P(y=1\mid x,\theta)$
- Bernoulli combined form: $P(y\mid x) = \hat{y}^{y}(1-\hat{y})^{1-y}$
- Binary cross-entropy: $L_{CE}(\hat{y},y) = -[y\log\hat{y} + (1-y)\log(1-\hat{y})]$
- Total loss over $m$ samples: $L_{CE} = -\dfrac{1}{m}\sum_{j=1}^m [y^{(j)}\log\hat{y}^{(j)} + (1-y^{(j)})\log(1-\hat{y}^{(j)})]$
- Gradient (vectorized): $\dfrac{\partial L_{CE}}{\partial \theta} = \dfrac{1}{m}X^T[\sigma(X\theta) - y]$
- Gradient descent update: $\theta \leftarrow \theta - \eta \,\nabla_\theta L$
- L2 regularization: $L + \lambda \sum_{i=1}^n \theta_i^2$ (gradient adds $+\lambda\,\dfrac{2}{n}\theta$)
- L1 regularization: $L + \lambda \sum_{i=1}^n |\theta_i|$
- Softmax: $P(y=c\mid x) = \dfrac{\exp(\theta_c^T x)}{\sum_{i=1}^C \exp(\theta_i^T x)}$
- Generalized cross-entropy (multiclass): $L_{CE}(\hat{y},y) = -\sum_{i=1}^C y_i \log \hat{y}_i$ ($y_i=1$ for correct class, 0 else)
- Layer-wise NN activation: $x^{[l]} = a^{[l]} = g(\theta^{[l]} x^{[l-1]})$, with weight matrix $\theta^{[l]} \in \mathbb{R}^{d^{[l]} \times d^{[l-1]}}$

## Worked Example / Canonical Trap
- **LR on a movie review** (6 features incl. # positive words = 3, # negative words = 2, etc.): with given $\theta$, $\theta^T x = 0.833 \Rightarrow \hat{y} = \sigma(0.833) = 0.7 \Rightarrow$ classify as positive. If true $y=1$: $L_{CE} = -\log 0.7 = 0.36$; if true $y=0$: $L_{CE} = -\log 0.3 = 1.2$. After one GD step ($\eta=0.1$), loss drops 0.36 → 0.12 → 0.075.
- **Effects of learning rate** on $L = x^2$, $\partial L/\partial x = 2x$ over 20 steps: $\eta=0.2$ converges nicely; $\eta=0.8$ slower but still converges; $\eta=1.0$ stuck oscillating between two points; $\eta=1.01$ diverges outward.
- **XOR not linearly separable** but solvable by stacking: XOR $= $ OR AND NAND. Each of OR, AND, NAND is a single LR-unit (perceptron with step function); stack into a 2-layer feedforward NN to compute XOR.

## Exam Traps & Misconceptions
- Sigmoid range is $(0,1)$, **not $[0,1]$** — the model can never output exactly 0 or 1. Therefore cross-entropy is **never exactly 0** even on a correct sample (Slide 43: "loss will be small," not zero).
- LR loss is **convex** (one global minimum) — but NN loss is **non-convex** (local minima, training harder).
- Random search for $\theta$ works in toy cases but is impractical → use gradient descent.
- The closed-form $\nabla L = 0$ has **no closed-form solution** for LR → must use iterative GD.
- **Scaling features changes $\theta$** values, but does not change LR's predictive performance (Slide 54: B is the True statement).
- Without **non-linear** activations, stacking layers collapses to a single linear transformation: $y = W_3W_2W_1 x + b' = W'x + b'$. Hidden layers therefore *require* non-linear activations.
- Linear activation is fine for **regression output layers**; sigmoid/softmax for **classification output layers**.
- Softmax with $C=2$ reduces to sigmoid (multiclass generalizes binary).
- "Stacked LR" with **step function** activation is technically a **Perceptron**, not a Logistic Regression unit.
- Overfitting *is* possible with only 1 feature — false intuition (Slide 54 A is wrong).
- Regularization typically **worsens training loss** (it constrains the model), but improves generalization.
- More neurons = more capacity = higher overfitting risk.

## Cross-References
- **Lecture 4/5 (Naive Bayes)**: NB is the canonical generative classifier; LR is its discriminative counterpart. Both need feature engineering (no composite features).
- **Lecture 7+ (Neural Networks deeper)**: this lecture sets up backpropagation (computing $\partial L/\partial \theta$ in deep nets), word embeddings, and architectures like CNN/RNN/Transformer that all reuse the LR-unit + non-linearity + GD recipe.
- **Lecture on word vectors (next)**: pre-lecture warm-up asks about sparse vs. dense vectors / tf-idf — links LR's hand-crafted features to learned dense representations.
