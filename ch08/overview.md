# Neural networks and deep learning

## 1. Topic at a glance

**Main idea:** learn a mapping by adjusting the strengths of connections between simple processing units. A neuron combines inputs and applies an activation function; hidden layers learn intermediate representations. Training compares predictions with labelled examples and adjusts parameters to reduce a loss. These materials concern **supervised classification and regression**, including prediction from images and sequences. A network that predicts a robot's action from labelled examples is still supervised learning; reinforcement learning is not developed here.

The main inductive bias depends on the architecture: a single summation neuron gives a linear decision boundary or linear regression function; a multilayer network composes nonlinear transformations; a CNN reuses local feature detectors across a grid; an RNN reuses a state transition across a sequence. More capacity allows more complex mappings but also makes memorisation possible. Representative data, suitable preprocessing, and control of overfitting remain essential.

**Scope:** perceptron, ADALINE, shallow/deep feedforward networks, functional link networks, product unit networks, cascade correlation networks, Elman/Jordan/multi-recurrent networks, time-delay networks, CNNs, and LSTMs. Radial basis function networks are named in the slides but not explained. The chapter provides the detailed CNN/LSTM coverage; the slides add several shallow architectures. [Slides, displayed slides 113–140](05errorBasedLearningTopic5.pdf); [textbook, §§8.2–8.4](fundamentals_of_ml_2ed_ch.pdf).

Keep these categories separate:

| Category | Items in these materials |
| --- | --- |
| Prediction models / architectures | Perceptron, ADALINE, feedforward NN and architectural variants, CNN, RNN, LSTM |
| Activation functions | Linear, step, ramp, sigmoid, tanh, ReLU and variants, ELU, GELU, Swish; softmax for an output vector |
| Objectives | Squared error, cross-entropy, regularised loss; MAE appears in a quiz |
| Gradient computation | Backpropagation; backpropagation through time (BPTT) |
| Parameter optimisation | Gradient descent, momentum, conjugate gradient/SCG, LeapFrog, metaheuristics such as PSO |
| Training organisation / control | Stochastic, mini-batch, full-batch updates; early stopping, dropout, active learning, pruning/growing |

**Source numbering:** the slides contain displayed slides **108–181**, corresponding to PDF pages **1–74** (PDF page = displayed slide − 107). The textbook extract contains printed pages **381–524**, corresponding to PDF pages **1–144** (PDF page = printed page − 380). References below use displayed slide numbers and printed textbook pages unless explicitly labelled otherwise.

## 2. Quick comparison

All architectures require numerical representations; categorical codes are not automatically meaningful numerical measurements.

| Model | Task / target type | Descriptive feature types and preparation | Main strength | Main limitation |
| --- | --- | --- | --- | --- |
| Perceptron | Binary categorical | Numeric / encoded categories; scale for stable updates | Simple linear classifier | Cannot perfectly separate nonlinear classes |
| ADALINE | Numeric; binary after thresholding | Numeric / encoded categories; scale | Simple squared-error linear model | Linear mapping; outlier influence |
| Feedforward NN / MLP | Numeric, binary, multiclass | Fixed vector; encode, impute, scale | Learns nonlinear representations | Capacity and optimisation need tuning |
| Functional link NN | As feedforward NN | Numerical function expansion, then network | Supplies useful nonlinear features | Expansion choice and dimension |
| Product unit NN | Numeric / categorical with suitable head | Numerical inputs with careful domain handling | Compact multiplicative interactions | Unstable gradients; zero/sign problems |
| Cascade correlation NN | Numeric / categorical with suitable head | Prepared fixed vector | Adds representational capacity as needed | Growing complexity; training detail limited locally |
| Time-delay NN | Numeric / categorical over time | Encoded lag window; preserve temporal order | Explicit finite history | Cannot access history outside its window |
| Simple RNN: Elman, Jordan, multi-recurrent | Sequence or whole-sequence targets | Ordered numeric / encoded sequence | Maintains evolving context | Long-range gradient instability |
| CNN | Image/grid classification or regression | Numeric grid; preserve layout/channels, scale | Shared local detectors reduce parameters | Locality bias; detail lost by pooling |
| LSTM | Sequence or whole-sequence targets | Ordered numeric / encoded sequence | Gated memory for longer dependencies | More parameters and sequential computation |

## 3. Individual model summaries

### Shared preparation and evaluation

The seven-row tables below distinguish model behaviour from possible remedies. Some practical mitigations are deductions from the models, rather than extensions explicitly taught: **imputation, missingness indicators, class weighting/resampling, and task-appropriate evaluation do not become automatic network capabilities**. Where no architecture-specific extension is supplied, the tables say so.

- Split data before fitting imputers, encoders, scalers, feature selection, or resampling. Fit them using the training split, or inside each cross-validation fold. Tune architecture, thresholds, regularisation, and stopping using validation data; reserve a test set for final assessment.
- Scale continuous and discrete measurements where their units/ranges differ. For nominal categories, use indicators such as one-hot encoding. Ordinal encoding represents order but also imposes numerical spacings; use it only when that assumption is reasonable. Mixed data become a combined numerical representation. Unseen categories need a planned encoding.
- For regression, assess error in meaningful target units after reversing any target scaling. For classification, inspect per-class errors; accuracy alone can hide minority-class failure. Precision/recall and macro-averaged measures help when classes are skewed. These metric suggestions are practical applications, not an additional evaluation chapter.
- For sequences, preserve order **within** each sequence. Shuffle independent training sequences or windows only when appropriate. Use time-respecting evaluation for forecasting and avoid leakage between overlapping windows or related subjects.

### 3.1 Perceptron

#### Main idea and assumptions

A classical perceptron is a **single weighted-sum unit with a step output**, learning one separating hyperplane. Linear separability is needed for perfect classification and the usual perceptron convergence guarantee, not for making predictions or running the update. Nonseparable data can make updates continue indefinitely. The manually configured McCulloch–Pitts neuron is a precursor, not a trained perceptron. [Slides 113–126; textbook §8.2.5, pp. 396–398.]

#### How it learns and predicts

For input $x=(x_1,\ldots,x_d)$, weights $w$, and bias $b$:

$$z=w^\top x+b,\qquad \hat y=\mathbb{1}[z\geq0].$$

Here $d$ is the number of numerical input features and $\mathbb{1}$ returns 1 when its condition holds. With targets $y\in\{0,1\}$:

1. Initialise parameters; repeatedly present labelled examples.
2. Compute $\hat y$ and error $e=y-\hat y$.
3. Update $w\leftarrow w+\eta e x$ and $b\leftarrow b+\eta e$, where $\eta>0$ is the learning rate.
4. Stop after a mistake-free pass or another stopping limit.

Prediction uses only the weighted sum and threshold. The update is the perceptron learning rule, **not differentiation through the step function**. With $b=-\theta$, this is the slides' threshold convention. Using an extra constant input $x_0=1$ folds $b$ into $w_0$; the slides also use a constant −1 convention. Keep signs consistent. [Slides 121–126.]

Example: $\hat y=\mathbb{1}[x_1+x_2-1.5\geq0]$ implements AND for binary inputs. XOR, and the quiz's XNOR table, cannot be separated by a single line.

#### Supported data and tasks

Numeric continuous/discrete features are direct inputs; nominal, ordinal, and mixed features need the preparation above. The standard unit predicts a binary label, with an uncalibrated margin $z$. It does not produce probabilities. Numeric regression requires a linear-output variant such as ADALINE; multiclass classification requires multiple units or another architecture.

#### Data quality and preprocessing

| Issue | Effect on the standard model and why | Relevant extension and how it helps | Preprocessing / mitigation and tradeoffs |
| --- | --- | --- | --- |
| Missing values | Missing inputs make the dot product undefined; missing targets prevent a supervised update. | No missing-data extension taught. | Impute inputs; indicators may preserve missingness information. Exclude unlabelled examples from supervised updates; imputation can distort boundaries. |
| Outliers | Large inputs can cause large updates and tilt the boundary. Targets are binary, so an extreme numeric target is inapplicable. | No robust perceptron extension taught. | Check measurement errors; scale or cap justified extremes, risking removal of useful rare cases. |
| Noise | Input noise moves points across the boundary; wrong labels can destroy separability and prevent convergence. | Nonlinear networks add capacity, but can memorise noise. | Check labels and use a finite stopping limit plus validation; extra capacity is not noise removal. |
| Skew class distributions | More majority examples influence updates; the rule does not encode class costs. | No imbalance-specific extension taught. | Rebalance training examples or use cost-aware updates as an additional design choice; assess minority errors and tradeoffs. |
| Irrelevant features | Add spurious directions and opportunities to fit chance patterns. | Feature selection is recommended in slide 129. | Select on training data; removing a feature can also remove an interaction useful to a richer model. |
| Different value ranges | Inputs scale their corresponding updates; units affect the optimisation path. | No scale-invariant training extension taught. | Scale numerical features using training statistics. |
| Different feature types | Weighted sums require numbers; arbitrary category codes impose false order and spacing. | Multiple layers change representational capacity, not native categorical support. | Encode nominal/ordinal features appropriately; one-hot expansion can be high dimensional. |

Redundant/correlated features make coefficients non-unique or unstable and duplicate their influence during training. High dimensionality can make training separation easy while generalisation remains poor. A small sample gives weak evidence about the boundary, especially in scarcely sampled regions.

#### Important choices and variants

Weights and bias are learned parameters; learning rate, stopping limit, input representation, and threshold are design choices. A sigmoid unit with an appropriate classification loss gives logistic regression; it is a different activation/training combination, discussed under feedforward networks. Replacing a step by a sigmoid does not create nonlinear decision boundaries in the original input space.

#### Strengths, limitations, and practical use

Useful as a cheap, interpretable linear baseline. It stores $d+1$ parameters and needs $O(d)$ arithmetic per example/update; a pass through $n$ examples costs $O(nd)$. These costs exclude preprocessing. A nonlinear problem needs transformed inputs or a richer architecture; simply training longer cannot fix representational limits.

### 3.2 ADALINE

#### Main idea and assumptions

ADALINE uses a linear output and learns by least mean squares (Widrow–Hoff learning). It fits a linear/affine relationship; classification applies a threshold **after** computing the continuous output. Unlike the perceptron, training error uses the unthresholded output. It needs no Gaussian-noise assumption to fit by squared error; such assumptions belong to statistical interpretations, not forward prediction. [Slide 115.]

#### How it learns and predicts

Compute $\hat y=w^\top x+b$ and minimise

$$J=\frac{1}{2n}\sum_{i=1}^n(y_i-\hat y_i)^2.$$

Here $i$ indexes the $n$ examples. The online LMS update is $w\leftarrow w+\eta(y_i-\hat y_i)x_i$, with the analogous bias update. Repeat passes until a stopping criterion. For regression, return the continuous value; for binary classification, threshold it. This is linear regression trained iteratively, not a nonlinear deep network.

#### Supported data and tasks

Continuous/discrete numeric inputs are direct; categorical and mixed inputs require encoding. It supports numeric targets, or binary categorical targets encoded numerically and thresholded. Its raw output is unbounded and is not a probability; a multiclass head requires an extension.

#### Data quality and preprocessing

| Issue | Effect on the standard model and why | Relevant extension and how it helps | Preprocessing / mitigation and tradeoffs |
| --- | --- | --- | --- |
| Missing values | Undefined input arithmetic; absent target gives no residual. | No missing-data extension taught. | Impute inputs; obtain labels or omit unlabelled rows from supervised fitting. |
| Outliers | Extreme inputs can have leverage; extreme targets/residuals contribute quadratically. | General weight regularisation is taught later; it does not make squared loss robust. | Investigate extremes; justified transformations may help but change the fitted relationship. |
| Noise | Input errors distort fitted slopes; target noise increases residual variance. | Capacity control in later NN material limits fitting noise, not input corruption itself. | Clean errors and validate. Zero-mean input noise can still bias the fit. |
| Skew class distributions | With classification coding, majority examples can dominate squared error. Not a class issue for numeric regression. | No ADALINE imbalance extension taught. | Reweight/resample if needed; assess each class and tune threshold on validation data. |
| Irrelevant features | Can receive spurious coefficients, especially with little data. | Later weight decay shrinks coefficients. | Training-only selection trades variance against possible omitted signal. |
| Different value ranges | Scale changes gradient sizes and conditioning. | No ADALINE-specific remedy taught. | Standardise measurements; adjust learning rate. |
| Different feature types | Requires a numerical vector, including encoded categories. | No native mixed-data extension taught. | One-hot nominal features; justify ordinal spacings. |

Collinearity can make weights non-unique while predictions remain similar; high dimensionality and few examples increase estimation uncertainty. Use a linear model when that simplicity is plausible, rather than assuming an iterative training rule makes it nonlinear.

#### Important choices and variants

Learned parameters are $w,b$. Learning rate, online versus batch updates, stopping, target coding, and classification threshold are choices. A too-large learning rate can cause divergence even though the squared-error linear problem is convex; too small a rate slows learning.

#### Strengths, limitations, and practical use

Simple, inexpensive, and interpretable; the slides mention adaptive signal processing. Arithmetic/storage costs match the perceptron per update/prediction. Its linearity underfits nonlinear mappings; squared error can overemphasise outliers.

### 3.3 Multilayer feedforward neural network (MLP; shallow or deep)

#### Main idea and assumptions

A feedforward network has no feedback cycles. A fully connected MLP links every unit to the next layer. Nonlinear hidden layers learn features that can combine into curved boundaries and complex regression functions. Hidden units do not need labelled intermediate targets: backpropagation assigns credit through the chain rule. Depth can represent hierarchical/compositional mappings efficiently, but extra depth is not automatically better.

There is no requirement that inputs be independent, normally distributed, or linearly separable. Generalisation does require relevant coverage of future cases; approximation theorems establish representational capacity, not that finite-data optimisation will find a good solution. [Textbook §§8.2.2–8.2.5, pp. 388–403.]

#### How it learns and predicts

For one example, let $a^{(0)}=x$. For layers $\ell=1,\ldots,L$:

$$z^{(\ell)}=W^{(\ell)}a^{(\ell-1)}+b^{(\ell)},\qquad
a^{(\ell)}=\phi_\ell(z^{(\ell)}).$$

$L$ counts processing layers, $W^{(\ell)}$ is the matrix of learned incoming weights, $b^{(\ell)}$ is the bias vector, $z^{(\ell)}$ is the preactivation, and $\phi_\ell$ is the chosen activation. The final activation is the prediction. Vectors for several examples can be processed as matrices. [Textbook §§8.2.3, 8.3.4.]

**Training sequence:** choose a task-compatible architecture/loss; initialise weights to break symmetry; shuffle independent examples; for each mini-batch, run a forward pass, calculate loss, backpropagate gradients, then update parameters. Repeat epochs, monitor validation performance, and retain the best parameters. Compute gradients using the current forward-pass parameters before changing them. An epoch is one pass over training data, not one weight update.

For half-squared error on one example, $E=\tfrac12\sum_k(a_k^{(L)}-y_k)^2$, and elementwise activations, define $\delta^{(\ell)}=\partial E/\partial z^{(\ell)}$. Then

$$\delta^{(L)}=(a^{(L)}-y)\odot\phi_L'(z^{(L)}),$$

$$\delta^{(\ell)}=((W^{(\ell+1)})^\top\delta^{(\ell+1)})\odot\phi_\ell'(z^{(\ell)}),$$

$$\frac{\partial E}{\partial W^{(\ell)}}=\delta^{(\ell)}(a^{(\ell-1)})^\top,
\qquad \frac{\partial E}{\partial b^{(\ell)}}=\delta^{(\ell)}.$$

$k$ indexes output units, $y$ is the target vector, $\odot$ denotes elementwise multiplication, and the prime denotes the activation derivative. For an average mini-batch objective $J$, average these example gradients and update each parameter $\theta\leftarrow\theta-\eta\,\partial J/\partial\theta$. This notation uses a **negative gradient update**; defining error as target minus prediction changes signs in intermediate formulas. [Textbook §§8.3.2–8.3.4, Eqs. 8.13–8.30, pp. 408–420; slides 143–149.]

**Prediction:** preprocess an input and run the forward pass only, with dropout disabled. There are no gradient updates at inference.

#### Supported data and tasks

Inputs are fixed-length numerical vectors, including continuous/discrete measurements and encoded nominal/ordinal features. Mixed inputs work after encoding and imputation; high-cardinality one-hot features can greatly increase parameter count.

| Task | Output and usual loss | Interpretation |
| --- | --- | --- |
| Numeric regression | Linear output with squared error | Numeric estimate; bounded output is an alternative when bounds are appropriate |
| Binary classification | One sigmoid output with binary cross-entropy | Estimated probability of class 1; apply a chosen threshold for a label |
| Mutually exclusive multiclass | $K$ softmax outputs with categorical cross-entropy | Probability distribution across $K$ classes; usually choose the largest |

For sigmoid, $\sigma(z)=1/(1+e^{-z})$. For multiclass logits $z_1,\ldots,z_K$:

$$p_k=\frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}},\qquad
E_{\mathrm{CE}}=-\sum_{k=1}^K y_k\log p_k=-\log p_c.$$

$p_k$ is the estimated probability of class $k$, $y$ is its one-hot target vector, and $c$ is the true class. Softmax plus cross-entropy gives the logit gradient $p-y$. Binary cross-entropy is $-[y\log p+(1-y)\log(1-p)]$. Logits are scores, not probabilities; even sigmoid/softmax probabilities need not be well calibrated. Softmax assumes mutually exclusive classes; multilabel outputs would need a different formulation, not developed here. [Textbook §8.4.3, pp. 463–467, Eqs. 8.65–8.72.]

Do not force an unbounded regression target through a sigmoid: either choose a linear output or deliberately transform targets and reverse the transformation. Class labels 0 and 1 remain legitimate sigmoid targets even though finite sigmoid outputs lie strictly inside $(0,1)$.

#### Data quality and preprocessing

| Issue | Effect on the standard model and why | Relevant extension and how it helps | Preprocessing / mitigation and tradeoffs |
| --- | --- | --- | --- |
| Missing values | Missing inputs propagate undefined values; missing targets cannot contribute the ordinary supervised loss. | Dropout is training regularisation, not native imputation. No missing-data model taught. | Impute inputs; indicators may help. Omit absent labels from the loss rather than inventing targets. Imputation may conceal uncertainty. |
| Outliers | Large inputs can saturate sigmoid/tanh or produce large ReLU activations/gradients. Extreme regression residuals dominate squared loss. | Regularisation controls capacity, not the influence function of squared loss; MAE is mentioned in quiz 6.3 Q15. | Verify anomalies; use justified scaling, transformations, or a deliberately changed loss. Keep meaningful rare examples; scaling alone does not remove outliers. |
| Noise | Input errors disturb activations; wrong labels or noisy numeric targets can be memorised by a large network. | Early stopping, weight penalties, smaller architectures, dropout reduce fitting of accidental detail. | Clean systematic errors, validate capacity. Zero-mean noise is not automatically harmless, especially in inputs or small samples. |
| Skew class distributions | Unweighted loss counts each example equally, so abundant classes dominate aggregate gradients and can hide minority errors. | No imbalance-specific architecture is taught; softmax does not correct imbalance. | Consider training-only class weighting/resampling and threshold tuning. These alter cost/probability tradeoffs and may overfit scarce minority examples. |
| Irrelevant features | Extra inputs can acquire chance associations and add incoming weights. Learning may downweight them but does not guarantee selection. | Weight decay, weight elimination, pruning, and architecture selection are taught. | Training-only feature selection; preserve useful interactions. Dropout does not reliably identify irrelevant inputs. |
| Different value ranges | Unequal gradient magnitudes hinder optimisation; large preactivations can saturate units. | Appropriate initialisation and activations improve gradient flow, not automatic unit invariance. | Standardise/normalise measurements. The slides' sigmoid range [−3,3] is a heuristic, not a requirement. |
| Different feature types | Dense arithmetic supports numbers; raw category identifiers impose unintended geometry. | Softmax/one-hot targets add multiclass support; no native categorical-input processing is taught. | Encode nominal, ordinal, and mixed inputs appropriately. One-hot increases dimension; discretising numeric features loses detail. |

Redundant inputs/multicollinearity can spread signal across many interchangeable weights, making individual weights hard to interpret. High dimensionality increases parameters: a dense layer with $d$ inputs and $h$ units has $h(d+1)$ parameters. Limited samples or sparse coverage can allow excellent training fit but unreliable interpolation/extrapolation. Regularisation reduces variance at the risk of underfitting; it cannot create missing coverage. [Slides 129, 172–175; textbook §8.4.4.]

#### Important choices and variants

- **Architecture:** depth and width determine capacity. Too small can underfit; too large can memorise. With only affine/linear hidden operations, layers collapse to one affine map: $W_2(W_1x+b_1)+b_2=(W_2W_1)x+(W_2b_1+b_2)$. A nonlinear output alone does not give rich hidden representations. [Textbook §8.2.4.]
- **Hidden activations:** sigmoid and tanh saturate; sigmoid's derivative is at most $1/4$, contributing to vanishing gradients through repeated products. ReLU, $\max(0,z)$, has derivative 1 on the positive side, but units can become inactive with zero gradient on all training cases. Leaky ReLU uses a fixed small negative-side slope; PReLU learns it. The slides additionally list ramp, ELU, GELU, and Swish without comparable development. [Slides 117–120; textbook §8.4.1, pp. 434–447.]
- **Initialisation:** avoid identical hidden-unit weights, saturation, and unstable activation/gradient magnitudes. Slide 151 uses a shallow-network uniform range $[-1/\sqrt f,1/\sqrt f]$, where $f$ is fan-in. The textbook discusses Xavier variance $2/(n_{\rm in}+n_{\rm out})$, a fan-in-only variance $1/n_{\rm in}$, and He variance $2/n_{\rm in}$ for ReLUs; these are **variances**, not uniform bounds or standard deviations. Here fan-in/out refer to the input/output dimensions of the relevant weight transformation. [Textbook §8.4.2, Eqs. 8.61–8.63, pp. 458–461; see source qualification below.]
- **Optimisation:** learning rate too small slows learning; too large can oscillate/diverge. A schedule reduces it over training. Batch size trades update noise against computation/memory. Full batch updates after all $n$ examples, stochastic after one, mini-batch after an intermediate number. Loss choice does not determine update mode.
- **Momentum:** with parameter vector $\theta$, gradient $g_t$, previous update $v_{t-1}$, learning rate $\eta$, and coefficient $\mu$, one convention is $v_t=\mu v_{t-1}-\eta g_t$, $\theta_{t+1}=\theta_t+v_t$. It smooths persistent directions but may overshoot. It is optional in stochastic and batch learning. [Slide 154; qualified against quiz 6.3 Q7.]
- **Capacity control:** minimise $J=J_{\rm data}+\lambda\Omega(W)$, where $\lambda$ is penalty strength and $\Omega$ a complexity penalty. Weight decay uses a sum of squared weights; weight elimination uses a saturating penalty to encourage removing weak connections. Neither makes all small weights irrelevant. Pruning removes units/weights; growing adds capacity when needed. Validate the effect. [Slides 173–175.]
- **Early stopping:** monitor a validation set with patience; restore the checkpoint with best validation error. One increase can be fluctuation. A maximum epoch limit or tiny gradient indicates stopping, not proof of good generalisation. [Slide 172; textbook §8.4.4, pp. 472–474.]
- **Dropout:** at training time replace activation $a$ by $m\odot a/q$, with independent Bernoulli keep indicators $m$ of probability $q$. This discourages dependence on particular units. Keep the full network at inference; too much dropping can underfit. The backward derivative through this operation also includes $m/q$. [Textbook §8.4.4, pp. 473–477; qualification of Algorithm 7 below.]
- **Adaptive sigmoid:** slide 176 learns slope $\alpha$ and amplitude $\gamma$ in $f(z)=\gamma\sigma(\alpha z)$. The slide calls the slope $\lambda$; $\alpha$ here avoids confusion with regularisation. With positive parameters, output is $(0,\gamma)$ and the maximum input derivative is $\gamma\alpha/4$. Increasing slope raises the central peak but narrows the active region; it does not increase gradients at every input. $\gamma$ scales amplitude, not a horizontal offset.

Weights/biases, PReLU slopes, and adaptive activation parameters are **learned parameters**. Layer sizes, optimiser settings, initialisation method, penalty strength, dropout probability, and patience are **hyperparameters** unless a separate selection procedure chooses them.

**Functional link and cascade variants are described separately below.** They retain the task-specific output/loss choices above.

#### Strengths, limitations, and practical use

Good for complex nonlinear mappings when sufficient representative data and computation are available. Hidden features can replace some manual feature engineering, but training is nonconvex and outcome depends on architecture and optimisation. Internal representations are generally less interpretable than a small linear model; raw weight magnitude is not a reliable feature-importance measure.

For a dense network with $P$ parameters, one forward pass and one backpropagation each take roughly $O(P)$ arithmetic per example; $e$ epochs on $n$ examples take $O(enP)$, ignoring validation and hardware effects. Model storage is $O(P)$; training additionally stores activations and optimiser state. Unlike instance-based prediction, inference does not require retaining every training example. Large models often need substantial compute and can be poor choices for small, simple datasets. [Textbook §8.5, pp. 521–523.]

### 3.4 Functional link neural network (FLNN)

#### Main idea and assumptions

Expand the original inputs into functions $h_1(x),\ldots,h_m(x)$, then learn a network on that representation. Useful chosen functions can expose nonlinear relationships directly. The slide shows this expansion **followed by a hidden layer**, so it should not be silently replaced by a different FLNN architecture with no hidden layer. The source does not specify the basis functions or whether every expansion parameter is fixed or learned. [Slide 134.]

#### How it learns and predicts

Construct $h(x)$, fit the downstream feedforward network using the procedure in §3.3, and apply the same expansion before every prediction. If the expansion itself has trainable parameters, gradient training additionally requires its derivatives; the local slide does not supply that learning procedure.

#### Supported data and tasks

The downstream output determines numeric, binary, or multiclass targets. Continuous/discrete numeric features and encoded nominal/ordinal/mixed inputs can be expanded, but functions must respect their meaning: squaring an arbitrary category ID does not create a sensible feature.

#### Data quality and preprocessing

| Issue | Effect on the standard model and why | Relevant extension and how it helps | Preprocessing / mitigation and tradeoffs |
| --- | --- | --- | --- |
| Missing values | A missing input can invalidate several expanded features; absent target still prevents supervised loss. | No FLNN-specific missing-data extension taught. | Impute before expansion; this can propagate imputation assumptions into many terms. |
| Outliers | Some expansions magnify extreme inputs; squared-error outputs retain target-outlier sensitivity. | General NN regularisation helps capacity, not extreme values directly. | Choose stable functions and transformations; inspect expanded ranges. |
| Noise | Nonlinear functions can amplify measurement errors; extra capacity can fit wrong labels. | General early stopping/regularisation applies. | Validate the expansion, not just the downstream network. |
| Skew class distributions | The expansion does not rebalance a classification loss. | No FLNN-specific remedy taught. | Same cost/representation tradeoffs as §3.3; not a class issue for regression. |
| Irrelevant features | Expansion may multiply irrelevant/redundant terms. | Feature selection and general pruning/penalties can reduce them. | Select useful terms within training folds; risk discarding interactions. |
| Different value ranges | Expansion can create much larger disparities than raw inputs. | No automatic scale correction taught. | Scale with regard to the chosen functions and inspect derived features. |
| Different feature types | Basis functions require numerical, meaningful arguments. | No native categorical extension taught. | Encode first; avoid pretending category codes are continuous quantities. |

Expansion can create collinearity and high dimensionality, increasing sample requirements. A useful bias can help with limited data, but a large arbitrary expansion can overfit it.

#### Important choices and variants

Choose the function family/number of terms, their domains and scale, downstream width/depth, and regularisation. Learned downstream weights differ from the chosen feature map. The local material is insufficient to name a preferred basis.

#### Strengths, limitations, and practical use

Useful if domain knowledge suggests informative transformations. Costs include evaluating the expansion and a larger downstream network. Poorly chosen functions add computation and instability without useful information; no empirical advantage is established in the local slide.

### 3.5 Product unit neural network (PUNN / $\Pi\Sigma$ network)

#### Main idea and assumptions

Use multiplicative hidden units to represent interactions compactly, then combine them with summation/output units. An exponent is a learned input-to-product-unit parameter. For example, one unit can represent $x_1^3x_2^7$, whereas summation units may need several units to approximate it. This is a representational advantage for suitable functions, not a universal superiority result. [Slides 135, 155–156; quiz 6.3 Q13–14.]

#### How it learns and predicts

For positive inputs, product unit $j$ computes

$$u_j=\prod_{i=1}^d x_i^{v_{ji}}
=\exp\!\left(\sum_{i=1}^d v_{ji}\log x_i\right).$$

$v_{ji}$ is the learned exponent on input $i$; a hidden activation may then be applied and outputs are combined by a summation layer. For positive inputs, $\partial u_j/\partial v_{ji}=u_j\log x_i$, explaining gradient instability when products or logarithms become extreme. Train the network against the selected output loss using the product-unit derivatives; prediction evaluates the learned products and output layer. [Slides 135, 155–156.]

**Domain qualification:** slide 135 uses $e^{\rho_j}\cos(\pi\varphi_j)$ for nonzero signed inputs, where $\rho_j=\sum_i v_{ji}\log|x_i|$ and $\varphi_j=\sum_i v_{ji}\mathbb{1}[x_i<0]$. This is a particular real-valued convention, not ordinary real exponentiation for arbitrary negative bases and real exponents. Zero inputs require separate treatment; the slide explicitly excludes them. Do not implement the simple logarithmic expression on arbitrary data.

#### Supported data and tasks

Numeric regression or categorical prediction is possible through the appropriate output layer. Continuous/discrete numerical inputs must satisfy the chosen power convention. Standard zero-containing one-hot encodings and signed standardised values are problematic for the displayed formulation. Nominal/ordinal/mixed data therefore need more than routine encoding: the product representation must remain defined and meaningful.

#### Data quality and preprocessing

| Issue | Effect on the standard model and why | Relevant extension and how it helps | Preprocessing / mitigation and tradeoffs |
| --- | --- | --- | --- |
| Missing values | An absent factor makes a product undefined; missing target gives no supervised error. | No missing-data extension taught. | Impute within valid input domains; errors can influence an entire product. |
| Outliers | Products/powers can amplify large inputs; near-zero inputs with negative powers can explode. Extreme regression targets dominate squared loss. | Signed-input convention addresses signs, not outliers. | Check domains and transformations; shifting inputs changes the interactions being represented. |
| Noise | Errors in factors propagate multiplicatively; incorrect labels can be fitted. | General capacity control applies, but no PUNN noise-specific extension is taught. | Validate perturbation sensitivity and reduce unnecessary units. |
| Skew class distributions | A classification loss still gives majority cases more aggregate influence. | No PUNN-specific imbalance extension taught. | Cost weighting/resampling is an additional design choice; assess minority performance. |
| Irrelevant features | An irrelevant factor can distort many products; zero exponent can remove a factor, but learning it is not guaranteed. | General architecture selection/regularisation can help. | Restrict/choose interactions using training evidence. |
| Different value ranges | Products and powers depend strongly on input scale, affecting activations and gradients. | Alternative optimisers are discussed, without a cure for scale/domain issues. | Scale with domain constraints; ordinary mean-centred standardisation may invalidate a positive-input formulation. |
| Different feature types | Requires meaningful numerical factors; zero/sign conventions complicate encoded categories. | Slide 135's signed extension permits a specified nonzero signed calculation. | Choose a valid representation or another architecture; arbitrary coding remains unjustified. |

Redundant factors can create interchangeable exponent combinations, and high-dimensional products can become unstable. Fewer units do not imply few learnable parameters or reliable fitting from scarce data.

#### Important choices and variants

Choose product-unit count, activations/output loss, sign/zero convention, initialisation, and optimisation settings. Exponents and output weights are learned. Large exponents can produce steep functions and gradients; gradient methods can be difficult despite compact representation.

#### Strengths, limitations, and practical use

Consider when multiplicative/high-order structure is plausible. The slides demonstrate fewer units for selected functions but warn of extremely large gradients. Product-unit prediction also requires powers/logarithms, so counting units alone is a poor runtime comparison. Interpretability of an explicit product may help locally, but mixed layers can remain difficult to explain.

### 3.6 Cascade correlation neural network (CCNN)

#### Main idea and assumptions

Build a feedforward architecture with direct input-to-output paths and a cascade of hidden units. Later hidden units can use original inputs and previous hidden outputs, progressively building features. The local slide establishes connectivity; growing networks are discussed later, but the detailed correlation-based candidate-selection/training procedure is not supplied. [Slides 140, 175.]

#### How it learns and predicts

The displayed structure can be written as $h_j=\phi_j(v_j^\top\tilde x+\sum_{r<j}s_{jr}h_r)$ and $\hat y=g(U\tilde x+Wh)$, where $\tilde x$ includes a constant bias input, $j$ orders hidden units, $r$ indexes earlier units, and $U,V,S,W$ denote connection weights. Prediction computes hidden units in cascade order, then the output. Training/growing seeks added capacity when the existing network underfits; the local source does not specify the full CCNN fitting algorithm.

#### Supported data and tasks

Prepared numerical fixed vectors, including encoded categories/mixed inputs. Regression or classification depends on the output head and loss; cascade connectivity itself supplies neither probabilities nor native category handling.

#### Data quality and preprocessing

| Issue | Effect on the standard model and why | Relevant extension and how it helps | Preprocessing / mitigation and tradeoffs |
| --- | --- | --- | --- |
| Missing values | Missing inputs invalidate paths; absent targets cannot supply supervised loss. | No CCNN missing-data extension taught. | Impute inputs; omit absent targets from supervised fitting. |
| Outliers | Extreme inputs affect direct and cascade paths; squared loss emphasises extreme targets. | Growing does not make the loss robust. | Verify extremes and validate chosen transformations. |
| Noise | New units can fit accidental residual patterns or wrong labels. | Validation stopping and pruning can limit excess capacity. | Check labels; growing on training error alone risks overfitting. |
| Skew class distributions | Majority-driven residual loss can guide capacity towards majority cases. | No CCNN-specific imbalance remedy taught. | Assess class-specific errors; cost weighting changes the fitting objective. |
| Irrelevant features | Inputs can enter many direct/cascade connections. | Pruning and feature selection are discussed generally. | Remove irrelevant/redundant inputs using training evidence. |
| Different value ranges | Units and gradients inherit disparities in the inputs. | No scale-invariant extension taught. | Scale numerical measurements. |
| Different feature types | Arithmetic requires numeric representations. | No native mixed-data extension taught. | Encode nominal/ordinal data meaningfully. |

Redundant inputs and successive correlated hidden features complicate interpretation. Growth increases parameter count and data needs; sparse coverage still limits generalisation.

#### Important choices and variants

Hidden-unit addition/stopping criteria, activation functions, output loss, and capacity limits matter. Too little growth underfits; continuing to chase residual training errors can overfit. Exact candidate-correlation and weight-freezing rules cannot be recovered from the local slide alone.

#### Strengths, limitations, and practical use

An approach to adapting capacity rather than fixing a large network initially. The direct paths can retain simple relationships while hidden units add complexity. Training/storage costs increase with growth; sequential cascade dependencies also constrain prediction parallelism.

### 3.7 Time-delay neural network (TDNN)

#### Main idea and assumptions

Supply a feedforward network with current inputs and explicit lagged inputs. Its bias is that a finite recent window contains useful predictive information. The slide shows a tapped-delay input structure; it does not require an internally recurrent state. [Slide 139.]

#### How it learns and predicts

Choose maximum lag $T$, form $\tilde x_t=[x_t,x_{t-1},\ldots,x_{t-T}]$, pair windows with correctly aligned targets, and train a feedforward network using §3.3. At prediction time form the same window. Here $t$ is time and $x_t$ can be a vector. Information before $t-T$ is unavailable unless summarised in supplied features.

#### Supported data and tasks

Sequences of numeric continuous/discrete or encoded categorical/mixed observations. Predict numeric values or categorical labels using the appropriate head; the target may be current or future, but the input window must respect what is available at prediction time. Fixed lag length makes the network input fixed-size.

#### Data quality and preprocessing

| Issue | Effect on the standard model and why | Relevant extension and how it helps | Preprocessing / mitigation and tradeoffs |
| --- | --- | --- | --- |
| Missing values | Missing observations create gaps in one or many windows; absent targets prevent supervised loss. | Recurrent alternatives encode history differently but do not automatically impute. | Impute causally for forecasting; represent gaps when useful. Future-based interpolation can leak information. |
| Outliers | One extreme input can appear in many overlapping windows; extreme regression targets dominate squared loss. | No TDNN-specific robust extension taught. | Check anomalies and their temporal propagation; capping can erase real events. |
| Noise | Noisy lags add misleading inputs; target/label noise can be memorised. | General early stopping and regularisation apply. | Validate lag selection; smoothing may erase useful short events. |
| Skew class distributions | Rare sequence events remain rare training targets. | No TDNN imbalance extension taught. | Weight/sample training windows carefully; avoid leaking near-duplicate windows across splits. |
| Irrelevant features | Irrelevant variables or lags add parameters and chance associations. | General feature selection/pruning helps. | Select lag range and variables using time-respecting validation. |
| Different value ranges | Same gradient-conditioning issues as the underlying MLP. | No native scale correction taught. | Use consistent training-fitted scales across corresponding lags. |
| Different feature types | Each lag needs meaningful numeric encoding. | No native categorical/time-gap handling taught. | Encode categories; define regular sampling or make elapsed time explicit where needed. |

Nearby lags are often redundant/collinear. A window over $d$ variables has $d(T+1)$ inputs before encoding, so long windows increase dimensionality and sample requirements. Overlapping windows do not provide as many independent examples as their raw count suggests.

#### Important choices and variants

Window length, sampling interval, forecast horizon, and MLP settings. Short windows may miss dependencies; long windows increase cost and overfitting. A TDNN's explicit history differs from an RNN's recursively compressed history.

#### Strengths, limitations, and practical use

Simple when a useful bounded history is known; no BPTT is required for the shown feedforward window model. It cannot dynamically retain events outside the window. Storage and arithmetic follow the expanded MLP input size, plus maintaining a lag buffer.

### 3.8 Simple recurrent neural networks: Elman, Jordan, multi-recurrent

#### Main idea and assumptions

Maintain a state that changes with each ordered input. The same transition weights are reused at every time-step, allowing variable-length sequences without a separate set of weights for every position. The state is a learned summary, not lossless storage of all earlier observations. Useful dependencies must be learnable from ordered, suitably sampled examples. [Slides 136–138; textbook §8.4.6.1, pp. 499–502.]

#### How it learns and predicts

For an Elman/simple RNN:

$$h_t=\phi(W_xx_t+W_hh_{t-1}+b_h),\qquad
\hat y_t=g(W_yh_t+b_y).$$

$x_t$ is the current numerical input, $h_t$ the hidden state, $W_x,W_h,W_y$ learned matrices, $b_h,b_y$ biases, and $\phi,g$ hidden/output activations. Initialise state at a sequence boundary; advance it in order. Return outputs at each step or use a final state for a sequence-level prediction. This is Eq. 8.104–8.105 with explicit output bias and task-specific output activation.

**Training:** unroll across the sequence; compute applicable output losses; backpropagate through the unrolled graph; sum contributions to each shared weight over time (and examples); update. This is **BPTT**, not a separate copy of the parameters at each step. A truncation horizon limits how far gradients travel and can sacrifice longer dependencies. Reset or carry state according to genuine sequence boundaries, avoiding contamination between unrelated examples. [Textbook §8.4.6.2, pp. 502–507.]

#### Supported data and tasks

Ordered numeric continuous/discrete vectors or encoded nominal/ordinal tokens and mixed observations. Outputs can be numeric predictions, binary/multiclass labels, or corresponding probabilities through the head in §3.3. Targets can occur per step or per sequence. Order is integral to the inputs; treating the time index as an ordinary feature of unordered rows is not equivalent to recurrence.

#### Data quality and preprocessing

| Issue | Effect on the standard model and why | Relevant extension and how it helps | Preprocessing / mitigation and tradeoffs |
| --- | --- | --- | --- |
| Missing values | Missing input can corrupt current and later states; absent target provides no loss at that step. | LSTM improves memory, not native missing-value handling. | Impute appropriately, mark gaps if useful, and mask absent targets. Distinguish padding from actual missing observations. |
| Outliers | Extreme inputs can alter state for later predictions; extreme numeric targets dominate squared error. | LSTM gates may regulate flow but are not guaranteed outlier rejection. | Verify events and use meaningful scales; removing extremes may remove the signal of interest. |
| Noise | Measurement errors can persist through state; wrong labels affect shared transitions across time. | Variational recurrent dropout uses a consistent mask through a sequence; early stopping also helps. | Clean systematic errors; validate on independent/time-appropriate sequences. |
| Skew class distributions | Frequent labels or longer labelled sequences may dominate summed loss. | No RNN-specific imbalance remedy taught. | Decide whether to weight steps or sequences; class reweighting trades errors and calibration. Numeric forecasting has no categorical imbalance unless defined as such. |
| Irrelevant features | May enter state repeatedly and waste limited memory/capacity. | General regularisation/pruning; gates in LSTM control information flow without guaranteeing feature selection. | Select useful variables; remove redundant channels cautiously. |
| Different value ranges | Large inputs can saturate units; repeated state transformations can amplify unstable dynamics. | Suitable initialisation and LSTM address parts of gradient flow. | Scale measurements; preserve meaningful timing and avoid future leakage. |
| Different feature types | Arithmetic needs encoded values; sequence position is structural information. | Recurrence handles order, not arbitrary category codes. | Encode nominal/ordinal features; mask padding in loss/state processing as appropriate. |

Correlated variables and redundant time information can waste state capacity. High-dimensional inputs or large state matrices need more data. Few independent sequences and missing coverage of histories limit generalisation even when there are many individual time-steps.

#### Important choices and variants

- **Elman:** feeds previous hidden activations to the hidden layer through a context buffer.
- **Jordan:** feeds previous output activations back as context, instead of the Elman hidden vector. Output mistakes can influence later predictions.
- **Multi-recurrent:** the displayed diagram combines hidden- and output-related feedback into a state layer; detailed update rules are not given. These are variants within the same recurrent family. [Slides 136–138, visually checked.]
- **State width, depth, activations, BPTT horizon, sequence loss reduction and state initialisation** affect capacity and accessible dependencies.
- Repeated Jacobian products in BPTT can shrink or amplify gradients. This is more precise than checking whether one individual recurrent weight exceeds 1. LSTM is the substantive gated extension below.
- Dropout masks changing at every step can disrupt memory; the chapter discusses **variational RNN dropout**, holding the recurrent mask fixed over a sequence. [Textbook §8.4.6.2, p. 507.]

#### Strengths, limitations, and practical use

Useful when order/history matters and sequence lengths vary. Parameter sharing saves parameters relative to independent time-position models, but a simple RNN often struggles with long dependencies. For input width $d$, state width $h$, and length $T$, the hidden transition costs roughly $O(T(dh+h^2))$; add output-layer costs. BPTT stores states/activations across unrolled steps; prediction can carry only the current state, but steps depend sequentially on prior states.

### 3.9 Convolutional neural network (CNN)

#### Main idea and assumptions

Use **local receptive fields** and **shared filters** to find the same local pattern at different positions of a grid. Later layers combine local features into larger patterns. Pooling subsamples feature maps. The inductive bias is meaningful locality and repeated spatial structure; arbitrary column order in a table usually does not supply this structure. [Textbook §8.4.5, pp. 477–498.]

#### How it learns and predicts

1. Apply each learned filter to local patches across the input; add its shared bias and apply an activation to generate a feature map.
2. Optionally pool nearby activations, for example by taking their maximum.
3. Repeat feature extraction and combine the resulting representation with an output head.
4. Train using forward propagation, task loss, and backpropagation. Each shared weight's gradient sums contributions from all spatial uses, plus training examples.

For one spatial dimension, output size is

$$N_{\rm out}=\left\lfloor\frac{N+2p-f}{s}\right\rfloor+1,$$

where $N$ is input size, $f$ filter size, $p$ padding on each side, and $s$ stride, assuming an ordinary filter with no dilation. Apply separately to image height and width. Example: a $6\times6$ image, $3\times3$ filter, stride 1, no padding yields $4\times4$; padding 1 yields $6\times6$. The formula expresses the window-counting mechanism illustrated in §8.4.5.3, pp. 486–489.

#### Supported data and tasks

Numeric grids such as continuous/discrete image intensities, with channels such as RGB. Categories can be represented as channels when grid structure is meaningful; integer category IDs are not intensities. Mixed non-grid features need a deliberately combined architecture, not native handling by a spatial filter. Classification and regression depend on the output head. Internal feature maps are representations, not class probabilities. [Textbook §§8.4.5.1–8.4.5.5.]

#### Data quality and preprocessing

| Issue | Effect on the standard model and why | Relevant extension and how it helps | Preprocessing / mitigation and tradeoffs |
| --- | --- | --- | --- |
| Missing values | Undefined pixels affect receptive fields; missing labels prevent supervised loss. Padding zeros are defined boundaries, not imputation. | No missing-pixel reconstruction extension taught. | Fill/mask missing data deliberately; resizing/filling can create artificial features. |
| Outliers | Unusual pixels can strongly activate filters; max pooling may retain isolated spikes. Extreme regression targets still dominate squared loss. | Pooling reduces resolution, not guaranteed outlier influence. | Check corrupted sensors/images; preserve real rare patterns. |
| Noise | Local noise alters detectors; incorrect labels can be memorised. | Weight sharing reduces parameter count; pooling/dropout/early stopping can reduce overfitting. | Clean artefacts and validate. Pooling can also discard weak useful signals. |
| Skew class distributions | Abundant image classes dominate an unweighted classification loss. | Convolution/pooling does not rebalance labels. | Class-aware training/assessment as additional choices; duplicated minority images do not create new information. |
| Irrelevant features | Shared filters may still learn irrelevant backgrounds or channels. | Local connectivity/weight sharing constrain capacity; dropout can reduce dependence on particular activations. | Check spurious backgrounds and remove irrelevant channels where justified. |
| Different value ranges | Channel ranges affect filter sums and gradients. | Initialisation/ReLU aid optimisation but do not automatically normalise input channels. | Consistent training-fitted intensity/channel scaling; maintain the same processing at inference. |
| Different feature types | Native representation is a numeric tensor with meaningful neighbourhoods. | Multiple channels support multiple numeric signals, not arbitrary raw mixed categories. | Encode categorical grids appropriately; do not invent locality by rearranging table columns. |

Sharing reduces parameter growth for high-dimensional grids, not the number of spatial activations. Correlated channels can remain redundant. Small or unrepresentative image datasets still invite overfitting; a CNN does not inherently understand unseen viewpoints or rare patterns.

#### Important choices and variants

Filter size and number, depth, stride, padding, activations, and pooling type/window affect receptive field, detail, cost, and capacity. Filters span the input channels; different filters produce different maps. A layer with $F$ filters of spatial size $f_h\times f_w$ over $C$ input channels has $F(f_hf_wC+1)$ parameters including one bias per filter, independent of image position count.

Shared filtering gives translation **equivariance**: translating a pattern translates its response, subject to boundary/stride effects. Pooling gives some tolerance to small positional changes; it does not guarantee complete translation invariance. Larger strides/pooling reduce cost but lose detail. [Textbook §§8.4.5.2–8.4.5.5, especially pp. 483, 486–492.]

#### Strengths, limitations, and practical use

Good for grid data with recurring local features; often more parameter-efficient than flattening into a dense network. Poorly matched when relationships have no spatial locality, or exact detail lost by pooling is essential. A convolution costs roughly the number of output positions times filters times weights per filter; low parameter count therefore does not guarantee cheap computation. Intermediate maps are somewhat inspectable, but the complete prediction can remain hard to interpret.

### 3.10 Long short-term memory network (LSTM)

#### Main idea and assumptions

An LSTM is a recurrent model with a separate **cell state** and learned forget, input, and output gates. The cell has an additive update path, making it easier to preserve information and error signals over longer intervals than a simple RNN. Gates control retention, writing, and exposure of information; they do not guarantee correct memory or eliminate all vanishing/exploding gradients. [Textbook §§8.4.6.3–8.4.6.4, pp. 507–521.]

#### How it learns and predicts

Let $r_t=[h_{t-1};x_t]$ concatenate prior hidden state and current input. A compact restatement of the **local textbook's formulation** is

$$f_t=\sigma(W_fr_t+b_f),\quad i_t=\sigma(W_ir_t+b_i),\quad
u_t=\tanh(W_ur_t+b_u),$$
$$c_t=f_t\odot c_{t-1}+i_t\odot u_t,$$
$$o_t=\sigma(W_or_t+b_o),\qquad
h_t=o_t\odot\tanh(W_c c_t+b_c).$$

$f_t,i_t,o_t$ are forget/input/output gate vectors, $u_t$ the candidate update, $c_t$ cell state, and $h_t$ exposed hidden state; each $W$ and $b$ is learned. Sigmoid gates lie between 0 and 1; tanh permits positive and negative candidate changes. The chapter includes a learned transformation $W_c$ before the output tanh (Eq. 8.115); this equation preserves that detail. Here $h_t$ names the output at time $t$ consistently; the chapter's Eq. 8.117 labels the propagated state $h_{t+1}$. [Textbook pp. 509–513, Eqs. 8.106–8.117; printed p. 513 visually checked.]

For each sequence: initialise hidden/cell states, advance the gates in temporal order, generate task outputs from $h_t$, and train through BPTT with shared parameters. Prediction carries $(h_t,c_t)$ forward without updating weights. Along the direct cell-state path, the derivative includes $f_t$, rather than repeated multiplication by a recurrent weight matrix; forget gates near 1 can preserve gradient flow. Gates near 0 still discard memory. [Textbook §8.4.6.4, pp. 520–521.]

#### Supported data and tasks

The same ordered numeric/encoded categorical/mixed sequences and per-step/per-sequence targets as an RNN. Output heads provide numeric estimates, labels, or estimated probabilities. Gates themselves are internal control signals, not task probabilities. Long memory capacity does not remove the need for enough examples of the relevant histories.

#### Data quality and preprocessing

| Issue | Effect on the standard model and why | Relevant extension and how it helps | Preprocessing / mitigation and tradeoffs |
| --- | --- | --- | --- |
| Missing values | Missing inputs can invalidate gates/state; absent targets cannot supply supervised loss. | No LSTM-specific missing-data extension taught; gates do not impute. | Input imputation/gap representation and masking as in §3.8; distinguish padding and real data. |
| Outliers | Large inputs may saturate gates or distort memory; squared-error targets retain quadratic influence. | Gating can filter information only if training learns that behaviour. | Check anomalies and scale; rare important events must remain learnable. |
| Noise | Corrupted inputs can be written into memory; incorrect labels affect gate learning. | Gating and sequence-consistent dropout may reduce unwanted dependence, without guaranteed denoising. | Clean errors, regularise, and validate independent sequences. |
| Skew class distributions | Standard losses still emphasise abundant labels/steps. | Memory gates do not correct class skew. | Choose step/sequence weighting and assess rare events; reweighting changes tradeoffs. |
| Irrelevant features | Gates may learn to suppress some signals, but extra inputs add parameters and spurious associations. | Forget/input gates control state; general penalties/pruning control capacity. | Feature selection and enough varied sequences remain useful. |
| Different value ranges | Large preactivations saturate sigmoid/tanh and slow gate learning. | Additive memory path improves temporal gradients, not input-scale invariance. | Scale inputs consistently; tune initialisation and learning rate. |
| Different feature types | Numerical gate computations require numerical encodings. | LSTM extends recurrent memory, not raw data-type support. | Encode nominal/ordinal/mixed sequences and handle lengths explicitly. |

Redundant inputs increase gate parameters; high dimensionality and few independent sequences can overfit more easily than with a small RNN. Sparse coverage of long-range dependencies is still a data limitation.

#### Important choices and variants

Hidden/cell width, stacking depth, sequence/truncation length, initial states, dropout, and the usual optimiser/output choices. Larger state can retain more information but increases parameters, computation, and overfitting risk. The sources develop LSTM rather than a survey of other gated architectures.

#### Strengths, limitations, and practical use

Prefer when long-range sequence dependencies are important and a simple RNN fails to learn them. Gates are more expensive than one simple transition; computation still proceeds sequentially over time. Training memory grows with unrolled length. More stable memory is a useful bias, not a guarantee of long-range generalisation or interpretability.

## 4. Exam and comparison points

### Core distinctions

1. **Activation versus net input:** weights/bias form $z$; activation maps $z$ to output. $\sigma(1.7)\approx0.8455$. Two measured inputs plus an augmented bias input give three input entries, even though the data have two descriptive features. [Quiz 6.2 Q1–2, Q6.]
2. **Nonlinearity versus depth:** linear hidden layers collapse algebraically. Nonlinear hidden units create intermediate features; depth permits successive composition. A single hidden layer can have powerful approximation capacity under suitable conditions; an input/output-only network is different. Universal approximation does not guarantee learnability, small sample requirements, or extrapolation. [Textbook §§8.2.4–8.2.5; quiz 6.2 Q12 names Hornik, Stinchcombe, and White. The original paper is not included.]
3. **Backpropagation versus gradient descent:** backpropagation calculates derivatives; the optimiser turns them into parameter updates. The loss is evaluated at outputs, then differentiated with respect to weights/biases throughout the network. Input sensing units have no trainable internal weights; “weights of input units” should mean outgoing connections into processing units. [Quiz 6.3 Q8–9; textbook §8.3.]
4. **Differentiability:** ordinary gradient training needs useful derivatives, not any arbitrary activation. ReLU's isolated kink is handled by a chosen derivative convention at zero; a step has zero derivative almost everywhere and is unsuitable for ordinary backpropagation. Monotonicity is not needed for chaining layers or for a derivative to exist. Strict monotonicity gives a unique inverse on the range; plain nondecreasing monotonicity does not. [Quiz 6.2 Q3–4; slides 117–120; textbook §8.4.1.]
5. **Training fit versus generalisation:** enough capacity can memorise labels/noise. Too many units or prolonged training can encourage overfitting, but neither guarantees it. Unrepresentative training data also harms generalisation; large learning rates primarily cause optimisation problems. Architecture selection is an iterative design process informed by data and task, not an irrevocable “first step.” [Quiz 6.2 Q5; quiz 6.3 Q3–4; slides 172–175.]
6. **Initialisation calculation:** quiz 6.3 Q11 selects approximately $[-0.45,0.45]$ for hidden units receiving four measurements **plus the bias input**: $1/\sqrt5\approx0.447$. Excluding the bias gives $1/\sqrt4=0.5$. State the fan-in convention; the saved answer is not a universal initialisation rule. [Slide 151.]
7. **Model choice:** MLP for fixed vectors; CNN for meaningful grids; TDNN for an explicit finite history; RNN for evolving sequence context; LSTM for more persistent gated memory; PUNN for suitable multiplicative mappings. None natively solves missing inputs, label errors, or class imbalance.

### Recorded answers and technical qualifications

| Source | Recorded course statement / answer | Qualified explanation |
| --- | --- | --- |
| Quiz 6.2 Q8 | “A single perceptron cannot separate non-linearly separable data” is marked **False**. | Under the standard weighted-sum threshold perceptron definition, the statement is **True**: its boundary is a hyperplane. Slides 122 and 127 and textbook §8.2.5 support this. Nonlinear feature transforms/product units change the representation or model. |
| Slide 127 | A single perceptron trained with gradient descent is logistic regression. | Correct specifically for a **sigmoid unit with the appropriate likelihood/loss**. A classical step perceptron and its mistake-driven rule are not logistic regression. The textbook §8.3, p. 403 states the sigmoid condition. |
| Quiz 6.2 Q5(a) | “Sigmoid gives an output in [0,1]” is unselected. | Its exact finite-input range is $(0,1)$; its output is nevertheless contained in $[0,1]$. If interpreted as an exact range, the closed-interval wording is imprecise; as a bound, the statement is valid. |
| Quiz 6.2 Q9–10; slides 128–129 | Targets must match output range; sigmoid inputs need scaling. | Choose a compatible output head or target transform. Input scaling helps optimisation but is not logically required by sigmoid's domain. Labels 0/1 are valid sigmoid classification targets; a linear regression output need not be bounded. |
| Quiz 6.2 Q11 | Sigmoid with 0/1 labels cannot reach squared loss exactly zero; explanation says weight adjustments can never be zero. | Finite real logits never give exactly 0 or 1, but loss may approach zero arbitrarily closely. Zero updates can arise from saturation, cancellation, zero incoming activation, or finite precision despite nonzero loss. “Cannot attain” differs from “cannot converge towards.” |
| Slide 129 | Zero-mean noise needs no treatment if architecture is not too large. | Independent zero-mean **target** noise can average out under squared loss with enough data, but increases uncertainty. Input noise can bias relationships; systematic, correlated, or severe noise and wrong labels are not harmless. Regularisation is mitigation, not immunity. |
| Quiz 6.3 Q2 | L2 SSE in linear regression is an example of batch learning. | Only if gradients/updates use the whole training set. SSE is an objective that can be optimised with stochastic, mini-batch, or batch updates. |
| Quiz 6.3 Q7; slide 154 | Momentum is required only for stochastic learning; batch sums smooth all oscillations. | Momentum is optional in both modes. Full-batch gradients remove sampling noise, but ill-conditioned curved valleys can still make plain gradient descent zigzag. Quiz Q12's left trajectory is smoother/higher-momentum under the diagram's intended comparison. |
| Quiz 6.3 Q10 | Cross-entropy > SSE > MSE for classification. | Cross-entropy naturally fits categorical likelihoods. With fixed sample/output count, SSE and MSE differ by a positive constant and have the same minimisers. Their gradient sizes differ, affecting learning-rate and regularisation scaling; there is no intrinsic SSE-over-MSE ranking. Slide 145 even labels a quantity divided by output count “SSE.” |
| Slides 158–161; quiz 6.3 Q15–16 | Conjugate gradient has an $n_w$-step guarantee; MAE does not qualify; Fletcher–Reeves adapts step size. | The classical finite-step result needs a positive-definite quadratic objective, exact arithmetic and suitable exact searches. An NN's squared-error loss is generally **not quadratic in all weights**; MAE is also nonsmooth/nonquadratic. Conjugacy is with respect to the Hessian $H$, $p_i^\top H p_j=0$, not the weight vector as written on slide 159. Nonlinear methods use local gradients/line searches without this global guarantee. |
| Quiz 6.3 Q17; slide 172 | Do not stop at the first rise in generalisation error. | Use **validation** error, patience, and restore the best checkpoint; true population error is not directly observed. |
| Quiz 6.3 Q18; slide 174 | Regularisation can create more local minima. | Some nonconvex penalties such as weight elimination can alter the landscape that way. It is not a universal consequence of regularisation; L2 adds smooth curvature. |
| Quiz 6.3 Q19 | Small weights should not automatically be pruned. | Contribution depends on input scale, downstream paths, interactions and rescaling. Test pruning's effect, rather than interpreting magnitude alone. |
| Quiz 6.3 Q20; slide 176 | Larger adaptive sigmoid slope gives larger gradients. | It raises the maximum central slope for positive amplitude, but at a fixed nonzero input can push the unit further into saturation. Amplitude scales the vertical range; it is not an independent translation. |
| Textbook §8.4.2, pp. 458–459 | Xavier discussion describes layer fan-out as the following layer's width. | For a weight matrix mapping a preceding layer into a current layer, the usual fan-in/fan-out dimensions are its input/current-output widths. Identify the actual transformation and implementation convention rather than mixing adjacent matrices' dimensions. |
| Textbook Algorithm 7, p. 476 | Inverted dropout scales the forward activation by $1/q$ but its printed backward line shows only the mask. | The derivative of $m\odot a/q$ is $m/q$; the backward path also requires that factor unless already absorbed elsewhere. The overview states the consistent chain-rule version. |
| Textbook §8.4.6.3, p. 511 | Using only positive candidate updates would make cell state increase monotonically. | Positive candidates prevent a negative additive write, but the forget gate can still reduce old state. Tanh enables signed candidate updates; monotonic increase is not implied when forgetting remains active. |

### Other assessed training ideas

- **Active learning:** the learner selects informative examples or regions. Slides 178–181 show selective replacement versus incremental addition to a training subset and use output sensitivity to input perturbations as one informativeness criterion. Subset size, selection, and termination matter. It is not ordinary random mini-batching; nor does it inevitably cause or prevent overfitting. [Quiz 6.3 Q21–22.]
- **Concept drift:** a fixed-capacity network can underfit a newly complex mapping. If only an additive loss level changes while gradient directions/minimiser location stay unchanged, gradient descent's geometry is unchanged; if minima move, continued adaptation may be needed. Resetting every weight is not automatically best. The quizzes assess these ideas, but the supplied slides do not develop a full drift-handling algorithm. [Quiz 6.3 Q5–6, Q23.]
- **Alternative optimisers:** slides 158–170 describe conjugate-gradient/Fletcher–Reeves, scaled conjugate gradient and LeapFrog; slide 158 labels conjugate-gradient details “for interest, not for examination,” although the saved quiz tests them. Treat this as a scope discrepancy. Quickprop is mentioned with adaptive momentum. These are training methods for networks, not new prediction models.
- **Metaheuristics / PSO:** each particle represents a candidate vector of **all weights and biases**, evaluated by the network's loss. No gradient is required, so differentiability of the loss is not essential. Population-based search can be expensive in a large parameter space and supplies no automatic guarantee of a global optimum. [Slide 171; quiz 6.3 Q24–25.]

The main mechanism, standard algorithm, important variants, supported descriptive/target types, and six data-quality prompts in the personal notes are answered in §§1–3; the required additional issues (irrelevance and scale, plus redundancy/data scarcity) are included there too.

## 5. Sources and unresolved questions

- [Course slides: 05errorBasedLearningTopic5.pdf](05errorBasedLearningTopic5.pdf), displayed slides 108–181 (PDF pp. 1–74). Main mechanism/architectures: 113–140; training: 141–171; performance and active learning: 172–181. Embedded equations/diagrams were visually checked where needed, including slides 134–135, 138, 140, 145 and 157.
- [Textbook chapter: fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), Chapter 8, printed pp. 381–524 (PDF pp. 1–144). Main references: §§8.2–8.3 for neurons, depth and backpropagation; §§8.4.1–8.4.4 for gradient stability, classification and regularisation; §8.4.5 for CNNs; §8.4.6 for RNNs/LSTMs. Printed p. 513 (PDF p. 133) was also checked visually to confirm the LSTM output transformation.
- [Personal notes: README.md](README.md), chapter outline and unanswered “Personal Summary Questions.” The notes contain questions rather than completed technical explanations.
- [Quiz 6.2](quiz-06-2-error-based-learning.md), Q1–12; [input-unit question diagram](quiz-06-2-error-based-learning-images/screenshot-from-2022-08-18-12-30-19.png).
- [Quiz 6.3](quiz-06-3-error-based-learning.md), Q1–25; [momentum diagram](quiz-06-3-error-based-learning-images/screenshot-from-2024-09-06-09-51-41.png); [adaptive sigmoid symbols](quiz-06-3-error-based-learning-images/6eded8c166c961a3.svg) and [slope symbol](quiz-06-3-error-based-learning-images/1f6af1368a500860.svg). Saved keys are reported as recorded, with substantive qualifications in §4.

**Material gaps:** radial basis function networks are only named (slide 132), so a mechanism cannot be established from these sources and no RBF model summary is invented. FLNN basis/learning details, full CCNN candidate training, and precise multi-recurrent updates are not supplied. The referenced recordings `NNtraining.mp4` and `PSOforNN.mp4`, the universal-approximation paper, and other original papers are absent. Detailed sign/zero treatment for arbitrary PUNN inputs is insufficient for a general implementation. Conjugate-gradient examination scope conflicts with the saved quiz. The saved perceptron answer conflicts directly with the standard definition; the intended marking rationale remains unknown.

No outside sources were needed. Practical preprocessing/evaluation suggestions and mathematical qualifications are identified as consequences or choices, rather than attributed to absent course extensions.
