# Error-Based Learning — Course Model Summary (`ch07`)

## 1. Topic at a glance

**Main idea and learning setting:**  
Error-based learning frames **supervised learning** (both regression and classification) as **parameterised error minimisation** ([`05errorBasedLearningTopic5.pdf`](05errorBasedLearningTopic5.pdf), Slides 4–13; [`fundamentals_of_ml_2ed_ch.pdf`](fundamentals_of_ml_2ed_ch.pdf), Sec. 7.1, PDF p. 1 / printed p. 311). Given a family of candidate models $\mathbb{M}_{\mathbf{w}}(\mathbf{d})$ parameterised by a weight vector $\mathbf{w} = (w_0, w_1, \dots, w_m)$, the learner searches for the weights $\mathbf{w}^*$ that minimise the total discrepancy (error/loss) between model predictions $\mathbb{M}_{\mathbf{w}}(\mathbf{d}_i)$ and ground-truth targets $t_i$ across the training dataset $\mathcal{D}$ of size $n$.

**Inductive bias:**  
- **Functional form bias:** Predictions are assumed to be well-approximated by a smooth parametric function of the inputs—specifically a linear combination of the original descriptive features $\mathbf{w} \cdot \mathbf{d}$, a linear combination of fixed non-linear basis functions $\sum_{k=0}^b w_k \phi_k(\mathbf{d})$, a maximum-margin hyperplane in a kernel-induced feature space, or a composition of weighted sums and non-linear activation functions in a neural network.
- **Smoothness / locality in parameter space:** Because parameters are updated iteratively in the direction of decreasing empirical error, nearby weight vectors are assumed to yield smoothly varying error surfaces (except in hard-threshold perceptrons).

**Distinguishing core components covered in this topic:**
- **Prediction models (covered below):**
  1. **Multivariable Linear Regression** (including **Simple Linear Regression** for $m=1$ and **Basis-Function / Polynomial Linear Regression** for non-linear relationships)
  2. **Logistic Regression** (including **Binary Logistic Regression**, **Non-Linear Basis-Function Logistic Regression**, and **One-vs-All Multinomial Logistic Regression**)
  3. **Support Vector Machines (SVMs)** (covered in textbook Sec. 7.4.7 and [`README.md`](README.md); includes **Hard-Margin Linear SVM**, **Soft-Margin SVM**, **Kernel SVM**, and **One-vs-All Multiclass SVM**)
  4. **Single Perceptron** (hard-threshold vs. continuous-activation single-unit classifier; listed in [`05errorBasedLearningTopic5.pdf`](05errorBasedLearningTopic5.pdf) Slide 2 & Slide 55–58, and tested in [`quiz5.md`](quiz5.md) Q8)
  5. **Multilayer (Shallow) Feedforward Neural Networks (FFNN / MLP)** (listed in [`05errorBasedLearningTopic5.pdf`](05errorBasedLearningTopic5.pdf) Slide 2 and examined in [`quiz5.md`](quiz5.md) Q9–Q15 and [`README.md`](README.md))
- **Training algorithms (optimisers), not models:**
  - **Generic iterative weight update** ([`05errorBasedLearningTopic5.pdf`](05errorBasedLearningTopic5.pdf), Slide 7): $\mathbf{w}(t+1) = \mathbf{w}(t) + \eta(t)\mathbf{q}(t)$, with step size $\eta(t)$ and search direction $\mathbf{q}(t)$ until stopping criteria are met.
  - **Batch Gradient Descent (GD):** Updates $\mathbf{w}$ once per epoch after summing error gradients over all $n$ training instances ([`fundamentals_of_ml_2ed_ch.pdf`](fundamentals_of_ml_2ed_ch.pdf), Sec. 7.4, PDF pp. 13–18 / printed pp. 323–328).
  - **Stochastic Gradient Descent (SGD) & Mini-Batch GD:** Referenced in Python implementations (`SGDRegressor`, `SGDClassifier`, [`05errorBasedLearningTopic5.pdf`](05errorBasedLearningTopic5.pdf), Slide 107) and neural network training ([`quiz5.md`](quiz5.md), Q11).
  - **Quadratic Programming (Convex Constrained Optimisation):** Used to solve for Lagrange multipliers $\alpha[i]$ in SVMs ([`fundamentals_of_ml_2ed_ch.pdf`](fundamentals_of_ml_2ed_ch.pdf), Sec. 7.4.7, PDF p. 52 / printed p. 362).
- **Objectives / loss functions, not models:**
  - **True (Generalisation) Error / Risk** ([`05errorBasedLearningTopic5.pdf`](05errorBasedLearningTopic5.pdf), Slide 10):
    $$\mathcal{E}_G(\Omega; \mathbf{w}) = \int \left(\hat{f}(\mathbf{x}, \mathbf{w}) - \mathbf{y}\right)^2 d\Omega(\mathbf{x}, \mathbf{y})$$
    Unavailable directly because the true joint distribution $\Omega(\mathbf{x}, \mathbf{y})$ is unknown.
  - **Empirical Training Error** ([`05errorBasedLearningTopic5.pdf`](05errorBasedLearningTopic5.pdf), Slide 10) and **Sum of Squared Errors ($L_2$ Loss)** ([`05errorBasedLearningTopic5.pdf`](05errorBasedLearningTopic5.pdf), Slide 25; [`fundamentals_of_ml_2ed_ch.pdf`](fundamentals_of_ml_2ed_ch.pdf), Eq. 7.4, PDF p. 9 / printed p. 319):
    $$\mathcal{E}_T(D_T; \mathbf{w}) = \frac{1}{n_T}\sum_{i=1}^{n_T}\left(\hat{f}(\mathbf{x}_i, \mathbf{w}) - \mathbf{y}_i\right)^2, \qquad L_2(\mathbb{M}_{\mathbf{w}}, \mathcal{D}) = \frac{1}{2}\sum_{i=1}^{n}\left(t_i - \mathbb{M}_{\mathbf{w}}(\mathbf{d}_i)\right)^2$$
    *(Note: the factor $\frac{1}{2}$ simplifies the derivative and does not change the location of the minimum $\mathbf{w}^*$.)*
  - **Maximum-Margin Objective:** Maximising $\frac{2}{\|\mathbf{w}\|}$ subject to margin constraints in SVMs ([`fundamentals_of_ml_2ed_ch.pdf`](fundamentals_of_ml_2ed_ch.pdf), PDF p. 52 / printed p. 362).
- **Feature transformations and kernels, not standalone models:**
  - **Basis Functions $\phi_k(\mathbf{d})$:** Explicit non-linear transformations (e.g., polynomial terms $d_1^2, d_1 d_2$) applied before a linear or logistic model.
  - **Kernel Functions $K(\mathbf{d}_i, \mathbf{d}_j)$:** Implicit inner products $\boldsymbol{\phi}(\mathbf{d}_i) \cdot \boldsymbol{\phi}(\mathbf{d}_j)$ in a high-dimensional space (Linear, Polynomial, Radial Basis Function / RBF) used in SVMs ([`fundamentals_of_ml_2ed_ch.pdf`](fundamentals_of_ml_2ed_ch.pdf), Table 7.6, PDF p. 56 / printed p. 366).
- **Statistical inference / evaluation methods:**
  - **Two-tailed $t$-test for weight significance** in linear regression ([`05errorBasedLearningTopic5.pdf`](05errorBasedLearningTopic5.pdf), Slides 42–44; [`fundamentals_of_ml_2ed_ch.pdf`](fundamentals_of_ml_2ed_ch.pdf), Sec. 7.4.1, PDF pp. 23–24 / printed pp. 333–334).

---

## 2. Quick comparison

| Model | Task / target type | Descriptive feature types and preparation | Main strength | Main limitation |
| --- | --- | --- | --- | --- |
| **Multivariable Linear Regression** | Regression (continuous numeric target) | Numeric (continuous/discrete); categorical requires one-hot / dummy encoding; scaling strongly aids GD convergence | Simple, fast, globally convex $L_2$ error surface, interpretable weights | Assumes linear relationship; sensitive to outliers ($L_2$ squares residuals) and collinearity |
| **Basis-Function / Polynomial Linear Regression** | Regression (continuous numeric target) | Numeric (or encoded categorical) mapped via explicit basis functions $\phi_k(\mathbf{d})$; scaling vital for high-degree polynomials | Models non-linear relationships while remaining linear in parameters $\mathbf{w}$ (convex $L_2$) | Combinatorial explosion of basis terms with feature count $m$ and degree $p$; prone to overfitting |
| **Logistic Regression (Binary & OvA Multinomial)** | Classification (binary $\{0, 1\}$ natively; multiclass via normalised One-vs-All) | Numeric; categorical requires one-hot / dummy encoding; feature scaling important to prevent sigmoid saturation | Outputs bounded continuous scores in $(0, 1)$; smooth gradient updates unlike hard threshold | Linear decision boundary in input space (unless basis functions added); course $L_2$-on-sigmoid loss has flat saturation plateaus |
| **Support Vector Machine (SVM)** | Classification (binary $\{-1, +1\}$ natively; multiclass via One-vs-All) | Numeric; categorical requires encoding; feature scaling mandatory so large-range features do not dominate margin/kernels | Maximum-margin boundary generalises well; kernel trick handles high-dimensional non-linear boundaries efficiently | Computationally expensive to train on very large $n$; black-box with non-linear kernels; outputs distances/scores rather than calibrated probabilities |
| **Single Perceptron** | Classification (binary; continuous score if sigmoid/linear activation used) | Numeric; categorical requires encoding; feature scaling aids convergence | Simplest neural building block; fast online/batch updates | Only converges to a separator on linearly separable data; hard step activation is non-differentiable at $0$ with zero gradient elsewhere |
| **Multilayer (Shallow) Neural Network (FFNN / MLP)** | Classification (binary/multiclass) or Regression (depending on output activation) | Numeric; categorical requires encoding; scaling inputs (and targets in regression) critical to avoid saturation and slow training | Universal function approximator capable of learning complex non-linear feature interactions automatically | Non-convex error surface with local minima/plateaus; prone to overfitting with too many hidden units; low interpretability |

---

## 3. Individual model summaries

### 3.1 Multivariable Linear Regression (and Basis-Function / Polynomial Regression)

#### Main idea and assumptions
Multivariable linear regression predicts a continuous target $t$ as a weighted linear combination of $m$ descriptive features $\mathbf{d} = (d_1, \dots, d_m)$ plus an intercept (bias) $w_0$ ([`05errorBasedLearningTopic5.pdf`](05errorBasedLearningTopic5.pdf), Slides 14–24; [`fundamentals_of_ml_2ed_ch.pdf`](fundamentals_of_ml_2ed_ch.pdf), Sec. 7.2–7.4, PDF pp. 2–17 / printed pp. 312–327). Geometrically, it fits a line ($m=1$), plane ($m=2$), or hyperplane ($m>2$) through the data that minimises the sum of squared vertical residuals.

- **Assumptions required to fit and predict:**
  - The target $t$ is reasonably approximated by a linear combination of the features $\mathbf{d}$ (or chosen basis functions $\phi_k(\mathbf{d})$).
  - Descriptive features are numeric (or numerically encoded) with no missing values.
- **Additional assumptions required for statistical inference ($t$-tests / standard errors)** ([`fundamentals_of_ml_2ed_ch.pdf`](fundamentals_of_ml_2ed_ch.pdf), Sec. 7.4.1, PDF pp. 23–24 / printed pp. 333–334):
  - Residuals $\epsilon_i = t_i - \mathbb{M}_{\mathbf{w}}(\mathbf{d}_i)$ are independent, identically distributed (homoscedastic with constant variance), and normally distributed with mean zero, and predictors are not collinear so the sampling distribution of each weight follows a $t$-distribution.

#### How it learns and predicts

- **Prediction rule:**  
  By introducing a dummy feature $d_0 = 1$ for every instance, the linear model is:
  $$\mathbb{M}_{\mathbf{w}}(\mathbf{d}) = w_0 + w_1 d_1 + \dots + w_m d_m = \sum_{j=0}^{m} w_j d_j = \mathbf{w} \cdot \mathbf{d}$$
  where $\mathbf{w} = (w_0, \dots, w_m)$ is the vector of learned weights and $\mathbf{d} = (1, d_1, \dots, d_m)$ is the augmented feature vector ([`05errorBasedLearningTopic5.pdf`](05errorBasedLearningTopic5.pdf), Slide 24; [`fundamentals_of_ml_2ed_ch.pdf`](fundamentals_of_ml_2ed_ch.pdf), Eq. 7.5, PDF p. 15 / printed p. 325).

- **Loss function ($L_2$ / Sum of Squared Errors):**  
  $$L_2(\mathbb{M}_{\mathbf{w}}, \mathcal{D}) = \frac{1}{2}\sum_{i=1}^{n}\left(t_i - \mathbb{M}_{\mathbf{w}}(\mathbf{d}_i)\right)^2$$
  Because $L_2$ is a quadratic function of $\mathbf{w}$, the error surface is **convex** (bowl-shaped) with a single global minimum ([`05errorBasedLearningTopic5.pdf`](05errorBasedLearningTopic5.pdf), Slide 29; [`fundamentals_of_ml_2ed_ch.pdf`](fundamentals_of_ml_2ed_ch.pdf), Sec. 7.3, PDF p. 11 / printed p. 321; visualised in [`notebooks/sum_squared_errors.ipynb`](notebooks/sum_squared_errors.ipynb)).

- **Partial derivative (error delta):**  
  For weight $w_j$ ($j \in \{0, \dots, m\}$), differentiating $L_2$ gives ([`05errorBasedLearningTopic5.pdf`](05errorBasedLearningTopic5.pdf), Slide 32; [`fundamentals_of_ml_2ed_ch.pdf`](fundamentals_of_ml_2ed_ch.pdf), Eqs. 7.9–7.12, PDF p. 17 / printed p. 327):
  $$\frac{\partial L_2(\mathbb{M}_{\mathbf{w}}, \mathcal{D})}{\partial w_j} = -\sum_{i=1}^{n}\left(t_i - \mathbb{M}_{\mathbf{w}}(\mathbf{d}_i)\right) d_{j,i} \equiv \operatorname{errorDelta}(\mathcal{D}, w_j)$$
  where $d_{j,i}$ is the value of feature $j$ in training instance $i$ (with $d_{0,i} = 1$).

- **Training algorithm (Batch Gradient Descent):**  
  1. **Initialise** weights $\mathbf{w} = (w_0, \dots, w_m)$ to small random values, and choose learning rate $\alpha > 0$ (written $\eta$ in some slides) and convergence threshold ([`05errorBasedLearningTopic5.pdf`](05errorBasedLearningTopic5.pdf), Slide 33).
  2. **Repeat (each iteration is one epoch):**
     - Compute predictions $\mathbb{M}_{\mathbf{w}}(\mathbf{d}_i)$ and residuals $e_i = t_i - \mathbb{M}_{\mathbf{w}}(\mathbf{d}_i)$ for all $i = 1, \dots, n$.
     - Update every weight $w_j$ simultaneously using the accumulated gradient over the entire dataset:
       $$w_j \leftarrow w_j - \alpha \times \operatorname{errorDelta}(\mathcal{D}, w_j) = w_j + \alpha \sum_{i=1}^{n}\left(t_i - \mathbb{M}_{\mathbf{w}}(\mathbf{d}_i)\right) d_{j,i}$$
  3. **Stop** when $L_2 \approx 0$, the change in $L_2$ (or weights) between epochs falls below a small threshold, or a maximum epoch limit is reached ([`05errorBasedLearningTopic5.pdf`](05errorBasedLearningTopic5.pdf), Slides 8–9, 33).

- **Worked mini-example (1 epoch trace):**  
  Suppose $m=1$ (`SIZE` $d_1=1$, target `PRICE` $t=2$), initial weights $w_0 = 0, w_1 = 0.5$, and learning rate $\alpha = 0.1$:
  - Prediction: $\mathbb{M}_{\mathbf{w}}(\mathbf{d}_1) = 0 + 0.5(1) = 0.5$; residual: $t_1 - \mathbb{M}_{\mathbf{w}}(\mathbf{d}_1) = 2 - 0.5 = 1.5$.
  - Gradients: $\operatorname{errorDelta}(w_0) = -1.5(1) = -1.5$; $\operatorname{errorDelta}(w_1) = -1.5(1) = -1.5$.
  - Updated weights: $w_0 \leftarrow 0 - 0.1(-1.5) = 0.15$; $w_1 \leftarrow 0.5 - 0.1(-1.5) = 0.65$. New prediction is $0.15 + 0.65(1) = 0.80$ (closer to $2$).
  *(See [`fundamentals_of_ml_2ed_ch.pdf`](fundamentals_of_ml_2ed_ch.pdf), Table 7.2, PDF pp. 19–21 / printed pp. 329–331 for the full 7-instance office-rental trace.)*

- **Interpreting weights via two-tailed $t$-test** ([`05errorBasedLearningTopic5.pdf`](05errorBasedLearningTopic5.pdf), Slides 41–44; [`fundamentals_of_ml_2ed_ch.pdf`](fundamentals_of_ml_2ed_ch.pdf), Sec. 7.4.1, Eqs. 7.18–7.20, PDF pp. 23–24 / printed pp. 333–334):  
  Raw weight magnitude $|w_j|$ **cannot** be used directly to rank feature importance because features have different measurement scales ([`quiz5.md`](quiz5.md), Q4). To test the null hypothesis $H_0: w_j = 0$ (feature $d_j$ has no linear relationship with $t$):
  $$se = \sqrt{\frac{\sum_{i=1}^{n}\left(t_i - \mathbb{M}_{\mathbf{w}}(\mathbf{d}_i)\right)^2}{n - 2}}, \qquad se(d_j) = \frac{se}{\sqrt{\sum_{i=1}^{n}\left(d_{j,i} - \bar{d}_j\right)^2}}, \qquad t_{\text{val}} = \left|\frac{w_j}{se(d_j)}\right|$$
  Compare $t_{\text{val}}$ against a critical threshold from the Student's $t$-distribution at significance level $\alpha_{\text{sig}}$ (e.g., $0.05$). If $t_{\text{val}} > t_{\text{critical}}$, reject $H_0$ and conclude $d_j$ is statistically significant.  
  *(See Section 4.2 below for a technical note on the course's $df = n-2$ formula versus $df = n-m-1$ in multivariable regression.)*

#### Supported data and tasks
- **Descriptive features:** Continuous and discrete **numeric** features natively. **Nominal categorical** features require **one-hot / dummy encoding**: a categorical feature with $k$ levels is replaced by $k$ binary $\{0, 1\}$ indicator features (or $k-1$ indicators by dropping one reference level; see [`05errorBasedLearningTopic5.pdf`](05errorBasedLearningTopic5.pdf), Slides 48–52; [`fundamentals_of_ml_2ed_ch.pdf`](fundamentals_of_ml_2ed_ch.pdf), Sec. 7.4.3, PDF pp. 26–28 / printed pp. 336–338). Assigning arbitrary integer codes (e.g., $1, 2, 3$) to nominal categories imposes a false linear ordering and equal spacing. **Ordinal categorical** features may use numeric codes only if equal spacing in target effect is a reasonable domain assumption; otherwise use dummy encoding.
- **Target features:** Continuous (or discrete) **numeric** targets for regression. Applying linear regression directly to binary classification targets $\{0, 1\}$ is flawed because predictions are unbounded $(-\infty, +\infty)$ and distant easy-to-classify points pull the regression line and shift the decision boundary ([`05errorBasedLearningTopic5.pdf`](05errorBasedLearningTopic5.pdf), Slide 55; [`fundamentals_of_ml_2ed_ch.pdf`](fundamentals_of_ml_2ed_ch.pdf), Fig. 7.11, PDF pp. 28–30 / printed pp. 338–340).
- **Outputs:** Unbounded real-valued numeric predictions $\hat{t} \in (-\infty, +\infty)$.

#### Data quality and preprocessing

| Issue | Effect on the standard model and why | Relevant extension and how it helps | Preprocessing / mitigation and tradeoffs |
| --- | --- | --- | --- |
| **1. Missing values** | Cannot compute $\mathbf{w} \cdot \mathbf{d}_i$ or gradient updates $e_i d_{j,i}$ if any $d_{j,i}$ is `NaN`; missing target $t_i$ prevents computing residual $t_i - \mathbb{M}_{\mathbf{w}}(\mathbf{d}_i)$. | None covered in course sources (linear models have no native surrogate-split mechanism). | Drop instances with missing targets $t_i$. For missing $d_{j,i}$, impute (mean/median for numeric, mode or missing-indicator category for categorical) using statistics fitted **only on the training split**; imputation can attenuate variance and distort correlations. |
| **2. Outliers** | **High sensitivity:** $L_2$ loss squares residuals $(t_i - \mathbb{M}_{\mathbf{w}}(\mathbf{d}_i))^2$, so extreme target outliers or high-leverage unusual inputs $d_{j,i}$ produce huge gradients $(t_i - \hat{t}_i)d_{j,i}$ that tilt the fitted hyperplane toward the outlier. | Not covered in Ch. 7 (outside course scope: Huber/$L_1$ loss or RANSAC reduce outlier influence). | Inspect residuals/leverage; clip/winsorise or remove verified data-entry errors in training data, or transform skewed targets (e.g., log transform). Removing valid extreme points risks losing tail behaviour. |
| **3. Noise** | Zero-mean random measurement noise in $t$ increases residual variance $se$ (wider confidence intervals) but does not asymptotically bias $\mathbf{w}^*$. Noise in descriptive features $d_j$ biases $|w_j|$ toward zero (attenuation bias). High-degree basis functions overfit noise. | Using lower polynomial degree $p$ (or regularisation, outside Ch. 7 scope) prevents fitting noise. | Smooth/filter noisy sensors or collect more training instances $n$ so random target errors cancel in the batch gradient sum $\sum_{i=1}^n e_i d_{j,i}$. |
| **4. Skew class distributions** | **Not directly applicable** (standard model is for continuous regression, not classification). However, **skewed target distributions** (long heavy tails) act like one-sided outliers under $L_2$, pulling predictions toward the tail. | Logistic regression or SVM for classification tasks. | For skewed continuous targets, apply a monotonic target transformation (e.g., $\log(t)$) before fitting and invert at prediction time (noting $\exp(\hat{t})$ estimates geometric rather than arithmetic mean). |
| **5. Irrelevant features** | Included in $\mathbf{w} \cdot \mathbf{d}$. In infinite data, irrelevant $d_j$ gets $w_j \to 0$, but on finite data, spurious sample correlations yield non-zero $w_j$, adding variance and hurting generalisation (especially with polynomial basis functions). | **Two-tailed $t$-test** ([`05errorBasedLearningTopic5.pdf`](05errorBasedLearningTopic5.pdf), Slides 42–44) identifies weights not significantly different from $0$ ($t_{\text{val}} \le t_{\text{critical}}$) for removal. | Perform feature selection (filter or wrapper selection, or backward elimination via $t$-tests on training data). Tradeoff: repeated $t$-testing across many features risks false positives (multiple testing). |
| **6. Features with different value ranges** | **Theoretical optimum:** An unregularised linear model with intercept can rescale $w_j$ to compensate for feature units, so scaling is **not strictly required for representability** ([`quiz5.md`](quiz5.md), Q16). **Gradient descent mechanics:** Since $\frac{\partial L_2}{\partial w_j} = -\sum e_i d_{j,i}$, a feature in $[500, 1000]$ has gradients $1000\times$ larger than a feature in $[0, 1]$, creating an elongated elliptical error surface where GD oscillates across the narrow valley unless $\alpha$ is tiny (e.g., $\alpha = 2 \times 10^{-8}$ in Textbook p. 330), causing very slow convergence. Also makes raw $|w_j|$ incomparable ([`quiz5.md`](quiz5.md), Q4). | **Learning rate decay** ($\alpha_\tau = \alpha_0 \frac{c}{c+\tau}$) dampens late oscillation, though it does not fix directional ill-conditioning across features. | Standardise ($z$-score) or normalise features (e.g., to $[0, 1]$ or $[-1, 1]$) using training-split parameters before running GD. Dramatically speeds up convergence and allows larger $\alpha$. |
| **7. Different descriptive feature data types** | Only numeric operations ($w_j d_j$) are defined; nominal strings cannot be multiplied by weights, and arbitrary integer codes impose a false linear order. | **One-hot / dummy encoding** ([`05errorBasedLearningTopic5.pdf`](05errorBasedLearningTopic5.pdf), Slides 48–52; Textbook Sec. 7.4.3): replaces a $k$-level categorical feature with $k$ (or $k-1$) binary $\{0, 1\}$ features, shifting the hyperplane intercept per category level. | Fit one-hot encoder on training categories. High-cardinality categoricals create many sparse binary columns, increasing dimensionality and overfitting risk. |

**Additional data-quality considerations:**
- **Redundant features and multicollinearity:** If two features are perfectly collinear (e.g., $d_2 = 2 d_1$, or keeping all $k$ one-hot indicators alongside an active intercept $w_0$ so $\sum_{l=1}^k d_{\text{cat}, l} = 1 = d_0$), the global minimum is no longer a single point $\mathbf{w}^*$ but an infinite line/subspace of equivalent weight combinations. Batch gradient descent from random initial weights still converges to one solution along that valley, but individual weights $w_j$ become unstable and difficult to interpret.
- **High dimensionality ($m \gg n$) and Polynomial Basis Expansion:** Adding polynomial basis functions of degree $p$ over $m$ features creates $\binom{m+p}{p}$ terms, rapidly causing the **curse of dimensionality** ([`05errorBasedLearningTopic5.pdf`](05errorBasedLearningTopic5.pdf), Slide 85; [`fundamentals_of_ml_2ed_ch.pdf`](fundamentals_of_ml_2ed_ch.pdf), Sec. 7.4.5, PDF p. 45 / printed p. 355). When the number of parameters $b+1 \ge n$, the model can interpolate training noise ($L_2 = 0$) and overfit severely.
- **Limited training data / sparse coverage:** Small $n$ reduces degrees of freedom ($n-2$ or $n-m-1$), inflates standard errors $se(d_j)$, and leaves regions of feature space unanchored, where high-degree polynomial basis functions extrapolate wildly.

#### Important choices and variants
- **Learned parameters:** Weight vector $\mathbf{w} = (w_0, \dots, w_m)$ (or $(w_0, \dots, w_b)$ with $b$ basis functions).
- **Hyperparameters:**
  - **Learning rate $\alpha$ (or $\eta$):** Controls gradient step size. Too small $\implies$ extremely slow convergence; too large $\implies$ overshooting the minimum and diverging ([`05errorBasedLearningTopic5.pdf`](05errorBasedLearningTopic5.pdf), Slides 39–40, 45; [`fundamentals_of_ml_2ed_ch.pdf`](fundamentals_of_ml_2ed_ch.pdf), Fig. 7.9, PDF p. 25 / printed p. 335).
  - **Learning rate decay schedule** ([`05errorBasedLearningTopic5.pdf`](05errorBasedLearningTopic5.pdf), Slide 47; [`fundamentals_of_ml_2ed_ch.pdf`](fundamentals_of_ml_2ed_ch.pdf), Eq. 7.21, PDF p. 25 / printed p. 335):
    $$\alpha_\tau = \alpha_0 \frac{c}{c + \tau}$$
    where $\alpha_0$ is the initial learning rate, $\tau$ is the current epoch number, and $c > 0$ is a constant controlling how quickly $\alpha_\tau$ decays. Allows large initial steps and fine-grained convergence near $\mathbf{w}^*$.
  - **Choice and count of Basis Functions $\phi_k(\mathbf{d})$ (Variant: Basis-Function / Polynomial Regression)** ([`05errorBasedLearningTopic5.pdf`](05errorBasedLearningTopic5.pdf), Slides 72–85; [`fundamentals_of_ml_2ed_ch.pdf`](fundamentals_of_ml_2ed_ch.pdf), Sec. 7.4.5, Eqs. 7.34–7.35, PDF pp. 41–45 / printed pp. 351–355):
    $$\mathbb{M}_{\mathbf{w}}(\mathbf{d}) = \sum_{k=0}^{b} w_k \phi_k(\mathbf{d}) = \mathbf{w} \cdot \boldsymbol{\phi}(\mathbf{d}) \quad (\text{with } \phi_0(\mathbf{d}) = 1)$$
    Even though $\phi_k(\mathbf{d})$ is non-linear in $\mathbf{d}$ (e.g., $d_1^2, d_1^3, d_1 d_2$), the model remains **linear in the weights $\mathbf{w}$**, so the $L_2$ error surface remains **convex** and the exact same gradient descent rule applies with $\phi_k(\mathbf{d}_i)$ replacing $d_{k,i}$ ([`quiz5.md`](quiz5.md), Q7). Higher degree $p$ reduces bias (underfitting) but rapidly increases variance (overfitting).

#### Strengths, limitations, and practical use
- **When to use:** First-choice baseline for continuous regression when relationships are approximately linear (or low-degree polynomial) and interpretability or statistical significance of predictors is required.
- **Interpretability:** High for standard multivariable regression ($w_j$ is the expected change in $t$ per unit increase in $d_j$, holding other features constant); lower when high-degree polynomial interaction terms $\phi_k(\mathbf{d})$ are used.
- **Computational & storage costs:**
  - **Training:** Each Batch GD epoch costs $\mathcal{O}(n \cdot m)$ (or $\mathcal{O}(n \cdot b)$ with $b$ basis functions). Total training for $E$ epochs is $\mathcal{O}(E \cdot n \cdot m)$.
  - **Prediction:** $\mathcal{O}(m)$ per query instance (dot product $\mathbf{w} \cdot \mathbf{d}$) — extremely fast.
  - **Storage:** $\mathcal{O}(m)$ — only the $m+1$ weights $\mathbf{w}$ are retained; training data $\mathcal{D}$ is discarded after training (eager learner).

---

### 3.2 Logistic Regression (Binary, Non-Linear Basis-Function, and OvA Multinomial)

#### Main idea and assumptions
Logistic regression adapts linear error-based learning to **classification** by passing the linear combination $\mathbf{w} \cdot \mathbf{d}$ through the smooth, S-shaped **logistic (sigmoid) function**, squashing outputs into $(0, 1)$ ([`05errorBasedLearningTopic5.pdf`](05errorBasedLearningTopic5.pdf), Slides 54–71; [`fundamentals_of_ml_2ed_ch.pdf`](fundamentals_of_ml_2ed_ch.pdf), Sec. 7.4.4, PDF pp. 28–40 / printed pp. 338–350).
- Unlike a hard-threshold step classifier (whose error surface consists of flat plateaus and discontinuous cliffs with zero gradient almost everywhere, [`05errorBasedLearningTopic5.pdf`](05errorBasedLearningTopic5.pdf), Slides 55–58), the continuous logistic transition makes the error surface differentiable so gradient descent can be applied.
- **Inductive bias and assumptions:** Assumes the two classes (with targets encoded as $t \in \{0, 1\}$) are separable by a linear hyperplane $\mathbf{w} \cdot \mathbf{d} = 0$ in the descriptive feature space (or in the transformed basis space $\mathbf{w} \cdot \boldsymbol{\phi}(\mathbf{d}) = 0$), and that class membership confidence varies smoothly with signed distance from the decision boundary.

#### How it learns and predicts

- **Logistic activation and prediction rule:**  
  With $\operatorname{logistic}(x) = \frac{1}{1 + e^{-x}}$, the binary logistic regression model is ([`05errorBasedLearningTopic5.pdf`](05errorBasedLearningTopic5.pdf), Slides 59–61; [`fundamentals_of_ml_2ed_ch.pdf`](fundamentals_of_ml_2ed_ch.pdf), Eqs. 7.26–7.27, PDF p. 33 / printed p. 343):
  $$\mathbb{M}_{\mathbf{w}}(\mathbf{d}) = \operatorname{logistic}(\mathbf{w} \cdot \mathbf{d}) = \frac{1}{1 + e^{-\mathbf{w} \cdot \mathbf{d}}} \in (0, 1)$$
  Using the derivative identity $\frac{d}{dx}\operatorname{logistic}(x) = \operatorname{logistic}(x)\left(1 - \operatorname{logistic}(x)\right)$ ([`05errorBasedLearningTopic5.pdf`](05errorBasedLearningTopic5.pdf), Slide 60; [`fundamentals_of_ml_2ed_ch.pdf`](fundamentals_of_ml_2ed_ch.pdf), Eq. 7.28, PDF p. 33 / printed p. 343), hard class predictions at threshold $0.5$ are:
  $$\hat{t} = \begin{cases} 1 & \text{if } \mathbb{M}_{\mathbf{w}}(\mathbf{d}) \ge 0.5 \iff \mathbf{w} \cdot \mathbf{d} \ge 0 \\ 0 & \text{if } \mathbb{M}_{\mathbf{w}}(\mathbf{d}) < 0.5 \iff \mathbf{w} \cdot \mathbf{d} < 0 \end{cases}$$
  Notice that $\mathbf{w} \cdot \mathbf{d} = 0$ defines the **linear decision boundary** where $\mathbb{M}_{\mathbf{w}}(\mathbf{d}) = 0.5$.

- **Loss function and Batch Gradient Descent update (Course / Textbook formulation):**  
  In Kelleher et al. and the Topic 5 slides, logistic regression is derived by minimising the **Sum of Squared Errors ($L_2$)** ([`05errorBasedLearningTopic5.pdf`](05errorBasedLearningTopic5.pdf), Slides 65–68; [`fundamentals_of_ml_2ed_ch.pdf`](fundamentals_of_ml_2ed_ch.pdf), Eqs. 7.31–7.33, PDF pp. 36–37 / printed pp. 346–347):
  $$L_2(\mathbb{M}_{\mathbf{w}}, \mathcal{D}) = \frac{1}{2}\sum_{i=1}^{n}\left(t_i - \mathbb{M}_{\mathbf{w}}(\mathbf{d}_i)\right)^2$$
  Applying the chain rule yields the partial derivative $\operatorname{errorDelta}(\mathcal{D}, w_j)$:
  $$\frac{\partial L_2}{\partial w_j} = -\sum_{i=1}^{n}\left(t_i - \mathbb{M}_{\mathbf{w}}(\mathbf{d}_i)\right)\mathbb{M}_{\mathbf{w}}(\mathbf{d}_i)\left(1 - \mathbb{M}_{\mathbf{w}}(\mathbf{d}_i)\right)d_{j,i}$$
  and the Batch Gradient Descent update rule (with learning rate $\alpha$ or $\eta$):
  $$w_j \leftarrow w_j + \alpha \sum_{i=1}^{n}\left(t_i - \mathbb{M}_{\mathbf{w}}(\mathbf{d}_i)\right)\mathbb{M}_{\mathbf{w}}(\mathbf{d}_i)\left(1 - \mathbb{M}_{\mathbf{w}}(\mathbf{d}_i)\right)d_{j,i}$$
  *(See Section 4.2 for an important note comparing this $L_2$-on-sigmoid formulation with standard binary cross-entropy loss.)*

- **Extension 1: Non-Linear Logistic Regression with Basis Functions** ([`05errorBasedLearningTopic5.pdf`](05errorBasedLearningTopic5.pdf), Slides 86–94; [`fundamentals_of_ml_2ed_ch.pdf`](fundamentals_of_ml_2ed_ch.pdf), Sec. 7.4.5, PDF pp. 45–47 / printed pp. 355–357):  
  Replaces $\mathbf{w} \cdot \mathbf{d}$ with a linear combination of basis functions $\sum_{k=0}^b w_k \phi_k(\mathbf{d})$:
  $$\mathbb{M}_{\mathbf{w}}(\mathbf{d}) = \operatorname{logistic}\left(\sum_{k=0}^{b} w_k \phi_k(\mathbf{d})\right) = \frac{1}{1 + \exp\left(-\sum_{k=0}^{b} w_k \phi_k(\mathbf{d})\right)}$$
  The decision boundary is linear in the $\phi$-space ($\sum_{k=0}^b w_k \phi_k(\mathbf{d}) = 0$) but curves non-linearly in the original feature space $\mathbf{d}$. In the gradient update, $d_{j,i}$ is simply replaced by $\phi_k(\mathbf{d}_i)$.

- **Extension 2: Multinomial Logistic Regression (Normalised One-vs-All)** ([`05errorBasedLearningTopic5.pdf`](05errorBasedLearningTopic5.pdf), Slides 95–105; [`fundamentals_of_ml_2ed_ch.pdf`](fundamentals_of_ml_2ed_ch.pdf), Sec. 7.4.6, Eqs. 7.36–7.38, PDF pp. 47–51 / printed pp. 357–361):  
  For a target with $r > 2$ classes $\operatorname{levels}(t) = \{1, \dots, r\}$:
  1. Construct $r$ binary One-vs-All (OvA) logistic models $\mathbb{M}_{\mathbf{w}_1}, \dots, \mathbb{M}_{\mathbf{w}_r}$, each with its own weight vector $\mathbf{w}_k$ and binary target indicator $t_{i,k} \in \{0, 1\}$ ($1$ if instance $i$ has class $k$, else $0$).
  2. At each epoch, normalise the raw outputs across all $r$ models so they sum to $1$ for every instance:
     $$\mathbb{M}'_{\mathbf{w}_k}(\mathbf{d}) = \frac{\mathbb{M}_{\mathbf{w}_k}(\mathbf{d})}{\sum_{l \in \operatorname{levels}(t)} \mathbb{M}_{\mathbf{w}_l}(\mathbf{d})}$$
  3. Update all $r$ models **in parallel** using the normalised predictions $\mathbb{M}'_{\mathbf{w}_k}(\mathbf{d}_i)$ ([`05errorBasedLearningTopic5.pdf`](05errorBasedLearningTopic5.pdf), Slide 99; note that Textbook Eq. 7.38 on p. 358 has a typographical `[1]` in $\mathbb{M}'_{\mathbf{w}_k}(\mathbf{d}_i[1])$ which Slide 99 corrects to $\mathbb{M}'_{\mathbf{w}_k}(\mathbf{d}_i)$):
     $$\mathbf{w}_k[j] \leftarrow \mathbf{w}_k[j] + \alpha \sum_{i=1}^{n}\left(t_{i,k} - \mathbb{M}'_{\mathbf{w}_k}(\mathbf{d}_i)\right)\mathbb{M}'_{\mathbf{w}_k}(\mathbf{d}_i)\left(1 - \mathbb{M}'_{\mathbf{w}_k}(\mathbf{d}_i)\right)d_{j,i}$$
  4. **Prediction:** Predict $\arg\max_{k} \mathbb{M}'_{\mathbf{w}_k}(\mathbf{d})$.

#### Supported data and tasks
- **Descriptive features:** Continuous and discrete **numeric** features natively; **categorical** features require one-hot / dummy encoding.
- **Target features:** **Binary categorical** ($t \in \{0, 1\}$) for standard logistic regression; **multiclass nominal categorical** ($r > 2$ levels) via One-vs-All multinomial logistic regression. Does **not** support continuous regression targets (unless targets are bounded in $[0, 1]$ or binned).
- **Outputs:** Continuous scores in $(0, 1)$ interpreted as estimated class-membership probabilities $P(t=1 \mid \mathbf{d})$ (or normalised class distribution $\mathbb{M}'_{\mathbf{w}_k}(\mathbf{d})$ summing to $1$ across $k=1,\dots,r$), which are thresholded (at $0.5$ for binary or $\arg\max_k$ for multiclass) to produce discrete class labels.

#### Data quality and preprocessing

| Issue | Effect on the standard model and why | Relevant extension and how it helps | Preprocessing / mitigation and tradeoffs |
| --- | --- | --- | --- |
| **1. Missing values** | Cannot evaluate $\mathbf{w} \cdot \mathbf{d}_i$ or gradient terms when $d_{j,i}$ or $t_i$ is missing. | None covered in course sources. | Drop rows with missing target $t_i$; impute missing descriptive values $d_{j,i}$ using training-split statistics before scaling/training. |
| **2. Outliers** | **Much less sensitive to well-classified extreme inputs than linear regression:** because $\operatorname{logistic}(\mathbf{w} \cdot \mathbf{d}_i)$ saturates at $0$ or $1$, an extreme point far on the *correct* side of the boundary has $t_i - \mathbb{M}_{\mathbf{w}}(\mathbf{d}_i) \approx 0$ and $\mathbb{M}(1-\mathbb{M}) \approx 0$, contributing almost zero gradient ([`05errorBasedLearningTopic5.pdf`](05errorBasedLearningTopic5.pdf), Slide 55; Textbook Fig. 7.11). However, extreme outliers on the *wrong* side of the boundary can either pull the boundary or (under the course $L_2$ loss) get stuck on a flat saturation plateau where $\mathbb{M}(1-\mathbb{M}) \approx 0$. | Non-linear basis functions can curve the boundary around isolated clusters, though they risk overfitting to outlier points. | Standardise features and inspect extreme values; clip or remove corrupted mislabelled outliers in the training split. |
| **3. Noise** | Measurement noise near the boundary $\mathbf{w} \cdot \mathbf{d} = 0$ causes misclassification because $\operatorname{logistic}(x)$ is steepest at $x=0$ (slope $0.25$). Label noise ($t_i$ flipped between $0$ and $1$) shifts the boundary toward the noisy instances. | Keeping basis-function complexity low (linear boundary) avoids carving pockets around noisy labels. | Clean mislabelled training examples; avoid high-degree polynomial basis functions when boundary noise is high. |
| **4. Skew class distributions** | Batch gradient sums over all $n$ instances; the majority class dominates the total loss and gradient, pushing the intercept $w_0$ so that the $\mathbb{M}_{\mathbf{w}}(\mathbf{d}) = 0.5$ boundary shifts toward (or completely past) the minority class, yielding poor minority recall. | None explicitly derived in Ch. 7 (though decision threshold tuning away from $0.5$ or class-weighted loss can be applied). | Undersample majority class, oversample minority class (on training split only), or lower the classification threshold below $0.5$ for the minority positive class (trading precision for recall). |
| **5. Irrelevant features** | Add noise to $\mathbf{w} \cdot \mathbf{d}$ and increase variance on finite samples, especially when expanded with polynomial basis functions. | None in Ch. 7 logistic section (outside Ch. 7: $L_1$/$L_2$ regularisation in `sklearn.linear_model.LogisticRegression`). | Apply feature selection on the training set prior to fitting. |
| **6. Features with different value ranges** | **High practical sensitivity during GD training:** Unscaled features with large magnitudes (e.g., $d_1 \sim 1000$) make $|\mathbf{w} \cdot \mathbf{d}| \gg 0$ even for small initial weights, pushing $\operatorname{logistic}(\mathbf{w} \cdot \mathbf{d})$ into its flat saturation regions ($\approx 0$ or $\approx 1$) where $\mathbb{M}(1-\mathbb{M}) \approx 0$ (**vanishing gradient**), stalling learning. Also distorts gradient directions across features. | Learning rate decay helps late convergence once in the basin, but cannot un-saturate initial weights if $\mathbf{w} \cdot \mathbf{d}$ is huge. | **Always scale/normalise numeric features** (e.g., $z$-score standardisation or $[-1, 1]$ scaling fitted on the training split) and initialise $\mathbf{w}$ to small values near $0$ so initial $\mathbf{w} \cdot \mathbf{d}_i$ lies in the active linear region of the sigmoid. |
| **7. Different descriptive feature data types** | Requires numeric inputs for dot product $\mathbf{w} \cdot \mathbf{d}$. | **One-hot / dummy encoding** for nominal categorical features, exactly as in linear regression. | Encode categoricals with training-fitted one-hot encoding before scaling/training. |

**Additional data-quality considerations:**
- **Complete linear separability (unregularised):** When classes are perfectly linearly separable, increasing $\|\mathbf{w}\| \to \infty$ pushes $\mathbb{M}_{\mathbf{w}}(\mathbf{d}_i)$ ever closer to $0$ and $1$, further reducing $L_2$ error without a finite weight norm minimum; stopping criteria (epoch limit or tiny $\Delta L_2$) halt training in practice.
- **Redundant features, high dimensionality, and sparse data:** Behave similarly to linear regression—collinear features create non-unique weight combinations, and high-dimensional basis expansions ($b \gg n$) overfit sparse regions.

#### Important choices and variants
- **Learned parameters:** Weight vector $\mathbf{w} = (w_0, \dots, w_m)$ (or $r$ weight vectors $\mathbf{w}_1, \dots, \mathbf{w}_r$ in OvA multinomial logistic regression).
- **Hyperparameters:** Initial learning rate $\alpha_0$, decay constant $c$, stopping threshold / max epochs, choice of basis functions $\phi_k(\mathbf{d})$, and classification decision threshold (default $0.5$).

#### Strengths, limitations, and practical use
- **When to use:** Strong baseline for binary or multiclass classification when a smooth decision score in $(0, 1)$, fast prediction, and interpretable feature effects are desired.
- **Interpretability:** High for linear features (the sign of $w_j$ indicates whether increasing $d_j$ increases or decreases the log-odds / score of class $1$).
- **Computational & storage costs:** Training costs $\mathcal{O}(E \cdot n \cdot m)$ for binary and $\mathcal{O}(E \cdot r \cdot n \cdot m)$ for $r$-class OvA. Prediction costs $\mathcal{O}(m)$ (or $\mathcal{O}(r \cdot m)$) per query; storage is $\mathcal{O}(m)$ (or $\mathcal{O}(r \cdot m)$).

---

### 3.3 Support Vector Machines (SVMs)

*(Covered in [`fundamentals_of_ml_2ed_ch.pdf`](fundamentals_of_ml_2ed_ch.pdf), Sec. 7.4.7, PDF pp. 51–57 / printed pp. 361–367 and [`README.md`](README.md); not included in the [`05errorBasedLearningTopic5.pdf`](05errorBasedLearningTopic5.pdf) slide deck.)*

#### Main idea and assumptions
While logistic regression finds *a* separating hyperplane when classes are linearly separable, infinitely many separating boundaries exist and logistic regression may place its boundary arbitrarily close to training instances ([`fundamentals_of_ml_2ed_ch.pdf`](fundamentals_of_ml_2ed_ch.pdf), Fig. 7.21a–b, PDF p. 51 / printed p. 361). A **Support Vector Machine (SVM)** searches for the unique **maximum-margin hyperplane**—the separating boundary that maximises the perpendicular distance (the **margin**) to the nearest training instances of either class ([`fundamentals_of_ml_2ed_ch.pdf`](fundamentals_of_ml_2ed_ch.pdf), Fig. 7.21c, PDF p. 51 / printed p. 361).
- **Support vectors:** The critical training instances that lie directly on the margin boundaries. Only the support vectors determine the orientation and position of the separating boundary; all other training instances further from the boundary have zero influence ($\alpha[i] = 0$).
- **Inductive bias:** Maximising the margin reduces structural risk / generalisation error because small perturbations to query instances near the boundary are less likely to push them across the decision surface.

#### How it learns and predicts

- **Target encoding and margin formulation (Hard-Margin Linear SVM):**  
  Binary target labels are encoded as $t_i \in \{-1, +1\}$. The separating hyperplane is $w_0 + \mathbf{w} \cdot \mathbf{d} = 0$, and the positive and negative margin boundaries are scaled to:
  $$w_0 + \mathbf{w} \cdot \mathbf{d} = +1 \quad \text{and} \quad w_0 + \mathbf{w} \cdot \mathbf{d} = -1$$
  Every training instance must satisfy the margin constraint ([`fundamentals_of_ml_2ed_ch.pdf`](fundamentals_of_ml_2ed_ch.pdf), Eqs. 7.39–7.41, PDF p. 52 / printed p. 362):
  $$t_i\left(w_0 + \mathbf{w} \cdot \mathbf{d}_i\right) \ge 1 \quad \forall i \in \{1, \dots, n\}$$
  The total width of the margin between the two boundaries is $\frac{2}{\|\mathbf{w}\|}$.

- **Training algorithm (Constrained Convex Optimisation / Quadratic Programming):**  
  1. **Primal objective:** Maximise $\frac{2}{\|\mathbf{w}\|}$ (equivalently, minimise $\frac{1}{2}\|\mathbf{w}\|^2$) subject to $t_i(w_0 + \mathbf{w} \cdot \mathbf{d}_i) \ge 1$.
  2. **Dual solution via Lagrange multipliers:** Because standard gradient descent is poorly suited to these inequality constraints, SVMs use **Quadratic Programming (QP)** to solve for non-negative **Lagrange multipliers** $\alpha[i] \ge 0$ for each training instance $i$ ([`fundamentals_of_ml_2ed_ch.pdf`](fundamentals_of_ml_2ed_ch.pdf), PDF pp. 52–53 / printed pp. 362–363).
  3. Only the **support vectors** (let $\mathcal{S}$ denote the set of support vectors) receive $\alpha[i] > 0$; all non-support vectors have $\alpha[i] = 0$ and can be discarded after training. The primal weight vector is $\mathbf{w} = \sum_{i \in \mathcal{S}} \alpha[i] t_i \mathbf{d}_i$ ([`fundamentals_of_ml_2ed_ch.pdf`](fundamentals_of_ml_2ed_ch.pdf), Eq. 7.43, PDF p. 53 / printed p. 363).

- **Prediction rule (Linear and Kernel SVM):**  
  For a query instance $\mathbf{q}$, the linear SVM classifies using only the support vectors $\mathcal{S}$ ([`fundamentals_of_ml_2ed_ch.pdf`](fundamentals_of_ml_2ed_ch.pdf), Eqs. 7.42 & 7.44, PDF pp. 52–53 / printed pp. 362–363):
  $$\mathbb{M}_{\boldsymbol{\alpha}, w_0}(\mathbf{q}) = \operatorname{sign}\left(w_0 + \sum_{i \in \mathcal{S}} \alpha[i]\, t_i\, (\mathbf{d}_i \cdot \mathbf{q})\right)$$
  returning $+1$ if the argument is $>0$ and $-1$ if $<0$.
  When **basis functions** $\boldsymbol{\phi}(\cdot)$ are used to map inputs into a higher-dimensional space, the prediction rule depends only on inner products $\boldsymbol{\phi}(\mathbf{d}_i) \cdot \boldsymbol{\phi}(\mathbf{q})$. Using the **kernel trick**, this inner product is replaced directly by a **kernel function** $K(\mathbf{d}_i, \mathbf{q}) = \boldsymbol{\phi}(\mathbf{d}_i) \cdot \boldsymbol{\phi}(\mathbf{q})$ without ever explicitly computing the high-dimensional coordinates $\boldsymbol{\phi}(\mathbf{d})$ ([`fundamentals_of_ml_2ed_ch.pdf`](fundamentals_of_ml_2ed_ch.pdf), Eqs. 7.46–7.48, Table 7.6, PDF pp. 55–56 / printed pp. 365–366):
  $$\mathbb{M}_{\boldsymbol{\alpha}, w_0}(\mathbf{q}) = \operatorname{sign}\left(w_0 + \sum_{i \in \mathcal{S}} \alpha[i]\, t_i\, K(\mathbf{d}_i, \mathbf{q})\right)$$
  **Standard kernels covered** ([`fundamentals_of_ml_2ed_ch.pdf`](fundamentals_of_ml_2ed_ch.pdf), Table 7.6, PDF p. 56 / printed p. 366):
  - **Linear kernel:** $K(\mathbf{d}_i, \mathbf{d}_j) = \mathbf{d}_i \cdot \mathbf{d}_j + c$
  - **Polynomial kernel:** $K(\mathbf{d}_i, \mathbf{d}_j) = (\mathbf{d}_i \cdot \mathbf{d}_j + c)^p$ (degree $p$)
  - **Radial Basis Function (RBF / Gaussian) kernel:** $K(\mathbf{d}_i, \mathbf{d}_j) = \exp\!\left(-\frac{\|\mathbf{d}_i - \mathbf{d}_j\|^2}{2\sigma^2}\right)$

#### Supported data and tasks
- **Descriptive features:** Continuous/discrete **numeric** features; categorical features require numerical encoding (one-hot encoding). Because margin width $\frac{2}{\|\mathbf{w}\|}$ and kernel functions ($\mathbf{d}_i \cdot \mathbf{d}_j$ and Euclidean distance $\|\mathbf{d}_i - \mathbf{d}_j\|^2$) are directly scale-dependent, **normalising/standardising descriptive features is essential**.
- **Target features:** **Binary categorical** encoded as $t \in \{-1, +1\}$ natively; **multiclass categorical** supported via **One-vs-All (OvA)** ensemble of binary SVMs ([`fundamentals_of_ml_2ed_ch.pdf`](fundamentals_of_ml_2ed_ch.pdf), PDF p. 56 / printed p. 366). *(Note: Support Vector Regression exists outside the Ch. 7 scope, which covers SVMs purely as classifiers.)*
- **Outputs:** Class labels $\{-1, +1\}$ and signed functional margin scores $w_0 + \sum_{i \in \mathcal{S}} \alpha[i] t_i K(\mathbf{d}_i, \mathbf{q})$. Does **not** natively output calibrated probabilities.

#### Data quality and preprocessing

| Issue | Effect on the standard model and why | Relevant extension and how it helps | Preprocessing / mitigation and tradeoffs |
| --- | --- | --- | --- |
| **1. Missing values** | Cannot compute inner products $\mathbf{d}_i \cdot \mathbf{d}_j$ or Euclidean distances $\|\mathbf{d}_i - \mathbf{d}_j\|^2$ if features are missing; missing $t_i$ breaks margin constraints. | None covered in course sources. | Drop instances with missing target $t_i$; impute missing $d_{j,i}$ using training-split statistics before scaling. |
| **2. Outliers** | **Far from boundary (correct side):** Zero effect ($\alpha[i] = 0$) because only support vectors on/inside the margin affect the hyperplane. **Near/across boundary:** In a **hard-margin SVM**, a single outlier inside the margin drastically narrows and tilts the margin, and a single outlier across the boundary makes the QP optimisation **infeasible** (no solution exists). | **Soft-margin SVM** ([`fundamentals_of_ml_2ed_ch.pdf`](fundamentals_of_ml_2ed_ch.pdf), Fig. 7.22, PDF p. 54 / printed p. 364) relaxes $t_i(w_0 + \mathbf{w}\cdot\mathbf{d}_i) \ge 1$, allowing some instances inside or on the wrong side of the margin so a wide, generalisable margin is preserved. | Use a soft-margin SVM (tune regularisation tradeoff) and clean obvious mislabelled outliers. |
| **3. Noise** | Overlapping classes or noisy labels near the boundary make data linearly non-separable, breaking hard-margin SVM or forcing high-degree/narrow-RBF kernels to wrap tightly around noisy points (overfitting). | **Soft-margin SVM** tolerates boundary overlap and noisy labels without collapsing the margin ([`fundamentals_of_ml_2ed_ch.pdf`](fundamentals_of_ml_2ed_ch.pdf), PDF p. 54 / printed p. 364). | Prefer soft-margin SVM with moderate kernel bandwidth $\sigma$ or lower polynomial degree $p$. |
| **4. Skew class distributions** | With imbalanced classes and an equal misclassification penalty in soft-margin SVM, the boundary is pushed toward the minority class so fewer majority instances violate the margin, degrading minority recall. | Class-weighted soft-margin penalties (outside Ch. 7 equations) or threshold adjustment on the signed margin score. | Resample training data or adjust decision threshold away from $0$. |
| **5. Irrelevant features** | Margin maximization ($L_2$ norm $\|\mathbf{w}\|^2$) handles moderate irrelevant features well in linear SVMs, but in **RBF kernels** $\|\mathbf{d}_i - \mathbf{d}_j\|^2 = \sum_{k=1}^m (d_{k,i} - d_{k,j})^2$, many noisy irrelevant features dominate pairwise Euclidean distances, degrading kernel similarity. | Using a linear kernel or feature selection when many irrelevant features are present. | Perform feature selection on the training split when using distance-based (RBF) kernels. |
| **6. Features with different value ranges** | **Highly sensitive:** Both the geometric margin $\frac{2}{\|\mathbf{w}\|}$ and kernel values ($\mathbf{d}_i \cdot \mathbf{d}_j$ and $\|\mathbf{d}_i - \mathbf{d}_j\|^2$) are dominated by features with the largest numeric ranges, effectively ignoring small-range features when maximising margin width. | None; scaling is a prerequisite of the geometric formulation. | **Mandatory:** Standardise or normalise all features to a common scale using training-split parameters before fitting or evaluating kernels. |
| **7. Different descriptive feature data types** | Requires numeric vectors for dot products and Euclidean distances. | **One-hot / dummy encoding** converts categorical features into binary numeric vectors. | Apply one-hot encoding (fitted on training split) and scale appropriately. |

**Additional data-quality considerations:**
- **High dimensionality ($m \gg n$) and Redundant Features:** SVMs excel in high-dimensional spaces because margin maximisation acts as strong capacity control, and the **kernel trick** computes inner products in $\mathcal{O}(m)$ time per pair instead of explicitly generating $\binom{m+p}{p}$ basis features ([`fundamentals_of_ml_2ed_ch.pdf`](fundamentals_of_ml_2ed_ch.pdf), PDF pp. 55–56 / printed pp. 365–366).
- **Limited / sparse data:** Because the boundary depends only on the support vectors near the class interface, SVMs often generalise well from relatively small datasets provided the support vectors adequately sample the boundary.

#### Important choices and variants
- **Learned parameters:** Lagrange multipliers $\alpha[i]$ for the support vectors $i \in \mathcal{S}$ and intercept $w_0$ (plus primal weights $\mathbf{w}$ if using a linear kernel).
- **Hyperparameters:**
  - **Hard-margin vs. Soft-margin tradeoff parameter:** Controls how strictly margin violations are penalised versus keeping the margin wide ([`fundamentals_of_ml_2ed_ch.pdf`](fundamentals_of_ml_2ed_ch.pdf), PDF p. 54 / printed p. 364).
  - **Kernel choice and kernel hyperparameters** ([`fundamentals_of_ml_2ed_ch.pdf`](fundamentals_of_ml_2ed_ch.pdf), Table 7.6, PDF p. 56 / printed p. 366): polynomial degree $p$ and constant $c$, or RBF bandwidth $\sigma$ (small $\sigma$ creates tight, spiky islands around training points $\implies$ overfitting; huge $\sigma$ flattens the kernel $\implies$ underfitting).

#### Strengths, limitations, and practical use
- **When to use:** High-accuracy binary/multiclass classification, especially with complex non-linear boundaries or high-dimensional feature spaces where explicit basis expansion is prohibitive.
- **Interpretability:** Linear SVM weights $\mathbf{w}$ are interpretable like logistic regression; non-linear kernel SVMs are relatively opaque (predictions depend on kernel similarities to support vectors).
- **Computational & storage costs:**
  - **Training:** Quadratic programming scales super-linearly (roughly $\mathcal{O}(n^2 \cdot m)$ to $\mathcal{O}(n^3)$) in dataset size $n$, making training slow on very large $n$.
  - **Prediction & Storage:** Retains only the $|\mathcal{S}|$ support vectors ($\mathcal{O}(|\mathcal{S}| \cdot m)$ storage and prediction cost per query; for linear SVM, $\mathbf{w}$ can be precomputed for $\mathcal{O}(m)$ prediction).

---

### 3.4 Single Perceptron

*(Listed in [`05errorBasedLearningTopic5.pdf`](05errorBasedLearningTopic5.pdf) Slide 2 Outline item 9, discussed in Slides 55–58 as the hard-threshold classifier, and examined in [`quiz5.md`](quiz5.md) Q8. Note: slides 108–181 of `05errorBasedLearningTopic5.pdf` are missing from the PDF file; see Section 5.)*

#### Main idea and assumptions
A **Single Perceptron** is the simplest artificial neural network unit: it computes a weighted sum of its descriptive inputs plus a bias/threshold term ($\text{net} = \mathbf{w} \cdot \mathbf{d}$) and passes that net input through an **activation function** $f(\text{net})$ to produce an output.
- In its classic hard-threshold form ([`05errorBasedLearningTopic5.pdf`](05errorBasedLearningTopic5.pdf), Slide 55; [`fundamentals_of_ml_2ed_ch.pdf`](fundamentals_of_ml_2ed_ch.pdf), Eq. 7.25, PDF p. 30 / printed p. 340), it uses a step function to output a binary class label ($0$ or $1$, or $-1$ and $+1$).
- When equipped with a continuous activation function (linear for regression, logistic sigmoid for binary classification), a single neuron is mathematically equivalent to multivariable linear or logistic regression.
- **Inductive bias and assumptions:** Strictly a **linear classifier**—it assumes the classes are **linearly separable** by a single hyperplane $\mathbf{w} \cdot \mathbf{d} = 0$ ([`quiz5.md`](quiz5.md), Q8). It cannot represent non-linearly separable functions such as **XOR**.

#### How it learns and predicts

- **Prediction rule:**  
  With augmented input $\mathbf{d} = (d_0, d_1, \dots, d_m)$ (where $d_0 = 1$, or $d_0 = -1$ in Engelbrecht's convention where $w_0 = \theta$ is the threshold) and weights $\mathbf{w} = (w_0, w_1, \dots, w_m)$:
  $$\hat{y} = f(\mathbf{w} \cdot \mathbf{d}), \qquad \text{where } f_{\text{step}}(x) = \begin{cases} 1 & \text{if } x \ge 0 \\ 0 & \text{if } x < 0 \end{cases}$$
- **Training algorithm (Perceptron Learning Rule / Delta Rule):**  
  1. Initialise weights $\mathbf{w}$ to small random values and choose learning rate $\eta \in (0, 1]$.
  2. For each training instance $(\mathbf{d}_i, t_i)$ (online/stochastic) or accumulated over the epoch (batch), update weights proportional to the prediction error $(t_i - \hat{y}_i)$:
     $$w_j \leftarrow w_j + \eta (t_i - \hat{y}_i) d_{j,i}$$
  3. **Convergence behaviour:** If the training data is **linearly separable**, the perceptron convergence theorem guarantees convergence to a separating hyperplane in finitely many steps. If the data is **not linearly separable**, the hard-threshold perceptron never converges and oscillates indefinitely (whereas a continuous sigmoid neuron trained with gradient descent converges to a minimum of the smooth error surface, [`05errorBasedLearningTopic5.pdf`](05errorBasedLearningTopic5.pdf), Slides 57–61).

#### Supported data and tasks
- **Descriptive features:** Numeric features; categorical features require one-hot / dummy encoding.
- **Target features:** Binary classification ($t \in \{0, 1\}$ or $\{-1, +1\}$) with step/sigmoid activation; continuous regression only if a linear activation function is used ([`quiz5.md`](quiz5.md), Q8).
- **Outputs:** Discrete binary labels $\{0, 1\}$ (step activation) or continuous scores in $(0, 1)$ (sigmoid activation).

#### Data quality and preprocessing

| Issue | Effect on the standard model and why | Relevant extension and how it helps | Preprocessing / mitigation and tradeoffs |
| --- | --- | --- | --- |
| **1. Missing values** | Cannot compute $\mathbf{w} \cdot \mathbf{d}_i$ or update $w_j$ when any $d_{j,i}$ or $t_i$ is missing. | None. | Drop instances with missing target $t_i$; impute missing descriptive values $d_{j,i}$ on the training split. |
| **2. Outliers** | If an outlier lies on the wrong side of the boundary (making the data linearly non-separable), the hard-threshold perceptron updates indefinitely trying to classify it and never converges. | Replacing the hard step function with a continuous **logistic sigmoid activation** (logistic regression) or using a **pocket algorithm / epoch limit** stops infinite oscillation. | Remove mislabelled outliers or switch to a continuous sigmoid activation / soft-margin SVM. |
| **3. Noise** | Boundary noise that causes class overlap breaks linear separability, preventing hard-threshold convergence and causing weights to thrash each epoch. | Continuous sigmoid activation (smooth gradient descent) + learning rate decay $\eta_\tau$ dampens oscillation around noisy boundary points. | Use learning rate decay and early stopping, or switch to logistic regression / soft-margin SVM. |
| **4. Skew class distributions** | Majority-class instances dominate weight updates, pushing the hyperplane toward the minority class. | Class-balanced updates or threshold adjustment. | Balance training classes (undersampling/oversampling on training split) or weight minority errors higher. |
| **5. Irrelevant features** | Add noise to the net input $\mathbf{w} \cdot \mathbf{d}$; can cause spurious updates on finite samples. | Weight decay / feature selection. | Select relevant features on the training split. |
| **6. Features with different value ranges** | Because $\Delta w_j = \eta (t_i - \hat{y}_i) d_{j,i}$, features with large numeric magnitudes receive disproportionately large weight updates, elongating convergence and skewing the hyperplane orientation during early epochs. | None native. | Standardise or normalise all numeric features using training-split parameters before training. |
| **7. Different descriptive feature data types** | Only numeric multiplication $w_j d_j$ is supported. | **One-hot / dummy encoding** for nominal categorical features. | Fit one-hot encoding on the training split. |

**Additional data-quality considerations:**
- **Redundant features & high dimensionality:** Collinear features yield infinitely many equivalent hyperplanes; when $m \ge n$, data is almost always linearly separable even on random noise, increasing overfitting risk.

#### Important choices and variants
- **Learned parameters:** Weights $\mathbf{w} = (w_0, \dots, w_m)$.
- **Hyperparameters:** Learning rate $\eta$, activation function (step vs. sigmoid vs. linear), batch vs. stochastic (online) updates, and maximum epochs.

#### Strengths, limitations, and practical use
- **When to use:** Primarily pedagogical or as the elementary building block of multilayer neural networks; for standalone binary classification, **Logistic Regression** or **Linear SVM** is almost always preferred because logistic regression provides smooth probability scores and SVMs maximise the margin.
- **Complexity:** $\mathcal{O}(n \cdot m)$ per epoch; $\mathcal{O}(m)$ prediction and storage.

---

### 3.5 Multilayer (Shallow) Feedforward Neural Networks (FFNN / MLP)

*(Listed in [`05errorBasedLearningTopic5.pdf`](05errorBasedLearningTopic5.pdf) Slide 2 Outline items 10–12, and directly examined in [`quiz5.md`](quiz5.md) Q9–Q15 and [`README.md`](README.md). Because slides 108–181 of `05errorBasedLearningTopic5.pdf` are missing from the PDF file, standard backpropagation equations from the listed reading—Engelbrecht 2007 Ch. 3 / Kelleher et al.—are provided below and clearly flagged; see Section 5.)*

#### Main idea and assumptions
A single perceptron or logistic regression model can only form a single linear boundary in its input space. A **Multilayer (Shallow) Feedforward Neural Network (FFNN)**—also called a **Multilayer Perceptron (MLP)**—overcomes this limitation by stacking an **input layer**, one (or more) **hidden layers** of non-linear neurons, and an **output layer** ([`quiz5.md`](quiz5.md), Q9–Q10).
- **Hidden units as learned basis functions:** Each hidden neuron computes a weighted sum of the inputs and applies a differentiable **non-linear activation function** (such as the logistic sigmoid or hyperbolic tangent $\tanh$). Whereas polynomial regression requires the practitioner to manually specify fixed basis functions $\phi_k(\mathbf{d})$ beforehand, a neural network **learns its own non-linear hidden feature representations** simultaneously with the output weights via error backpropagation.
- **Why non-linear hidden activations are essential** ([`quiz5.md`](quiz5.md), Q10, Q13): If every hidden unit used a purely linear activation function, the composition of linear layers $\mathbf{W}^{(2)}(\mathbf{W}^{(1)}\mathbf{d})$ would collapse algebraically into a single linear transformation $\mathbf{W}'\mathbf{d}$, making the multilayer network no more expressive than single-layer linear regression.
- **Inductive bias:** Smooth, hierarchical composition of learned non-linear features. By the universal approximation property, an FFNN with even a single hidden layer containing enough non-linear hidden units can approximate any continuous function on a compact domain to arbitrary accuracy.

#### How it learns and predicts

- **Architecture and Forward Pass (Prediction Rule)** *(standard 1-hidden-layer FFNN notation from Engelbrecht 2007, Ch. 3, cited on Slide 3)*:  
  Given input vector $\mathbf{d} = (d_0, d_1, \dots, d_m)$ (with bias input $d_0 = 1$), $J$ hidden units with hidden-layer weights $v_{j,i}$ (connecting input $i$ to hidden unit $j$), hidden activation function $f_h$, and $K$ output units with output-layer weights $w_{k,j}$ (connecting hidden unit $j$ to output unit $k$, with hidden bias $z_0 = 1$) and output activation $f_o$:
  1. **Hidden layer activations ($j = 1, \dots, J$):**
     $$\text{net}_{h,j} = \sum_{i=0}^{m} v_{j,i} d_i, \qquad z_j = f_h(\text{net}_{h,j})$$
  2. **Output layer predictions ($k = 1, \dots, K$):**
     $$\text{net}_{o,k} = \sum_{j=0}^{J} w_{k,j} z_j, \qquad o_k = f_o(\text{net}_{o,k})$$
  - For **regression**, the output activation $f_o$ is typically **linear** ($f_o(x) = x$) so outputs are unbounded, or sigmoid/$\tanh$ if targets are scaled into $(0, 1)$ or $(-1, 1)$.
  - For **binary classification**, a single output unit ($K=1$) with **logistic sigmoid** activation outputs $o_1 \in (0, 1)$.
  - For **$K$-class classification**, $K$ output units represent the $K$ classes (using one-hot target encoding $\mathbf{t}_p \in \{0, 1\}^K$).

- **Training Algorithm (Backpropagation with Stochastic or Batch Gradient Descent)** ([`quiz5.md`](quiz5.md), Q11, Q12, Q14, Q15):  
  Minimises the Sum of Squared Errors ($L_2$) across output units: $E_p = \frac{1}{2}\sum_{k=1}^{K}(t_{p,k} - o_{p,k})^2$ for pattern $p$ (or summed over all $n$ patterns in Batch GD).
  1. **Initialise** all weights $v_{j,i}$ and $w_{k,j}$ to **small random values** near zero (e.g., uniform in $\left[-\frac{1}{\sqrt{\text{fan\_in}}}, \frac{1}{\sqrt{\text{fan\_in}}}\right]$) ([`quiz5.md`](quiz5.md), Q14).
     - *Why not initialise all weights to zero or an identical constant?* Because every hidden unit would compute the exact same net input and receive the exact same backpropagation gradient, remaining identical forever (**symmetry problem**).
     - *Why not initialise to large random values?* Because large net inputs $|\text{net}_{h,j}| \gg 0$ push sigmoid/$\tanh$ activations into their flat saturation tails where the derivative $f'_h(\text{net}_{h,j}) \approx 0$ (**premature saturation / vanishing gradient**), paralysing weight updates ([`quiz5.md`](quiz5.md), Q14).
  2. **Repeat for each epoch** (either **Stochastic/Online GD** updating after each pattern $p$, or **Batch GD** accumulating gradients over all $n$ patterns before updating, [`quiz5.md`](quiz5.md), Q11):
     - **Forward pass:** Compute hidden activations $z_j$ and output predictions $o_k$.
     - **Backward pass (Error Backpropagation via Chain Rule):**
       - Compute **output unit error signal** $\delta_{o,k}$:
         $$\delta_{o,k} = -(t_k - o_k) f'_o(\text{net}_{o,k})$$
         *(For sigmoid output, $f'_o(\text{net}_{o,k}) = o_k(1 - o_k)$; for linear output, $f'_o(\text{net}_{o,k}) = 1$.)*
       - Propagate error backward to compute **hidden unit error signal** $\delta_{h,j}$ ($j = 1, \dots, J$):
         $$\delta_{h,j} = f'_h(\text{net}_{h,j}) \sum_{k=1}^{K} w_{k,j} \delta_{o,k}$$
     - **Weight updates** (with learning rate $\eta$ and optional momentum $\alpha_{\text{mom}} \Delta w(t-1)$):
       $$w_{k,j} \leftarrow w_{k,j} - \eta \delta_{o,k} z_j, \qquad v_{j,i} \leftarrow v_{j,i} - \eta \delta_{h,j} d_i$$
  3. **Stop** using validation-set early stopping, minimum error threshold, or maximum epochs.

#### Supported data and tasks
- **Descriptive features:** Continuous/discrete **numeric** features; **categorical** features require one-hot / dummy encoding (or binary encoding). **Scaling inputs (e.g., to $[0, 1]$, $[-1, 1]$, or $z$-scores) is essential** to prevent hidden-unit saturation.
- **Target features:** Supports **both categorical targets** (binary and multiclass classification, using one-hot target vectors for $K$ classes) and **continuous numeric targets** (regression, including multi-output regression). When sigmoid or $\tanh$ output activations are used for regression, targets **must be scaled** into the active range of the output activation function (e.g., $[0.1, 0.9]$ for sigmoid) so the network can actually reach the target values without saturating ([`quiz5.md`](quiz5.md), Q16).
- **Outputs:** Continuous numeric values (regression) or continuous class scores in $(0, 1)$ (classification, converted to labels via threshold or $\arg\max_k o_k$).

#### Data quality and preprocessing

*(Directly addresses all seven "Personal Summary Questions - Neural Networks" in [`README.md`](README.md).)*

| Issue | Effect on the standard model and why | Relevant extension and how it helps | Preprocessing / mitigation and tradeoffs |
| --- | --- | --- | --- |
| **1. Missing values** | Cannot compute $\text{net}_{h,j} = \sum_{i=0}^m v_{j,i} d_i$ if any $d_i$ is missing (`NaN` propagates through every hidden and output unit); missing $t_k$ prevents computing output error $\delta_{o,k}$. | None in standard FFNN. | Drop rows with missing targets; impute missing descriptive features using training-split statistics (optionally adding a binary missingness indicator feature). |
| **2. Outliers** | **Extreme input outliers ($d_i$):** Push $\text{net}_{h,j}$ into the flat tails of sigmoid/$\tanh$ hidden activations where $f'_h(\text{net}_{h,j}) \approx 0$, saturating hidden units and stalling learning. **Extreme target outliers ($t_k$ in regression):** Under $L_2$ loss and high network capacity ($J$ hidden units), the network bends its non-linear surface to interpolate outliers, causing severe overfitting. | **Weight decay ($L_2$ regularisation)**, **early stopping**, or using **fewer hidden units $J$** restricts the curvature of the fitted function so it cannot spike to match isolated target outliers ([`quiz5.md`](quiz5.md), Q15). | Clip/winsorise or remove corrupted outliers in training data; scale inputs robustly (e.g., using median/IQR if extreme inputs exist) on the training split. |
| **3. Noise** | Because an over-parameterised FFNN is a universal approximator, training too long or with too many hidden units $J$ causes the network to **memorise both input measurement noise and incorrect target labels** (overfitting, [`quiz5.md`](quiz5.md), Q15), resulting in low training error $\mathcal{E}_T$ but high generalisation error $\mathcal{E}_G$. | **Early stopping** (halting training when validation error starts rising), **reducing hidden units $J$**, and **weight decay (regularisation)** prevent the weights from growing large enough to fit high-frequency noise ([`quiz5.md`](quiz5.md), Q15). | Use a validation set for early stopping and tune $J$ conservatively. |
| **4. Skew class distributions** | Majority-class patterns dominate the cumulative gradient updates in each epoch; the network learns to output high scores for the majority class and low scores for the minority class across most of the input space. | Cost-sensitive error weighting (multiplying $\delta_{o,k}$ by a higher weight for minority-class patterns) or adjusting the classification threshold. | Oversample minority class / undersample majority class in the training split, or use stratified mini-batches. |
| **5. Irrelevant features** | Every input $d_i$ is connected to every hidden unit $j$ via $v_{j,i}$ ($J$ extra parameters per irrelevant feature). Irrelevant features inject random variations into $\text{net}_{h,j}$, create extra local minima on the non-convex error surface, and increase overfitting when $n$ is limited. | **Weight decay ($L_2$ / $L_1$ regularisation)** shrinks weights $v_{j,i}$ connected to unhelpful inputs toward zero. | Perform feature selection on the training set before training to reduce parameter count and improve generalisation. |
| **6. Features with different value ranges** | **Severely degrades learning:** (1) Unscaled inputs with large magnitudes make $|\text{net}_{h,j}| \gg 0$ at initialization, immediately **saturating** sigmoid/$\tanh$ hidden units ($f'_h \approx 0$, vanishing gradients); (2) Because $\frac{\partial E}{\partial v_{j,i}} = \delta_{h,j} d_i$, gradients for large-scale features are orders of magnitude larger than for small-scale features, creating steep ravines on the error surface where gradient descent oscillates or crawls. | Adaptive per-weight learning rates / momentum help navigate ravines, but cannot prevent initial activation saturation if raw inputs are large. | **Mandatory in practice:** Standardise ($z$-score) or normalise all descriptive features to $[0, 1]$ or $[-1, 1]$ (using training-split statistics), and scale regression targets if bounded output activations are used ([`quiz5.md`](quiz5.md), Q16). |
| **7. Different descriptive feature data types** | Only numeric multiplications $v_{j,i} d_i$ are defined. | **One-hot / dummy encoding** for nominal categorical features (or binary encoding for two-level categories). | Encode categoricals into binary $\{0, 1\}$ (or $\{-1, +1\}$) indicator columns fitted on the training split. |

**Additional data-quality considerations:**
- **Redundant / collinear features:** Unlike unregularised OLS matrix inversion, backpropagation with small random initial weights handles collinear features without crashing, splitting the weight contribution across the redundant inputs. However, redundant features still increase the number of weights and slow down training.
- **High dimensionality and limited training data:** An FFNN with $m$ inputs, $J$ hidden units, and $K$ outputs has $(m+1)J + (J+1)K$ learned weights. When $m$ is large and $n$ is small, the parameter count easily exceeds $n$, leading to severe overfitting unless $J$ is very small or strong regularisation / early stopping is applied ([`quiz5.md`](quiz5.md), Q15).

#### Important choices and variants
- **Learned parameters:** Input-to-hidden weights and biases $v_{j,i}$ and hidden-to-output weights and biases $w_{k,j}$.
- **Hyperparameters:**
  - **Number of hidden units $J$ (and hidden layers):** Primary control on model capacity ([`quiz5.md`](quiz5.md), Q15). Too few hidden units $\implies$ **underfitting** (cannot capture the true non-linear target function); too many hidden units $\implies$ **overfitting** (memorises training noise).
  - **Activation functions $f_h, f_o$:** Must be non-linear and differentiable in the hidden layer (e.g., logistic sigmoid, $\tanh$, ReLU) to learn non-linear functions ([`quiz5.md`](quiz5.md), Q10, Q12, Q13).
  - **Weight initialisation range:** Small random values around $0$ to break symmetry while avoiding saturation ([`quiz5.md`](quiz5.md), Q14).
  - **Training mode (Stochastic / Online vs. Batch GD)** ([`quiz5.md`](quiz5.md), Q11):
    - **Batch GD:** Computes exact gradient over all $n$ instances before updating; smooth trajectory but can get trapped in shallow local minima and is slow per update when $n$ is large.
    - **Stochastic (Online) GD:** Updates weights after every single instance; noisy gradient steps help **escape local minima** and redundancy in large datasets makes per-instance updates much faster, though it fluctuates around the minimum unless $\eta$ decays.
  - **Learning rate $\eta$ and Momentum:** Control convergence speed and dampen oscillations across narrow error-surface valleys.

#### Strengths, limitations, and practical use
- **When to use:** Complex non-linear regression or classification tasks where manual basis-function engineering is difficult and sufficient training data is available.
- **Limitations:** **Non-convex error surface** with many local minima and flat plateaus (no guarantee of finding the global minimum); slow iterative training requiring careful hyperparameter tuning (scaling, initialisation, $\eta$, $J$, early stopping); **black-box** predictions with low interpretability.
- **Computational & storage costs:** Each forward-backward epoch over $n$ instances costs $\mathcal{O}(n \cdot (m J + J K))$, and training often requires hundreds or thousands of epochs $E$. Prediction is fast at $\mathcal{O}(m J + J K)$ per query, and storage is $\mathcal{O}(m J + J K)$ weights.

---

## 4. Exam and comparison points

### 4.1 Answers to Personal Summary Questions ([`README.md`](README.md))

#### Part A: Linear Regression, Logistic Regression, and Support Vector Machines
1. **Can these models handle both numeric and categorical descriptive features?**  
   - **Natively, no:** All three models compute dot products $\mathbf{w} \cdot \mathbf{d}$ (or Euclidean distances $\|\mathbf{d}_i - \mathbf{d}_j\|^2$ in RBF SVMs), which are only defined for **numeric** features.
   - **With preprocessing, yes:** Nominal categorical features with $k$ levels are converted into $k$ (or $k-1$) binary $\{0, 1\}$ indicator features via **one-hot / dummy encoding** ([`05errorBasedLearningTopic5.pdf`](05errorBasedLearningTopic5.pdf), Slides 48–52; [`fundamentals_of_ml_2ed_ch.pdf`](fundamentals_of_ml_2ed_ch.pdf), Sec. 7.4.3, PDF pp. 26–28 / printed pp. 336–338). Assigning a single integer code ($1, 2, \dots, k$) to a nominal feature is incorrect because it imposes a false linear ordering and equal spacing between categories.
2. **Can these models handle both numeric and categorical target features?**  
   - **Multivariable Linear Regression** (and Polynomial/Basis-Function Linear Regression) predicts **continuous numeric targets** (regression). Applying it directly to categorical targets fails because predictions are unbounded $(-\infty, +\infty)$ and distant well-classified instances pull the decision boundary ([`05errorBasedLearningTopic5.pdf`](05errorBasedLearningTopic5.pdf), Slide 55).
   - **Logistic Regression** predicts **categorical targets**: binary $\{0, 1\}$ natively via the logistic sigmoid $\operatorname{logistic}(\mathbf{w} \cdot \mathbf{d}) \in (0, 1)$, and multiclass ($r > 2$ levels) via normalised **One-vs-All (OvA)** models trained in parallel ([`05errorBasedLearningTopic5.pdf`](05errorBasedLearningTopic5.pdf), Slides 95–100).
   - **Support Vector Machines (SVMs)** (as taught in Ch. 7 Sec. 7.4.7) predict **categorical targets**: binary $\{-1, +1\}$ natively and multiclass via One-vs-All.
3. **How do these models handle the 7 data quality issues?**  
   *(See the detailed tables in Sections 3.1, 3.2, and 3.3 above; key contrasts summarised below:)*
   - **Missing values:** None of the three models handle missing values natively; rows with missing targets are dropped and missing descriptive values must be imputed (fitted on the training split).
   - **Outliers:** **Linear regression** is highly sensitive because $L_2$ squares residuals. **Logistic regression** is much less sensitive to outliers on the *correct* side of the boundary because $\operatorname{logistic}(x)$ saturates at $0$ and $1$ ([`05errorBasedLearningTopic5.pdf`](05errorBasedLearningTopic5.pdf), Slide 55). **Hard-margin SVM** is unaffected by outliers far on the correct side ($\alpha[i]=0$) but breaks completely if an outlier crosses the margin; **Soft-margin SVM** tolerates margin violations to preserve a wide margin ([`fundamentals_of_ml_2ed_ch.pdf`](fundamentals_of_ml_2ed_ch.pdf), PDF p. 54 / printed p. 364).
   - **Noise:** Target noise inflates residual variance $se$ in linear regression; boundary label noise shifts logistic boundaries and requires **soft-margin** relaxation in SVMs; high-degree basis functions or very narrow RBF kernels ($\sigma \to 0$) overfit noise.
   - **Skew class distributions:** Not applicable to linear regression (though skewed continuous targets pull $L_2$ fits). In logistic regression and SVMs, severe class imbalance pulls the decision boundary toward the minority class unless mitigated by resampling, threshold tuning, or class weighting.
   - **Irrelevant features:** In linear regression, a **two-tailed $t$-test** ($H_0: w_j = 0$) identifies statistically insignificant features for removal ([`05errorBasedLearningTopic5.pdf`](05errorBasedLearningTopic5.pdf), Slides 42–44). In RBF-kernel SVMs and polynomial basis expansions, many irrelevant features corrupt pairwise distances or explode parameter counts, requiring prior feature selection.
   - **Features with different value ranges:** In **linear regression**, unscaled features do not change the theoretical optimum $\mathbf{w}^*$, but they make the $L_2$ error surface an elongated valley where gradient descent oscillates unless $\alpha$ is tiny ([`quiz5.md`](quiz5.md), Q16), and they prevent comparing $|w_j|$ directly ([`quiz5.md`](quiz5.md), Q4). In **logistic regression**, large unscaled features saturate the sigmoid ($\mathbb{M}(1-\mathbb{M}) \approx 0$). In **SVMs**, unscaled features directly distort the geometric margin $\frac{2}{\|\mathbf{w}\|}$ and kernel functions, making scaling mandatory.
   - **Different descriptive feature data types:** Handled across all three models by one-hot / dummy encoding categorical features into binary numeric indicators.

#### Part B: Neural Networks (Single Perceptron & Shallow FFNN)
- All seven data-quality questions from [`README.md`](README.md) (*missing values, outliers, noise, skew class distributions, irrelevant features, features with different value ranges, and different descriptive feature data types*) are answered in full in the tables in **Section 3.4** and **Section 3.5** above.

---

### 4.2 Quiz-Tested Distinctions, Common Misconceptions, and Course Nuances

1. **Misconception: "The descriptive feature with the largest weight $|w_j|$ is the most predictive feature in a linear regression model."** ([`quiz5.md`](quiz5.md), Q4; [`05errorBasedLearningTopic5.pdf`](05errorBasedLearningTopic5.pdf), Slide 41; [`fundamentals_of_ml_2ed_ch.pdf`](fundamentals_of_ml_2ed_ch.pdf), p. 333)  
   - **Why it is false:** Weight magnitudes depend directly on the measurement units/scale of the features (e.g., measuring area in $\text{m}^2$ vs. $\text{cm}^2$ changes $w_j$ by a factor of $10{,}000$). Furthermore, even if features are normalised, sample variance affects certainty. The course uses a **two-tailed $t$-test** ($t_{\text{val}} = |w_j / se(d_j)|$) to test whether $w_j$ is statistically significantly different from zero.

2. **Course Formula vs. Standard Statistical Theory: Degrees of Freedom in the Multivariable $t$-Test** ([`05errorBasedLearningTopic5.pdf`](05errorBasedLearningTopic5.pdf), Slides 43–44; [`fundamentals_of_ml_2ed_ch.pdf`](fundamentals_of_ml_2ed_ch.pdf), Eq. 7.18, PDF p. 24 / printed p. 334)  
   - **Recorded course formula:** Both the slides and textbook define overall standard error as $se = \sqrt{\frac{\sum_{i=1}^n (t_i - \mathbb{M}_{\mathbf{w}}(\mathbf{d}_i))^2}{n - 2}}$ and state that $t_{\text{val}}$ is compared against a $t$-distribution with **$n - 2$ degrees of freedom**, with $se(d_j) = \frac{se}{\sqrt{\sum_{i=1}^n (d_{j,i} - \bar{d}_j)^2}}$.
   - **Technically qualified explanation:** $df = n - 2$ is exact for **simple (univariate) linear regression** ($m = 1$, which estimates $2$ parameters: $w_0$ and $w_1$). In general **multivariable linear regression** with $m$ descriptive features ($m + 1$ parameters including $w_0$), the residual degrees of freedom are **$df = n - m - 1$** (or $n - p$ where $p = m + 1$), and $se(w_j)$ also scales by $\sqrt{\text{VIF}_j}$ when predictors are correlated. **For course assessments, remember the course formula uses $n - 2$**, while noting $n - m - 1$ is the general multiple-regression adjustment.

3. **Course Formulation vs. Standard ML Practice: Logistic Regression Loss Function** ([`05errorBasedLearningTopic5.pdf`](05errorBasedLearningTopic5.pdf), Slides 65–68; [`fundamentals_of_ml_2ed_ch.pdf`](fundamentals_of_ml_2ed_ch.pdf), Eqs. 7.31–7.33, PDF pp. 36–37 / printed pp. 346–347)  
   - **Recorded course formulation:** Derives logistic regression by minimising the **Sum of Squared Errors ($L_2$)** on the logistic output, producing the gradient term $(t_i - \mathbb{M}_{\mathbf{w}}(\mathbf{d}_i))\mathbb{M}_{\mathbf{w}}(\mathbf{d}_i)(1 - \mathbb{M}_{\mathbf{w}}(\mathbf{d}_i))d_{j,i}$. Similarly, multinomial logistic regression (Sec. 7.4.6, Slides 96–100) normalises $r$ independent sigmoids via $\mathbb{M}'_{\mathbf{w}_k}(\mathbf{d}) = \frac{\mathbb{M}_{\mathbf{w}_k}(\mathbf{d})}{\sum_l \mathbb{M}_{\mathbf{w}_l}(\mathbf{d})}$ and applies the $L_2$ update rule.
   - **Technically qualified explanation:** Composing $L_2$ loss with the logistic sigmoid creates a **non-convex** error surface with flat saturation plateaus: if an instance is strongly misclassified ($t_i = 1$ but $\mathbb{M}_{\mathbf{w}}(\mathbf{d}_i) \approx 0$), the factor $\mathbb{M}_{\mathbf{w}}(\mathbf{d}_i)(1 - \mathbb{M}_{\mathbf{w}}(\mathbf{d}_i)) \approx 0$ causes the gradient to vanish even though the error is maximum! In standard statistical and library implementations (including `sklearn.linear_model.LogisticRegression` shown on Slide 107), binary logistic regression minimises **binary cross-entropy (negative log-likelihood)**—which is strictly **convex** and cancels the $\mathbb{M}(1-\mathbb{M})$ saturation factor so the gradient is simply $-\sum_{i=1}^n (t_i - \mathbb{M}_{\mathbf{w}}(\mathbf{d}_i))d_{j,i}$—and multinomial logistic regression uses the **softmax** function with multiclass cross-entropy. **Use the $L_2$ formula with $\mathbb{M}(1-\mathbb{M})$ when answering course derivations**, as it directly previews the sigmoid backpropagation delta rule used in neural networks.

4. **Misconception: "Adding polynomial basis functions $\phi_k(\mathbf{d})$ makes linear regression non-linear in its weights and destroys convexity."** ([`quiz5.md`](quiz5.md), Q7)  
   - **Why it is false:** Even when $\phi_k(\mathbf{d})$ is non-linear in the descriptive features $\mathbf{d}$ (e.g., $d_1^2, d_1 d_2$), the model $\mathbb{M}_{\mathbf{w}}(\mathbf{d}) = \sum_{k=0}^b w_k \phi_k(\mathbf{d})$ remains **linear in the parameters $\mathbf{w}$**. Therefore, the $L_2$ error surface with respect to $\mathbf{w}$ is still a convex bowl with a single global minimum.

5. **Misconception: "Stacking multiple hidden layers of linear units allows a neural network to learn non-linear relationships."** ([`quiz5.md`](quiz5.md), Q10, Q13)  
   - **Why it is false:** A linear combination of linear combinations is still a single linear combination ($\mathbf{W}^{(2)}(\mathbf{W}^{(1)}\mathbf{d}) = (\mathbf{W}^{(2)}\mathbf{W}^{(1)})\mathbf{d}$). Hidden units **must** use **non-linear** (and differentiable, for gradient-based backpropagation, [`quiz5.md`](quiz5.md), Q12) activation functions to learn non-linear boundaries.

6. **Scaling in Linear Regression vs. Neural Networks / Logistic Regression / SVMs** ([`quiz5.md`](quiz5.md), Q16)  
   - **Recorded quiz distinction:** In [`quiz5.md`](quiz5.md) Q16 (*"Consider a linear regression model with gradient descent. Is it necessary to scale the descriptive features and targets?"*), the recorded answer is **`False`** because a linear model can theoretically absorb feature and target scales into $w_j$ and $w_0$, unlike neural networks with bounded sigmoid/$\tanh$ activations where unscaled inputs cause **premature saturation** ($f' \approx 0$) and unscaled regression targets outside $(0, 1)$ or $(-1, 1)$ cannot be represented by bounded output units.
   - **Practical qualification:** While not *theoretically required for representability* in linear regression, failing to scale descriptive features with widely different ranges makes gradient descent badly ill-conditioned (requiring tiny learning rates like $\alpha = 2 \times 10^{-8}$ in Textbook p. 330 and many more epochs). And for **SVMs**, scaling descriptive features is mandatory because the margin $\frac{2}{\|\mathbf{w}\|}$ and kernel functions are scale-sensitive.

---

## 5. Sources and unresolved questions

### Local sources used
- [`05errorBasedLearningTopic5.pdf`](05errorBasedLearningTopic5.pdf) — Topic 5 Lecture Slides (*Error-based Learning*, 107 PDF pages present, corresponding to slides `1 / 181` through `107 / 181`): covers error-based learning overview, generic learning algorithm, true vs. empirical error, simple & multivariable linear regression, batch gradient descent, $t$-test interpretation, learning rate decay, categorical encoding, logistic regression, basis functions, multinomial logistic regression, and `sklearn` classes.
- [`fundamentals_of_ml_2ed_ch.pdf`](fundamentals_of_ml_2ed_ch.pdf) — Kelleher, Mac Namee, & D'Arcy, *Fundamentals of Machine Learning for Predictive Data Analytics* (2nd ed., 2020), Chapter 7: *Information-Based Learning* $\to$ **Chapter 7: Error-Based Learning** (PDF pp. 1–61 / printed pp. 311–371): covers Sec. 7.1–7.4.7 including Linear Regression, Gradient Descent, Weight Interpretation ($t$-test), Learning Rate Decay, Categorical Features, Logistic Regression, Basis Functions, Multinomial Logistic Regression, and Support Vector Machines (Sec. 7.4.7).
- [`README.md`](README.md) — Personal revision notes on Chapter 7 (Linear Regression through SVMs) and the two sets of *Personal Summary Questions* (Linear/Logistic/SVM and Neural Networks).
- [`quiz5.md`](quiz5.md) — Topic 5 self-test quiz (16 questions and recorded answer key covering Error-Based Learning, Linear/Logistic Regression, Perceptrons, and Shallow Neural Networks).
- [`notebooks/sum_squared_errors.ipynb`](notebooks/sum_squared_errors.ipynb) — Jupyter notebook visualising simple linear regression lines, residuals, and the convex parabolic 1D error surface ($w_1$ vs. SSE with fixed $w_0$).

### Source gaps and resolved conflicts
1. **Truncated slide deck (`05errorBasedLearningTopic5.pdf` ends at page `107 / 181`):**  
   Slide 2 of [`05errorBasedLearningTopic5.pdf`](05errorBasedLearningTopic5.pdf) lists 12 sections in the Topic 5 outline, including **9. Single Perceptron**, **10. Multilayer (Shallow) Neural Networks**, **11. Training Algorithms**, and **12. Performance Issues** (drawn from Engelbrecht, *Computational Intelligence: An Introduction*, 2nd ed., 2007, Ch. 2, 3, & 7, cited on Slide 3). However, the PDF file in this directory stops at page 107 (Slide `107 / 181`, Section 8). Sections 3.4 and 3.5 above synthesise all concepts tested in [`quiz5.md`](quiz5.md) (Q8–Q16), [`README.md`](README.md), and Slides 55–58, supplemented with the standard mathematical notation from Engelbrecht (2007, Ch. 2–3) for the perceptron and 1-hidden-layer backpropagation equations.
2. **Truncated textbook exercises (`fundamentals_of_ml_2ed_ch.pdf` ends at PDF p. 61 / printed p. 371):**  
   The textbook chapter includes the full text of Sections 7.1–7.6 (printed pp. 311–370) and the first page of Section 7.7 (*Exercises*, printed p. 371); subsequent exercise pages are not in the PDF. No model explanations or equations are affected.
3. **Textbook typographical error in Multinomial Logistic Regression update equation:**  
   In [`fundamentals_of_ml_2ed_ch.pdf`](fundamentals_of_ml_2ed_ch.pdf) Eq. 7.38 (PDF p. 48 / printed p. 358), the normalised model output inside the summation is printed as $\mathbb{M}'_{\mathbf{w}_k}(\mathbf{d}_i[1])$ instead of $\mathbb{M}'_{\mathbf{w}_k}(\mathbf{d}_i)$. [`05errorBasedLearningTopic5.pdf`](05errorBasedLearningTopic5.pdf) Slide 99 gives the corrected equation using $\mathbb{M}'_{\mathbf{w}_k}(\mathbf{d}_i)$.
