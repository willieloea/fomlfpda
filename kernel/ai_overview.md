# Kernel-Based Learning — Course Model Summary

---

## 1. Topic at a Glance

**Kernel-based learning** comprises supervised learning models that construct linear decision boundaries or regression functions—either directly in the original input space or implicitly in a higher-dimensional transformed feature space—using pairwise similarity (inner products) between training instances ([`06kernelBasedLearningTopic6.pdf`, Slide 4](06kernelBasedLearningTopic6.pdf)).

* **Learning setting:** Supervised learning for binary classification, multi-class classification (via decomposition), and continuous regression ([`06kernelBasedLearningTopic6.pdf`, Slides 2, 5, 42, 49](06kernelBasedLearningTopic6.pdf)).
* **Inductive bias:**
  * **Classification (Support Vector Machines, SVM):** Prefers the **maximum-margin separating hyperplane**—the decision boundary equidistant from the closest opposing class instances (**support vectors**) that maximises the perpendicular safety margin $\frac{2}{\|\mathbf{w}\|}$ between classes ([`06kernelBasedLearningTopic6.pdf`, Slides 7–9, 13–14](06kernelBasedLearningTopic6.pdf); [`fundamentals_of_ml_2ed_ch.pdf`, pp. 361–365 (PDF pp. 1–5)](fundamentals_of_ml_2ed_ch.pdf)). Unlike logistic regression, which also fits a linear boundary, an SVM places the boundary solely using the boundary-defining support vectors to maximise generalisation reliability ([`fundamentals_of_ml_2ed_ch.pdf`, p. 362 (PDF p. 2)](fundamentals_of_ml_2ed_ch.pdf)).
  * **Regression (Support Vector Regression, SVR):** Prefers the **flattest function** (minimum weight norm $\frac{1}{2}\|\mathbf{w}\|^2$) that encloses as many training targets as possible inside a tube of tolerated error radius $\epsilon$, penalising only residuals that exceed $\epsilon$ linearly ([`06kernelBasedLearningTopic6.pdf`, Slides 51–53, 66](06kernelBasedLearningTopic6.pdf)).
  * **Non-linear extension:** Assumes that classes or functional relationships that are non-linear in the $N$-dimensional input space $\mathbb{R}^N$ become linearly separable (or linearly approximable) when mapped into a higher-dimensional feature space $\mathbb{R}^D$ ($D > N$) via a basis transformation $\phi(\mathbf{x})$ ([`06kernelBasedLearningTopic6.pdf`, Slides 4, 33](06kernelBasedLearningTopic6.pdf); [`fundamentals_of_ml_2ed_ch.pdf`, pp. 365–366 (PDF pp. 5–6)](fundamentals_of_ml_2ed_ch.pdf)).

### Taxonomy of Concepts Covered

To avoid conflating models with mathematical machinery, distinguish the following roles:

| Category | Specific items covered in this topic | Role |
| --- | --- | --- |
| **Prediction models & variants** | **Hard-Margin Linear SVM**, **Soft-Margin SVM**, **Non-Linear (Kernel) SVM**, **Multi-Class SVM (OvR, OvO)**, **Support Vector Regression (Linear & Kernel SVR)** | Map query feature vectors $\mathbf{x} \in \mathbb{R}^N$ to class labels $\hat{y} \in \{-1, +1\}$, class indices $\{1, \dots, K\}$, or continuous predictions $\hat{y} \in \mathbb{R}$ ([`06kernelBasedLearningTopic6.pdf`, Slides 2, 12, 37, 43–44, 58](06kernelBasedLearningTopic6.pdf)). |
| **Objectives & loss functions** | Maximum-margin objective ($\min \frac{1}{2}\|\mathbf{w}\|^2$), soft-margin slack penalty ($C\sum_i \xi_i$, hinge loss), **$\epsilon$-insensitive loss** ($L_\epsilon(r) = \max\{0, |r| - \epsilon\}$) vs. Ordinary Least Squares (**OLS** $L_2$ loss $\sum_i (y_i - \hat{y}_i)^2$) | Define what constitutes an optimal hyperplane and how margin violations or residuals are penalised ([`06kernelBasedLearningTopic6.pdf`, Slides 15, 26–27, 53, 66](06kernelBasedLearningTopic6.pdf)). |
| **Training algorithms & optimisation machinery** | Constrained **Convex Quadratic Programming (QP)**, **Lagrangian multiplier method** (primal-to-dual reformulation), **Karush–Kuhn–Tucker (KKT) conditions**, **Sequential Minimal Optimization (SMO, Platt 1998)** | Solve for the optimal Lagrange multipliers ($\alpha_i$ or $\alpha_i, \alpha_i^*$), weights $\mathbf{w}$, and bias $b$ without iterative gradient-descent error updates ([`06kernelBasedLearningTopic6.pdf`, Slides 15–23, 60–64](06kernelBasedLearningTopic6.pdf); [`quiz-08-1-kernel-based-learning.md`, Q8, Q9, Q12](quiz-08-1-kernel-based-learning.md)). |
| **Similarity measures & kernels** | Standard dot product ($\mathbf{x}_i \cdot \mathbf{x}_j$), **Linear kernel**, **Polynomial kernel**, **Gaussian Radial Basis Function (RBF) kernel**, **Sigmoid (hyperbolic tangent) kernel** | Compute inner products $\phi(\mathbf{x}_i) \cdot \phi(\mathbf{x}_j)$ directly in input space (**the kernel trick**) without explicitly constructing $\phi(\mathbf{x})$ ([`06kernelBasedLearningTopic6.pdf`, Slides 34, 40, 65](06kernelBasedLearningTopic6.pdf); [`fundamentals_of_ml_2ed_ch.pdf`, p. 366 (PDF p. 6)](fundamentals_of_ml_2ed_ch.pdf)). |
| **Evaluation & theoretical bounds** | Vapnik's expected generalisation error bound via support-vector ratio ($E[P(\text{error})] \le \frac{E[n_{SV}]}{n_T}$) | Bounds expected out-of-sample error using the fraction of training instances that become support vectors ([`quiz-08-1-kernel-based-learning.md`, Q1](quiz-08-1-kernel-based-learning.md)). |

---

## 2. Quick Comparison

| Model | Task / target type | Descriptive feature types and preparation | Main strength | Main limitation |
| --- | --- | --- | --- | --- |
| **Hard-Margin Linear SVM** | Binary classification ($y_i \in \{-1, +1\}$) | Numeric only; requires encoding for categoricals, imputation for missing values, and z-score scaling | Convex global optimum; maximum margin yields strong generalisation on clean, linearly separable data | Fails completely (infeasible optimisation) if classes overlap or contain boundary-crossing noise |
| **Soft-Margin Linear SVM** | Binary classification ($y_i \in \{-1, +1\}$) | Numeric only; requires encoding, imputation, and z-score scaling | Tolerates overlapping classes, label noise, and moderate outliers via slack variables $\xi_i$ and penalty $C$ | Decision boundary remains strictly linear in input space; sensitive to choice of $C$ and class imbalance |
| **Non-Linear (Kernel) SVM** | Binary classification ($y_i \in \{-1, +1\}$) | Numeric only; requires encoding, imputation, and z-score scaling (critical for distance/dot-product kernels) | Captures complex non-linear boundaries via the kernel trick while preserving convex optimisation in dual space | Computationally expensive for large $n_T$; low interpretability; sensitive to kernel choice, kernel parameters ($\gamma, p$), and $C$ |
| **Multi-Class SVM (OvR / OvO)** | Multi-class classification ($K > 2$ classes) | Same as underlying binary SVM (numeric, scaled, complete) | Extends binary SVMs to $K > 2$ classes using either $K$ models (OvR) or $\frac{K(K-1)}{2}$ pairwise models (OvO) | OvR introduces severe class imbalance per subproblem; OvO scales quadratically in the number of classes $K$ |
| **Support Vector Regression (SVR — Linear & Kernel)** | Continuous regression ($y_i \in \mathbb{R}$) | Numeric only; requires encoding, imputation, and z-score scaling | Sparse solution defined by support vectors outside the $\epsilon$-tube; linear $\epsilon$-insensitive penalty is more robust to moderate outliers and noise than OLS | Sensitive to extreme outliers and hyperparameter tuning ($C, \epsilon$, kernel parameters); expensive on large datasets |

---

## 3. Individual Model Summaries

### 3.1 Shared Mathematical Foundations: Duality, KKT Conditions, and Kernels

Because both SVM and SVR rely on the same constrained optimisation and kernel machinery, these core concepts are defined once here and referenced in Sections 3.2 and 3.3.

#### Notation Reconciliation (Slides vs. Textbook)

| Concept | Lecture Slides ([`06kernelBasedLearningTopic6.pdf`](06kernelBasedLearningTopic6.pdf)) | Textbook ([`fundamentals_of_ml_2ed_ch.pdf`, Sec. 7.4.7](fundamentals_of_ml_2ed_ch.pdf)) |
| --- | --- | --- |
| Training / query feature vector | $\mathbf{x}_i, \mathbf{x} \in \mathbb{R}^N$ | $\mathbf{d}_i, \mathbf{q} \in \mathbb{R}^m$ |
| Binary target label | $y_i \in \{-1, +1\}$ | $t_i \in \{-1, +1\}$ |
| Weight vector and bias (intercept) | $\mathbf{w}$ and $b$ | $\mathbf{w}$ and $w_0$ |
| Number of training instances / support vectors | $n_T = |D_T|$ and $n_{SV}$ | $n$ and $s$ |
| Basis mapping and kernel function | $\phi(\mathbf{x})$ (or $\Phi(\mathbf{x})$) and $k(\mathbf{x}_i, \mathbf{x}_j)$ | $\boldsymbol{\phi}(\mathbf{d})$ and $\text{kernel}(\mathbf{d}_i, \mathbf{q})$ |

#### Why Lagrangian Duality and KKT Conditions Are Used

Both SVM and SVR are formulated as **primal** convex quadratic minimisation problems over weight vector $\mathbf{w}$ and bias $b$, subject to linear inequality constraints on the training instances ([`06kernelBasedLearningTopic6.pdf`, Slides 15, 55](06kernelBasedLearningTopic6.pdf)). Rather than solving the primal directly, the problem is converted to its **Lagrangian dual** for two reasons ([`06kernelBasedLearningTopic6.pdf`, Slide 16](06kernelBasedLearningTopic6.pdf)):

1. **Simpler constraints:** The $n_T$ data-dependent inequality constraints are replaced by simple box or non-negativity bounds on the Lagrange multipliers ($\alpha_i \ge 0$ or $0 \le \alpha_i \le C$).
2. **Dot-product representation (enabling the kernel trick):** In the dual objective and prediction rule, training feature vectors $\mathbf{x}_i$ appear **only inside pairwise dot products** $\mathbf{x}_i \cdot \mathbf{x}_j$, making non-linear generalisation seamless.

At the saddle point ($\min_{\mathbf{w}, b} \max_{\boldsymbol{\alpha}} L$) of the convex programme, the **Karush–Kuhn–Tucker (KKT) conditions** hold ([`06kernelBasedLearningTopic6.pdf`, Slides 12, 18, 21, 23, 62–63](06kernelBasedLearningTopic6.pdf); [`quiz-08-1-kernel-based-learning.md`, Q9](quiz-08-1-kernel-based-learning.md)):

* **Stationarity (stability conditions):** Partial derivatives of the Lagrangian $L$ with respect to primal variables vanish ($\frac{\partial L}{\partial \mathbf{w}} = \mathbf{0}, \frac{\partial L}{\partial b} = 0, \frac{\partial L}{\partial \xi_i} = 0$).
* **Primal feasibility (called "2nd KKT" on Slides 12 & 14):** The original margin/tube constraints on each training instance are satisfied.
* **Dual feasibility (called "1st KKT" on Slide 21):** Lagrange multipliers are non-negative ($\alpha_i \ge 0, \mu_i \ge 0$).
* **Complementary slackness (called "3rd KKT" on Slide 23):** The product of each Lagrange multiplier and its corresponding inequality constraint is zero (e.g., $\alpha_i [y_i(\mathbf{w}\cdot\mathbf{x}_i + b) - 1] = 0$ in hard-margin SVM). Consequently, $\alpha_i > 0$ is possible **only** when the constraint is active ($y_i(\mathbf{w}\cdot\mathbf{x}_i + b) = 1$). Instances with $\alpha_i > 0$ are the **support vectors**; all other training instances have $\alpha_i = 0$ and play no role in the learned model ([`06kernelBasedLearningTopic6.pdf`, Slides 22–23](06kernelBasedLearningTopic6.pdf)).

#### The Kernel Trick and Standard Kernel Functions

To model non-linear relationships, an input vector $\mathbf{x} \in \mathbb{R}^N$ is mapped into a higher-dimensional feature space $\mathbb{R}^D$ ($D > N$) via a non-linear basis transformation $\phi(\mathbf{x}): \mathbb{R}^N \to \mathbb{R}^D$ ([`06kernelBasedLearningTopic6.pdf`, Slide 33](06kernelBasedLearningTopic6.pdf)). Explicitly computing $\phi(\mathbf{x})$ and taking dot products $\phi(\mathbf{x}_i) \cdot \phi(\mathbf{x}_j)$ in $\mathbb{R}^D$ can be computationally prohibitive or infinite-dimensional ([`06kernelBasedLearningTopic6.pdf`, Slide 34](06kernelBasedLearningTopic6.pdf); [`fundamentals_of_ml_2ed_ch.pdf`, p. 366 (PDF p. 6)](fundamentals_of_ml_2ed_ch.pdf)).

The **kernel trick** replaces every inner product $\phi(\mathbf{x}_i) \cdot \phi(\mathbf{x}_j)$ with a **kernel function** $k(\mathbf{x}_i, \mathbf{x}_j) = \phi(\mathbf{x}_i) \cdot \phi(\mathbf{x}_j)$ evaluated directly on the $N$-dimensional inputs $\mathbf{x}_i, \mathbf{x}_j$, so $\phi(\mathbf{x})$ never needs to be computed explicitly ([`06kernelBasedLearningTopic6.pdf`, Slide 34](06kernelBasedLearningTopic6.pdf)).

* **Worked Example of the Kernel Trick ([`06kernelBasedLearningTopic6.pdf`, Slide 36](06kernelBasedLearningTopic6.pdf)):**
  Suppose $\mathbf{x} = (x_1, x_2)^T \in \mathbb{R}^2$ is mapped to $\mathbb{R}^3$ via $\Phi(\mathbf{x}) = (x_1^2, \sqrt{2}x_1 x_2, x_2^2)^T$. The dot product in $\mathbb{R}^3$ is:
  $$\Phi(\mathbf{x}_i) \cdot \Phi(\mathbf{x}_j) = x_{i1}^2 x_{j1}^2 + 2 x_{i1} x_{i2} x_{j1} x_{j2} + x_{i2}^2 x_{j2}^2 = (x_{i1} x_{j1} + x_{i2} x_{j2})^2 = (\mathbf{x}_i \cdot \mathbf{x}_j)^2 = k(\mathbf{x}_i, \mathbf{x}_j)$$
  Thus, squaring the 2D dot product computes the exact 3D feature-space inner product without constructing the 3D vectors.

**Common Kernel Functions Covered ([`06kernelBasedLearningTopic6.pdf`, Slides 38, 40, 65](06kernelBasedLearningTopic6.pdf); [`fundamentals_of_ml_2ed_ch.pdf`, p. 366 (PDF p. 6)](fundamentals_of_ml_2ed_ch.pdf)):**

| Kernel | Formula | Hyperparameters | Notes |
| --- | --- | --- | --- |
| **Linear** | $k(\mathbf{x}, \mathbf{x}_i) = \mathbf{x} \cdot \mathbf{x}_i + c$ | Optional constant $c$ (often $c=0$) | Equivalent to no basis mapping ($\phi(\mathbf{x}) = \mathbf{x}$ when $c=0$). |
| **Polynomial** | $k(\mathbf{x}, \mathbf{x}_i) = (\mathbf{x} \cdot \mathbf{x}_i + 1)^p$ | Degree $p \in \mathbb{N}$ | Maps inputs to polynomial combinations of features up to degree $p$. |
| **Gaussian Radial Basis Function (RBF)** | $k(\mathbf{x}, \mathbf{x}_i) = \exp\left(-\frac{\|\mathbf{x} - \mathbf{x}_i\|^2}{2\sigma^2}\right) = \exp\left(-\gamma \|\mathbf{x} - \mathbf{x}_i\|^2\right)$ | Bandwidth $\sigma > 0$ or scale $\gamma = \frac{1}{2\sigma^2} > 0$ | Local similarity measure based on squared Euclidean distance; $k(\mathbf{x}, \mathbf{x}_i) \in (0, 1]$, approaching $1$ when $\mathbf{x} \approx \mathbf{x}_i$ and $0$ when $\|\mathbf{x} - \mathbf{x}_i\|$ is large. |
| **Sigmoid (Hyperbolic Tangent)** | $k(\mathbf{x}, \mathbf{x}_i) = \tanh(\kappa \mathbf{x} \cdot \mathbf{x}_i - \delta)$ | Scale $\kappa$, offset $\delta$ | Listed for classification on Slide 40; behaves similarly to a two-layer perceptron kernel. |

---

### 3.2 Support Vector Machines (SVM) for Classification

#### Main Idea and Assumptions

Developed by Cortes and Vapnik, the **Support Vector Machine (SVM)** is a binary classifier that separates two classes ($y_i \in \{-1, +1\}$) using a hyperplane $\mathbf{w} \cdot \mathbf{x} + b = 0$ ([`06kernelBasedLearningTopic6.pdf`, Slides 5, 12](06kernelBasedLearningTopic6.pdf)). While infinitely many hyperplanes can separate two linearly separable classes, an SVM selects the unique hyperplane that **maximises the margin**—the perpendicular distance $d_+ + d_- = \frac{2}{\|\mathbf{w}\|}$ between the closest positive instance(s) and closest negative instance(s) ([`06kernelBasedLearningTopic6.pdf`, Slides 8–9, 13–14](06kernelBasedLearningTopic6.pdf)).

* **Core intuition:** A wider margin provides a larger geometric buffer against slight perturbations in new query instances, yielding lower generalisation error than a boundary passing close to training points ([`fundamentals_of_ml_2ed_ch.pdf`, pp. 361–362 (PDF pp. 1–2)](fundamentals_of_ml_2ed_ch.pdf)).
* **Sparsity via Support Vectors:** The decision boundary is completely determined by the **support vectors**—the training instances that lie on the margin boundaries (or inside/across the margin in soft-margin SVMs). Moving or removing any non-support vector (where $\alpha_i = 0$) has zero effect on the hyperplane ([`06kernelBasedLearningTopic6.pdf`, Slides 9–10, 22](06kernelBasedLearningTopic6.pdf)).
* **Assumptions:**
  * *To fit/predict (Hard-Margin Linear SVM):* The training classes must be **strictly linearly separable** in input space; otherwise, the constrained optimisation problem has no feasible solution ([`06kernelBasedLearningTopic6.pdf`, Slide 25](06kernelBasedLearningTopic6.pdf); [`fundamentals_of_ml_2ed_ch.pdf`, p. 366 (PDF p. 6)](fundamentals_of_ml_2ed_ch.pdf)).
  * *To fit/predict (Soft-Margin & Kernel SVM):* Does **not** require linear separability in input space. Soft-margin SVM allows margin violations via slack variables $\xi_i \ge 0$, and Kernel SVM assumes the classes become approximately linearly separable after non-linear mapping $\phi(\mathbf{x})$ ([`06kernelBasedLearningTopic6.pdf`, Slides 5, 26, 33](06kernelBasedLearningTopic6.pdf)).
  * *For theoretical generalisation guarantees:* Training and test instances are assumed to be drawn independently and identically distributed (i.i.d.) from the same underlying probability distribution (required for Vapnik's structural risk / leave-one-out support-vector bound; see Section 4). Unlike parametric statistical tests, SVMs make no Gaussian distributional assumptions on descriptive features or residuals.

#### How It Learns and Predicts

##### 1. Target Formatting and Canonical Hyperplanes

1. Encode the binary target labels as $y_i \in \{-1, +1\}$ ([`06kernelBasedLearningTopic6.pdf`, Slide 12](06kernelBasedLearningTopic6.pdf); [`quiz-08-1-kernel-based-learning.md`, Q11](quiz-08-1-kernel-based-learning.md)).
2. Rescale $\mathbf{w}$ and $b$ so that the **positive margin** hyperplane is $\mathbf{w} \cdot \mathbf{x} + b = +1$ and the **negative margin** hyperplane is $\mathbf{w} \cdot \mathbf{x} + b = -1$, with the decision boundary halfway between them at $\mathbf{w} \cdot \mathbf{x} + b = 0$ ([`06kernelBasedLearningTopic6.pdf`, Slides 12–14](06kernelBasedLearningTopic6.pdf)).
3. Because the perpendicular distance from any point $\mathbf{x}$ to $\mathbf{w} \cdot \mathbf{x} + b = 0$ is $d = \frac{|\mathbf{w}\cdot\mathbf{x} + b|}{\|\mathbf{w}\|}$, support vectors on the positive and negative margins lie at distances $d_+ = \frac{1}{\|\mathbf{w}\|}$ and $d_- = \frac{1}{\|\mathbf{w}\|}$, giving a total margin width of ([`06kernelBasedLearningTopic6.pdf`, Slide 14](06kernelBasedLearningTopic6.pdf); [`fundamentals_of_ml_2ed_ch.pdf`, pp. 364–365 (PDF pp. 4–5)](fundamentals_of_ml_2ed_ch.pdf)):
   $$d_+ + d_- = \frac{2}{\|\mathbf{w}\|}$$
   Maximising $\frac{2}{\|\mathbf{w}\|}$ is equivalent to minimising $\frac{1}{2}\|\mathbf{w}\|^2$ (where the $\frac{1}{2}$ factor simplifies differentiation: $\frac{\partial}{\partial \mathbf{w}}\left(\frac{1}{2}\|\mathbf{w}\|^2\right) = \mathbf{w}$) ([`06kernelBasedLearningTopic6.pdf`, Slide 15](06kernelBasedLearningTopic6.pdf)).

##### 2. Training Formulations (Primal and Dual)

* **Standard Hard-Margin Linear SVM ([`06kernelBasedLearningTopic6.pdf`, Slides 15–21](06kernelBasedLearningTopic6.pdf)):**
  * *Primal problem:*
    $$\min_{\mathbf{w}, b} \frac{1}{2}\|\mathbf{w}\|^2 \quad \text{subject to} \quad y_i(\mathbf{w} \cdot \mathbf{x}_i + b) - 1 \ge 0, \quad \forall i = 1, \dots, n_T$$
  * *Lagrangian and KKT stationarity:* Forming $L(\mathbf{w}, b, \boldsymbol{\alpha}) = \frac{1}{2}\|\mathbf{w}\|^2 - \sum_{i=1}^{n_T} \alpha_i [y_i(\mathbf{w} \cdot \mathbf{x}_i + b) - 1]$ with $\alpha_i \ge 0$ and setting $\frac{\partial L}{\partial \mathbf{w}} = \mathbf{0}$ and $\frac{\partial L}{\partial b} = 0$ yields:
    $$\mathbf{w} = \sum_{i=1}^{n_T} \alpha_i y_i \mathbf{x}_i \quad \text{and} \quad \sum_{i=1}^{n_T} \alpha_i y_i = 0$$
  * *Dual QP problem:* Substituting these back into $L$ gives:
    $$\max_{\boldsymbol{\alpha}} W(\boldsymbol{\alpha}) = \sum_{i=1}^{n_T} \alpha_i - \frac{1}{2}\sum_{i=1}^{n_T}\sum_{j=1}^{n_T} \alpha_i \alpha_j y_i y_j (\mathbf{x}_i \cdot \mathbf{x}_j) \quad \text{subject to} \quad \alpha_i \ge 0 \;\; (\forall i), \quad \sum_{i=1}^{n_T} \alpha_i y_i = 0$$

* **Soft-Margin and Non-Linear (Kernel) SVM ([`06kernelBasedLearningTopic6.pdf`, Slides 26–32, 37–41](06kernelBasedLearningTopic6.pdf)):**
  * *Primal problem (with slack variables $\xi_i \ge 0$ and regularisation hyperparameter $C > 0$):*
    $$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_{i=1}^{n_T} \xi_i \quad \text{subject to} \quad y_i(\mathbf{w} \cdot \phi(\mathbf{x}_i) + b) \ge 1 - \xi_i, \quad \xi_i \ge 0, \quad \forall i = 1, \dots, n_T$$
    Here, $\xi_i = 0$ if $\mathbf{x}_i$ is on or outside the correct margin; $0 < \xi_i < 1$ if $\mathbf{x}_i$ lies inside the margin on the correct side of the boundary; and $\xi_i > 1$ if $\mathbf{x}_i$ is misclassified (so $\sum_i \xi_i$ bounds the number of training errors) ([`06kernelBasedLearningTopic6.pdf`, Slide 26](06kernelBasedLearningTopic6.pdf)).
  * *Stationarity wrt $\xi_i$:* Introducing multipliers $\mu_i \ge 0$ for $\xi_i \ge 0$ gives $\frac{\partial L}{\partial \xi_i} = C - \alpha_i - \mu_i = 0 \implies \alpha_i = C - \mu_i \le C$ ([`06kernelBasedLearningTopic6.pdf`, Slides 31–32](06kernelBasedLearningTopic6.pdf)).
  * *General Dual QP Problem (solved e.g. via Platt's SMO algorithm):*
    $$\max_{\boldsymbol{\alpha}} W(\boldsymbol{\alpha}) = \sum_{i=1}^{n_T} \alpha_i - \frac{1}{2}\sum_{i=1}^{n_T}\sum_{j=1}^{n_T} \alpha_i \alpha_j y_i y_j k(\mathbf{x}_i, \mathbf{x}_j) \quad \text{subject to} \quad 0 \le \alpha_i \le C \;\; (\forall i), \quad \sum_{i=1}^{n_T} \alpha_i y_i = 0$$

##### 3. Computing the Bias $b$ and Making Predictions

1. **Identify support vectors:** Training instances with $\alpha_i > 0$ ([`06kernelBasedLearningTopic6.pdf`, Slide 22](06kernelBasedLearningTopic6.pdf)). In soft-margin SVM, instances with $0 < \alpha_i < C$ have $\mu_i > 0 \implies \xi_i = 0$, meaning they lie **exactly on the margin** ($y_i f(\mathbf{x}_i) = 1$) ([`06kernelBasedLearningTopic6.pdf`, Slides 38–39](06kernelBasedLearningTopic6.pdf)).
2. **Calculate bias $b$:** Using the complementary slackness KKT condition on each free support vector $i \in S = \{i : 0 < \alpha_i < C\}$, solve $y_i\left(\sum_{j=1}^{n_T} y_j \alpha_j k(\mathbf{x}_j, \mathbf{x}_i) + b_i\right) = 1$. Since $y_i \in \{-1, +1\} \implies y_i^{-1} = y_i$, this gives $b_i = y_i - \sum_{j=1}^{n_T} y_j \alpha_j k(\mathbf{x}_j, \mathbf{x}_i)$. For numerical stability, average $b_i$ over all $n_{SV}$ margin support vectors ([`06kernelBasedLearningTopic6.pdf`, Slides 23, 38–39](06kernelBasedLearningTopic6.pdf)):
   $$b = \frac{1}{n_{SV}} \sum_{i \in S} \left( y_i - \sum_{j=1}^{n_T} y_j \alpha_j k(\mathbf{x}_j, \mathbf{x}_i) \right)$$
   *(Note: Slide 23 writes $b_{i_o} = \frac{1}{y_{i_o}} - y_{i_o}\mathbf{w}\cdot\mathbf{x}_{i_o}$, which contains a typographical extra $y_{i_o}$ on the second term; Slide 38–39 gives the algebraically exact expression above. See Section 4.)*
3. **Predict for a new query $\mathbf{x}$:** Evaluate the decision function using only the support vectors ($\alpha_i > 0$) without ever needing to compute $\mathbf{w}$ explicitly ([`06kernelBasedLearningTopic6.pdf`, Slides 24, 37](06kernelBasedLearningTopic6.pdf)):
   $$\hat{y} = \text{sign}\left( \sum_{i \in SV} y_i \alpha_i k(\mathbf{x}, \mathbf{x}_i) + b \right)$$

##### 4. Worked Example (Linear SVM Prediction)

Adapted from the generators dataset in [`fundamentals_of_ml_2ed_ch.pdf`, p. 365 (PDF p. 5)](fundamentals_of_ml_2ed_ch.pdf), where `good` $= +1$ and `faulty` $= -1$, with three support vectors:
* $\mathbf{d}_1 = (-0.225, 0.217), \; t_1 = +1, \; \alpha_1 = 23.056$
* $\mathbf{d}_2 = (-0.066, -0.069), \; t_2 = -1, \; \alpha_2 = 6.998$
* $\mathbf{d}_3 = (-0.273, -0.080), \; t_3 = -1, \; \alpha_3 = 16.058$
* Bias weight $w_0 = -0.1838$, query instance $\mathbf{q}_1 = (-0.314, -0.251)$.

Compute the dot products $\mathbf{d}_i \cdot \mathbf{q}_1$:
* $\mathbf{d}_1 \cdot \mathbf{q}_1 = (-0.225)(-0.314) + (0.217)(-0.251) = 0.07065 - 0.054467 = 0.016183$
* $\mathbf{d}_2 \cdot \mathbf{q}_1 = (-0.066)(-0.314) + (-0.069)(-0.251) = 0.020724 + 0.017319 = 0.038043$
* $\mathbf{d}_3 \cdot \mathbf{q}_1 = (-0.273)(-0.314) + (-0.080)(-0.251) = 0.085722 + 0.020080 = 0.105802$

Weighted sum over support vectors:
$$\sum_{i=1}^3 t_i \alpha_i (\mathbf{d}_i \cdot \mathbf{q}_1) = (+1)(23.056)(0.016183) + (-1)(6.998)(0.038043) + (-1)(16.058)(0.105802) \approx 0.3731 - 0.2662 - 1.6990 = -1.5921$$

* **Standard SVM formula ([`06kernelBasedLearningTopic6.pdf`, Slide 24](06kernelBasedLearningTopic6.pdf)):** Adding $w_0 = -0.1838$ once outside the sum gives $-1.5921 - 0.1838 = -1.7759$.
* **Textbook Eq. (7.41) calculation ([`fundamentals_of_ml_2ed_ch.pdf`, p. 365 (PDF p. 5)](fundamentals_of_ml_2ed_ch.pdf)):** Because Eq. (7.41) places $+ w_0$ inside the summation $\sum_{i=1}^s (t_i \alpha_i (\mathbf{d}_i \cdot \mathbf{q}) + w_0)$, the textbook adds $w_0 = -0.1838$ three times (once per support vector): $-1.5921 + 3(-0.1838) = -2.1435 \approx -2.145$.
* In both cases, the score is negative ($< -1$), so $\hat{y} = -1$ (`faulty`). *(See Section 4 for the formula conflict note.)*

#### Supported Data and Tasks

* **Descriptive features:**
  * **Numeric (continuous/discrete):** Supported natively via dot products $\mathbf{x}_i \cdot \mathbf{x}_j$ and Euclidean distances $\|\mathbf{x}_i - \mathbf{x}_j\|$ ([`06kernelBasedLearningTopic6.pdf`, Slide 45](06kernelBasedLearningTopic6.pdf)).
  * **Categorical (nominal/ordinal) and mixed data:** **Not supported natively**—all features must be numerical before fitting ([`06kernelBasedLearningTopic6.pdf`, Slide 45](06kernelBasedLearningTopic6.pdf); [`fundamentals_of_ml_2ed_ch.pdf`, p. 367 (PDF p. 7)](fundamentals_of_ml_2ed_ch.pdf)). Nominal features require one-hot / dummy encoding (as arbitrary integer codes impose a false linear ordering and distort dot products/distances); ordinal features may use order-preserving numeric encoding if spacing is meaningful, followed by scaling.
* **Target features:**
  * **Binary categorical:** Standard SVM task; target must be formatted as $y_i \in \{-1, +1\}$ ([`06kernelBasedLearningTopic6.pdf`, Slides 5, 12](06kernelBasedLearningTopic6.pdf); [`quiz-08-1-kernel-based-learning.md`, Q11](quiz-08-1-kernel-based-learning.md)).
  * **Multi-class categorical ($K > 2$):** Requires decomposition extensions—**One-vs-Rest (OvR)** or **One-vs-One (OvO)** ([`06kernelBasedLearningTopic6.pdf`, Slides 42–44](06kernelBasedLearningTopic6.pdf); [`fundamentals_of_ml_2ed_ch.pdf`, p. 367 (PDF p. 7)](fundamentals_of_ml_2ed_ch.pdf)).
  * **Continuous numeric:** Requires the **Support Vector Regression (SVR)** variant (Section 3.3).
* **Outputs:**
  * **Class labels:** $\hat{y} = \text{sign}(f(\mathbf{x})) \in \{-1, +1\}$ (or class $k \in \{1, \dots, K\}$ in OvR/OvO).
  * **Real-valued decision scores:** $f(\mathbf{x}) = \sum_{i \in SV} y_i \alpha_i k(\mathbf{x}, \mathbf{x}_i) + b \in \mathbb{R}$, proportional to the signed perpendicular distance from the hyperplane in feature space. Used directly in OvR ($\hat{y} = \arg\max_k f_k(\mathbf{x})$) ([`06kernelBasedLearningTopic6.pdf`, Slide 43](06kernelBasedLearningTopic6.pdf)).
  * **Probabilities:** **Not produced natively.** Decision scores $f(\mathbf{x})$ are unbounded signed margin distances, not calibrated probabilities (unlike logistic regression).

#### Data Quality and Preprocessing (SVM)

| Issue | Effect on the standard model and why | Relevant extension and how it helps | Preprocessing / mitigation and tradeoffs |
| --- | --- | --- | --- |
| **1. Missing values** | **Cannot be handled natively.** Both training and prediction require computing pairwise inner products $\mathbf{x}_i \cdot \mathbf{x}_j$ and Euclidean distances $\|\mathbf{x}_i - \mathbf{x}_j\|^2$ inside kernel functions; any `NaN` coordinate makes the dot product or distance undefined ([`06kernelBasedLearningTopic6.pdf`, Slide 45](06kernelBasedLearningTopic6.pdf)). Missing targets prevent constraint formation $y_i(\mathbf{w}\cdot\mathbf{x}_i+b) \ge 1$. | None covered in the course materials. | Drop instances/features with missing values or impute missing descriptive features (e.g., mean/median/k-NN imputation fitted strictly on the training split to prevent data leakage). Drop unlabelled rows (in supervised SVM). Imputation can distort inter-point distances and margin geometry if many values are missing. |
| **2. Outliers** | **Depends on location:** Outliers far on the **correct** side of the margin satisfy $y_i(\mathbf{w}\cdot\mathbf{x}_i + b) > 1$, have $\alpha_i = 0$, and do **not** become support vectors, so they have **zero effect** on the boundary ([`06kernelBasedLearningTopic6.pdf`, Slide 47](06kernelBasedLearningTopic6.pdf)). Outliers that **violate** the margin (or lie on the wrong side) make **hard-margin SVM** infeasible or severely skew its boundary. *(Extreme target values are N/A for binary labels $\{-1, +1\}$.)* | **Soft-margin SVM** introduces slack variables $\xi_i \ge 0$ and penalty $C$. Rather than shifting the hyperplane drastically to accommodate a margin-violating outlier, the optimiser can assign it a large $\xi_i$ and cap its multiplier at $\alpha_i = C$ ([`06kernelBasedLearningTopic6.pdf`, Slide 47](06kernelBasedLearningTopic6.pdf)). | Choose a **smaller $C$** so margin violations are penalised lightly (wider margin, more robust to outliers) and avoid overly large $\gamma$ in Gaussian RBF kernels (large $\gamma$ creates tight local pockets around individual outliers) ([`06kernelBasedLearningTopic6.pdf`, Slides 29, 47](06kernelBasedLearningTopic6.pdf)). *Tradeoff:* Too small a $C$ increases bias and underfits genuine boundary structure ([`06kernelBasedLearningTopic6.pdf`, Slide 28](06kernelBasedLearningTopic6.pdf)). |
| **3. Noise** | **Hard-margin SVM is not robust to noise** ([`06kernelBasedLearningTopic6.pdf`, Slide 46](06kernelBasedLearningTopic6.pdf)):<br>• *Label noise (incorrect $y_i$):* A single mislabelled point inside the opposite class makes linear separation impossible or forces a tiny margin.<br>• *Descriptive feature noise:* Perturbations to non-support vectors have no effect, but perturbations to points near the boundary change the support vectors and shift the margin ([`06kernelBasedLearningTopic6.pdf`, Slide 46](06kernelBasedLearningTopic6.pdf)). | **Soft-margin SVM** is robust to moderate label and feature noise because slack variables $\xi_i$ allow mislabelled or perturbed boundary instances to violate margin constraints without dominating the hyperplane, provided $C$ is not too large ([`06kernelBasedLearningTopic6.pdf`, Slide 46](06kernelBasedLearningTopic6.pdf); [`quiz-08-1-kernel-based-learning.md`, Q6](quiz-08-1-kernel-based-learning.md)). | Tune $C$ and kernel width ($\gamma$ or $\sigma$) via cross-validation: moderate/small $C$ and smooth kernels prevent fitting noisy fluctuations. Filtering severe label noise on the training split can help, though aggressive filtering risks removing genuine boundary support vectors. |
| **4. Skew class distributions** | **Not robust to imbalanced classes.** In soft-margin SVM, both classes share the same error weight $C$. Because majority-class instances outnumber minority instances, reducing majority-class slack errors decreases the objective $C\sum_i \xi_i$ more than correctly classifying the few minority instances, pushing the decision boundary toward the minority class and favouring majority predictions ([`06kernelBasedLearningTopic6.pdf`, Slide 48](06kernelBasedLearningTopic6.pdf)). Also arises artificially in **OvR** multi-class SVM ([`06kernelBasedLearningTopic6.pdf`, Slide 43](06kernelBasedLearningTopic6.pdf)). | **Class-weighted Soft-Margin SVM:** Assign separate regularisation penalties $C_+$ for the minority class and $C_-$ for the majority class with $C_+ > C_-$ (so $\min \frac{1}{2}\|\mathbf{w}\|^2 + C_+\sum_{y_i=+1}\xi_i + C_-\sum_{y_i=-1}\xi_i$), penalising minority misclassifications more heavily ([`06kernelBasedLearningTopic6.pdf`, Slide 48](06kernelBasedLearningTopic6.pdf)). | Use class-specific penalties $C_+ > C_-$ (e.g. inversely proportional to class frequencies), or resample the training split (oversampling minority / undersampling majority) inside each cross-validation fold. *Tradeoff:* Heavy minority weighting ($C_+ \gg C_-$) can increase false positives if the minority class has noisy boundary instances. |
| **5. Irrelevant features** | **No built-in feature selection ($L_2$ penalty $\frac{1}{2}\|\mathbf{w}\|^2$ shrinks weights smoothly rather than setting them to zero).** SVMs still perform well with a **few** irrelevant features, but performance deteriorates when there are **many** irrelevant features ([`06kernelBasedLearningTopic6.pdf`, Slide 47](06kernelBasedLearningTopic6.pdf)) because every feature contributes equally to dot products $\mathbf{x}_i \cdot \mathbf{x}_j$ and Euclidean distances $\|\mathbf{x}_i - \mathbf{x}_j\|^2$, corrupting kernel similarities with uninformative variation. | None covered in the course materials. | Apply feature selection or dimensionality reduction (fitted on the training split only) prior to training when many irrelevant features are suspected. *Tradeoff:* Aggressive filtering risks discarding features that are only informative via non-linear kernel interactions. |
| **6. Features with different value ranges** | **Highly sensitive to scale differences.** Features with large numeric ranges dominate both inner products $\mathbf{x}_i \cdot \mathbf{x}_j = \sum_{d=1}^N x_{id} x_{jd}$ and Euclidean distances $\|\mathbf{x}_i - \mathbf{x}_j\|^2 = \sum_{d=1}^N (x_{id} - x_{jd})^2$ in kernels, as well as the $L_2$ margin penalty $\frac{1}{2}\|\mathbf{w}\|^2$ ([`06kernelBasedLearningTopic6.pdf`, Slide 45](06kernelBasedLearningTopic6.pdf)). | None (scaling is a preprocessing requirement, not a model variant). | **Standardise features using z-score normalisation** ($z = \frac{x - \mu}{\sigma}$, with $\mu, \sigma$ fitted strictly on the training split). The lectures explicitly recommend **z-score normalisation over min-max scaling** because min-max scaling is sensitive to outliers and squashes normal values into a tiny range ([`06kernelBasedLearningTopic6.pdf`, Slide 45](06kernelBasedLearningTopic6.pdf)). |
| **7. Different descriptive feature data types** | **Requires all features to be numerical** ([`06kernelBasedLearningTopic6.pdf`, Slide 45](06kernelBasedLearningTopic6.pdf)). Categorical features (nominal or ordinal strings/symbols) cannot be multiplied in dot products or subtracted in Euclidean distances. | Continuous target types are handled by **SVR** (Section 3.3); categorical descriptive features require encoding rather than a distinct SVM algorithm ([`fundamentals_of_ml_2ed_ch.pdf`, p. 367 (PDF p. 7)](fundamentals_of_ml_2ed_ch.pdf)). | Convert nominal features via one-hot/dummy encoding and ordinal features via appropriate numeric mapping, then apply z-score scaling ([`fundamentals_of_ml_2ed_ch.pdf`, p. 367 (PDF p. 7)](fundamentals_of_ml_2ed_ch.pdf)). *Tradeoff:* One-hot encoding high-cardinality categoricals increases dimensionality and sparsity. |

* **Additional data considerations:**
  * **High dimensionality ($N$ large):** SVMs generalise well in high-dimensional spaces ([`fundamentals_of_ml_2ed_ch.pdf`, p. 367 (PDF p. 7)](fundamentals_of_ml_2ed_ch.pdf)) because the dual optimisation complexity ($n_T$ variables) depends on the number of training instances $n_T$ rather than the feature dimension $D$, and $\frac{1}{2}\|\mathbf{w}\|^2$ regularises capacity via margin maximisation.
  * **Redundant features and multicollinearity:** Unlike Ordinary Least Squares, which fails when $\mathbf{X}^T\mathbf{X}$ is singular, SVM's strictly convex quadratic penalty $\frac{1}{2}\|\mathbf{w}\|^2$ guarantees a unique weight vector $\mathbf{w}$ even with collinear features, distributing weight across correlated features. However, in distance-based kernels (e.g., Gaussian RBF), duplicated/redundant features over-weight that underlying factor in $\|\mathbf{x}_i - \mathbf{x}_j\|^2$.
  * **Limited training data vs. large datasets:** Because only support vectors define the boundary, SVMs can work well on small-to-moderate datasets if the boundary region is represented; however, SVMs scale poorly to **very large datasets** ($n_T \gg 10^4$) because the dual QP involves an $n_T \times n_T$ kernel matrix ([`06kernelBasedLearningTopic6.pdf`, Slide 6](06kernelBasedLearningTopic6.pdf)).

#### Important Choices and Variants

* **Learned parameters vs. Hyperparameters:**
  * *Learned during training:* Lagrange multipliers $\alpha_i \in [0, C]$ (and the resulting subset of support vectors), primal weight vector $\mathbf{w} = \sum_{i \in SV} \alpha_i y_i \phi(\mathbf{x}_i)$ (explicit only for linear SVM), and bias $b$ ([`06kernelBasedLearningTopic6.pdf`, Slides 21–23, 37–39](06kernelBasedLearningTopic6.pdf)).
  * *Hyperparameters (chosen by the user / tuned via cross-validation):* Regularisation parameter $C > 0$, class weights ($C_+, C_-$), choice of kernel function $k(\cdot,\cdot)$, and kernel parameters (polynomial degree $p$; Gaussian RBF $\gamma = \frac{1}{2\sigma^2}$ or $\sigma$; sigmoid $\kappa, \delta$) ([`06kernelBasedLearningTopic6.pdf`, Slides 6, 27, 40, 48](06kernelBasedLearningTopic6.pdf); [`fundamentals_of_ml_2ed_ch.pdf`, p. 366 (PDF p. 6)](fundamentals_of_ml_2ed_ch.pdf)).
* **Effect of $C$ (Soft-Margin Regularisation) ([`06kernelBasedLearningTopic6.pdf`, Slides 27–29](06kernelBasedLearningTopic6.pdf); [`quiz-08-1-kernel-based-learning.md`, Q5](quiz-08-1-kernel-based-learning.md)):**
  * **Large $C$ (weak regularisation):** Heavily penalises slack errors $\sum_i \xi_i$, forcing a **narrower margin** with fewer training violations. Lower bias, higher variance $\to$ **increases risk of overfitting** (sensitive to noisy boundary points/outliers, though effective if boundary points reflect true fine-grained separation as in "Generalization Data 1" on Slide 29).
  * **Small $C$ (strong regularisation):** Tolerates more margin violations/misclassifications to achieve a **wider, flatter margin** (smaller $\|\mathbf{w}\|^2$). Higher bias, lower variance $\to$ more robust to noisy outliers ("Generalization Data 2" on Slide 29), but overly small $C$ **increases risk of underfitting**.
* **Effect of Kernel Hyperparameters ([`06kernelBasedLearningTopic6.pdf`, Slides 38, 40, 47](06kernelBasedLearningTopic6.pdf); [`fundamentals_of_ml_2ed_ch.pdf`, p. 366 (PDF p. 6)](fundamentals_of_ml_2ed_ch.pdf)):**
  * **Gaussian RBF $\gamma = \frac{1}{2\sigma^2}$:** Controls the radius of influence of each support vector. A **large $\gamma$** (small $\sigma$) makes $\exp(-\gamma\|\mathbf{x}-\mathbf{x}_i\|^2)$ decay rapidly with distance, creating tight, complex decision islands around individual points (high variance, sensitive to outliers, prone to overfitting) ([`06kernelBasedLearningTopic6.pdf`, Slide 47](06kernelBasedLearningTopic6.pdf)). A **small $\gamma$** (large $\sigma$) produces smooth, broad similarity neighbourhoods (higher bias, more robust to outliers).
  * **Polynomial degree $p$:** $p=1$ gives a linear boundary; larger $p$ yields increasingly flexible curved boundaries (higher variance/overfitting risk). The textbook recommends starting with a linear or low-degree polynomial kernel before trying more complex kernels ([`fundamentals_of_ml_2ed_ch.pdf`, p. 366 (PDF p. 6)](fundamentals_of_ml_2ed_ch.pdf)).
* **Multi-Class SVM Variants ($K > 2$ classes) ([`06kernelBasedLearningTopic6.pdf`, Slides 42–44](06kernelBasedLearningTopic6.pdf)):**
  1. **One-vs-Rest (OvR / One-vs-All):** Trains **$K$ binary SVMs**. Classifier $k$ treats class $k$ as positive ($+1$) and all other $K-1$ classes combined as negative ($-1$). A query $\mathbf{x}$ is assigned to the class with the largest real-valued decision score:
     $$\hat{y} = \arg\max_{k \in \{1,\dots,K\}} f_k(\mathbf{x})$$
     *Tradeoff:* Simple and trains only $K$ models, but each binary problem has $1$ class against $K-1$ classes, creating **highly imbalanced training data** ([`06kernelBasedLearningTopic6.pdf`, Slide 43](06kernelBasedLearningTopic6.pdf)).
  2. **One-vs-One (OvO):** Trains **$\frac{K(K-1)}{2}$ binary SVMs**, one for every pair of classes. For a query $\mathbf{x}$, each pairwise classifier casts one vote, and **majority voting** selects the predicted class.
     *Tradeoff:* Each subproblem is smaller, more balanced, and easier to separate (often yielding higher accuracy), but training and evaluating $\frac{K(K-1)}{2}$ models is computationally much more expensive when $K$ is large ([`06kernelBasedLearningTopic6.pdf`, Slide 44](06kernelBasedLearningTopic6.pdf)).

#### Strengths, Limitations, and Practical Use

* **Strengths ([`06kernelBasedLearningTopic6.pdf`, Slide 6](06kernelBasedLearningTopic6.pdf); [`fundamentals_of_ml_2ed_ch.pdf`, p. 367 (PDF p. 7)](fundamentals_of_ml_2ed_ch.pdf)):**
  * **Convex optimisation problem:** Guaranteed global optimum with no local minima.
  * **Strong generalisation and theoretical guarantees:** Margin maximisation resists overfitting even in high-dimensional spaces.
  * **Geometric intuition and sparsity:** Predictions depend only on the $n_{SV}$ support vectors ($n_{SV} \le n_T$).
* **Limitations ([`06kernelBasedLearningTopic6.pdf`, Slide 6](06kernelBasedLearningTopic6.pdf); [`fundamentals_of_ml_2ed_ch.pdf`, p. 367 (PDF p. 7)](fundamentals_of_ml_2ed_ch.pdf)):**
  * **Poor scalability to large datasets ($n_T$ large):** Solving the highly constrained dual QP requires computing pairwise kernel entries $k(\mathbf{x}_i, \mathbf{x}_j)$ across $n_T$ instances ($O(n_T^2)$ kernel evaluations/storage and super-quadratic QP solver time).
  * **Black-box predictions (low interpretability):** Unlike linear/logistic regression or decision trees, non-linear kernel SVMs operate in an implicit feature space $\phi(\mathbf{x})$ where individual feature contributions cannot be easily inspected.
  * **Hyperparameter sensitivity:** Performance depends heavily on choosing an appropriate kernel, kernel parameters ($\gamma, p$), regularisation $C$, and feature scaling.
  * **Binary by nature:** Requires OvR or OvO wrappers for multi-class problems.

---

### 3.3 Support Vector Regression (SVR)

#### Main Idea and Assumptions

**Support Vector Regression (SVR)** adapts the maximum-margin and kernel principles of SVMs to continuous target prediction ($y_i \in \mathbb{R}$) ([`06kernelBasedLearningTopic6.pdf`, Slides 49–53](06kernelBasedLearningTopic6.pdf)).

* **Core intuition ($\epsilon$-tube and flatness):** Instead of minimising the sum of squared residuals across all data points like Ordinary Least Squares (OLS), SVR places a symmetrical **$\epsilon$-insensitive tube** of half-width $\epsilon \ge 0$ around the regression function $f(\mathbf{x}) = \mathbf{w} \cdot \mathbf{x} + b$. Any training instance whose residual $r_i = y_i - \hat{y}_i$ falls **inside** the tube ($|y_i - f(\mathbf{x}_i)| \le \epsilon$) is considered sufficiently accurate and incurs **zero loss** ([`06kernelBasedLearningTopic6.pdf`, Slides 51–52, 66](06kernelBasedLearningTopic6.pdf)). Simultaneously, SVR minimises $\frac{1}{2}\|\mathbf{w}\|^2$ to make the regression function as **flat** as possible, while linearly penalising deviations outside the tube via slack variables $\xi_i, \xi_i^* \ge 0$ ([`06kernelBasedLearningTopic6.pdf`, Slides 52–53](06kernelBasedLearningTopic6.pdf)).
* **Comparison of Loss Functions — OLS vs. SVR ([`06kernelBasedLearningTopic6.pdf`, Slides 50, 52, 66](06kernelBasedLearningTopic6.pdf)):**
  * **OLS ($L_2$ squared error loss):** Minimises $\sum_{i=1}^{n_T} (y_i - \hat{y}_i)^2$ (closed-form $\hat{\mathbf{w}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{Y}$). Every non-zero residual contributes to the fit, and large residuals grow **quadratically**, making OLS highly sensitive to outliers.
  * **SVR ($\epsilon$-insensitive loss):** Uses $L_\epsilon(r) = \max\{0, |r| - \epsilon\}$ where $r = y - \hat{y}$. Residuals with $|r| \le \epsilon$ contribute $0$ loss; residuals outside the tube ($|r| > \epsilon$) are penalised **linearly** rather than quadratically, reducing sensitivity to moderate outliers and small noise.
* **Assumptions:** Assumes a linear relationship $f(\mathbf{x}) = \mathbf{w}\cdot\mathbf{x} + b$ in input space (Linear SVR) or in the kernel-induced feature space $f(\mathbf{x}) = \mathbf{w}\cdot\phi(\mathbf{x}) + b$ (Kernel SVR) ([`06kernelBasedLearningTopic6.pdf`, Slides 49, 54](06kernelBasedLearningTopic6.pdf)), with an error tolerance band of $\pm\epsilon$.

#### How It Learns and Predicts

##### 1. Primal Optimisation Problem ([`06kernelBasedLearningTopic6.pdf`, Slides 51–55](06kernelBasedLearningTopic6.pdf))

For each training instance $(\mathbf{x}_i, y_i)$, introduce **two** non-negative slack variables: $\xi_i \ge 0$ for positive deviations **above** the upper tube boundary ($y_i - f(\mathbf{x}_i) > \epsilon$) and $\xi_i^* \ge 0$ for negative deviations **below** the lower tube boundary ($f(\mathbf{x}_i) - y_i > \epsilon$):

$$\min_{\mathbf{w}, b, \boldsymbol{\xi}, \boldsymbol{\xi}^*} \underbrace{\frac{1}{2}\|\mathbf{w}\|^2}_{\text{flatness / max margin}} + \underbrace{C\sum_{i=1}^{n_T} (\xi_i + \xi_i^*)}_{\text{min tube violation error}}$$
$$\text{subject to} \quad \begin{cases} y_i - \mathbf{w}\cdot\mathbf{x}_i - b \le \epsilon + \xi_i & (\text{above positive tube boundary}) \\ \mathbf{w}\cdot\mathbf{x}_i + b - y_i \le \epsilon + \xi_i^* & (\text{below negative tube boundary}) \\ \xi_i, \xi_i^* \ge 0 & \forall i = 1, \dots, n_T \end{cases}$$

* **Primal problem dimensionality:** $N + 2n_T + 1$ variables ($N$ weights in $\mathbf{w}$, $2n_T$ slack variables $\xi_i, \xi_i^*$, and $1$ bias $b$) ([`06kernelBasedLearningTopic6.pdf`, Slide 55](06kernelBasedLearningTopic6.pdf)).

##### 2. Lagrangian and Dual Problem ([`06kernelBasedLearningTopic6.pdf`, Slides 56–61, 65](06kernelBasedLearningTopic6.pdf))

1. Introduce Lagrange multipliers $\alpha_i \ge 0$ (for the upper constraint), $\alpha_i^* \ge 0$ (for the lower constraint), and $\mu_i, \mu_i^* \ge 0$ (for $\xi_i, \xi_i^* \ge 0$) ([`06kernelBasedLearningTopic6.pdf`, Slide 56](06kernelBasedLearningTopic6.pdf)).
2. Setting partial derivatives of the Lagrangian $L(\mathbf{w}, b, \boldsymbol{\xi}, \boldsymbol{\xi}^*, \boldsymbol{\alpha}, \boldsymbol{\alpha}^*, \boldsymbol{\mu}, \boldsymbol{\mu}^*)$ to zero yields the KKT stationarity conditions ([`06kernelBasedLearningTopic6.pdf`, Slide 60](06kernelBasedLearningTopic6.pdf)):
   $$\frac{\partial L}{\partial \mathbf{w}} = \mathbf{0} \implies \mathbf{w} = \sum_{i=1}^{n_T} (\alpha_i - \alpha_i^*)\mathbf{x}_i, \qquad \frac{\partial L}{\partial b} = 0 \implies \sum_{i=1}^{n_T} (\alpha_i - \alpha_i^*) = 0$$
   $$\frac{\partial L}{\partial \xi_i} = 0 \implies C - \alpha_i - \mu_i = 0 \implies 0 \le \alpha_i \le C, \qquad \frac{\partial L}{\partial \xi_i^*} = 0 \implies C - \alpha_i^* - \mu_i^* = 0 \implies 0 \le \alpha_i^* \le C$$
3. Substituting these into the Lagrangian (and applying the kernel trick $\mathbf{x}_i \cdot \mathbf{x}_j \to k(\mathbf{x}_i, \mathbf{x}_j)$) yields the **dual optimisation problem** over $2n_T$ variables ($\alpha_i, \alpha_i^*$) ([`06kernelBasedLearningTopic6.pdf`, Slides 61, 65](06kernelBasedLearningTopic6.pdf)):
   * **As printed on Slides 61 and 65:**
     $$\max_{\alpha_i, \alpha_i^*} W(\boldsymbol{\alpha}, \boldsymbol{\alpha}^*) = \frac{1}{2}\sum_{i,j}^{n_T} (\alpha_i - \alpha_i^*)(\alpha_j - \alpha_j^*) k(\mathbf{x}_i, \mathbf{x}_j) - \epsilon \sum_{i=1}^{n_T} (\alpha_i + \alpha_i^*) + \sum_{i=1}^{n_T} y_i (\alpha_i - \alpha_i^*)$$
   * **Technically qualified sign note:** When substituting $\mathbf{w} = \sum_i (\alpha_i - \alpha_i^*)\mathbf{x}_i$ into $\frac{1}{2}\|\mathbf{w}\|^2 - \mathbf{w}\cdot\sum_i(\alpha_i - \alpha_i^*)\mathbf{x}_i$ in the Lagrangian on Slide 59, the quadratic term simplifies to $\frac{1}{2}\|\mathbf{w}\|^2 - \|\mathbf{w}\|^2 = -\frac{1}{2}\|\mathbf{w}\|^2$. Thus, the standard concave dual maximisation objective has a **leading minus sign** $-\frac{1}{2}\sum_{i,j}(\alpha_i - \alpha_i^*)(\alpha_j - \alpha_j^*)k(\mathbf{x}_i, \mathbf{x}_j)$ (matching the $-\frac{1}{2}$ in the SVM dual on Slides 21 and 41). In both cases, the constraints are:
     $$\sum_{i=1}^{n_T} (\alpha_i - \alpha_i^*) = 0 \quad \text{and} \quad 0 \le \alpha_i, \alpha_i^* \le C \quad (\forall i = 1, \dots, n_T)$$

##### 3. Multiplier Behaviour, Bias $b$, and Prediction ([`06kernelBasedLearningTopic6.pdf`, Slides 57–58, 62–64](06kernelBasedLearningTopic6.pdf))

Because an instance cannot simultaneously lie above the upper boundary and below the lower boundary, at least one of $(\alpha_i, \alpha_i^*)$ is always zero ($\alpha_i \alpha_i^* = 0$) ([`06kernelBasedLearningTopic6.pdf`, Slides 52, 57](06kernelBasedLearningTopic6.pdf)):
* **Inside the $\epsilon$-tube ($|y_i - f(\mathbf{x}_i)| < \epsilon$):** $\xi_i = \xi_i^* = 0$ and $\alpha_i = \alpha_i^* = 0$ ($\implies \alpha_i - \alpha_i^* = 0$). These points **contribute nothing** to $f(\mathbf{x})$.
* **Above the $\epsilon$-tube ($y_i - f(\mathbf{x}_i) \ge \epsilon$, prediction $\hat{y}_i$ too small):** $\alpha_i > 0, \alpha_i^* = 0$. Becomes a **support vector** with positive weight $+\alpha_i$ that pulls the regression function **upward**.
* **Below the $\epsilon$-tube ($f(\mathbf{x}_i) - y_i \ge \epsilon$, prediction $\hat{y}_i$ too large):** $\alpha_i = 0, \alpha_i^* > 0$. Becomes a **support vector** with negative weight $-\alpha_i^*$ that pulls the regression function **downward**.
* **Calculating bias $b$ ([`06kernelBasedLearningTopic6.pdf`, Slides 62–64](06kernelBasedLearningTopic6.pdf)):** By complementary slackness, support vectors with $0 < \alpha_i < C$ have $\xi_i = 0$ and lie **on the upper $\epsilon$-boundary** ($y_i - \mathbf{w}\cdot\mathbf{x}_i - b_i = \epsilon \implies b_i = y_i - \epsilon - \mathbf{w}\cdot\mathbf{x}_i$). Support vectors with $0 < \alpha_i^* < C$ have $\xi_i^* = 0$ and lie **on the lower $\epsilon$-boundary** ($\mathbf{w}\cdot\mathbf{x}_i + b_{i^*} - y_i = \epsilon \implies b_{i^*} = y_i + \epsilon - \mathbf{w}\cdot\mathbf{x}_i$). Average over both boundary sets $SV = \{i \mid 0 < \alpha_i < C\}$ and $SV^* = \{i \mid 0 < \alpha_i^* < C\}$:
  $$b = \frac{1}{2}\left( \frac{1}{n_{SV}}\sum_{i \in SV} b_i + \frac{1}{n_{SV^*}}\sum_{i \in SV^*} b_{i^*} \right)$$
* **Prediction rule ([`06kernelBasedLearningTopic6.pdf`, Slide 58](06kernelBasedLearningTopic6.pdf)):**
  $$f(\mathbf{x}) = \sum_{i=1}^{n_T} (\alpha_i - \alpha_i^*) k(\mathbf{x}_i, \mathbf{x}) + b$$

#### Supported Data and Tasks

* **Descriptive features:** Numeric only (continuous or discrete); categorical features must be one-hot/numerically encoded and all features should be z-score standardised, for the same dot-product and kernel-distance reasons as SVM ([`06kernelBasedLearningTopic6.pdf`, Slides 45, 65](06kernelBasedLearningTopic6.pdf)).
* **Target feature:** Continuous numeric target $y_i \in \mathbb{R}$ ([`06kernelBasedLearningTopic6.pdf`, Slide 49](06kernelBasedLearningTopic6.pdf)). (For binary/multi-class targets, use SVM instead.)
* **Outputs:** Continuous real-valued predictions $\hat{y} = f(\mathbf{x}) \in \mathbb{R}$.

#### Data Quality and Preprocessing (SVR)

| Issue | Effect on the standard model and why | Relevant extension and how it helps | Preprocessing / mitigation and tradeoffs |
| --- | --- | --- | --- |
| **1. Missing values** | **Cannot be handled natively.** Kernel evaluations $k(\mathbf{x}_i, \mathbf{x}_j)$ and dot products $\mathbf{x}_i \cdot \mathbf{x}_j$ are undefined when descriptive features contain missing values ([`06kernelBasedLearningTopic6.pdf`, Slide 45](06kernelBasedLearningTopic6.pdf)); missing targets $y_i$ cannot be placed in the dual objective $\sum_i y_i(\alpha_i - \alpha_i^*)$ or tube constraints. | None covered in the course materials. | Impute missing descriptive features (fitted on the training split) or drop affected instances/features; drop rows with missing targets. |
| **2. Outliers** | **More robust to moderate outliers than OLS, but still sensitive to extreme outliers ([`06kernelBasedLearningTopic6.pdf`, Slide 67](06kernelBasedLearningTopic6.pdf)):**<br>• *Target outliers ($y_i$):* Points outside the $\epsilon$-tube become support vectors ($|\alpha_i - \alpha_i^*| > 0$) and pull the tube margins toward the outlier. However, because the $\epsilon$-insensitive loss grows **linearly** ($|r| - \epsilon$) rather than quadratically ($r^2$) and multipliers are capped at $C$, moderate outliers have bounded influence compared to OLS ([`06kernelBasedLearningTopic6.pdf`, Slides 66–67](06kernelBasedLearningTopic6.pdf)).<br>• *Feature outliers ($\mathbf{x}_i$):* Affect linear SVR similarly to target outliers if they fall outside the tube. | **Gaussian RBF Kernel SVR with small $\gamma$:** On Slide 67, the lectures note that when a Gaussian kernel is used and $\gamma$ is small (or more generally, when $\|\mathbf{x}_i - \mathbf{x}_j\|$ is very large), the kernel similarity $k(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma\|\mathbf{x}_i - \mathbf{x}_j\|^2) \approx 0$ for distant feature outliers, isolating their influence on predictions elsewhere ([`06kernelBasedLearningTopic6.pdf`, Slide 67](06kernelBasedLearningTopic6.pdf)). | Choose a smaller regularisation parameter $C$ (caps the maximum pull $\alpha_i, \alpha_i^* \le C$ of any outlier) or inspect/remove extreme data-entry outliers on the training split. Use z-score scaling (or robust scaling) rather than min-max scaling ([`06kernelBasedLearningTopic6.pdf`, Slide 45](06kernelBasedLearningTopic6.pdf)). *Tradeoff:* Too small a $C$ underfits the main trend. |
| **3. Noise** | **Generally robust to small-to-moderate noise ([`06kernelBasedLearningTopic6.pdf`, Slide 67](06kernelBasedLearningTopic6.pdf)):**<br>• *Target noise ($y_i$):* Noisy instances whose target perturbations keep them **inside** the $\epsilon$-tube ($|y_i - f(\mathbf{x}_i)| \le \epsilon$) incur **zero loss**, have $\alpha_i = \alpha_i^* = 0$, and contribute no penalty to the optimisation ([`06kernelBasedLearningTopic6.pdf`, Slides 52, 67](06kernelBasedLearningTopic6.pdf)).<br>• *Feature noise ($\mathbf{x}_i$):* Small input perturbations that leave points inside the $\epsilon$-tube do not change the support vectors or regression surface. | The **$\epsilon$-insensitive tube ($\epsilon$)** and **soft-margin regularisation ($C$)** are built into standard SVR to absorb target/measurement noise of magnitude up to $\epsilon$ and limit the impact of larger noise spikes ([`06kernelBasedLearningTopic6.pdf`, Slides 51–53, 67](06kernelBasedLearningTopic6.pdf)). | Set $\epsilon$ proportional to the expected measurement noise level in $y$ and tune $C$ via cross-validation. *Tradeoff:* Setting $\epsilon$ too large to hide noise causes underfitting and loses predictive precision ([`quiz-08-1-kernel-based-learning.md`, Q2](quiz-08-1-kernel-based-learning.md)). |
| **4. Skew class distributions** | **Not applicable as class imbalance** (target $y_i \in \mathbb{R}$ is continuous). *Related regression analogue:* If the target distribution is heavily skewed or has sparse tail regions, extreme tail targets may lie outside the $\epsilon$-tube and act like target outliers pulling the tube, or sparse regions may be under-represented if $C$ is small. | None covered in the course materials. | Target transformations (e.g., log transform for heavily right-skewed positive targets, fitted/inverted carefully) can stabilise residual magnitudes relative to a constant tube width $\epsilon$. |
| **5. Irrelevant features** | Same mechanism as SVM: $\frac{1}{2}\|\mathbf{w}\|^2$ is an $L_2$ penalty that does not zero out irrelevant coordinates, and irrelevant features add noise to $\mathbf{x}_i \cdot \mathbf{x}_j$ and $\|\mathbf{x}_i - \mathbf{x}_j\|^2$ in kernel calculations. Tolerates a few irrelevant features, but degrades when many are present ([`06kernelBasedLearningTopic6.pdf`, Slide 47](06kernelBasedLearningTopic6.pdf)). | None covered in the course materials. | Perform feature selection or dimensionality reduction within the training split before fitting SVR. |
| **6. Features with different value ranges** | **Highly sensitive to feature scales**, exactly as in SVM: large-scale features dominate dot products $\mathbf{x}_i \cdot \mathbf{x}_j$, kernel distances $\|\mathbf{x}_i - \mathbf{x}_j\|^2$, and weight regularisation $\frac{1}{2}\|\mathbf{w}\|^2$ ([`06kernelBasedLearningTopic6.pdf`, Slide 45](06kernelBasedLearningTopic6.pdf)). | None. | Apply **z-score standardisation** fitted on the training split (preferred over min-max scaling due to outlier sensitivity) ([`06kernelBasedLearningTopic6.pdf`, Slide 45](06kernelBasedLearningTopic6.pdf)). |
| **7. Different descriptive feature data types** | **Requires all descriptive features to be numerical** to evaluate dot products and kernels ([`06kernelBasedLearningTopic6.pdf`, Slide 45](06kernelBasedLearningTopic6.pdf)). | Non-linear relationships among numeric features are handled via **Kernel SVR** (polynomial, Gaussian RBF) ([`06kernelBasedLearningTopic6.pdf`, Slide 65](06kernelBasedLearningTopic6.pdf)). | Encode nominal features via one-hot encoding and ordinal features via numeric mapping, then z-score standardise. |

* **Additional data considerations (SVR):**
  * **Multicollinearity and redundant features:** Whereas OLS $(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{Y}$ ([`06kernelBasedLearningTopic6.pdf`, Slide 50](06kernelBasedLearningTopic6.pdf)) becomes numerically unstable or non-invertible under severe multicollinearity, SVR minimises $\frac{1}{2}\|\mathbf{w}\|^2 + C\sum_i(\xi_i + \xi_i^*)$, where the $L_2$ norm term $\frac{1}{2}\|\mathbf{w}\|^2$ acts like ridge regularisation and guarantees a well-posed convex optimum even with collinear features.
  * **Dimensionality vs. sample size:** Notice the complexity comparison on Slides 55 and 61: the **primal** SVR problem has $N + 2n_T + 1$ variables, whereas the **dual** problem has $2n_T$ variables regardless of the feature-space dimensionality $D$ ([`06kernelBasedLearningTopic6.pdf`, Slides 55, 61](06kernelBasedLearningTopic6.pdf)). Hence, dual SVR handles very high-dimensional feature spaces ($D \gg n_T$) efficiently, but scales poorly when $n_T$ is very large.

#### Important Choices and Variants

* **Learned parameters:** Dual multipliers $\alpha_i, \alpha_i^* \in [0, C]$ ($2n_T$ parameters, nonzero only for support vectors outside/on the $\epsilon$-tube), primal weights $\mathbf{w} = \sum_{i=1}^{n_T}(\alpha_i - \alpha_i^*)\mathbf{x}_i$ (in linear SVR), and bias $b$ ([`06kernelBasedLearningTopic6.pdf`, Slides 60–64](06kernelBasedLearningTopic6.pdf)).
* **Hyperparameters:**
  * **Tube width $\epsilon \ge 0$:** Sets the radius of the error-insensitive zone ([`06kernelBasedLearningTopic6.pdf`, Slide 51](06kernelBasedLearningTopic6.pdf)).
    * *Larger $\epsilon$:* More training points fall inside the tube ($\alpha_i = \alpha_i^* = 0$), yielding **fewer support vectors**, a flatter function, and fewer constraint violations—but making $\epsilon$ too large causes severe **underfitting** (low accuracy) ([`quiz-08-1-kernel-based-learning.md`, Q2, Q4](quiz-08-1-kernel-based-learning.md)).
    * *Smaller $\epsilon$:* Narrower tube forces more points outside the tube to become support vectors, fitting the data more tightly—but if $\epsilon$ is too small (especially with large $C$), the model fits noise (**overfitting**).
  * **Regularisation parameter $C > 0$:** Controls the trade-off between function flatness ($\frac{1}{2}\|\mathbf{w}\|^2$) and deviations outside the $\epsilon$-tube ($\sum_i(\xi_i + \xi_i^*)$) ([`06kernelBasedLearningTopic6.pdf`, Slide 53](06kernelBasedLearningTopic6.pdf)). Large $C$ penalises outside-tube errors heavily (higher variance / overfitting risk); small $C$ prioritises flatness (higher bias / underfitting risk).
  * **Kernel function and parameters:** Linear ($\mathbf{x}_1 \cdot \mathbf{x}_2$), Polynomial ($(\mathbf{x}\cdot\mathbf{x}_i + 1)^p$), or Gaussian RBF ($e^{-\|\mathbf{x}-\mathbf{x}_i\|^2 / 2\sigma^2}$) ([`06kernelBasedLearningTopic6.pdf`, Slide 65](06kernelBasedLearningTopic6.pdf)).

#### Strengths, Limitations, and Practical Use

* **When SVR is a good choice:** Continuous regression problems with non-linear relationships, high-dimensional inputs, multicollinearity, or moderate noise/outliers where OLS squared-error loss would over-react to residuals.
* **When SVR is a poor choice:** Very large datasets ($n_T$ large, where dual QP over $2n_T$ variables is slow), datasets with extreme uncleaned target outliers, or applications requiring transparent coefficient interpretation (when non-linear kernels are used).

---

## 4. Exam and Comparison Points

### Key Comparisons

1. **SVM vs. Logistic Regression ([`fundamentals_of_ml_2ed_ch.pdf`, pp. 362, 367 (PDF pp. 2, 7)](fundamentals_of_ml_2ed_ch.pdf)):**
   * Both learn a linear decision boundary $\mathbf{w}\cdot\mathbf{x} + b = 0$ (or non-linear via basis functions), but their **inductive biases** and **outputs** differ:
   * Logistic regression fits all training instances by optimising a smooth log-loss/sum-of-squared-error objective via gradient descent and outputs calibrated **probabilities** $\in (0, 1)$, with interpretable feature weights.
   * SVM solves a constrained **quadratic programme** to **maximise the margin** $\frac{2}{\|\mathbf{w}\|}$ using **only the support vectors**, outputs uncalibrated **signed margin scores** / class labels $\{-1, +1\}$, works particularly well in high dimensions, and is much less interpretable when kernels are used.
2. **SVR vs. Ordinary Least Squares (OLS) Linear Regression ([`06kernelBasedLearningTopic6.pdf`, Slides 50, 66–67](06kernelBasedLearningTopic6.pdf)):**
   * OLS minimises $L_2$ sum of squared errors $\sum_{i=1}^{n_T}(y_i - \hat{y}_i)^2$: every training point influences the fit, and residuals are penalised **quadratically**, making OLS highly sensitive to outliers.
   * SVR minimises $\frac{1}{2}\|\mathbf{w}\|^2 + C\sum_{i=1}^{n_T}(\xi_i + \xi_i^*)$ using the **$\epsilon$-insensitive loss** $L_\epsilon(r) = \max\{0, |r| - \epsilon\}$: points inside the $\epsilon$-tube incur **zero loss** ($\alpha_i = \alpha_i^* = 0$), and points outside the tube are penalised **linearly** and capped by $\alpha_i, \alpha_i^* \le C$.
3. **One-vs-Rest (OvR) vs. One-vs-One (OvO) Multi-Class SVM ([`06kernelBasedLearningTopic6.pdf`, Slides 42–44](06kernelBasedLearningTopic6.pdf)):**
   * **OvR:** $K$ classifiers; class $k$ ($+1$) vs. all other $K-1$ classes ($-1$); predicts $\hat{y} = \arg\max_k f_k(\mathbf{x})$. Fast ($K$ models), but suffers from severe **class imbalance**.
   * **OvO:** $\frac{K(K-1)}{2}$ classifiers; one per pair of classes; predicts via **majority voting**. Avoids artificial class imbalance and solves easier subproblems (higher accuracy), but scales as $O(K^2)$ in number of models.

### Quiz-Tested Concepts, Misconceptions, and Source Conflicts

Every question in [`quiz-08-1-kernel-based-learning.md`](quiz-08-1-kernel-based-learning.md) tests a specific conceptual trap or course convention:

* **Misconception 1: "SVMs can only classify linearly separable data because they use hyperplanes" ([`quiz-08-1-kernel-based-learning.md`, Q7](quiz-08-1-kernel-based-learning.md) — False).**
  * *Why it is wrong:* Soft-margin SVMs handle overlapping classes in input space via slack variables $\xi_i \ge 0$, and Kernel SVMs find a linear hyperplane in a higher-dimensional feature space $\mathbb{R}^D$ that corresponds to a **non-linear** decision boundary in the original input space $\mathbb{R}^N$ ([`06kernelBasedLearningTopic6.pdf`, Slides 4, 25–26, 33](06kernelBasedLearningTopic6.pdf)).
* **Misconception 2: "If an input dataset has noise in the input features, an SVM cannot be used" ([`quiz-08-1-kernel-based-learning.md`, Q6](quiz-08-1-kernel-based-learning.md) — False).**
  * *Why it is wrong:* Only hard-margin SVM fails when noise causes class overlap. Soft-margin SVM tolerates feature and label noise via slack variables $\xi_i$ and regularisation $C$, and small perturbations to non-support vectors do not change the hyperplane at all ([`06kernelBasedLearningTopic6.pdf`, Slide 46](06kernelBasedLearningTopic6.pdf)).
* **Misconception 3: "A large value of $C$ guarantees good generalisation because errors are strongly punished" ([`quiz-08-1-kernel-based-learning.md`, Q5](quiz-08-1-kernel-based-learning.md) — False; it increases the chance of overfitting).**
  * *Why it is wrong:* Large $C$ forces $\sum_i \xi_i$ to be tiny, shrinking the margin width $\frac{2}{\|\mathbf{w}\|}$ to accommodate noisy training points and outliers, which increases variance and overfitting risk ([`06kernelBasedLearningTopic6.pdf`, Slides 27–28](06kernelBasedLearningTopic6.pdf)).
* **Misconception 4: "In SVR, $\epsilon$ should be as large as possible to reduce constraint violations, or the width of the $\epsilon$-tube is maximised during training so all points fall inside it" ([`quiz-08-1-kernel-based-learning.md`, Q2, Q4](quiz-08-1-kernel-based-learning.md) — Both False).**
  * *Why it is wrong:* $\epsilon$ is a fixed **hyperparameter** chosen before training, not an optimisation variable maximised during training. Making $\epsilon$ huge trivially eliminates constraint violations by ignoring large prediction errors, resulting in severe underfitting ([`06kernelBasedLearningTopic6.pdf`, Slides 51–53](06kernelBasedLearningTopic6.pdf)).
* **Misconception 5: "For 1D non-linear regression, $\phi(x_1) = (x_1, 2x_1)$ is a good basis transformation" ([`quiz-08-1-kernel-based-learning.md`, Q3](quiz-08-1-kernel-based-learning.md) — False).**
  * *Why it is wrong:* Because $2x_1$ is a linear multiple of $x_1$, all mapped points lie on a 1D line in $\mathbb{R}^2$. Any linear model $w_1 x_1 + w_2(2x_1) + b = (w_1 + 2w_2)x_1 + b$ remains strictly linear in $x_1$. A useful basis transformation must be **non-linear** (e.g., $\phi(x_1) = (x_1, x_1^2)$).
* **Misconception 6: "Acceptable initialisation values for Gaussian means $\mu_i(0)$ and standard deviations $\sigma_i(0)$ are $[-1, 1]$" ([`quiz-08-1-kernel-based-learning.md`, Q10](quiz-08-1-kernel-based-learning.md) — False).**
  * *Why it is wrong:* A standard deviation $\sigma_i$ measures spread/width and **must be strictly positive ($\sigma_i > 0$)**; zero or negative values in $[-1, 1]$ are invalid. (Furthermore, in standard Gaussian RBF kernels $k(\mathbf{x}, \mathbf{x}_i) = \exp(-\|\mathbf{x}-\mathbf{x}_i\|^2 / 2\sigma^2)$, the kernel centres are the training vectors $\mathbf{x}_i$ themselves rather than randomly initialised parameters $\mu_i(0)$, which belong to RBF neural networks.)
* **Tested Rule 7: Target formatting $y_i \in \{-1, +1\}$ ([`quiz-08-1-kernel-based-learning.md`, Q11](quiz-08-1-kernel-based-learning.md) — True) and Bias calculation via KKT Complementary Slackness ([`quiz-08-1-kernel-based-learning.md`, Q9](quiz-08-1-kernel-based-learning.md) — `complementary slackness condition`).**
  * Target labels must be $\{-1, +1\}$ so that $y_i(\mathbf{w}\cdot\mathbf{x}_i + b) \ge 1$ combines both class constraints ([`06kernelBasedLearningTopic6.pdf`, Slides 12–13](06kernelBasedLearningTopic6.pdf)). The bias $b$ is solved from support vectors ($0 < \alpha_i < C$) using the 3rd KKT condition $\alpha_i[y_i(\mathbf{w}\cdot\mathbf{x}_i + b) - 1] = 0$, which is **complementary slackness** ([`06kernelBasedLearningTopic6.pdf`, Slide 23](06kernelBasedLearningTopic6.pdf)).

#### Flagged Conflicts and Questionable Quiz / Source Statements

> [!IMPORTANT]
> Separate the **recorded course/quiz answer** from the **technically qualified explanation** on the following five points:

1. **Quiz Q1 — Support Vector Count and Generalisation Accuracy ([`quiz-08-1-kernel-based-learning.md`, Q1](quiz-08-1-kernel-based-learning.md)):**
   * **Scenario:** An SVM is fitted to $n_T = 10\,000$ observations with $n_{SV} = 500$ expected support vectors.
   * **Recorded course answer:** `c. The classification accuracy will be at least 95%`.
   * **Underlying formula and technical qualification:** This comes from Vapnik's leave-one-out bound (covered in the supplementary reading `kroon2004.pdf` referenced on Slide 3): if a support vector is removed, the hyperplane changes; if any of the $n_T - n_{SV}$ non-support vectors is removed, the hyperplane is unchanged and classifies that left-out point correctly ([`06kernelBasedLearningTopic6.pdf`, Slides 9–10](06kernelBasedLearningTopic6.pdf)). Thus, the leave-one-out error rate is bounded by:
     $$E[P(\text{error})] \le \frac{E[n_{SV}]}{n_T} = \frac{500}{10\,000} = 0.05 \quad (5\%)$$
     giving an expected accuracy $\ge 1 - 0.05 = 95\%$. Technically, this bounds the **expected** generalisation error over i.i.d. draws (and assumes hard-margin or counted margin errors), rather than guaranteeing $\ge 95\%$ accuracy deterministically on every arbitrary test sample.
2. **Quiz Q8 vs. Textbook Chapter 7 — Is SVM "Error-Based Learning"? ([`quiz-08-1-kernel-based-learning.md`, Q8](quiz-08-1-kernel-based-learning.md) vs. [`fundamentals_of_ml_2ed_ch.pdf`, p. 361 (PDF p. 1)](fundamentals_of_ml_2ed_ch.pdf)):**
   * **Recorded course answer (Q8):** `False` to *"Kernel-based learning uses the calculated error (e.g. the error between $y$ and $\hat{y}$) to update model parameters"*.
   * **Textbook classification:** Kelleher et al. place SVMs in Chapter 7 (*Error-Based Learning*, p. 361).
   * **Reconciliation:** In the Stellenbosch course lectures ([`06kernelBasedLearningTopic6.pdf`, Slide 4](06kernelBasedLearningTopic6.pdf)), kernel-based learning is treated as its own paradigm resulting in similarity-based dual models trained by **convex quadratic programming** over KKT margin constraints, **not** by iteratively calculating $(y - \hat{y})$ and taking gradient-descent weight steps. For exams/quizzes, remember that kernel methods solve a constrained QP rather than performing error-driven weight updates.
3. **Quiz Q12 — Origin of $\max\left(\frac{2}{\|\mathbf{w}\|}\right)$ ([`quiz-08-1-kernel-based-learning.md`, Q12](quiz-08-1-kernel-based-learning.md)):**
   * **Question:** *"The maximisation problem of $\max\left(\frac{2}{\|\mathbf{w}\|}\right)$ for finding the optimal hyperplane is a result of applying..."*
   * **Recorded course answer:** `ii. Karush–Kuhn–Tucker conditions` (marked correct over `iv. Lagrangian multiplier method`).
   * **Technically qualified explanation:** On [`06kernelBasedLearningTopic6.pdf`, Slide 14](06kernelBasedLearningTopic6.pdf), the slide header reads: *"Objective is to maximize the margin, i.e. $\max \frac{2}{\|\mathbf{w}\|}$, s.t. satisfaction of the 2nd KKT condition. How did we get this?"* followed by the Euclidean perpendicular distance formula $d = \frac{|\mathbf{w}^T\mathbf{x}+b|}{\|\mathbf{w}\|}$ evaluated at the canonical margin boundaries $\mathbf{w}^T\mathbf{x}+b = \pm 1$. Mathematically, $\frac{2}{\|\mathbf{w}\|}$ comes from **Euclidean geometry** (perpendicular distance between parallel hyperplanes) combined with the canonical scaling of the primal separability constraint (which Slides 12 and 14 label "the 2nd KKT condition"), while the **Lagrangian multiplier method** and **KKT conditions** are subsequently applied on Slides 15–23 to *solve* the resulting constrained optimisation problem.
4. **Textbook Equations (7.41), (7.46), (7.47) vs. Lecture Slides 24 & 37 — Placement of Bias $b$ ($w_0$):**
   * **Textbook ([`fundamentals_of_ml_2ed_ch.pdf`, pp. 363, 365, 366 (PDF pp. 3, 5, 6)](fundamentals_of_ml_2ed_ch.pdf)):** Writes $\mathbb{M}_{\boldsymbol{\alpha}, w_0}(\mathbf{q}) = \sum_{i=1}^s (t_i \times \boldsymbol{\alpha}[i] \times (\mathbf{d}_i \cdot \mathbf{q}) + w_0)$, placing $+ w_0$ **inside** the summation over support vectors (and adding $w_0$ three times in the worked example on p. 365).
   * **Lecture Slides ([`06kernelBasedLearningTopic6.pdf`, Slides 24, 37](06kernelBasedLearningTopic6.pdf)) & Standard Mathematical Definition:** Because $\mathbf{w} = \sum_{i=1}^{n_T} y_i \alpha_i \phi(\mathbf{x}_i)$, substituting $\mathbf{w}$ into $\mathbf{w}\cdot\phi(\mathbf{x}) + b$ yields $\left(\sum_{i=1}^{n_T} y_i \alpha_i k(\mathbf{x}, \mathbf{x}_i)\right) + b$, with the bias $b$ added **once outside** the summation. Always place $+ b$ outside the summation when writing SVM/SVR prediction formulas.
5. **Slide Formula Typos to Watch For ([`06kernelBasedLearningTopic6.pdf`, Slides 23, 61, 65](06kernelBasedLearningTopic6.pdf)):**
   * **Slide 23 (Linear SVM bias $b$):** Writes $b_{i_o} = \frac{1}{y_{i_o}} - y_{i_o}\mathbf{w}\cdot\mathbf{x}_{i_o}$ from $y_{i_o}(\mathbf{w}\cdot\mathbf{x}_{i_o} + b_{i_o}) = 1$. Dividing by $y_{i_o} \in \{-1, +1\}$ (where $\frac{1}{y_{i_o}} = y_{i_o}$) actually gives $b_{i_o} = y_{i_o} - \mathbf{w}\cdot\mathbf{x}_{i_o}$, which matches Slide 38 ($b = y_i - \sum_j y_j \alpha_j k(\mathbf{x}_j, \mathbf{x}_i)$).
   * **Slides 61 and 65 (SVR Dual Objective):** Omit the minus sign before $\frac{1}{2}\sum_{i,j}(\alpha_i - \alpha_i^*)(\alpha_j - \alpha_j^*)k(\mathbf{x}_i, \mathbf{x}_j)$ when maximising $W(\boldsymbol{\alpha})$. Compare with Slide 59 (Lagrangian) and Slides 21/41 (SVM dual, which correctly has $-\frac{1}{2}$).

### Revision Questions (Synthesised for Self-Testing)

*(Note: No personal `README.md` file was present in this directory; the questions below cover the standard course synthesis prompts for this topic.)*

1. **Why do we reformulate the primal SVM/SVR problem into the Lagrangian dual?**
   * It replaces $n_T$ data-dependent margin/tube inequality constraints with simple box bounds ($0 \le \alpha_i \le C$) and expresses training data purely as pairwise dot products $\mathbf{x}_i \cdot \mathbf{x}_j$, enabling the kernel trick ([`06kernelBasedLearningTopic6.pdf`, Slide 16](06kernelBasedLearningTopic6.pdf)).
2. **What happens to the SVM decision boundary if you remove all training instances that have $\alpha_i = 0$?**
   * Nothing changes: the weight vector $\mathbf{w} = \sum_{i=1}^{n_T} \alpha_i y_i \mathbf{x}_i$ and decision function depend solely on the support vectors ($\alpha_i > 0$) ([`06kernelBasedLearningTopic6.pdf`, Slides 9–10, 22](06kernelBasedLearningTopic6.pdf)).
3. **Why is z-score normalisation preferred over min-max scaling for SVMs and SVRs?**
   * SVMs and SVRs rely on dot products and Euclidean distances that are distorted by feature scale differences; min-max scaling is sensitive to outliers and compresses non-outlier values into a narrow range, whereas z-score standardisation centres and scales by standard deviation ([`06kernelBasedLearningTopic6.pdf`, Slide 45](06kernelBasedLearningTopic6.pdf)).
4. **How do $\alpha_i$ and $\alpha_i^*$ behave for points above, below, and inside the $\epsilon$-tube in SVR?**
   * Above the tube ($y_i - \hat{y}_i > \epsilon$): $\alpha_i > 0, \alpha_i^* = 0$ (pulls function up). Below the tube ($\hat{y}_i - y_i > \epsilon$): $\alpha_i = 0, \alpha_i^* > 0$ (pulls function down). Strictly inside the tube ($|y_i - \hat{y}_i| < \epsilon$): $\alpha_i = \alpha_i^* = 0$ (contributes nothing) ([`06kernelBasedLearningTopic6.pdf`, Slides 57–58](06kernelBasedLearningTopic6.pdf)).

---

## 5. Sources and Unresolved Questions

### Local Sources Used

* [`06kernelBasedLearningTopic6.pdf`](06kernelBasedLearningTopic6.pdf) — Lecture slides: *Machine Learning 441/741, Topic 6: Kernel-based Learning*, A.P. Engelbrecht, Stellenbosch University, 2026 (68 slides, PDF pp. 1–68). Primary source for course scope, mathematical derivations, SVR formulation, multi-class strategies, and data quality analyses.
* [`fundamentals_of_ml_2ed_ch.pdf`](fundamentals_of_ml_2ed_ch.pdf) — Textbook excerpt: J.D. Kelleher, B. Mac Namee, and A. D'Arcy, *Fundamentals of Machine Learning for Predictive Data Analytics*, Chapter 7 (*Error-Based Learning*), Section 7.4.7 (*Support Vector Machines*) and Section 7.5 (*Summary*), printed pp. 361–367 (PDF pp. 1–7). Primary source for SVM intuition, margin diagrams, and the generators worked example.
* [`quiz-08-1-kernel-based-learning.md`](quiz-08-1-kernel-based-learning.md) (and SVG equation assets in [`quiz-08-1-kernel-based-learning-images/`](quiz-08-1-kernel-based-learning-images/)) — Saved Moodle quiz and answer key (Questions 1–12).

### Source Gaps and Unresolved Questions

* **Missing supplementary articles referenced on Slide 3:** [`06kernelBasedLearningTopic6.pdf`, Slide 3](06kernelBasedLearningTopic6.pdf) lists two additional reading files—`kroon2004.pdf` and `SmoSch04.pdf` (Smola & Schölkopf, 2004, *A Tutorial on Support Vector Regression*)—which are not present in this directory. In particular, Quiz Question 1 (bounding generalisation accuracy via the support-vector ratio $\frac{n_{SV}}{n_T}$) and Quiz Question 10 (Gaussian parameter initialisation notation $\mu_i(0), \sigma_i(0)$) draw on external/supplementary material not printed in the slides or textbook excerpt.
* **No personal notes (`README.md`):** This topic directory does not contain a personal `README.md` or personal summary questions list.
