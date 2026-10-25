# Ensemble Learning — Course Overview

---

## 1. Topic at a glance

Rather than training a single predictive model, **ensemble learning** (*decision by committee*) trains multiple models and aggregates their outputs to make predictions ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 4; [fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), Section 4.4.5, p. 158 / PDF p. 1). Its core premise is that "two heads are better than one": different models make errors on different data patterns, so combining their knowledge improves generalisation—**provided the ensemble members are diverse** (different in their prediction behaviour) and avoid "group think" ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slides 5–7; [fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), p. 158 / PDF p. 1).

### How ensemble diversity is created
Course materials highlight six ways to induce different prediction behaviours ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 7):
1. **Data sampling:** Train the same model type on different subsets of instances (e.g., bootstrap sampling, sub-bagging, weighted boosting distributions).
2. **Feature sampling (*subspace sampling*):** Train the same model type using different subsets of descriptive features (e.g., random forests, extra trees).
3. **Model family diversity:** Combine different types of learning algorithms (*heterogeneous ensembles* / *mixture of heterogeneous experts*).
4. **Training algorithm diversity:** Use different optimisation or learning algorithms for the same model class (e.g., shallow neural networks trained with different optimisers).
5. **Hyperparameter diversity:** Vary control parameter values across members (e.g., SVM ensembles with different regularisation $C$ or kernels; $k$-NN with different $k$).
6. **Architecture diversity:** Vary model structure (e.g., neural networks with different hidden-layer topologies).

### Models, meta-algorithms, objectives, and evaluation methods covered

- **Ensemble meta-frameworks (training & aggregation strategies):**
  - **Bagging (*Bootstrap Aggregating*)** and **Sub-bagging** (parallel training on random instance samples; variance reduction).
  - **Boosting** (sequential training where each model focuses on prior errors; bias and variance reduction).
  - **Stacking (*Stacked Generalisation*)** (two-stage hierarchical framework using a *meta-learner* trained on out-of-sample base predictions).
  - **Homogeneous vs. Heterogeneous Ensembles** (including **Neural Network Ensembles** and **SVM Ensembles**).
- **Specific ensemble models:**
  - **Random Forests** (bagging + subspace feature sampling + unpruned decision trees).
  - **Extra Trees (*Extremely Randomized Trees*)** (full training set + subspace feature sampling + random split thresholds).
  - **Isolation Forests** (unsupervised anomaly detection using sub-bagged random binary *isolation trees*).
  - **Adaptive Boosting (AdaBoost)** (sequential weighted-error boosting, typically with **decision stumps**).
  - **Gradient Boosting Machines (GBM)** (sequential additive models fitted to residuals / negative gradients of a loss function).
- **Algorithms, objectives, and evaluation methods (distinct from the prediction models):**
  - **Sampling schemes:** Bootstrap sampling (with replacement), sub-bagging (without replacement), subspace sampling (feature subsets), and weighted replication.
  - **Weak-learner training algorithm:** Cumulative weighted sum algorithm for finding optimal decision stumps ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slides 51–54).
  - **Objectives / loss functions:** Weighted misclassification error $\epsilon_t$ (AdaBoost), Mean Squared Error / $L_2$ loss (GBM regression), and Log-Likelihood loss (GBM classification).
  - **Evaluation method:** **Out-of-Bag (OOB) error** for bootstrap-based ensembles ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slides 33–35).

---

## 2. Quick comparison

| Model | Task / target type | Descriptive feature types and preparation | Main strength | Main limitation |
| --- | --- | --- | --- | --- |
| **Bagging (and Sub-bagging)** | Supervised: classification (categorical) & regression (numeric) | Depends on base learner (e.g., trees handle mixed types natively; NNs/SVMs require numeric encoding & scaling) | Reduces variance of complex, unstable base learners; parallelisable training and fast prediction | Does not reduce base-learner bias; loses single-model interpretability |
| **Random Forest** | Supervised: classification (categorical) & regression (numeric) | Numeric and categorical features (native in theoretical trees; library implementations often require encoding); no scaling needed | Strong out-of-the-box generalisation via bootstrap + subspace decorrelation; OOB error estimation | Cannot extrapolate beyond training target range in regression; black-box ensemble |
| **Extra Trees** | Supervised: classification (categorical) & regression (numeric) | Same as Random Forest (numeric and categorical; no feature scaling required) | Computationally faster and lower variance than Random Forest due to random split points | Higher bias than Random Forest; degrades when many noisy/irrelevant features exist |
| **Isolation Forest** | Unsupervised: anomaly / outlier detection (no target feature) | Continuous numeric features (categorical features require encoding; monotonic scaling does not alter split order) | Isolates anomalies near the root in few random splits without density or distance estimation | Struggles when anomalies are not isolated in sparse regions or when many irrelevant features mask splits |
| **AdaBoost (Adaptive Boosting)** | Supervised: binary classification $y \in \{-1, +1\}$ (extensions like AdaBoost.R2 support regression) | Depends on weak learner (decision stumps handle numeric and categorical features without scaling) | Combines high-bias weak learners (e.g., decision stumps) into complex boundaries; reduces bias and variance | Highly sensitive to label noise and outliers due to exponential weight concentration on hard instances |
| **Gradient Boosting Machine (GBM)** | Supervised: regression (numeric via MSE) & classification (categorical via log-likelihood) | Shallow regression trees handle numeric and categorical features without scaling | Directly minimises differentiable loss functions via residual fitting; high predictive accuracy | Sequential training cannot be parallelised across trees; prone to overfitting without shrinkage ($\nu$) or early stopping |
| **Stacking & Heterogeneous Ensembles** | Supervised: classification & regression | Determined by the most restrictive base learner (often requires imputation, numeric encoding, and scaling) | Combines complementary inductive biases across diverse model families via a learned meta-learner | High training, deployment, and maintenance complexity; requires held-out data $D_h$ to avoid meta-level leakage |

---

## 3. Individual model summaries

### 3.1 Bagging (Bootstrap Aggregating) and Sub-bagging

#### Main idea and assumptions
**Bagging** (*bootstrap aggregating*) trains multiple instances of a base learner independently on random bootstrap samples of the training data and aggregates their predictions ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), Section 4.4.5.1, p. 159 / PDF p. 2; [07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slides 9–11).

- **Inductive bias and bias–variance mechanism:** Using the expected squared error decomposition for an estimator $\hat{f}(x)$,
  $$E\left[(y - \hat{f}(x))^2\right] = \text{Bias}^2[\hat{f}(x)] + \text{Var}[\hat{f}(x)] + \sigma^2$$
  bagging reduces **variance** ($\text{Var}[\hat{f}(x)]$) through voting or averaging, making the ensemble less sensitive to small fluctuations in the training set ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slides 24–25). However, **bootstrapping does not reduce the bias of base learners**: because bagging does not change the capacity or functional form of the base model, the average bias across base learners equals the bias of a single learner ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 25).
- **Assumptions:** Base learners should be **high-variance, low-bias** (complex models that tend to overfit, such as unpruned decision trees or neural networks) and **unstable** (small changes in training data produce substantially different models) so that bootstrap perturbations yield diverse predictions ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slides 24, 27, 31; [fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), p. 159 / PDF p. 2).
- **Theoretical independence assumption (Condorcet's jury theorem):** For a binary classification ensemble of $M$ **independent** classifiers each with error probability $p < 0.5$, the probability that simple majority voting makes an incorrect prediction is given by the binomial tail ([quiz-07-ensemble-learning.md](quiz-07-ensemble-learning.md), Q22; [fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), p. 158 / PDF p. 1):
  $$P(\text{ensemble error}) = \sum_{k = \lceil M/2 \rceil}^{M} \binom{M}{k} p^k (1 - p)^{M - k}$$
  *(In practice, base models trained on overlapping samples from the same dataset are only partially independent, so actual error reduction is bounded by residual correlation between trees.)*

#### How it learns and predicts

**Training:**
1. **Create $M$ training subsets** from the original dataset $D$ of $N$ instances:
   - **Standard Bagging (Bootstrap sampling):** For each model $i \in \{1, \dots, M\}$, sample $N$ instances **with replacement** from $D$ to form $D_i$ ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slides 10–11). The probability that a specific instance $x$ is included at least once in $D_i$ is:
     $$P_r(x \in D_i) = 1 - \left(1 - \frac{1}{N}\right)^N \approx 1 - e^{-1} \approx 0.6322$$
     Thus, on average slightly less than $2/3$ ($63.22\%$) of unique instances appear in each bootstrap sample, while the remaining $\approx 36.78\%$ slots are filled by **duplicates** (and $\approx 36.78\%$ of unique instances are omitted, or *out-of-bag*). Different bootstrap samples contain different duplicates, causing each model to emphasise classifying its duplicated instances correctly and thereby creating model diversity ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 11).
   - **Sub-bagging:** Sample a smaller subset of instances **without replacement** ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 12; [fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), Footnote 26, p. 159 / PDF p. 2).
2. **Train base learners:** Induce one base model $M_i$ independently on each subset $D_i$.

**Prediction:**
- **Classification:** Aggregate individual predictions via **majority voting** (or weighted majority voting).
- **Regression:** Aggregate individual numeric predictions using a measure of central tendency—specifically the **median** (or mean). Course materials emphasise that the **median is preferred over the mean** because the mean is more heavily affected by outlier predictions from individual models ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 9; [fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), p. 159 / PDF p. 2).

**Out-of-Bag (OOB) Error Evaluation:**
Because each bootstrap sample omits $\approx 36.8\%$ of the training instances, generalisation performance can be estimated without a separate validation set or cross-validation ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slides 33–35):
1. For a specific training instance $x_j$, identify the subset of ensemble models that did **not** include $x_j$ in their bootstrap sample (*out-of-bag models*).
2. Aggregate the predictions of only those out-of-bag models for $x_j$.
3. Compare the aggregated OOB prediction with the true target $y_j$, and average this error over all training instances $j = 1, \dots, N$.

> **Worked Example (OOB Evaluation & Independent Majority Voting — Quiz 7 Q19 & Q22):**
> - **OOB model count ([quiz-07-ensemble-learning.md](quiz-07-ensemble-learning.md), Q19):** Suppose 3 bootstrap samples of size 10 are drawn from instances $0\text{–}9$. If instance `6` appears twice in Dataset 1, zero times in Dataset 2, and zero times in Dataset 3, then **2 models** (Models 2 and 3) did not see instance `6` during training and can be used to verify generalisation performance on instance `6`.
> - **Majority voting error ([quiz-07-ensemble-learning.md](quiz-07-ensemble-learning.md), Q22):** With $M = 25$ independent binary classifiers each having error rate $p = 0.35$, the ensemble errs if at least $13$ models err:
>   $$P(\text{error}) = \sum_{k=13}^{25} \binom{25}{k} (0.35)^k (0.65)^{25-k} \approx 0.0604 \approx 0.06$$

#### Supported data and tasks
- **Descriptive features:** Bagging is a meta-algorithm; supported feature types depend entirely on the base learner:
  - *Decision-tree bagging:* Supports continuous/discrete numeric and nominal/ordinal categorical features natively without scaling.
  - *Neural network or SVM bagging ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slides 43, 46; [quiz-07-ensemble-learning.md](quiz-07-ensemble-learning.md), Q12, Q15):* Requires numeric inputs, one-hot/dummy encoding for nominal categorical features, and feature scaling (standardisation/normalisation).
- **Target features:** Categorical (binary and multiclass classification) and continuous numeric (regression).
- **Outputs:** Class labels (via majority vote), class vote proportions (scores/estimated probabilities across committee members), or continuous numeric values (median or mean across members).

#### Data quality and preprocessing
Because bagging wraps a chosen base learner, its data quality behaviour combines the **sampling/aggregation mechanics of bagging** with the **properties of the base learner** (see Section 3.2 for decision-tree specifics).

| Issue | Effect on the standard model and why | Relevant extension and how it helps | Preprocessing / mitigation and tradeoffs |
| --- | --- | --- | --- |
| **1. Missing values** | Bagging itself does not impute missing values. If base learners cannot handle missing descriptive values (e.g., NNs, SVMs), training or prediction fails. Missing targets cannot be used in supervised aggregation. | Using decision trees with missing-value handling (e.g., surrogate splits or fractional routing) allows the ensemble to proceed without dropping rows. | Drop instances with missing targets; impute missing descriptive values within each training split/fold (trades off potential imputation bias against data retention). |
| **2. Outliers** | *Input outliers:* Base learners sensitive to extreme inputs may produce erratic predictions on some bags, though bootstrap omission excludes a given outlier from $\approx 36.8\%$ of bags. *Target outliers (regression):* Extreme target values skew individual regression models and pull mean aggregation. | **Median aggregation** for regression ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 9; [fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), p. 159) prevents extreme predictions in a minority of models from dominating the ensemble output. | Filter extreme erroneous outliers in training data or use robust base learners; median aggregation mitigates outlier predictions at inference time. |
| **3. Noise** | Measurement or label noise increases the variance of complex base learners that fit noisy training instances. Bagging averages out unsystematic variance across bootstrap samples, making it much more resilient to noise than boosting. | **Sub-bagging** or **subspace sampling** (Random Forests) further decorrelates base models so fewer models memorise the same noisy points. | Clean obvious label errors; avoid base learners with excessive capacity if dataset size is tiny. |
| **4. Skew class distributions** | Each bootstrap sample preserves the training set's class imbalance on average; minority instances may even be underrepresented or absent in small sub-bags, causing majority voting to favour the majority class. | **Weighted majority voting** or stratified/balanced bootstrap sampling (outside standard course formulation) adjusts class influence. | Apply class-balanced resampling (oversampling/undersampling) inside the training split or tune classification vote thresholds using precision/recall/F1. |
| **5. Irrelevant features** | If base learners are sensitive to irrelevant features (e.g., $k$-NN or unregularised models), bootstrap sampling alone gives every model the same full feature set, so irrelevant features affect all members similarly. | **Subspace sampling** (sampling feature subsets, as in Random Forests or SVM subspace ensembles, [quiz-07-ensemble-learning.md](quiz-07-ensemble-learning.md), Q15c) varies the features seen by each model/split. | Perform feature selection inside the training pipeline or use base learners that select informative splits (e.g., decision trees). |
| **6. Features with different value ranges** | Bootstrap sampling and voting/median aggregation are scale-invariant, so sensitivity depends solely on the base learner (scale-invariant for trees; scale-sensitive for SVMs, NNs, and distance-based learners). | None needed for tree base learners; homogeneous NN/SVM ensembles require scaled inputs. | Standardise or normalise features using training-split statistics when using scale-sensitive base learners. |
| **7. Different descriptive feature data types** | Bagging operates on instance indices and model outputs; it has no native mechanism to convert data types for base learners that only accept numeric vectors. | Using decision trees as base learners natively supports mixed numeric and categorical features ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 31). | Encode nominal features (e.g., one-hot) and ordinal features appropriately when using numeric-only base learners (NNs, SVMs). |

- **Redundant features & multicollinearity:** In standard bagging without feature sampling, dominant or collinear features are available to every bootstrap model, making base models more correlated and limiting variance reduction.
- **High dimensionality & limited training data:** Small datasets combined with sub-bagging leave very few unique instances per model, increasing individual model bias ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 26). High dimensionality is handled well if combined with subspace sampling.

#### Important choices and variants
- **Learned parameters vs. hyperparameters:** Learned parameters are the internal parameters of all $M$ base learners. Hyperparameters include the **number of ensemble members ($M$)**, the **sampling method** (bootstrapping vs. sub-bagging), the **subset/bag size**, and the **base learner hyperparameters**.
- **Effect of bag size in Sub-bagging ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slides 12, 26–27):**
  - **Small sub-bags:** Base learners train on less information and do not see the full data distribution, which **increases bias** (and averaging cannot cancel out this bias). However, base learners are **more diverse** (stronger decorrelation), yielding stronger variance reduction from averaging. Beneficial when the base learner has **very high variance and low bias** ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 27).
  - **Large sub-bags:** More information is used to train each base model, resulting in **lower bias**, but base learners become **more correlated**, giving less variance reduction than small bags. Preferred when the base learner has **moderate variance and higher bias** ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slides 26–27).
  - **Bootstrapping vs. Sub-bagging:** Sub-bagging is faster to train (fewer instances per model), but **bootstrap sampling tends to outperform sub-bagging** for the same number of ensemble members because sub-bagging increases bias as a function of bag size ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slides 12, 27).
- **Homogeneous Neural Network and SVM Bagging Ensembles:**
  - *Neural Network Ensembles ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 43; [quiz-07-ensemble-learning.md](quiz-07-ensemble-learning.md), Q12):* Combining multiple neural networks via bagging improves generalisation only when members behave differently—achieved by training on different bootstrap subsets, varying network architectures, or using different weight initialisations/optimisers. Training identical architectures on the exact same data subset defeats the purpose of ensembling ([quiz-07-ensemble-learning.md](quiz-07-ensemble-learning.md), Q12).
  - *SVM Ensembles ([quiz-07-ensemble-learning.md](quiz-07-ensemble-learning.md), Q15):* Because SVMs are deterministic and relatively stable, diversity is increased by training members on different data subsets, using different regularisation parameter $C$ values, using different kernels across members, or—for non-linearly separable problems—omitting kernels and training linear SVMs on small subsets of instances and descriptive features ([quiz-07-ensemble-learning.md](quiz-07-ensemble-learning.md), Q15).

#### Strengths, limitations, and practical use
- **Strengths:** Effectively stabilises high-variance models and reduces overfitting; base models train **independently in parallel**; fast inference when parallelised ([quiz-07-ensemble-learning.md](quiz-07-ensemble-learning.md), Q20); OOB error provides free validation when bootstrapping is used.
- **Limitations:** Cannot fix an underfitting (high-bias) base learner; increases memory footprint and deployment maintenance by $M\times$ compared to a single model; sacrifices single-model interpretability.

---

### 3.2 Random Forests

#### Main idea and assumptions
A **Random Forest** is a homogeneous bagging ensemble that uses **unpruned decision trees** as base learners and adds **subspace sampling** (random feature subset selection at each split) to further decorrelate the trees ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slides 28–30; [fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), Section 4.4.5.1, p. 159 / PDF p. 2).

- **Why decision trees are ideal for bagging ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 31; [fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), p. 159 / PDF p. 2):**
  1. Decision trees are **high-variance, low-bias** learners that are very sensitive to changes in the training set: a small change in the bootstrap sample can change the root or upper-level split feature, creating a ripple effect throughout the subtrees.
  2. They support both classification and regression, work with numeric and categorical features, and make **few statistical assumptions** (no normality or linearity assumptions).
- **Why subspace sampling is needed:** Even with bootstrap sampling, if a dataset has a few very strong predictive features, every decision tree would select those same features near the root, producing highly correlated trees. Restricting each node split to a random subset of $m$ features forces trees to explore different features and patterns, increasing diversity and reducing ensemble variance ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slides 29–30, 32).

#### How it learns and predicts

**Training ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 30):**
For each tree $T_i$ ($i = 1, \dots, M$):
1. **Bootstrap sample:** Draw a bootstrap sample $D_i$ of size $n$ by sampling **with replacement** from the training set $D$.
2. **Induce an unpruned tree with subspace sampling:** At each node to be split:
   - Randomly select a subset of $m$ features from all $d$ descriptive features (typically $m = \sqrt{d}$ for classification, [07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 30 Footnote 1).
   - Among those $m$ candidate features, select the feature and split point that yield the best split performance (e.g., highest information gain / Gini reduction for classification, or variance/sum-of-squared-errors reduction for regression).
   - Continue growing the tree without post-pruning until every leaf is pure (contains instances of the same class, i.e., overfit to $D_i$) or reaches a minimum node size ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 30; [quiz-07-ensemble-learning.md](quiz-07-ensemble-learning.md), Q9).

**Prediction:**
- **Classification:** Pass the query instance down all $M$ trees and return the **majority vote** (or weighted vote) across the trees ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 9; [quiz-07-ensemble-learning.md](quiz-07-ensemble-learning.md), Q21).
- **Regression:** Return the **median** (preferred over the mean for outlier robustness) or mean of the $M$ tree predictions ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 9; [fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), p. 159 / PDF p. 2).

> **Worked Example (Random Forest Majority Voting — Quiz 7 Q21):**
> Given 3 decision trees predicting *Heart Disease* (`True` / `False`) for query `Chest Pain = Yes, Blood Pressure = High, Age = Above 50` ([quiz-07-ensemble-learning.md](quiz-07-ensemble-learning.md), Q21; [8159d73f86354ae7.png](quiz-07-ensemble-learning-images/8159d73f86354ae7.png)):
> - **Tree 1:** Splits on `Chest Pain` $\rightarrow$ `Yes` $\rightarrow$ predicts **`True`**.
> - **Tree 2:** Splits on `Blood Pressure` (`Medium, High`) $\rightarrow$ `Chest Pain` (`Yes`) $\rightarrow$ predicts **`True`**.
> - **Tree 3:** Splits on `Age` (`>= 50`) $\rightarrow$ `Blood Pressure` (`Medium, High`) $\rightarrow$ predicts **`False`**.
> - **Majority vote:** 2 votes for `True`, 1 vote for `False` $\rightarrow$ Ensemble predicts **`True`**.

#### Supported data and tasks
- **Descriptive features:** Continuous and discrete numeric features, and nominal and ordinal categorical features ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 31). No feature scaling is required because axis-aligned split criteria depend only on value ordering or category membership. *(Note: While decision tree algorithms theoretically split categorical features directly into multi-way or subset branches, libraries such as `scikit-learn` require nominal categorical features to be one-hot or ordinal encoded prior to fitting.)*
- **Target features:** Categorical (binary and multiclass classification) and continuous numeric (regression) ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 31).
- **Outputs:** Class labels, class probability estimates (fraction of trees voting for each class or average leaf class distributions), or continuous numeric predictions.

#### Data quality and preprocessing
Slide 36 ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf)) explicitly prompts for the effects of the seven course data quality issues on Random Forests:

| Issue | Effect on the standard model and why | Relevant extension and how it helps | Preprocessing / mitigation and tradeoffs |
| --- | --- | --- | --- |
| **1. Missing values** | Decision trees are listed as working well with missing values ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 31): during tree induction, impurity reduction can be scaled by the fraction of non-missing instances, and instances with missing split values can be routed via majority/fractional branches or surrogate splits. Missing targets cannot be used in supervised training. | Surrogate splits, fractional instance routing (C4.5-style), or proximity-matrix iterative imputation in Random Forest implementations. | Impute missing descriptive values (e.g., median/mode or k-NN imputation inside the training split) if using an implementation that does not support missing values natively; drop rows with missing targets. |
| **2. Outliers** | *Input outliers:* Classification trees are robust to input outliers ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 31) because axis-aligned thresholds depend on rank order/proportions rather than Euclidean distances, and an extreme point falls into an existing outer leaf. *Target outliers (regression):* Extreme target values distort variance-reduction splits in the ~63.2% of trees whose bag includes the outlier. | **Median aggregation** across trees ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 9; [fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), p. 159) shields regression predictions from trees distorted by target outliers. | Inspect and remove corrupted target outliers in regression training splits; use median aggregation or robust loss criteria (e.g., MAE split criterion). |
| **3. Noise** | Deep unpruned trees grown to pure leaves ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 30) will fit noisy instances in their individual bootstrap sample, but because different trees see different bootstrap samples and feature subsets, isolated noisy splits cancel out under majority voting ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slides 31–32). | Increasing minimum sample split or limiting maximum tree depth prevents individual trees from isolating single noisy instances into 1-sample leaves. | Tune `min_samples_split` / `min_samples_leaf` (trading off slightly higher tree bias for lower sensitivity to label noise). |
| **4. Skew class distributions** | In skewed classification datasets, bootstrap samples contain mostly majority-class instances. Standard information-gain/Gini splits and unweighted majority voting favour the majority class, causing poor recall on the minority class. | **Class-weighted splits and weighted voting**, or **Balanced Random Forests** (undersampling the majority class in each bootstrap sample). | Resample the training split (oversample minority / undersample majority), use class weights, or adjust the voting probability threshold using precision–recall curves. |
| **5. Irrelevant features** | At each node, only $m$ randomly selected features are evaluated ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 30). If $m$ contains at least one informative feature, the impurity criterion ignores the irrelevant ones. However, if the proportion of irrelevant features is extremely high, some random subsets of size $m$ will contain *only* irrelevant features, forcing noisy splits. | Increasing $m$ (the subspace sample size) when many features are irrelevant increases the probability that at least one informative feature is included at each split. | Filter clearly irrelevant features beforehand or tune $m$ via OOB/cross-validation error (increasing $m$ reduces bad splits but increases correlation between trees). |
| **6. Features with different value ranges** | **Insensitive (invariant) to monotonic feature scaling.** Each tree evaluates one feature at a time using threshold comparisons ($x_j \le \theta$), and information gain / Gini / variance reduction depends only on how instances partition across the threshold, not on the feature's units or magnitude. | None required; standard Random Forests are natively scale-invariant. | No normalisation or standardisation is needed. |
| **7. Different descriptive feature data types** | Natively supports continuous/discrete numeric and nominal/ordinal categorical features ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 31) via threshold splits ($x_j \le \theta$) for numeric/ordinal data and category-membership splits for nominal data. | High-cardinality nominal features can bias unregularised information gain toward many-way splits; using **information gain ratio** or binary subset splits mitigates this. | In software libraries requiring numeric matrices, one-hot encode low-cardinality nominal features (noting that extreme one-hot expansion can dilute subspace sampling of size $m$). |

- **Redundant features & multicollinearity:** Highly correlated features do not cause numerical instability in trees; moreover, subspace sampling allows different collinear features to be chosen in different trees, spreading importance across them.
- **High dimensionality:** Scales very well to large numbers of input features ([quiz-07-ensemble-learning.md](quiz-07-ensemble-learning.md), Q8) because each split only evaluates $m \approx \sqrt{d}$ features rather than all $d$ features.
- **Extrapolation limitation in regression:** Because every regression tree leaf predicts a constant derived from training targets in that region, **Random Forests cannot extrapolate** outside the minimum and maximum target values seen in the training data, even when combined in an ensemble ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 31).

#### Important choices and variants
- **Number of trees ($M$):** Increase $M$ until the **out-of-bag (OOB) error** stops reducing ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 33). Adding more trees does not cause a Random Forest to overfit (it further stabilises the variance reduction), though it increases training and prediction time linearly.
- **Subset (bag) size:** Using smaller subsamples reduces training time per tree but produces smaller trees with less capacity to overfit—which is **undesirable in bagging** because smaller trees have higher bias—and requires more trees in the ensemble ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 33).
- **Number of random features ($m$):** Controls the bias–variance tradeoff of the forest. Usually set to $m = \sqrt{d}$ (where $d$ is total features); Slide 33 notes that Random Forest performance is generally insensitive to small variations around $m \approx 3, 4$ ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slides 30, 33). Very small $m$ decorrelates trees strongly (lower variance) but increases individual tree bias; $m = d$ reduces Random Forest to standard bagged decision trees.
- **Minimum sample split (`min_samples_split`) & Pruning:** Standard Random Forests grow trees until leaves are pure without post-pruning ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 30; [quiz-07-ensemble-learning.md](quiz-07-ensemble-learning.md), Q9). Stopping splits earlier produces smaller trees and less overfitting of individual trees *(see Section 4 for a note on the wording on Slide 33)*.

#### Strengths, limitations, and practical use
- **Strengths:** Captures complex non-linear decision boundaries (e.g., spiral boundary with just 6 trees, [07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 32); handles mixed data types and large feature spaces; parallelisable across trees; built-in OOB generalisation estimate; insensitive to feature scaling.
- **Limitations:** Loses the transparent rule-set interpretability of a single decision tree; higher memory and inference cost than a single tree; cannot extrapolate trends in regression ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 31).

---

### 3.3 Extra Trees (Extremely Randomized Trees)

#### Main idea and assumptions
**Extra Trees** (*Extremely Randomized Trees*) are a tree-ensemble variant similar to Random Forests that introduces **stronger randomisation** during split construction to further decorrelate base learners and reduce variance, while using the **entire training dataset** rather than bootstrap bags ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slides 37–38).

- **Bias–variance tradeoff vs. Random Forests ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 38):**
  - **Higher bias** than Random Forests because split thresholds are chosen at random rather than optimised over all possible thresholds.
  - **Lower variance** than Random Forests because random split points produce stronger decorrelation across trees (and using the full dataset avoids the sample-omission bias of bootstrap bags).

#### How it learns and predicts

**Training ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 37):**
For each tree in the ensemble:
1. **Use all available training data** (no bootstrap sampling / bagging).
2. At each node to be split:
   - Randomly select a subset of $m = \sqrt{\text{\#features}}$ candidate features.
   - For each selected feature, **randomly select split points** (rather than searching for the optimal threshold).
   - Evaluate those candidate random splits and select the split that is **best among the $m$ features**.
3. **Stop splitting** when the resultant number of samples in a node is less than a given minimum threshold (or the leaf is pure).

**Prediction:**
- Identical to Random Forests: majority voting for classification and median/mean aggregation for regression.

#### Supported data and tasks
- **Descriptive features:** Continuous/discrete numeric and categorical features; no feature scaling required.
- **Target features:** Categorical (classification) and numeric (regression).
- **Outputs:** Class labels, class vote proportions, or numeric predictions.

#### Data quality and preprocessing
Extra Trees share the decision-tree data properties of Random Forests (Section 3.2), with two important differences caused by **full-dataset training** and **random split points**:

| Issue | Effect on the standard model and why | Relevant extension and how it helps | Preprocessing / mitigation and tradeoffs |
| --- | --- | --- | --- |
| **1. Missing values** | Same as Random Forests: requires tree-level missing-value routing or prior imputation; no OOB samples exist since all data is used. | Surrogate or fractional routing in tree induction. | Impute missing descriptive values inside the training split; drop missing targets. |
| **2. Outliers** | *Input outliers:* Uniformly drawing a random split point across a feature's min–max range is affected if an extreme input outlier stretches the feature domain, making random thresholds more likely to fall in the empty gap between the bulk of data and the outlier. *Target outliers:* Same as Random Forests. | Clipping or winsorising extreme input domains, or using median aggregation for regression targets. | Remove or clip extreme input outliers before training so random split selection is not wasted in sparse outlier tails. |
| **3. Noise** | Random split thresholds combined with a stopping sample-count threshold ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 37) prevent trees from placing exact optimal cuts around noisy points, reducing variance. | Increasing the minimum sample threshold for stopping splits smooths leaf predictions further. | Tune the minimum sample threshold to balance bias and noise resilience. |
| **4. Skew class distributions** | Full-dataset training preserves the exact class skew in every tree; majority voting still favours the dominant class. | Class-weighted impurity evaluation when choosing the best among the $m$ random splits. | Resample training data or apply class weights. |
| **5. Irrelevant features** | **More sensitive to noisy/irrelevant features than Random Forests ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 38):** because split thresholds are chosen randomly even on informative features, an informative feature with a poor random threshold may fail to stand out clearly from noisy features. | Increasing $m$ (up to $m = d$) gives each node more informative candidates to compare. | Perform feature selection to remove noisy/irrelevant features before training, or prefer Random Forests when many noisy features are present ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 38). |
| **6. Features with different value ranges** | **Insensitive to linear rescaling:** random thresholds drawn within each feature's domain and impurity evaluations are unaffected by units or linear scaling. | None needed. | No scaling required. |
| **7. Different descriptive feature data types** | Supports numeric and categorical features like Random Forests. | Random subset partitioning for categorical features. | Encode categorical features if required by the library. |

#### Important choices and variants
- **Hyperparameters:** Number of trees ($M$), number of random features per split ($m$, default $\sqrt{\text{\#features}}$), and the **stopping threshold** (minimum number of samples required to split a node, [07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 37).
- **Note on OOB error:** Because standard Extra Trees use **all available training data** (not bootstrap bags, [07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 37), out-of-bag error is not available unless bootstrapping is explicitly enabled; cross-validation is used instead.

#### Strengths, limitations, and practical use
- **Strengths:** **Computationally less expensive** than Random Forests (no exhaustive search over sorted feature values for optimal thresholds at every node) with very similar predictive performance and lower variance ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 38).
- **Limitations:** Higher bias than Random Forests; **performs slightly worse than Random Forests when many noisy/irrelevant features are present** ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 38).

---

### 3.4 Isolation Forests

#### Main idea and assumptions
An **Isolation Forest** is an **unsupervised** ensemble learning model designed specifically for **anomaly (outlier) detection** ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 39).

- **Core assumption and inductive bias:** Anomalies are "few and different." When a dataset is recursively partitioned using random binary splits, **anomalies require fewer splits to isolate** than normal points—meaning anomalies are isolated **closer to the root** of an isolation tree (short path length), whereas normal points lie in dense clusters and require **more splits** to isolate (deep path length) ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slides 39, 42).

#### How it learns and predicts

**Training (Building Isolation Trees, [07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 40):**
For each isolation tree in the ensemble:
1. **Sub-bagging:** Randomly select a data subsample of size $n$ without replacement from the dataset.
2. **Grow a random binary tree:** Recursively split the subsample by:
   - Randomly selecting a descriptive feature.
   - Randomly selecting a split point as any value in the domain (between the minimum and maximum) of that feature.
   - Partitioning points into left and right child nodes until **every data point in the subsample is isolated** in its own leaf (or a maximum tree height limit is reached).

**Prediction / Anomaly Scoring ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slides 40–41):**
1. Parse each data point $\mathbf{x}$ through every isolation tree and record its **path length (depth) $h(\mathbf{x})$**, defined as the number of edges traversed from the root node to the terminating (external) node where $\mathbf{x}$ is isolated.
2. Calculate the **average depth $\overline{h(\mathbf{x})}$** (or $E[h(\mathbf{x})]$) of $\mathbf{x}$ across all isolation trees in the forest.
3. Compute the **normalization constant $c(n)$**, which is the expected average path length of an unsuccessful search in a Binary Search Tree (BST) with $n$ samples ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 41):
   $$c(n) = 2H(n - 1) - \frac{2(n - 1)}{n}$$
   where $H(k)$ is the $k$-th harmonic number *(written as "harmonic mean" on Slide 41)*:
   $$H(k) = 1 + \frac{1}{2} + \frac{1}{3} + \dots + \frac{1}{k} \approx \ln(k) + 0.5772156649$$
4. Calculate the **anomaly score $s(\mathbf{x}, n) \in (0, 1]$** ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 40):
   $$s(\mathbf{x}, n) = 2^{-\frac{\overline{h(\mathbf{x})}}{c(n)}}$$

**Interpretation of the Anomaly Score ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slides 40, 42):**
- **$\overline{h(\mathbf{x})} \ll c(n) \implies s(\mathbf{x}, n) \to 1$:** Strong **anomaly** (isolated very close to the root).
- **$\overline{h(\mathbf{x})} \approx c(n) \implies s(\mathbf{x}, n) \approx 0.5$:** **Borderline case** (or, if *all* scores in the dataset are close to $0.5$, the entire sample does not have clearly distinct anomalies).
- **$\overline{h(\mathbf{x})} \gg c(n) \implies s(\mathbf{x}, n) \ll 0.5$ (closer to $0$):** **Normal observation** (deep inside dense clusters).

> **Worked Example (Isolation Score Calculation):**
> Suppose sub-bag size is $n = 6$ (as in the 6-point diagram on [07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 42).
> 1. Harmonic number: $H(5) = 1 + \frac{1}{2} + \frac{1}{3} + \frac{1}{4} + \frac{1}{5} = \frac{137}{60} \approx 2.2833$.
> 2. Normalization constant: $c(6) = 2(2.2833) - \frac{2(5)}{6} = 4.5667 - 1.6667 = 2.90$.
> 3. For the red outlier point $(13, 11)$ isolated at depth $1$ in the first split ($\overline{h(\mathbf{x})} = 1.0$):
>    $$s(\mathbf{x}, 6) = 2^{-1.0 / 2.90} = 2^{-0.3448} \approx 0.79 > 0.5 \quad (\text{flagged as an anomaly})$$
> 4. For a normal point $(8, 4)$ isolated at average depth $\overline{h(\mathbf{x})} = 4.0$:
>    $$s(\mathbf{x}, 6) = 2^{-4.0 / 2.90} = 2^{-1.379} \approx 0.38 < 0.5 \quad (\text{normal observation})$$

#### Supported data and tasks
- **Learning setting & task:** **Unsupervised anomaly / outlier detection** ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 39). Target features are **not applicable** (no labels are used during training).
- **Descriptive features:** Continuous/discrete numeric features (since splits select a random value in the domain $[\min, \max]$ of a feature, [07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 40). Categorical features require encoding before use.
- **Outputs:** Continuous **anomaly score** $s(\mathbf{x}, n) \in (0, 1]$ (which is a normalized path-length score, *not* a calibrated probability) and binary anomaly/normal flags after applying a threshold (e.g., $> 0.5$ or top-$k\%$ contamination).

#### Data quality and preprocessing

| Issue | Effect on the standard model and why | Relevant extension and how it helps | Preprocessing / mitigation and tradeoffs |
| --- | --- | --- | --- |
| **1. Missing values** | *Missing targets:* Not applicable (unsupervised). *Missing descriptive values:* A point cannot be routed down a random threshold test $x_j < \theta$ when $x_j$ is missing. | Routing instances with missing values down both branches weighted by child subtree sizes (outside course slides). | Impute missing values prior to fitting; note that naive mean/median imputation pulls incomplete points toward the dense centre, making them appear more "normal" (deeper path). |
| **2. Outliers** | **Outliers are the detection signal itself** rather than a nuisance: points in sparse extremes of feature domains are isolated in very few random cuts ($\overline{h(\mathbf{x})}$ small, score $\to 1$). However, if anomalies are abundant (*swamping* / *masking* in large samples), they can form dense clusters that take more steps to isolate. | **Sub-bagging** ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 40) with a small subsample size $n$ (typically $n = 256$) isolates anomalies effectively by preventing anomaly clusters from becoming dense inside a single tree. | Keep sub-bag size $n$ moderate rather than using the full dataset when anomalies may cluster. |
| **3. Noise** | Random measurement noise on normal instances shifts their feature values; extreme measurement errors become indistinguishable from true anomalies and receive high anomaly scores. Label noise is not applicable. | Averaging path lengths $\overline{h(\mathbf{x})}$ over many isolation trees stabilises scores against single lucky/unlucky random cuts. | Increase the number of isolation trees to stabilise $\overline{h(\mathbf{x})}$; domain-validate high-scoring points to separate sensor glitches from true anomalies. |
| **4. Skew class distributions** | Supervised class imbalance is **not applicable**. In anomaly detection, extreme rarity of anomalies relative to normal data is **expected and beneficial**, as rare points are easily isolated near the root. | Not applicable (sub-bagging handles rare anomalies naturally). | Set the decision threshold on $s(\mathbf{x}, n)$ according to the expected anomaly rate or domain false-positive tolerance. |
| **5. Irrelevant features** | Because each split selects a feature **completely at random** ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 40) without an impurity criterion, many irrelevant noise features cause trees to waste splits on dimensions where anomalies look normal, inflating path lengths $\overline{h(\mathbf{x})}$ and masking anomalies. | Feature selection or weighting features by kurtosis/kurtosis-based subsampling (beyond course scope). | Remove irrelevant or constant features before fitting an Isolation Forest, especially in high-dimensional data. |
| **6. Features with different value ranges** | **Insensitive to linear rescaling:** each split first picks a feature $j$ uniformly at random and then picks a threshold uniformly within $[\min(x_j), \max(x_j)]$. Multiplying or shifting $x_j$ scales its domain identically without changing split probabilities or tree topology. | None needed. | No feature standardisation or normalisation is required. |
| **7. Different descriptive feature data types** | Standard Isolation Forest assumes numeric domains to sample split points between $\min$ and $\max$ ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 40). Nominal features do not have a continuous interval domain. | Extended categorical isolation trees (splitting on random category subsets). | Encode categorical features numerically (noting that one-hot encoding creates many sparse binary features that can dilute random feature selection). |

#### Important choices and variants
- **Hyperparameters:** Number of isolation trees ($M$, e.g., $100$) and **sub-bag sample size ($n$)** (controls $c(n)$ and how finely dense vs. sparse regions are resolved), plus the anomaly score threshold (around $0.5$ or tuned to domain contamination).
- **No learned parameters in the parametric sense:** Each tree stores only the randomly chosen feature indices, random split thresholds, and subtree depths.

#### Strengths, limitations, and practical use
- **Strengths:** Does not require labelled anomalies; avoids expensive pairwise distance or density calculations (unlike $k$-NN or kernel density anomaly detectors), making it very fast ($O(M \cdot n \log n)$ training, $O(M \log n)$ scoring) and memory-efficient.
- **Limitations:** Struggles when many irrelevant features dilute random feature selection, or when anomalies lie inside local hollows surrounded by normal points along axis-parallel projections.

---

### 3.5 Adaptive Boosting (AdaBoost) and Sampling-Based Boosting

#### Main idea and assumptions
In **boosting**, base learners are trained **sequentially**, where each new learner in the sequence focuses on the training instances that previous learners misclassified, and predictions are combined via a **confidence-weighted vote** ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slides 13–15, 50, 55–56; [fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), Section 4.4.5.2, pp. 159–161 / PDF pp. 2–4).

- **Inductive bias and bias–variance mechanism:** Unlike bagging (which uses complex, low-bias, high-variance learners), boosting uses **simple weak learners**—most commonly **decision stumps** (one-level decision trees consisting of a single root split that immediately terminates into leaves, [07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slides 47–48). A single decision stump underfits severely, having **high bias and low variance** ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 24; [quiz-07-ensemble-learning.md](quiz-07-ensemble-learning.md), Q10). By sequentially adding weak learners that target remaining errors and combining them via weighted voting, boosting **reduces bias** (less underfitting) for the overall ensemble while **also reducing variance** through weighted averaging over the weak learners ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slides 24, 49, 63; [quiz-07-ensemble-learning.md](quiz-07-ensemble-learning.md), Q11).
- **Other possible weak learners ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 47):** Besides decision stumps, weak learners can be linear regression models, logistic regression, or linear SVM/SVR.

#### How it learns and predicts
The course materials present two closely related formulations of discrete boosting:
1. **Sampling-based Boosting with normalised multiplicative updates** ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), Section 4.4.5.2, pp. 160–162 / PDF pp. 3–5; [07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slides 14–21), where instance weights $\mathbf{w}$ define a probability distribution used to sample a *replicated training set* of size $n$ at each iteration.
2. **Standard AdaBoost optimizing weighted training error directly** ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slides 50–62), where all $n$ instances (100% of the dataset, i.e., proportion `1`, [quiz-07-ensemble-learning.md](quiz-07-ensemble-learning.md), Q4) are passed to the weak learner with weights $w_t(i)$ and updated exponentially before normalisation.

*(Mathematically, the two weight-update formulas produce the **exact same normalised weights** at every step.)*

**Algorithm (AdaBoost / Boosting for Binary Classification, $y_i \in \{-1, +1\}$):**
1. **Initialise instance weights:** Assign an equal weight to all $n$ training instances ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slides 17, 56; [fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), p. 160 / PDF p. 3):
   $$w_1(i) = \frac{1}{n} \quad \text{for } i = 1, \dots, n$$
2. **For each iteration $t = 1, \dots, T$:**
   - **a. Train weak classifier $h_t$:**
     - *Direct weighted-error formulation ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slides 51–52, 56):* Select the feature $j^*$ and threshold $\theta^*$ (and inequality direction $p_j$) that minimise the weighted misclassification error over all $n$ training instances:
       $$(j^*, \theta^*) = \operatorname*{argmin}_{j, \theta} \sum_{i=1}^{n} w_t(i) \, \mathbb{I}(y_i \ne h_t(x_i))$$
       where $\mathbb{I}(\cdot)$ is the indicator function ($1$ if misclassified, $0$ if correct).
     - *Cumulative-sum stump-training algorithm ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slides 53–54):* For each dimension $j = 1, \dots, d$, sort samples $x_i$ in ascending order, compute the $n$ cumulative weighted label sums $w_{\text{cum}}^j(i) = \sum_{k=1}^{i} w_k y_k$, and place threshold $\theta_j$ at the extremum (maximum absolute peak) of $w_{\text{cum}}^j$ (whose sign gives the inequality direction $p_j$). The global extremum across all $d$ dimensions gives $(\theta^*, j^*)$.
     - *Sampling-based formulation ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), p. 160; [07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 14):* Sample a replicated training set of $n$ instances proportional to $w_t(i)$ and induce $h_t$ on that sample.
   - **b. Compute weighted error rate $\epsilon_t$** on the complete training set ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slides 15, 56; [fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), p. 161 / PDF p. 4, Footnote 29):
     $$\epsilon_t = \sum_{i=1}^{n} w_t(i) \, \mathbb{I}(y_i \ne h_t(x_i))$$
   - **c. Compute model voting weight (confidence factor) $\alpha_t$** ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slides 15, 56; [fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), Eq. 4.14, p. 161 / PDF p. 4):
     $$\alpha_t = \frac{1}{2} \ln\left(\frac{1 - \epsilon_t}{\epsilon_t}\right)$$
     *Key properties of $\alpha_t$ ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 57; [quiz-07-ensemble-learning.md](quiz-07-ensemble-learning.md), Q16c):*
     - $\alpha_t$ is a **non-linear** decreasing function of $\epsilon_t$.
     - If $\epsilon_t < 0.5$, $\alpha_t > 0$ (higher voting weight as $\epsilon_t \to 0$).
     - If $\epsilon_t = 0.5$ (random guessing), $\alpha_t = \frac{1}{2}\ln(1) = 0$.
     - If $\epsilon_t > 0.5$ (e.g., $55\%$ error rate), $\alpha_t < 0$ (**negative voting weight**, effectively inverting the binary classifier's prediction).
   - **d. Update and normalise instance weights** so that $\sum_{i=1}^n w_{t+1}(i) = 1$:
     - *Exponential form + explicit normalisation ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 56):*
       $$w_{t+1}(i) = \begin{cases} w_t(i) \, e^{\alpha_t} & \text{if } y_i \ne h_t(x_i) \text{ (misclassified — weight increased)} \\ w_t(i) \, e^{-\alpha_t} & \text{if } y_i = h_t(x_i) \text{ (correctly classified — weight decreased)} \end{cases}$$
       followed by dividing each $w_{t+1}(i)$ by $\sum_{k=1}^n w_{t+1}(k)$ so the weights sum to $1$.
     - *Equivalent self-normalising multiplicative form ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), Eqs. 4.12–4.13, p. 161 / PDF p. 4; [07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 15):*
       $$w_{t+1}(i) = \begin{cases} w_t(i) \times \frac{1}{2 \epsilon_t} & \text{if misclassified} \\ w_t(i) \times \frac{1}{2 (1 - \epsilon_t)} & \text{if correctly classified} \end{cases}$$
       *(Because $\sum_{\text{wrong}} w_t(i) = \epsilon_t$ and $\sum_{\text{right}} w_t(i) = 1 - \epsilon_t$, the updated weights automatically sum to $\epsilon_t \frac{1}{2\epsilon_t} + (1-\epsilon_t)\frac{1}{2(1-\epsilon_t)} = \frac{1}{2} + \frac{1}{2} = 1$, [fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), Footnote 30, p. 161).*

**Prediction ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slides 56, 61–62):**
For a query instance $\mathbf{x}$, each weak classifier outputs $h_t(\mathbf{x}) \in \{-1, +1\}$, and the ensemble predicts the sign of the confidence-weighted sum:
$$H(\mathbf{x}) = \operatorname{sign}\left(\sum_{t=1}^{T} \alpha_t h_t(\mathbf{x})\right)$$
(Predict $-1$ when $\sum_{t=1}^T \alpha_t h_t(\mathbf{x}) < 0$, else predict $+1$; for general categorical targets, return the class with the highest sum of confidence weights $\alpha_t$, [fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), p. 161 / PDF p. 4).

> **Worked Example 1 (AdaBoost Error, Weight Update, and Prediction — Slides 59–62 & Quiz 7 Q17–Q18):**
> - **Iteration 1 on 10 instances ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 59):** Each instance starts with $w_0(i) = 0.1$. Stump $h_0$ misclassifies instances 8, 9, and 10:
>   - Weighted error: $\epsilon_0 = 0.1 + 0.1 + 0.1 = 0.3$.
>   - Voting weight: $\alpha_0 = \frac{1}{2} \ln\left(\frac{1 - 0.3}{0.3}\right) = \frac{1}{2}\ln(2.3333) \approx 0.4236$.
>   - Updated normalised weights:
>     - Correctly classified (instances 1–7): $w_1(i) = 0.1 \times \frac{1}{2(1 - 0.3)} = \frac{0.1}{1.4} \approx 0.0714$ (`0.071`).
>     - Misclassified (instances 8–10): $w_1(i) = 0.1 \times \frac{1}{2(0.3)} = \frac{0.1}{0.6} \approx 0.1667$ (`0.167`).
> - **Quiz 7 Q17 (Calculating $\epsilon_t$):** Given 4 instances with weights $[0.3, 0.25, 0.15, 0.3]$, actual labels $Y = [1, -1, -1, 1]$, and predictions $H = [1, 1, 1, 1]$, instances 2 and 3 are misclassified ($Y \ne H$). Thus $\epsilon = 0.25 + 0.15 = \mathbf{0.40}$ ([quiz-07-ensemble-learning.md](quiz-07-ensemble-learning.md), Q17).
> - **Quiz 7 Q18 (AdaBoost Weighted Vote Prediction):** With labels `-1 = Faulty`, `1 = Good`, predictions $[h_1, h_2, h_3] = [-1, -1, 1]$, and voting weights $[\alpha_1, \alpha_2, \alpha_3] = [0.25, 0.20, 0.50]$:
>   $$\sum_{t=1}^3 \alpha_t h_t(\mathbf{x}) = (0.25)(-1) + (0.20)(-1) + (0.50)(1) = -0.25 - 0.20 + 0.50 = +0.05 > 0 \implies \text{Predict } +1 \text{ (\textbf{Good})}$$
>   Even though two of the three models voted `Faulty` ($-1$), Model 3 had a higher confidence weight ($0.50 > 0.25 + 0.20$) and outweighed them ([quiz-07-ensemble-learning.md](quiz-07-ensemble-learning.md), Q18).

#### Supported data and tasks
- **Descriptive features:** Depends on the weak learner. With **decision stumps** (the standard course choice), numeric (continuous/discrete) and categorical (nominal/ordinal) features are supported without scaling. With linear/logistic regression or linear SVM weak learners ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 47), numeric encoding and feature scaling are required.
- **Target features:** Standard AdaBoost is formulated for **binary classification** with $y_i \in \{-1, +1\}$ ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slides 51–62). The textbook notes that boosting can be adapted to **continuous regression targets** (e.g., **AdaBoost.R2**, Drucker 1997) by basing weight updates on prediction error magnitude rather than misclassification and aggregating via a **weighted mean** ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), p. 161 / PDF p. 4, Footnote 31).
- **Outputs:** Binary class label $H(\mathbf{x}) \in \{-1, +1\}$ and uncalibrated real-valued margin score $\sum_{t=1}^T \alpha_t h_t(\mathbf{x})$ (or weighted mean for regression extensions).

#### Data quality and preprocessing
Slides 64–65 ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf)) and Quiz 7 (Q6, Q16) specifically highlight AdaBoost's behaviour under data quality issues:

| Issue | Effect on the standard model and why | Relevant extension and how it helps | Preprocessing / mitigation and tradeoffs |
| --- | --- | --- | --- |
| **1. Missing values** | Decision stumps cannot evaluate $x_{ij} \le \theta$ if $x_{ij}$ is missing unless tree missing-value routing is implemented. Missing target labels $y_i$ cannot be used because every weight update depends on $\mathbb{I}(y_i \ne h_t(x_i))$. | Stumps with surrogate/default branch routing or missingness indicator splits. | Drop rows with missing targets; impute missing descriptive features inside the training split before boosting. |
| **2. Outliers** | **Highly sensitive to outliers ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slides 64–65; [quiz-07-ensemble-learning.md](quiz-07-ensemble-learning.md), Q6, Q16b):** an outlier that lies on the wrong side of the class boundary is repeatedly misclassified by early weak learners, so its weight $w_t(i)$ grows exponentially ($w_t(i) e^{\alpha_t}$). Subsequent weak learners are forced to construct specific splits just to classify that outlier, distorting the decision boundary. | Gentle AdaBoost / BrownBoost or switching to **Gradient Boosting** with a learning rate $\nu$ or sign-of-error loss ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), p. 168 / PDF p. 11). | **Remove outliers and noisy instances from the training data prior to training AdaBoost** ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 64), or cap the number of boosting iterations $T$. |
| **3. Noise** | **Highly sensitive to label and measurement noise ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slides 64–65; [quiz-07-ensemble-learning.md](quiz-07-ensemble-learning.md), Q16d):** mislabelled or noisy boundary instances are consistently misclassified by low-error models, accumulating massive instance weights and causing later stumps to overfit the noise. | Early stopping (limiting $T$) prevents the ensemble from dedicating dozens of late-stage stumps to isolated noisy points. | Clean mislabelled training instances or apply outlier/noise filtering inside the training fold; prefer Random Forests when data is heavily noisy. |
| **4. Skew class distributions** | Raised on Slide 64 ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf)): because initial weights are uniform ($w_1(i) = 1/n$), the first weak learner can achieve a low weighted error $\epsilon_1$ simply by favouring the majority class. Although misclassified minority instances receive increased weights in later iterations, minority boundary points can also be treated like noise/outliers if the class overlap is high. | Cost-sensitive AdaBoost (AdaCost) or initializing $w_1(i)$ inversely proportional to class frequencies. | Balance class weights at initialisation ($w_1(i)$) or resample the training split before boosting; evaluate with F1/ROC-AUC rather than accuracy. |
| **5. Irrelevant features** | With decision stumps, each iteration searches all $d$ features and selects the single feature $j^*$ that minimises weighted error ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slides 51–53), naturally ignoring irrelevant features as long as informative features achieve lower weighted error. | None needed for decision stumps (implicitly performs greedy feature selection at each step). | If $T$ is excessively large and data is noisy, irrelevant features may eventually be picked to fit noise; use early stopping on validation error. |
| **6. Features with different value ranges** | With decision stumps ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slides 48, 53), each split sorts instances along one dimension $j$ at a time and finds the cumulative weight extremum; this ordering is **completely invariant to feature scaling**. (If linear/SVM weak learners are used instead, scaling is required.) | None needed for decision stumps. | No scaling required for stump-based AdaBoost; standardise features if using linear/logistic/SVM weak learners ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 47). |
| **7. Different descriptive feature data types** | Decision stumps handle continuous/discrete numeric features via threshold cuts ($x_j \le \theta$) and categorical features via category-equality tests. Linear/SVM weak learners require numeric features. | None needed for decision stumps. | Encode nominal features if required by the software library or when using linear weak learners. |

#### Important choices and variants
- **Learned parameters:** For each of the $T$ weak learners (stumps), the selected feature $j^*$, threshold $\theta^*$, split polarity $p_j$, and the model voting weight $\alpha_t$.
- **Hyperparameters:**
  - **Number of weak learners / iterations ($T$):** Increasing $T$ reduces ensemble bias and allows simple stumps to approximate complex boundaries ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 63; [quiz-07-ensemble-learning.md](quiz-07-ensemble-learning.md), Q11). However, unlike Random Forests, **increasing $T$ too far in AdaBoost can lead to overfitting** ([quiz-07-ensemble-learning.md](quiz-07-ensemble-learning.md), Q16f), especially when noise or outliers are present.
  - **Weak learner complexity:** Course materials emphasise **preferring simple models (decision stumps) over complex models** ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 64). Simple models speed up training and inference without hurting ensemble accuracy, whereas complex base learners overfit quickly and leave few meaningful errors for subsequent iterations to correct.

#### Strengths, limitations, and practical use
- **Strengths:** Turns very fast, simple weak learners ($O(d \cdot n \log n)$ to sort features once for cumulative-sum stump finding, [07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 53) into a highly accurate classifier with low bias and low variance; performs automatic feature selection when using stumps.
- **Limitations:** **Sequential training** (model $t$ depends on weights from model $t-1$, so training cannot be parallelised across models); vulnerable to outliers and label noise ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slides 64–65).

---

### 3.6 Gradient Boosting Machines (GBM)

#### Main idea and assumptions
A **Gradient Boosting Machine (GBM)** is a sequential boosting algorithm that minimises bias error by training each new weak learner (typically a shallow regression tree or regression decision stump) directly on the **errors (*model residuals*)**—more generally, the negative gradient of a differentiable loss function—of the preceding ensemble, rather than reweighting or resampling training instances ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slides 66–68; [fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), Section 4.4.5.3, pp. 164–168 / PDF pp. 7–11).

- **Key distinction from AdaBoost ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 66; [fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), pp. 164, 168 / PDF pp. 7, 11):**
  1. **No instance weights $w[i]$** are adjusted at each iteration as in AdaBoost.
  2. **No explicit voting/averaging aggregation step:** instead, GBM is an **additive model** where the final prediction is the initial naive prediction plus the sum of sequential residual corrections ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), Eq. 4.18 & Footnote 33, p. 165 / PDF p. 8).

#### How it learns and predicts

**Training & Prediction (Regression Mode — MSE / $L_2$ Loss, [07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slides 68–77; [fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), pp. 164–168 / PDF pp. 7–11):**
1. **Step 1 — Initialise with a naive base model ($F_0$ / $\mathbb{M}_0$):** Predict the overall average target value across all $n$ training instances ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 70; [fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), Eq. 4.15, p. 164 / PDF p. 7):
   $$F_0(\mathbf{x}) = \bar{y} = \frac{1}{n} \sum_{i=1}^{n} y_i \quad \left(\text{written in textbook as } \mathbb{M}_0(\mathbf{d}) = \frac{1}{n}\sum_{i} t_i\right)$$
2. **Step 2 — Iteratively fit trees to residuals ($m = 1, \dots, M$):**
   - **a. Compute residuals $r_m$** (the errors of the current ensemble model $F_{m-1}$ on each training instance $i$, [07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slides 71, 74; [fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), p. 164):
     $$r_{m, i} = y_i - F_{m-1}(\mathbf{x}_i)$$
   - **b. Train a regression tree $\gamma_m(\mathbf{x})$ (or $\mathbb{M}_{\Delta m}(\mathbf{d})$)** using the descriptive features $\mathbf{x}_i$ as inputs and the computed residuals $r_{m, i}$ as the continuous target values ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slides 68, 72, 75).
   - **c. Update the additive ensemble model** by adding the new tree's residual prediction scaled by the **learning rate (*incremental shrinkage*)** $\nu \in (0, 1]$ (denoted $\nu$ on [07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 73 and $\alpha$ in [fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), Eq. 4.19, p. 168 / PDF p. 11):
     $$F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \nu \, \gamma_m(\mathbf{x}) \quad \left(\text{unscaled when } \nu = 1\text{: } \mathbb{M}_m(\mathbf{d}) = \mathbb{M}_{m-1}(\mathbf{d}) + \mathbb{M}_{\Delta m}(\mathbf{d})\right)$$
     *(Note: Slide 76 contains a slide typo writing $F_2 = F_1 + \nu F_0$ in the header, whereas the actual formula shown on Slide 77 and textbook Eq. 4.19 is $F_2 = F_1 + \nu \gamma_2$.)*
3. **Final Additive Prediction ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), Eqs. 4.18–4.19, pp. 165, 168 / PDF pp. 8, 11):**
   $$F_M(\mathbf{x}) = F_0(\mathbf{x}) + \sum_{m=1}^{M} \nu \, \gamma_m(\mathbf{x})$$

**Objectives for Regression vs. Classification ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 67; [fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), p. 168 / PDF p. 11):**
- **Gradient Boosting Regression:** Minimises **Mean Squared Error (MSE / $L_2$ loss)**, where the negative gradient with respect to $F(\mathbf{x})$ is proportional to the residual $y - F(\mathbf{x})$.
- **Gradient Boosting Classifier:** Minimises **Log-Likelihood ($LL$) loss** ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 67):
  - For binary target $y \in \{0, 1\}$, transform to $\bar{y} = 2y - 1 \in \{-1, +1\}$.
  - Given model log-odds / raw score prediction $\hat{y}$, the loss is:
    $$LL = \log\left(1 + e^{-2\bar{y}\hat{y}}\right)$$
  - In classification, the initial model predicts the prior likelihood (or log-odds) of the target levels, and subsequent regression trees predict continuous corrections to those likelihoods/log-odds ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), p. 168 / PDF p. 11).

> **Worked Example (GBM Regression Step-by-Step — Textbook Table 4.15, pp. 165–166 & Slides 70–75):**
> - **Textbook Bike Rentals ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), Table 4.15, p. 166 / PDF p. 9, with $\nu = 1$):**
>   - Average rentals across the 10 days is $\mathbb{M}_0(\mathbf{d}) = 1287.1$.
>   - For Day 1 ($\text{TEMP} = 4, t_1 = 602$), the initial residual is $t_1 - \mathbb{M}_0 = 602 - 1287.1 = -685.1$.
>   - First regression stump $\mathbb{M}_{\Delta 1}$ splits at $\text{TEMP} \le 9.5$, predicting $-460.9$ for $\text{TEMP} \le 9.5$ and $+691.4$ for $\text{TEMP} > 9.5$.
>   - Updated ensemble prediction for Day 1: $\mathbb{M}_1(\mathbf{d}_1) = 1287.1 + (-460.9) = 826.2$.
>   - Next residual for Day 1: $t_1 - \mathbb{M}_1(\mathbf{d}_1) = 602 - 826.2 = -224.2$, which becomes the target for $\mathbb{M}_{\Delta 2}$.
> - **Slides Regression Example ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slides 72–73, with learning rate $\nu$):**
>   - Initial prediction $F_0 = \bar{y}$. First tree splits at $x \le 49.5$ with leaf values $\gamma_1 = 6.0$ (if $x \le 49.5$) and $\gamma_1 = -5.9$ (otherwise).
>   - Combined model after 1 tree: $F_1(x) = F_0 + 6.0\nu$ if $x \le 49.5$, and $F_0 - 5.9\nu$ otherwise.

#### Supported data and tasks
- **Descriptive features:** Continuous/discrete numeric and nominal/ordinal categorical features (via the base regression trees); no feature scaling is required.
- **Target features:** Continuous numeric targets (regression) and categorical targets (binary and multiclass classification; note that **even for classification, the weak learners $\gamma_m$ are regression trees** because they predict continuous residuals / likelihood corrections, [fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), p. 168 / PDF p. 11).
- **Outputs:** Continuous numeric predictions for regression; raw log-odds scores, class likelihoods/probabilities, and class labels for classification.

#### Data quality and preprocessing

| Issue | Effect on the standard model and why | Relevant extension and how it helps | Preprocessing / mitigation and tradeoffs |
| --- | --- | --- | --- |
| **1. Missing values** | Base regression trees require missing-value routing (e.g., surrogate/default branch splits) or prior imputation. Missing target values cannot be used because residuals $y_i - F_{m-1}(\mathbf{x}_i)$ cannot be computed. | Tree-level missing-value routing (standard in modern GBM libraries). | Drop instances with missing targets; impute missing descriptive features inside the training split if the tree implementation lacks native missingness routing. |
| **2. Outliers** | *Input outliers:* Handled well by axis-aligned tree splits. *Target outliers (regression):* Under standard **MSE ($L_2$) loss**, a single extreme target $y_i$ produces a huge residual $r_{m,i} = y_i - F_{m-1}(\mathbf{x}_i)$ that dominates squared-error split criteria, causing early trees to chase the outlier ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), p. 168 / PDF p. 11). | **Two textbook extensions ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), p. 168 / PDF p. 11):** (1) **Learning rate / incremental shrinkage ($\alpha$ or $\nu \in (0, 1]$)** dampens the impact of later error corrections; (2) **Training trees to predict only the sign of the errors** ($\operatorname{sign}(y_i - F_{m-1}(\mathbf{x}_i))$, corresponding to $L_1$/MAE loss) rather than their magnitudes prevents large outlier residuals from dominating. | Use a small learning rate $\nu$, use sign-of-residual ($L_1$/Huber) loss for heavy-tailed regression targets, or filter corrupted target outliers in the training split. |
| **3. Noise** | Because each new tree directly fits the remaining residuals, continuing to add trees for too many iterations causes the ensemble to memorise measurement and label noise (overfitting). | **Incremental shrinkage ($\nu \in (0, 1]$)** combined with **early stopping** (stopping when validation error stops improving, [07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 76) and shallow trees (decision stumps). | Tune $M$ and $\nu$ jointly using validation data or cross-validation; restrict tree depth (e.g., stumps or shallow trees). |
| **4. Skew class distributions** | In log-likelihood classification ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 67), the initial log-odds $F_0$ is heavily skewed toward the majority class, and total log-likelihood gradient is dominated by the majority class unless weighted. | Instance/class-weighted log-likelihood loss. | Apply class weights to the loss function, resample the training split, or tune the probability decision threshold. |
| **5. Irrelevant features** | Shallow regression trees / stumps greedily pick the feature that best reduces residual variance at each split, ignoring irrelevant features during early iterations. | Subspace feature sampling (stochastic gradient boosting) and early stopping prevent late iterations from splitting on irrelevant features to fit tiny noisy residuals. | Use early stopping and shallow trees so late trees do not fit residual noise using irrelevant features. |
| **6. Features with different value ranges** | **Insensitive to feature scaling** because the base learners are decision trees/stumps whose threshold splits depend only on feature ordering. | None needed. | No feature scaling required. |
| **7. Different descriptive feature data types** | Base regression trees support continuous/discrete numeric and categorical descriptive features natively. | Ordered/target encoding or native categorical subset splits in modern GBM variants. | Encode nominal features if required by the library. |

- **Extrapolation limitation:** Like Random Forests, tree-based GBMs sum constant leaf values from axis-aligned regions and therefore **cannot extrapolate linear trends** beyond the bounding box of training feature values.

#### Important choices and variants
- **Learned parameters:** Initial constant prediction $F_0$, and the split features, thresholds, and leaf correction values $\gamma_m$ of all $M$ trees.
- **Hyperparameters:**
  - **Learning rate / incremental shrinkage ($\nu$ or $\alpha \in (0, 1]$):** Scales each tree's correction ($F_m = F_{m-1} + \nu \gamma_m$, [07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 73; [fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), Eq. 4.19, p. 168 / PDF p. 11). Smaller $\nu$ slows down learning and makes the ensemble more robust to outliers and overfitting, but requires a larger number of trees $M$.
  - **Number of trees / iterations ($M$):** Adding too many trees overfits the training residuals; stopped when validation prediction stops improving ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 76).
  - **Base tree depth:** Usually restricted to very shallow trees or **decision stumps** (1 level deep, [07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 68; [fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), p. 165 / PDF p. 8) to keep individual learners weak (low variance, high bias).
  - **Loss function:** MSE ($L_2$) for regression, sign-of-error ($L_1$) for outlier-robust regression ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), p. 168 / PDF p. 11), or Log-Likelihood ($LL$) for classification ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 67).

#### Strengths, limitations, and practical use
- **Strengths:** Directly optimises differentiable loss functions; achieves state-of-the-art accuracy on tabular classification and regression tasks; flexible extensions for outlier robustness (shrinkage and sign-of-residuals).
- **Limitations:** Strictly sequential training (each tree requires the residuals of the previous tree, so trees cannot be trained in parallel); more hyperparameters to tune ($\nu, M$, tree depth) than Random Forests.

---

### 3.7 Stacking (Stacked Generalisation) and Heterogeneous Ensembles

#### Main idea and assumptions
- **Homogeneous vs. Heterogeneous Ensembles ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slides 45–46):**
  - Every machine learning algorithm has an **inductive bias**, so no single algorithm is most accurate for all problems (No Free Lunch).
  - A **mixture of homogeneous experts** (combining models of the same family, such as Random Forests or Neural Network ensembles) improves accuracy over a single model, but its predictive power is **still limited by the shared inductive bias** of that single algorithm family ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 45).
  - A **mixture of heterogeneous experts** (*heterogeneous ensemble*) combines **different machine learning algorithms** (e.g., Logistic Regression + Neural Networks + Classification Trees; or Naive Bayes + SVM + Classification Tree + $k$-NN with different $k$ + shallow feedforward NNs with different optimisers, [07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 46) to leverage complementary strengths and counteract individual inductive biases.
- **Stacking (*Stacked Generalisation*, [07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slides 22–23; [quiz-07-ensemble-learning.md](quiz-07-ensemble-learning.md), Q1):**
  - Rather than combining base models via fixed majority voting or median aggregation, **stacking trains a machine learning model called a *meta-learner*** to learn how to combine the predictions of the base models ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 22).
  - Stacking ensembles can be **either homogeneous or heterogeneous** ([quiz-07-ensemble-learning.md](quiz-07-ensemble-learning.md), Q1), though they are most commonly used with a diverse (often heterogeneous) group of base learners ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 22).
  - **Crucial assumption against data leakage:** The meta-learner is trained on the **predictions made by the base models** (not the raw descriptive features) using a **held-out dataset $D_h$ that was NOT used to train the base learners** ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slides 22–23). If the meta-learner were trained on in-sample training predictions, it would simply assign all weight to whichever base model overfitted (memorised) the training set most severely.

#### How it learns and predicts

**Training ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 23):**
1. **Split the dataset:** Divide the available data into a base **training set** and a **hold-out set $D_h$** (containing $n_h$ instances).
2. **Train base learners:** Train a user-specified number $K$ of base learners $M_1, \dots, M_K$ on the base training set.
3. **Generate out-of-sample meta-features:** Use each trained base learner $M_k$ to generate an out-of-sample prediction $\hat{y}_{ik}$ for every instance $i \in \{1, \dots, n_h\}$ in the hold-out set $D_h$, forming a meta-dataset ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 23):
   $$D_{\text{meta}} = \begin{bmatrix} \hat{y}_{11} & \hat{y}_{12} & \dots & \hat{y}_{1K} & t_1 \\ \hat{y}_{21} & \hat{y}_{22} & \dots & \hat{y}_{2K} & t_2 \\ \vdots & \vdots & \ddots & \vdots & \vdots \\ \hat{y}_{n_h 1} & \hat{y}_{n_h 2} & \dots & \hat{y}_{n_h K} & t_{n_h} \end{bmatrix}$$
   *(Note: Slide 23 illustrates $K=3$ base learners with a typo in the column headers showing $\hat{y}_1, \hat{y}_3, \hat{y}_3$ instead of $\hat{y}_1, \hat{y}_2, \hat{y}_3$.)*
4. **Train the meta-learner:** Train the meta-learner on $D_{\text{meta}}$, using the base predictions $(\hat{y}_{i1}, \dots, \hat{y}_{iK})$ as descriptive features and the true hold-out targets $t_i$ as the target feature.

**Prediction:**
1. Given a new query instance $\mathbf{x}$, compute the prediction $\hat{y}_k = M_k(\mathbf{x})$ from each of the $K$ base learners.
2. Feed the vector of base predictions $(\hat{y}_1, \dots, \hat{y}_K)$ into the trained meta-learner to produce the final learned prediction.

#### Supported data and tasks
- **Descriptive features:**
  - *Base level:* Each base learner processes the original dataset, so a heterogeneous ensemble requires preprocessing that satisfies its most restrictive member (or separate model-specific pipelines). For example, combining Classification Trees, SVMs, $k$-NN, Naive Bayes, and Neural Networks ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 46) requires numeric encoding and feature scaling for the SVM, $k$-NN, and Neural Network members.
  - *Meta-learner level:* The meta-learner receives only the base predictions $\hat{y}_{ik}$ (numeric values, class probabilities, or predicted class labels), which are already on a commensurate scale (e.g., $[0, 1]$ probabilities or target units).
- **Target features:** Categorical (classification) and continuous numeric (regression).
- **Outputs:** Determined by the meta-learner (class labels, calibrated probabilities, or numeric predictions).

#### Data quality and preprocessing

| Issue | Effect on the standard model and why | Relevant extension and how it helps | Preprocessing / mitigation and tradeoffs |
| --- | --- | --- | --- |
| **1. Missing values** | If any base learner cannot handle missing descriptive values (e.g., SVM, $k$-NN, Neural Network, Logistic Regression), it cannot produce a prediction $\hat{y}_{ik}$, leaving a missing input for the meta-learner. | Model-specific preprocessing pipelines or imputation before base training. | Impute missing descriptive values using statistics fitted **only on the base training split** (never across $D_h$ or test data); drop missing targets. |
| **2. Outliers** | Outlier-sensitive base learners (e.g., linear regression, $k$-NN, AdaBoost) may produce distorted predictions, but the meta-learner can learn to down-weight unreliable base learners on $D_h$. | Using a regularised or robust meta-learner (e.g., ridge regression or logistic regression) prevents extreme base predictions from dominating. | Clean training outliers or include outlier-robust base learners (e.g., Random Forests) in the heterogeneous mix. |
| **3. Noise** | Base models that overfit noise on the base training set will perform poorly on the out-of-sample hold-out set $D_h$; the meta-learner observes their poor accuracy on $D_h$ and assigns them lower weight. | **Out-of-fold / cross-validated stacking** (generating out-of-sample predictions across $k$ folds rather than a single small hold-out $D_h$) gives the meta-learner a larger, less noisy training set. | Ensure strict separation between the base training set and $D_h$ ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 23) so overfitted base models are penalised by the meta-learner. |
| **4. Skew class distributions** | If $D_h$ is small and unstratified, rare classes may have very few examples in $D_h$, causing the meta-learner to underfit minority-class combinations. | Stratified splitting when creating $D_h$, and passing predicted **class probabilities** rather than hard labels to the meta-learner. | Use **stratified sampling** when dividing the training set and hold-out set $D_h$, and apply class weights or threshold tuning at the meta-learner level. |
| **5. Irrelevant features** | Irrelevant raw features affect sensitive base learners (e.g., $k$-NN), lowering their accuracy on $D_h$; the meta-learner only sees the $K$ base predictions, not the raw features ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 22). | Including feature-selecting base learners (trees, L1-regularised models) allows the meta-learner to rely on models that ignored the irrelevant features. | Apply feature selection for distance-based or unregularised base learners within the base training split. |
| **6. Features with different value ranges** | Scale-sensitive base learners (SVMs, $k$-NN, NNs) are distorted by unscaled raw features; however, base predictions $\hat{y}_{ik}$ fed into the meta-learner all predict the same target $t$ and share the same scale. | Per-model feature scaling pipelines at the base level. | Standardise/normalise raw features using base-training-split parameters for scale-sensitive base learners. |
| **7. Different descriptive feature data types** | In heterogeneous ensembles ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 46), tree/Naive Bayes models handle categorical data naturally, whereas SVMs and NNs require numeric inputs. | Separate feature encoding pipelines per base model family. | Encode categorical features for numeric-only base learners; if base learners output categorical class labels, encode them for the meta-learner or pass continuous class probabilities instead. |

- **Multicollinearity at the meta-learner level:** Because all base learners $M_1, \dots, M_K$ are trying to predict the same target $t$, their predictions $(\hat{y}_1, \dots, \hat{y}_K)$ in $D_{\text{meta}}$ are often **highly correlated (multicollinear)**. Using a simple, regularised meta-learner (such as ridge/lasso linear or logistic regression) avoids coefficient instability and meta-level overfitting.
- **Limited training data:** Splitting off a hold-out set $D_h$ ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 23) reduces the data available to train the base learners and leaves only $n_h$ instances to train the meta-learner, which is problematic on small datasets unless out-of-fold cross-validation stacking is used.

#### Important choices and variants
- **Choice of base learners:** Should have **diverse inductive biases** (e.g., combining information-based trees, similarity-based $k$-NN, probabilistic Naive Bayes, and error-based SVMs/Neural Networks, [07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slides 45–46).
- **Choice of meta-learner:** Usually a simple, low-variance model (e.g., logistic regression for classification, linear/ridge regression for regression) because the base learners have already extracted the complex non-linear patterns.
- **Hold-out split proportion ($D_h$):** Must balance giving base learners enough training data against giving the meta-learner enough out-of-sample predictions $n_h$.

#### Strengths, limitations, and practical use
- **Strengths:** Overcomes the inductive-bias ceiling of any single algorithm family ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 45); learns *which* base models are reliable without relying on naive equal-weight voting ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 80).
- **Limitations:** Highest deployment, latency, and software maintenance cost of all ensemble methods (every base model must be maintained and executed at inference time before the meta-learner can predict, [07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 80); completely black-box.

---

## 4. Exam and comparison points

### Key distinctions across ensemble families

| Dimension | Bagging (e.g., Random Forest) | Boosting (AdaBoost / GBM) | Stacking |
| --- | --- | --- | --- |
| **Training order** | **Parallel / independent** (models can be trained simultaneously) | **Sequential** (each model depends on the previous model's errors/residuals) | **Two-stage hierarchical** (base models in parallel $\rightarrow$ meta-learner on $D_h$) |
| **Typical base learners** | **Complex, low-bias, high-variance** (e.g., unpruned decision trees, deep NNs) | **Simple, high-bias, low-variance weak learners** (e.g., decision stumps) | **Diverse models** (often heterogeneous families, or homogeneous with varied hyperparameters) |
| **Primary bias–variance effect** | **Reduces variance** (less overfitting); **does not reduce bias** ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slides 24–25) | **Reduces bias** (less underfitting) **and reduces variance** via weighted averaging ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 24) | Combines inductive biases to reduce both bias and variance |
| **How predictions are combined** | Majority vote (classification) or **median** / mean (regression) | Confidence-weighted vote $\sum \alpha_t h_t(\mathbf{x})$ (AdaBoost) or additive residual sum $F_0 + \sum \nu \gamma_m(\mathbf{x})$ (GBM) | **Trained meta-learner** taking base predictions as inputs ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slides 22–23) |
| **Sensitivity to noise & outliers** | **Resilient** (outliers omitted from $\approx 36.8\%$ of bags; median aggregation resists outlier trees) | **AdaBoost is highly sensitive** (exponential weight growth on outliers); **GBM mitigated via $\nu$ and sign-of-error** | Resilient if the meta-learner is trained on clean out-of-sample $D_h$ |
| **Fast prediction / parallel inference** | **Preferred for fast parallel training & prediction** ([quiz-07-ensemble-learning.md](quiz-07-ensemble-learning.md), Q20) | Fast per stump at test time, but sequential during training | Slowest (must run all heterogeneous base models + meta-learner) |

### Common misconceptions and quiz-tested traps

1. **Trap: "Duplicate instances in bootstrap samples should be avoided because they cause ensemble overfitting."**
   - **Why it is false ([quiz-07-ensemble-learning.md](quiz-07-ensemble-learning.md), Q2; [07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 11; [fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), p. 159 / PDF p. 2):** Sampling with replacement *intentionally* creates duplicates ($\approx 36.8\%$ of each sample) so that each bootstrap sample omits $\approx 36.8\%$ of the dataset and emphasises different duplicated instances. This difference between samples is precisely what makes the base models diverse.
2. **Trap: "Decision trees in a Random Forest must be post-pruned to prevent overfitting."**
   - **Why it is false ([quiz-07-ensemble-learning.md](quiz-07-ensemble-learning.md), Q3, Q9; [07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slides 24, 30–31):** In bagging and Random Forests, it is **desirable** for individual trees to grow unpruned until leaves are pure (overfit, low bias, high variance), because voting/averaging across decorrelated trees reduces the variance whereas post-pruning would increase bias that bagging cannot remove.
3. **Trap: "In AdaBoost, a model with an error rate above 50% (e.g., 55%) gets a weight of zero."**
   - **Why it is false ([quiz-07-ensemble-learning.md](quiz-07-ensemble-learning.md), Q16c; [07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 57):** Since $\alpha_t = \frac{1}{2}\ln\left(\frac{1-\epsilon_t}{\epsilon_t}\right)$, when $\epsilon_t = 0.55 > 0.5$, $\frac{1-\epsilon_t}{\epsilon_t} = \frac{0.45}{0.55} < 1$, so $\alpha_t = \frac{1}{2}\ln(0.818) \approx -0.10 < 0$. It receives a **negative voting weight** (flipping its binary prediction).
4. **Trap: "The first model in a boosting ensemble trains on a small subsample of the dataset."**
   - **Why it is false ([quiz-07-ensemble-learning.md](quiz-07-ensemble-learning.md), Q4):** The first model receives a proportion of **`1`** ($100\%$ of the dataset size, with uniform initial weights $w_1(i) = 1/n$).
5. **Trap: "Regression ensembles use majority voting."**
   - **Why it is false ([quiz-07-ensemble-learning.md](quiz-07-ensemble-learning.md), Q14; [07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 9; [fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), p. 158 / PDF p. 1):** Majority voting is for categorical targets (classification). Regression ensembles aggregate continuous predictions using the **median** (preferred in bagging/random forests for outlier resistance) or **weighted mean / additive sum** (in boosting/GBM).
6. **Trap: "Stacking ensembles must be heterogeneous."**
   - **Why it is false ([quiz-07-ensemble-learning.md](quiz-07-ensemble-learning.md), Q1):** Stacking can combine **homogeneous** base models (e.g., same algorithm with different hyperparameters or feature subsets) as well as **heterogeneous** base models.
7. **Trap: "Extra Trees use bootstrap samples just like Random Forests."**
   - **Why it is false ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 37):** Standard Extra Trees use **all available training data** (not bootstrap bags) and achieve decorrelation through random feature subsets ($m = \sqrt{d}$) *plus* **randomly selected split points**.

### Flagged conflicts and slide/quiz ambiguities

> [!IMPORTANT]
> Separate the **recorded course / Moodle answer** from the **technically rigorous explanation** on the following points:

1. **Quiz 7 Question 3 Moodle Explanation ("Bias from overfitting"):**
   - **Recorded quiz key ([quiz-07-ensemble-learning.md](quiz-07-ensemble-learning.md), Q3):** Marks *"When training a bagging ensemble learner, it is better for the individual learners overfit the training data than to underfit"* as **True**, with the Moodle explanation: *"The bias from overfitting will be reduced during the voting"*.
   - **Technical qualification ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slides 24–25):** The True answer is correct, but the Moodle explanation confuses **bias** with **variance**. Overfitting corresponds to **high variance** (and low bias), whereas underfitting corresponds to **high bias**. As Slides 24–25 explicitly prove via the bias–variance decomposition, **bootstrapping/bagging reduces variance** through voting/averaging and **does not reduce the bias of base learners**.
2. **Quiz 7 Question 5 ("Learning rate in a boosting ensemble"):**
   - **Recorded quiz key ([quiz-07-ensemble-learning.md](quiz-07-ensemble-learning.md), Q5):** Marks *"During the learning process for a boosting ensemble, the learning rate must be controlled so that the weights assigned to instances do not become excessively large"* as **False**, with the Moodle explanation: *"No learning rate, weights sum to 1"*.
   - **Technical qualification:** In **AdaBoost** ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 56; [fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), p. 161), there is indeed no learning rate parameter, and instance weights $w_t(i)$ are normalised to sum to $1$ at every iteration. In **Gradient Boosting Machines (GBM)** ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 73; [fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), Eq. 4.19, p. 168), there *is* a learning rate $\nu$ (or $\alpha$, incremental shrinkage), but GBM does not use instance weights at all—its learning rate scales the additive contribution $\nu \gamma_m(\mathbf{x})$ of each residual tree. Thus the statement in Q5 is false under both algorithms.
3. **Slide 33 wording on `Minimum sample split` in Random Forests:**
   - **Slide text ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf), Slide 33):** States *"Reducing the minimum number of samples required to consider creating a split leads to smaller trees, and less overfitting"*.
   - **Technical qualification:** In standard decision-tree terminology (and `scikit-learn`'s `min_samples_split`), **increasing** the minimum number of samples required to split a node stops splitting earlier, producing smaller trees and less overfitting (whereas *reducing* it down to 2 allows splitting until leaves are pure, producing larger trees). Compare Slide 37 ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf)), which states the stopping rule accurately: *"Splitting stops if resultant number of samples is less than given threshold"*.
4. **Minor notation/typographical errors in the lecture slides:**
   - **Slide 23 ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf)):** The stacking meta-feature table header writes $\hat{y}_1, \hat{y}_3, \hat{y}_3$ instead of $\hat{y}_1, \hat{y}_2, \hat{y}_3$.
   - **Slide 41 ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf)):** Calls $H(n) = 1 + \frac{1}{2} + \frac{1}{3} + \dots + \frac{1}{n}$ the "$n$-th harmonic mean"; mathematically, $H(n)$ is the **$n$-th harmonic number**.
   - **Slides 59–60 ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf)):** In Step 5A / Step 4A text captions, the prose swaps $e^{\alpha_t}$ (written for "correctly classified") and $e^{-\alpha_t}$ (written for "incorrectly classified"), whereas **Slide 56** and the **actual computed numbers in the tables on Slides 59–60** (`0.071` for correct, `0.167` for incorrect) correctly use $w_t(i)e^{-\alpha_t}$ (decrease) for correctly classified instances and $w_t(i)e^{\alpha_t}$ (increase) for misclassified instances.
   - **Slide 76 ([07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf)):** Writes $F_2 = F_1 + \nu F_0$ in the slide header; the actual GBM update shown on Slide 77 and textbook Eq. 4.19 is $F_2 = F_1 + \nu \gamma_2$.

---

## 5. Sources and unresolved questions

### Local sources used
- **Lecture Slides:** [07ensembleLearningTopic7.pdf](07ensembleLearningTopic7.pdf) — *Machine Learning 441/741, Topic 8/9: Ensemble Learning*, AP Engelbrecht, Stellenbosch University, 2026 (80 slides, PDF pp. 1–80).
- **Textbook Chapter Excerpt:** [fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf) — John D. Kelleher, Brian Mac Namee, and Aoife D'Arcy, *Fundamentals of Machine Learning for Predictive Data Analytics*, Chapter 4 (*Information-Based Learning*), Section 4.4.5 (*Model Ensembles*, printed pp. 158–168 / PDF pp. 1–11).
- **Course Quiz and Answer Key:** [quiz-07-ensemble-learning.md](quiz-07-ensemble-learning.md) (Questions 1–22) and diagram [8159d73f86354ae7.png](quiz-07-ensemble-learning-images/8159d73f86354ae7.png).

### Material gaps
- **Personal notes (`README.md`):** No `README.md` or personal summary questions file is present in this directory; all slide discussion prompts (such as Slide 36 on Random Forest data quality issues and Slide 64 on AdaBoost noise, outliers, and skew class distributions) and all 22 quiz questions have been addressed directly within this overview.
