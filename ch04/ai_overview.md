# Information-based learning: revision overview

## 1. Topic at a glance

A decision tree predicts by asking a sequence of questions about an instance. Each internal node contains a test; each branch is an outcome; each leaf contains a prediction. Training recursively partitions labelled data so that the target becomes more homogeneous within each region. These are **supervised, non-parametric models**: the tree structure is learned rather than fixed in advance. Non-parametric does not mean that a fitted tree has no parameters.

The central inductive bias is that a small collection of informative, conditional tests will generalise. Ordinary single-feature numeric tests produce **axis-parallel boundaries**. Greedy induction chooses the best immediate split, without guaranteeing the globally smallest or most accurate tree. Growing until leaves are pure encourages fitting noise; simplicity preferences and pruning counteract this tendency. No normality, linear relationship, or independent-feature assumption is needed for ordinary tree prediction. Representative training data are still essential for generalisation. ([Slides](04informationBasedLearningTopic4.pdf), slides 4–6, 10–18, 40–41; [textbook](fundamentals_of_ml_2ed_ch.pdf), §§4.2.1, 4.3, 4.5.)

**Scope and terminology:**

- **Prediction models:** classification trees, constant-leaf regression trees, linear/nonlinear model trees, and oblique trees. Probabilistic decision trees are named on slide 6 but not developed.
- **Induction algorithms:** ID3 is the textbook baseline; C4.5 is the lecture focus; CART is introduced in the textbook summary. Genetic programming is another way to search for tree structures/models, not another prediction task.
- **Split criteria:** entropy/information gain, gain ratio, Gini impurity, mean squared error (MSE), variance reduction, and standard deviation reduction (SDR) are measures or objectives, not separate prediction models.
- **Complexity control:** early stopping and post-pruning modify training. Windowing changes which training cases are initially used.
- **Ensemble models/training methods:** bagging, random forests, reweighting-based boosting, and gradient boosting are covered in textbook §4.4.5 and partly in the personal notes. The slides defer fuller ensemble treatment to a later topic.

## 2. Quick comparison

| Model | Task / target type | Descriptive feature types and preparation | Main strength | Main limitation |
| --- | --- | --- | --- | --- |
| Classification tree: ID3; C4.5/CART variants | Binary or multiclass categorical target | ID3: categorical; extensions: numeric thresholds and mixed data | Readable conditional rules; nonlinear interactions | Greedy, unstable; small leaves overfit |
| Constant-leaf regression tree | Numeric target | Numeric thresholds and categorical tests; implementation may require encoding | Simple local averages; nonlinear response | Stepwise predictions; extreme targets distort squared loss |
| Linear/nonlinear model tree | Numeric target | Tree tests plus numeric/encoded inputs for leaf models | Models local trends with fewer regions | Small leaf datasets cannot support complex models |
| Linear oblique tree | Classification; same split idea can serve regression | Numeric/encoded features; scaling depends on split learner | Compact representation of sloping boundaries | Harder split search and explanations |
| Bagged trees | Classification or regression | Inherits base-tree requirements | Reduces prediction variability | More computation/storage; correlated errors remain |
| Random forest | Classification or regression | Bagged trees plus feature subsampling | More tree diversity | Can miss useful features in a sampled subset |
| Reweighting-based boosted trees | Classification; regression needs changed updates | Inherits base-tree requirements | Focuses learning on difficult cases | Can repeatedly emphasise noise/mislabels |
| Gradient-boosted trees | Regression; classification with suitable loss | Inherits base-tree requirements | Learns successive error corrections | Loss and tuning strongly affect overfitting/outliers |

The rows describe substantive mechanisms. Changing entropy to Gini does not itself create a new model family.

## 3. Individual model summaries

### 3.1 Classification trees

#### Main idea and assumptions

A leaf represents a region of descriptive feature space and predicts its dominant class. Tests later on a path are conditional on earlier answers, so trees can represent interactions without specifying them beforehand. The basic ID3 formulation assumes categorical descriptive features, categorical targets, and complete observations. C4.5 extends input handling and pruning; these abilities must not be attributed automatically to every tree implementation.

Single-feature threshold trees prefer axis-aligned partitions. ID3's information-gain heuristic favours informative early tests, but does not solve a global optimisation problem. Perfectly contradictory examples cannot be separated using the available features; purity is therefore not an unconditional stopping guarantee. ([Textbook](fundamentals_of_ml_2ed_ch.pdf), §4.3, printed pp. 133–135; §4.5, p. 169.)

#### How it learns and predicts

**Training, basic ID3:**

1. At the current node, use only the training cases reaching it. If empty, use the immediate parent's majority class; if pure, return that class; if no descriptive features remain, return the current majority class.
2. Score each available feature, select the largest information gain, and create one branch per categorical outcome.
3. Recurse on each partition, removing that categorical feature from further tests **on that path**. The same feature may still occur on other paths.
4. With extensions, stop earlier or grow and then prune. When no admissible split remains, use a majority leaf.

**Prediction:** follow the tests for a query until a leaf, then return its label. Leaf class proportions can also estimate probabilities, but tiny/pure leaves can yield overconfident estimates; a probability estimate is not a guarantee of calibration. Basic ID3 needs an explicit policy for unseen categorical levels or missing test inputs.

For a node dataset $D$ with $n=|D|$ cases and $K$ possible classes, let $p_k$ be the fraction in class $k$. For a candidate test $s$, let $D_o$ be child $o$ and $q_o=|D_o|/n$. Using base 2 throughout:

$$
H(D)=-\sum_{k=1}^{K}p_k\log_2 p_k,\qquad
\operatorname{Rem}(s,D)=\sum_o q_oH(D_o),
$$

$$
IG(s,D)=H(D)-\operatorname{Rem}(s,D).
$$

Use $0\log_2 0=0$. Entropy is the **expected** information of the class outcome; individual-outcome information is $-\log_2 p_k$. Information gain measures the reduction in class uncertainty after observing a test outcome. Weight child entropy by child size, not by an unweighted average. ([Textbook](fundamentals_of_ml_2ed_ch.pdf), §§4.2.2–4.2.3, equations 4.1–4.4, printed pp. 126–130; [slides](04informationBasedLearningTopic4.pdf), 16–18.)

**Small example:** a node contains two positive and two negative cases, so $H(D)=1$ bit. A test makes children containing $(2+,0-)$ and $(0+,2-)$: remainder $0$, gain $1$. Another test makes two children each containing $(1+,1-)$: remainder $1$, gain $0$. In the slides' film example, Country has gain $0.2756$, Big Star $0.0349$, and Genre about $0.17$, so Country is selected (slides 20–22).

Alternative criteria, with the same child notation:

$$
\operatorname{SplitInfo}(s,D)=-\sum_o q_o\log_2q_o,\qquad
GR(s,D)=\frac{IG(s,D)}{\operatorname{SplitInfo}(s,D)},
$$

$$
G(D)=1-\sum_kp_k^2,\qquad
G_{\text{after}}(s,D)=\sum_o q_oG(D_o).
$$

Maximise gain ratio, or minimise weighted child Gini (equivalently maximise Gini reduction at a fixed parent). A zero split-information denominator gives no usable ratio. Gain ratio discourages high-cardinality partitions; it does not guarantee that an identifier will be ignored. Gini is the probability of disagreement between two independently sampled class labels from the node distribution, not the error rate of its majority classifier. ([Textbook](fundamentals_of_ml_2ed_ch.pdf), §4.4.1, equations 4.8–4.9, printed pp. 142–145; slides 23–30. See the formula conflict in §4 below.)

#### Supported data and tasks

- **Targets:** binary or multiclass nominal labels; ordinal targets can be treated as classes, but ordinary class impurity ignores the ordering. Numeric prediction requires a regression variant.
- **Descriptive features:** original ID3 uses categorical equality tests. Nominal values need no numerical ordering in a native categorical tree. Ordinal values can use ordered thresholds when their order is meaningful. Numeric continuous or discrete values can use threshold tests. Mixed data combine these test types.
- **Numeric extension:** sort distinct observed values, test midpoints between adjacent values, and select the best-scoring threshold jointly with other features. Keep numeric features available for later splits on the same path. Slides 33 considers all adjacent distinct boundaries; textbook §4.4.2 narrows classification candidates to class-transition boundaries. Do not split tied feature values apart; testing all distinct boundaries avoids ambiguity in mixed-label ties.
- A library that accepts only numeric arrays may require nominal encoding. Arbitrary category numbers used with thresholds impose a potentially misleading order; one-hot encoding changes the available tests and can enlarge the tree. Encoding is an implementation requirement, not evidence that native categorical trees cannot handle categories.

#### Data quality and preprocessing

| Issue | Effect on the standard model and why | Relevant extension and how it helps | Preprocessing / mitigation and tradeoffs |
| --- | --- | --- | --- |
| Missing values | Basic ID3 cannot score/routinely route unknown inputs. Missing targets cannot contribute supervised class counts. | C4.5 discounts gains for missing inputs and fractionally routes cases; see below. This does not supply missing labels. | Impute inputs or use a supported missing-value method. Exclude unlabelled cases from supervised fitting or obtain labels; do not silently impute target labels. Imputation can hide missingness patterns. |
| Outliers | An unusual numeric magnitude does not dominate a distance or squared target loss, but an isolated case can induce a small branch. Extreme numeric targets are inapplicable to classification; a rare valid class is not automatically an outlier. | Pruning can remove unsupported exceptional branches; removal is not guaranteed. | Check errors and cap complexity; avoid deleting valid rare cases merely because they are unusual. |
| Noise | Measurement errors change test assignments; wrong class labels can induce spurious splits and pure tiny leaves. | Early stopping/post-pruning reduce fitting of weak patterns; bagging reduces variability. | Correct known errors and validate pruning; systematic noise or many wrong labels can survive pruning. |
| Skew class distributions | Counts, impurity gains, majority leaves, and ordinary validation accuracy may favour common classes; small valid minority branches may be pruned. | Class/case weighting changes induction and, where supported, pruning decisions (quiz Q22). | Consider training-only resampling or weights and stratified validation. Check minority recall/precision and balanced or macro metrics; balancing is not mandatory for every objective and can increase false positives. |
| Irrelevant features | Greedy selection often leaves them unused, but chance gains, especially among many candidates, can select them. Unique IDs can appear perfectly informative. | Gain ratio reduces many-values bias; pruning removes some spurious tests. | Remove identifiers/leakage; validate feature selection. Removing weak marginal features can lose conditional interactions. |
| Different value ranges | Axis-aligned split scores depend on partitions, not distances or coefficient magnitudes. Changing units by positive affine rescaling preserves candidate partitions. | No scaling extension is needed for this mechanism. | Normalisation/standardisation usually unnecessary. Rounding/binning can change partitions and lose information. |
| Different descriptive types | Basic categorical ID3 does not interpret a continuous measurement as ordered. Numeric codes do not confer meaningful order on nominal values. | Threshold extension handles numeric/discrete ordered values; C4.5 combines categorical and numeric tests. | Declare types correctly; encode only as required by the implementation. Discretisation simplifies data but discards within-bin information. |

**C4.5 missing-input mechanism:** let $F$ be the fraction of node weight whose candidate feature is unknown. Evaluate the information gain on the known cases and discount it by $1-F$; the split-information calculation includes unknown as an extra outcome. This is a test-selection penalty, not median imputation. For branch frequencies $q_o$ estimated from known cases, a case with incoming weight $w_i$ and an unknown test value contributes $w_iq_o$ to each child. Known outcomes send the full incoming weight to their matching child. Prediction with unknown inputs combines the class distributions from possible branches using routing weights, then selects the most probable class. Preserve weights through subsequent levels. Slides 35–38 describe this method; the notation $H(D)$ on slide 36 is abbreviated and should not be read as comparing full-data parent entropy with known-only child entropy.

**Redundancy and sparse data:** correlated/redundant features do not create the coefficient-identification problem of linear regression, but substitute splits can make the chosen tree unstable and its apparent feature importance misleading. Many features create more opportunities for accidental gain. Small datasets and deep leaves yield unreliable class proportions, especially for rare combinations absent from training. Pruning trades specificity for better-supported regions; it cannot manufacture missing coverage. ([Textbook](fundamentals_of_ml_2ed_ch.pdf), §§4.4.4, 4.5, printed pp. 154–155, 169–170.)

#### Important choices and variants

**Learned:** topology, selected features, thresholds, branch outcomes, leaf labels/distributions. **Chosen hyperparameters:** criterion, maximum depth, minimum cases per split/leaf, minimum gain, pruning policy/strength, and class costs/weights. A deeper tree or smaller leaf minimum increases flexibility and usually lowers training error, while increasing variance; stronger restrictions can underfit.

- **C4.5:** gain-ratio-based induction, numeric thresholds, missing-input handling, and post-pruning. J48 is an open-source implementation mentioned in textbook §4.5. These are not guarantees shared by the local ID3 code.
- **CART:** binary classification/regression trees; textbook §4.5 associates classification CART with Gini. Its regression criterion is not class entropy.
- **Pre-pruning:** stop for small subsets, limited depth, insufficient gain, or statistical tests such as chi-squared tests. Efficient, but can suppress interactions only revealed after several splits.
- **Post-pruning / reduced-error pruning:** grow a large tree; work bottom-up, replacing a subtree with the **majority class of training cases reaching its root**, not the majority vote of its leaves. Retain the replacement if validation error does not increase. The textbook and local code allow ties; slide 31 describes retaining a prune when generalisation improves. Keep a final test set separate from pruning/tuning data. ([Textbook](fundamentals_of_ml_2ed_ch.pdf), §4.4.4, printed pp. 154–158; [local pruning code](src/ch04/reduced_error_pruning.py), `reduced_error_pruning`.)
- **Categorical subset splits:** a test such as colour $\in\{\text{red, green}\}$ versus the remaining levels groups categories. It can reduce fragmentation compared with one branch per value, but does not guarantee a smaller completed tree for every dataset (quiz Q20).
- **Windowing:** induce on a random training subset, add misclassified cases from the remaining training pool, and refit until no outside exceptions remain. Reduces initial induction cost; persistent noise can force the window to grow. Those outside cases are still training data, not an independent test set (slides 39, 47).

#### Strengths, limitations, and practical use

Useful for explainable conditional rules, mixed tabular data, nonlinear effects and interactions. Each root-to-leaf path yields one conjunctive rule, so the direct rule extraction has one rule per leaf. One fitted tree can support both prediction and explanation (slides 5, 8–9; quiz Q8, Q14). A rule explains the model's decision, not a causal relationship.

Large trees lose readability, and small data changes can alter an early split and the whole downstream tree. Numeric diagonal boundaries can require many axis-aligned splits. Training evaluates many candidate splits repeatedly; prediction requires only the tests on one path. With depth $h$ and $L$ total nodes, ordinary traversal takes $O(h)$ tests and tree storage is $O(L)$, excluding stored training examples. Use held-out or cross-validated classification performance; examine a confusion matrix and class-specific metrics when classes are skewed. Interpretability supports scrutiny in high-stakes applications but does not itself establish validity or suitability.

### 3.2 Constant-leaf regression trees

#### Main idea and assumptions

Partition feature space into regions whose numeric targets are similar, then predict one constant per region. Under squared-error loss, the best constant is the region's mean. No global linearity or normality is required. For a mean to represent the expected response well, enough representative observations are needed; zero-mean random target noise may average out, while systematic error will not. ([Slides](04informationBasedLearningTopic4.pdf), 42–48; [textbook](fundamentals_of_ml_2ed_ch.pdf), §4.4.3.)

#### How it learns and predicts

**Training:** search candidate tests as for classification, choose the largest reduction in numeric dispersion, recurse subject to stopping conditions, and optionally prune using numeric validation loss. **Prediction:** traverse to leaf $R$ and return

$$
\hat y_R=\bar y_R=\frac{1}{n_R}\sum_{i\in R}y_i,\qquad
I(R)=\frac{1}{n_R}\sum_{i\in R}(y_i-\bar y_R)^2,
$$

where $R$ is the set of training indices reaching the leaf, $n_R=|R|$, and $y_i$ is target $i$. A squared-error split minimises $\sum_o(n_o/n)I(D_o)$, equivalently total child squared error. For example, targets $[2,4,8]$ give a prediction $14/3$; changing $8$ to $80$ shifts it to $86/3$, showing target-outlier sensitivity.

Slides 43 also gives $SDR=\sigma(D)-\sum_o q_o\sigma(D_o)$, where $\sigma$ is target standard deviation. SDR and MSE reduction are **different criteria** and can rank splits differently. The textbook's equation 4.10 uses sample variance with denominator $n-1$ rather than node MSE's $n$. Weighted sample variance is not numerically identical to weighted MSE; singleton sample variance needs an explicit convention. Use the criterion specified by the question or implementation, rather than silently equating them. (Slides 43; textbook §4.4.3, printed pp. 149–152, equations 4.10–4.11.)

#### Supported data and tasks

Numeric targets, including continuous measurements and discrete counts treated numerically; the standard leaf mean need not be an integer or respect physical constraints. Descriptive features can be continuous/discrete numeric, nominal or ordinal with appropriate tests, or mixed, subject to implementation support as in §3.1. A numeric code for a class label does not turn classification into meaningful regression. Output is a numeric estimate, not a class probability. Predictions form a step function and constant leaves cannot extrapolate a trend beyond observed targets.

#### Data quality and preprocessing

| Issue | Effect on the standard model and why | Relevant extension and how it helps | Preprocessing / mitigation and tradeoffs |
| --- | --- | --- | --- |
| Missing values | Ordinary threshold traversal needs a test value; missing targets cannot enter means or losses. | Slides 49 teaches surrogate splits using another feature that reproduces the primary partition. Not native to every regression tree. | Impute inputs or choose supported surrogates; obtain/exclude missing labels. Surrogates need observed related features and add search cost. |
| Outliers | Extreme inputs do not directly dominate distances, but can induce rare splits. Extreme targets distort means, MSE and SDR, and can drive extra splitting. | Pruning limits fragmentation but still pools extreme targets into means; it does not remove their influence. | Verify anomalies; transformations or justified cleaning may help but change interpretation or discard valid extremes. |
| Noise | Noisy inputs alter routing; large target errors inflate dispersion and encourage deep fits. | Stopping/pruning pools more cases; small zero-mean target noise can average out with sufficient data. | Use sufficient leaf support and numeric validation loss. Biased or correlated noise is not removed by averaging. |
| Skew class distributions | Class imbalance is not applicable: there are no target classes. Sparse extreme-response regions may nevertheless have poor support. | No special rare-response extension developed here. | Inspect error across target ranges; do not automatically balance a numeric target as if it were classes. |
| Irrelevant features | Chance dispersion reductions can produce spurious regions. | Pruning and bagging can reduce resulting variance. | Remove IDs/leakage and validate selection; useful interactions can have weak marginal effects. |
| Different value ranges | Axis-aligned input thresholds do not require comparable units. Scaling the target rescales MSE/SDR and related stopping tolerances. | No input-scaling extension needed. | Usually leave input units intact; adjust loss thresholds if target units change. |
| Different descriptive types | Numeric tests need order; nominal categories need equality/subset tests or appropriate encoding. | Numeric/categorical split mechanisms from §3.1 apply. | Encode according to the implementation; do not impose arbitrary numeric order. |

Redundant features create alternative unstable splits rather than singular mean estimates. High dimensionality and little data increase accidental splits and sparse leaves. All preprocessing must be fitted within the training fold. Slides 48–49 are the principal data-quality source.

**Surrogate qualification:** a surrogate predicts the outcome of a primary **input test**, not the unknown target. Slides 49 says to ignore cases missing the primary input when assessing its split, then learn alternative splits with similar assignments. Its final fallback wording, “randomly assign to subset with smallest deviation from the mean target feature value,” is ambiguous; a new case has no known target to compare. The supplied materials do not establish an operational fallback when every surrogate is missing. Do not present that phrase as a universal prediction algorithm.

#### Important choices and variants

Learned parameters are tests, topology and leaf means; hyperparameters include depth, leaf support, dispersion/gain tolerances, criterion and pruning. Lower tolerances and smaller leaves allow more detailed, potentially noisy partitions. MSE tolerances depend on target units. Classification and regression share depth/size limits, but class purity is replaced by numeric loss/dispersion criteria. A singleton can have zero MSE: numeric targets do not make zero within-leaf error impossible. Model trees below change the leaf predictor rather than just its stopping tolerance.

#### Strengths, limitations, and practical use

Useful for interpretable nonlinear numeric relationships with local interactions. Predictions can jump at thresholds and cannot continue a slope outside the training range. Pruning improves complexity control but does not make mean prediction resistant to extreme targets. Traversal/storage costs match ordinary classification trees; evaluation uses numeric errors such as MSE/RMSE or MAE according to the objective, rather than class accuracy. Depth effects are illustrated in slides 44–46.

### 3.3 Linear and nonlinear model trees

#### Main idea and assumptions

A model tree partitions input space, then uses a **local regression model** at each leaf instead of a constant mean. Linear leaves represent local trends; nonlinear leaves represent more complicated local functions. This can reduce the number of regions needed, but increases the data and estimation requirements within each region. An axis-aligned model tree may still use single-feature internal tests: complex leaves do not make the internal tests oblique. ([Slides](04informationBasedLearningTopic4.pdf), 50–56.)

#### How it learns and predicts

**Training:** induce regression partitions, fit the selected model to each region, evaluate numeric error, and control both partition and local-model complexity. The slides describe regression-tree-style induction; they do not provide one complete optimiser for every model-tree variant.

For a linear leaf $R$, a representative squared-error fit is

$$
\hat{\boldsymbol\beta}_R=\arg\min_{\boldsymbol\beta}
\sum_{i\in R}(y_i-\beta_0-\sum_{j=1}^{p}\beta_jx_{ij})^2,
\qquad
\hat y(\mathbf x)=\hat\beta_{R,0}+\sum_{j=1}^{p}\hat\beta_{R,j}x_j.
$$

Here $\mathbf x$ has $p$ encoded numeric inputs; $x_{ij}$ is input $j$ for case $i$; $\beta_0$ is the intercept and $\beta_j$ a slope. First traverse to $R$, then evaluate its fitted function. Nonlinear leaves replace this linear expression with their chosen function. Unique unregularised least-squares coefficients require a full-rank leaf design; normal residuals are not needed just to compute predictions. Distributional assumptions for confidence intervals are separate and not developed here.

#### Supported data and tasks

The lecture's model trees predict **numeric targets**, with numeric local-model outputs. Internal tests can handle mixed features, while leaf regressions need meaningful numeric inputs or categorical encoding; ordinal spacing must not be assumed without justification. Linear versus nonlinear describes the leaf function. Quiz Q25 uses “model trees for classification” for nonlinear oblique tests; that terminology should not erase the distinction between a leaf model and an internal-node model.

#### Data quality and preprocessing

| Issue | Effect on the standard model and why | Relevant extension and how it helps | Preprocessing / mitigation and tradeoffs |
| --- | --- | --- | --- |
| Missing values | Missing values can block both routing and leaf evaluation; missing targets prevent fitting numeric loss. | Regression-tree surrogates address routing only; slides do not specify a complete missing-input leaf estimator. | Fit input imputation within training folds; obtain/exclude missing labels. A routed case may still lack a leaf-model input. |
| Outliers | Extreme targets distort squared-error leaves; unusual numeric inputs can also have high leverage on fitted slopes. | Pruning reduces tree complexity but does not make leaf least squares outlier-resistant. No robust leaf estimator is developed here. | Check anomalies and local fit support; avoid unwarranted extrapolation. |
| Noise | Small regions and complex functions can memorise measurement/target noise. | Simpler linear leaves and restricted depth reduce flexibility; nonlinear leaves increase it. | Validate both leaf-model and tree complexity. More pruning alone may leave an overcomplex leaf model. |
| Skew class distributions | Not applicable to the numeric-target model. Rare input regimes can receive too few cases. | No specific sparse-regime extension developed here. | Examine errors by regime and require enough data to fit each leaf. |
| Irrelevant features | Harmful split candidates and unnecessary leaf coefficients both increase variance. | Restrict leaf inputs/function complexity; no specific regularised leaf procedure is supplied. | Feature selection needs validation; conditional predictive inputs may matter locally. |
| Different value ranges | Axis-aligned routing is scale-insensitive, but leaf fitting can have numerical/optimisation issues. Exact unpenalised linear least squares is invariant to invertible unit changes after coefficient adjustment. | Behaviour depends on the chosen leaf learner; nonlinear/penalised learners may be scale-sensitive. | Scale encoded numeric leaf inputs if needed; keep fitting within training folds. |
| Different descriptive types | Leaf arithmetic cannot use raw nominal strings or arbitrary category codes meaningfully. | Mixed split tests plus encoded leaf regressions support mixed data. | Use suitable dummy/ordinal encodings; increasing encoded dimension increases local data requirements. |

**Multicollinearity is especially relevant here:** redundant columns can make local linear coefficients non-unique or unstable. Splitting reduces the available sample size, so a globally adequate dataset may still yield underdetermined leaf regressions. Nonlinear bases add further degrees of freedom; restricting the function family or pooling regions trades flexibility for stability. These are consequences of the fitted leaf mechanism, qualifying slide 50's statement that all regression-tree aspects apply.

#### Important choices and variants

Choose tree depth/leaf support, linear versus nonlinear functions, included predictors/basis terms, and the leaf learner's fitting/complexity settings. Learned quantities include split thresholds and local coefficients/functions. Linear leaves are a substantive alternative to constant leaves; nonlinear leaves belong under the same parent but add greater flexibility and overfitting risk.

Slides 53–56 illustrate evolved linear/nonlinear expressions for abalone and a synthetic piecewise dynamical system, citing Potgieter and Engelbrecht (2008). Quiz Q16 establishes that genetic programming can evolve a tree. The actual search operators, settings and comparative results are not available in the local papers, which are absent.

#### Strengths, limitations, and practical use

Useful when different input regions follow different smooth trends. A local equation may explain a region more compactly than many constant leaves, as slides 51–52 illustrate. Complexity and extrapolation can make the explanation less transparent and predictions less stable. Prediction includes tree traversal **plus** leaf-function evaluation; for a dense linear leaf this adds $O(p)$ arithmetic. Storage includes all local coefficients. Evaluate numeric error and stability on held-out data; inspect boundary discontinuities and unsupported local extrapolation.

### 3.4 Oblique decision trees

#### Main idea and assumptions

Change the **internal test** to a combination of features. A sloping boundary may then need one test where an axis-aligned tree needs many. Fewer nodes do not guarantee less overfitting: each node now learns more coefficients and can search a richer family of boundaries. Slides 57–58 develop linear oblique tests; quiz Q25 additionally mentions nonlinear tests without supplying an induction procedure.

#### How it learns and predicts

For $p$ numeric/encoded inputs, test

$$
\mathbf a^\top\mathbf x+b>0,
$$

where $\mathbf a=(a_1,\ldots,a_p)$ contains learned coefficients and $b$ is an intercept. **Training:** search for a useful separating hyperplane, partition cases by the sign, and recurse. Slide 58 names logistic regression and support vector machines as possible split learners; it does not teach their full algorithms here. **Prediction:** evaluate each combination along a path and return the leaf's prediction. For example, $x_1+x_2>1$ separates diagonally in two dimensions.

#### Supported data and tasks

The lecture examples and quiz focus on classification; the oblique partition mechanism can also be combined with numeric leaves. Labels/probabilities versus numeric predictions depend on those leaves. Linear tests need numeric arithmetic: continuous/discrete measurements can enter directly; nominal categories require appropriate encoding, and ordinal encoding requires a defensible order/spacing. Raw mixed data are not natively handled by a linear combination.

#### Data quality and preprocessing

| Issue | Effect on the standard model and why | Relevant extension and how it helps | Preprocessing / mitigation and tradeoffs |
| --- | --- | --- | --- |
| Missing values | Any missing participating input can make a hyperplane test undefined; missing targets cannot train a supervised split/leaf. | No dedicated oblique missing-value method is developed here. | Fit input imputation within folds; obtain/exclude missing targets. Do not assume C4.5's single-feature routing transfers unchanged. |
| Outliers | Extreme inputs may influence fitted hyperplanes, depending on the split learner. Numeric leaves retain target-outlier risks. | No robust oblique fitting extension is developed here. | Inspect anomalies and validate the chosen learner; ordinary-tree outlier claims are insufficient. |
| Noise | Noisy inputs perturb sums and learned coefficients; wrong labels can distort separating tests. | Complexity control can limit fitting noise, without removing it. | Validate depth and split-learner settings; complex nonlinear tests can memorise rapidly (quiz Q25). |
| Skew class distributions | Split fitting and leaf decisions can favour common classes. For regression, class imbalance is inapplicable. | Class weighting is a possible choice only if the actual split learner/leaf method supports it; no oblique-specific recipe supplied. | Check class-specific metrics and training-only balancing where justified. |
| Irrelevant features | Additional inputs add possible hyperplane coefficients, increasing estimation/search variance. | Restrict participating inputs or split complexity; no particular selection algorithm is supplied. | Validate feature selection and avoid IDs/leakage; do not infer automatic sparsity. |
| Different value ranges | Ideal hyperplanes can compensate with inverse coefficient changes, but regularised/numerical split searches may depend on units. | Depends on the chosen split learner. | Standardise numeric inputs when that learner needs it; fit scaling on training data. |
| Different descriptive types | The linear combination requires numeric inputs; arbitrary nominal codes impose artificial geometry. | Suitable encoding allows categorical participation. | Use meaningful encoding; extra columns increase search and interpretation costs. |

Correlated predictors may yield unstable coefficients even when a separating plane predicts similarly. High dimensionality and small node samples make rich split searches particularly prone to overfitting. Unlike single-feature threshold trees, neither coefficient stability nor scaling behaviour can be inferred from class counts alone.

#### Important choices and variants

Choose linear versus nonlinear node tests, split learner and its hyperparameters, number of participating predictors, depth, leaf size and pruning. Coefficients and intercepts are learned parameters. Linear oblique tests permit hyperplanes; nonlinear node functions permit curved boundaries and much higher flexibility. An oblique tree need not have regression models in its leaves.

#### Strengths, limitations, and practical use

A good candidate when boundaries are sloped and many ordinary splits would be needed. Explanations contain combinations rather than simple one-feature conditions; a smaller tree may still be harder to explain. Split optimisation costs more than scoring a single threshold. With dense tests using $p$ inputs at depth $h$, traversal takes $O(hp)$ arithmetic. Compare held-out performance and complexity, rather than assuming that fewer leaves guarantees better generalisation.

### 3.5 Bagged trees and random forests

#### Main idea and assumptions

Average away part of the instability of individual trees by fitting diverse models. Bagging changes the training cases; random forests add feature subsampling to encourage further diversity. The benefit is smaller when trees make highly correlated errors. This is an ensemble prediction model, not a replacement impurity metric. ([Textbook](fundamentals_of_ml_2ed_ch.pdf), §4.4.5.1, printed pp. 159–160.)

#### How it learns and predicts

**Training:** for $B$ trees, independently sample $n$ cases with replacement from the $n$ training cases and fit one tree per bootstrap sample. Duplicates and omissions create different samples. Random subspaces restrict the candidate features used by trees; the chapter illustrates selecting a feature subset for each bootstrap sample. **Subagging** uses subsets sampled without replacement.

**Prediction:** combine class votes for classification or numeric predictions for regression. The chapter explicitly uses majority vote and **median** regression aggregation, favouring the median's resistance to extreme individual predictions. Mean aggregation is another possible design, with different robustness; do not silently replace the chapter's median convention. A vote fraction is not automatically a calibrated probability. More trees reduce sampling variability of the aggregate, but do not repair a shared bias or missing coverage.

#### Supported data and tasks

Categorical targets with classification trees, numeric targets with regression trees. Feature support and missing-input behaviour are inherited from the base learner; bootstrapping does not encode categories, fit a missing-value policy, or supply missing targets. Random forests are grouped here because their main change is diversity through feature sampling.

#### Data quality and preprocessing

| Issue | Effect on the standard model and why | Relevant extension and how it helps | Preprocessing / mitigation and tradeoffs |
| --- | --- | --- | --- |
| Missing values | Sampling and voting do not resolve missing input tests or targets. | Use a base-tree missing-input extension; no new ensemble missing-target method is supplied. | Fit any imputation within training folds; obtain/exclude missing labels. |
| Outliers | Unusual cases can occur repeatedly in bootstrap samples. Mean-leaf target sensitivity remains. | Aggregation reduces some idiosyncratic fits; chapter's median limits extreme tree predictions. | Verify anomalies; aggregation cannot guarantee resistance to systematic contamination. |
| Noise | Averaging/voting reduces variability from independently fitted noisy branches; shared mistakes remain. | Random subspaces can reduce tree correlation. | Retain sufficient leaf support and validate on representative data; wrong labels persist across samples. |
| Skew class distributions | Bootstraps preserve skew in expectation and may omit scarce classes; votes can favour common labels. Not applicable as class imbalance for regression. | Class weighting of supported base trees can change their costs; no special balanced-forest algorithm is developed here. | Consider training-only resampling/weights and minority metrics; duplication does not add independent minority evidence. |
| Irrelevant features | Chance splits remain possible; sampled feature sets may exclude strong predictors. | Random forests decorrelate trees, but do not guarantee irrelevant features are ignored. | Remove IDs/leakage; tune feature sampling, especially when signal is sparse. |
| Different value ranges | Inherits scale-insensitive axis-aligned split partitions. | No new scaling extension needed for ordinary base trees. | Scale only if another pipeline component needs it. |
| Different descriptive types | Inherits base learner's native tests or encoding requirements. | Same categorical/numeric tree extensions apply. | Encode correctly; feature subsampling over one-hot columns can change how a categorical variable is represented. |

Redundant predictors can keep trees correlated and distribute apparent importance across substitutes. With high dimension and few cases, feature subsampling can help diversity but also miss rare signal. Bagging cannot create information in regions absent from the original training data.

#### Important choices and variants

Choose $B$, base-tree complexity, sampling size, feature-subset size and aggregation. Smaller feature subsets tend to reduce correlation but can weaken each tree. More trees increase computation/storage; increasing $B$ is different from increasing each tree's depth. Trees' structures and leaf predictions are learned. Feature sampling settings are hyperparameters. The source's feature-subset illustration should not be treated as a complete specification of every random-forest implementation.

#### Strengths, limitations, and practical use

Useful when a single tree is unstable and aggregate predictive performance matters more than a short rule set. Trees can be trained independently and in parallel. Storage and inference grow with the number of trees: roughly $B$ traversals, rather than one. A collection of individually readable trees is much less transparent as a whole. Evaluate the final aggregate, using the task-appropriate held-out metrics from §3.1/§3.2.

### 3.6 Reweighting-based boosted trees

#### Main idea and assumptions

Fit models sequentially, giving more influence to cases previous models got wrong. This differs from independent bootstrap fits. Difficult genuine cases can benefit; noisy labels can also receive increasing attention. The chapter gives AdaBoost-style classification updates. The binary formula below requires a learner better than weighted random guessing, not arbitrary error rates. ([Textbook](fundamentals_of_ml_2ed_ch.pdf), §4.4.5.2, printed pp. 159–163.)

#### How it learns and predicts

For $n$ labelled cases start with normalised weights $w_i=1/n$. At round $m$:

1. Fit classifier $h_m$ using weights or a sample drawn according to them.
2. Compute **weighted training error** $\epsilon_m=\sum_iw_i\mathbf1[h_m(\mathbf x_i)\ne y_i]$, where the indicator is one for an error and zero otherwise.
3. Set $\alpha_m=\tfrac12\ln((1-\epsilon_m)/\epsilon_m)$.
4. Multiply each misclassified case's weight by $1/(2\epsilon_m)$ and each correctly classified case's weight by $1/(2(1-\epsilon_m))$.

For $0<\epsilon_m<1/2$, misclassified cases gain relative weight and $\alpha_m>0$; these updates preserve a total weight of one. At zero error the expressions are singular and need termination/special handling; at error at least one half the usual positive-weight weak-learner condition fails. Do not apply the equations blindly.

For binary labels $y\in\{-1,+1\}$, predict $\operatorname{sign}(\sum_{m=1}^{B}\alpha_mh_m(\mathbf x))$. The sum is a **score**, not a probability. The chapter describes weighted voting more generally, but multiclass boosting requires a suitable formulation; the binary conditions are not universal multiclass rules. Numeric-target boosting changes both error-based weight updates and aggregation; the chapter mentions weighted-mean regression aggregation, not the unchanged misclassification algorithm. (Equations 4.12–4.14, printed p. 161.)

#### Supported data and tasks

The explicit algorithm is classification with categorical labels and tree-compatible descriptive inputs. Continuous/discrete numeric and nominal/ordinal/mixed inputs depend on the base learner's tests or encoding. Regression is a modified variant, not an interpretation of categorical mistakes as numeric residuals. Output is a voted label or score; probabilities require a justified mapping.

#### Data quality and preprocessing

| Issue | Effect on the standard model and why | Relevant extension and how it helps | Preprocessing / mitigation and tradeoffs |
| --- | --- | --- | --- |
| Missing values | Weights cannot make a missing input routable or a missing label usable for error calculation. | Base-tree input-handling extensions may be used; no missing-target method is developed. | Fit input preprocessing within folds; obtain/exclude missing labels. |
| Outliers | Hard anomalous cases can keep attracting weight; numeric variants can be distorted by extreme target errors. | No dedicated robust reweighting variant is developed here. | Check anomalies and limit rounds/base complexity; difficult valid cases should not simply be deleted. |
| Noise | Wrong labels can be repeatedly upweighted; measurement noise can create inconsistent splits. | Weak/simple trees limit individual fits but do not guarantee noise resistance. | Validate stopping and inspect heavily weighted cases. |
| Skew class distributions | Equal initial case weights do not give equal total weight to each class. Errors may shift weight to minority cases without ensuring balanced performance. | Course class-weighting idea can inform a compatible base learner, but no complete imbalance-specific boosting procedure is given. | Assess minority metrics and validated cost/resampling choices; altering weights changes the learning objective. |
| Irrelevant features | Sequential fitting may exploit accidental features to correct residual mistakes. | Restrict base-tree complexity; no guaranteed automatic feature rejection. | Remove IDs/leakage and validate selection. |
| Different value ranges | Ordinary base-tree thresholds remain insensitive to input units. | No additional scaling mechanism needed for those trees. | Other base learners may require scaling; do not generalise tree behaviour to all boosting. |
| Different descriptive types | Support follows the base tree; the weight updates themselves do not interpret feature types. | Numeric/categorical tree extensions apply. | Encode as required without imposing arbitrary nominal order. |

High dimensionality, correlated substitutes and sparse leaf support retain the base-tree risks; sequential emphasis on difficult cases can amplify them. More rounds reuse the same finite evidence, rather than creating new independent observations.

#### Important choices and variants

Choose number of rounds, tree depth/leaf support and a boosting formulation appropriate to the target. Tree parameters, sample weights and confidence factors are learned during fitting. More/deeper trees permit stronger correction and potentially overfitting. In the notes, the phrase “$\alpha$ increases as $\epsilon$” is a transcription error: the formula makes confidence increase as error **decreases**. The data used for round-error weighting are training data, even though the notes call them a “validation set.”

#### Strengths, limitations, and practical use

Can combine simple learners into complex boundaries by concentrating on previous errors. Less suitable when stubborn errors mostly reflect bad labels or insufficient features. Rounds depend on previous rounds, so training is less straightforward to parallelise than bagging. Prediction evaluates a weighted ensemble; storage and inference grow with $B$. Validate the aggregate's classification performance and minority errors, not merely the declining weighted training errors.

### 3.7 Gradient-boosted trees

#### Main idea and assumptions

Build an additive predictor by fitting successive corrections. For squared error, corrections predict numeric residuals. More generally, the loss determines which errors are emphasised. Gradient boosting is not the same as resampling misclassified cases. ([Textbook](fundamentals_of_ml_2ed_ch.pdf), §4.4.5.3, printed pp. 164–168.)

#### How it learns and predicts

For regression inputs $\mathbf x_i$ and numeric targets $y_i$, initialise $F_0(\mathbf x)=\bar y$, the training target mean. At round $m$, compute

$$
r_{im}=y_i-F_{m-1}(\mathbf x_i),\qquad
F_m(\mathbf x)=F_{m-1}(\mathbf x)+\eta g_m(\mathbf x),
$$

where $g_m$ is a regression tree fitted to residuals $r_{im}$, $0<\eta\leq1$ is a chosen learning rate, and $m=1,\ldots,B$. **Prediction** sums the initial value and all $B$ fitted corrections. Smaller $\eta$ usually requires more rounds; it slows correction rather than removing outlier influence. These are the chapter's squared-error equations 4.15–4.19 with consistent notation.

With other differentiable losses, corrections follow negative loss gradients rather than necessarily raw target-minus-prediction residuals. The textbook mentions other losses and sign-of-error corrections; these alter what is fitted. Classification uses an appropriate classification loss and score-to-probability interpretation, not naive squared-error residuals of arbitrary class numbers.

#### Supported data and tasks

The worked version predicts numeric targets. A classification variant predicts class scores/probabilities with suitable loss/link, then labels. Input support is inherited from base trees: numeric/discrete, categorical/ordinal and mixed features need their corresponding tests or encodings. The additive numeric regression output is not a probability.

#### Data quality and preprocessing

| Issue | Effect on the standard model and why | Relevant extension and how it helps | Preprocessing / mitigation and tradeoffs |
| --- | --- | --- | --- |
| Missing values | Residual fitting still needs routable inputs and known targets. | A missing-capable base learner may help inputs; no universal missing-value feature is implied. | Fit input imputation within folds; obtain/exclude missing labels. |
| Outliers | Squared-error residuals give extreme targets large corrections; unusual inputs may create small regions. | Textbook describes sign-based corrections/alternative losses to reduce magnitude influence; shrinkage slows updates. | Validate loss, depth and learning rate. Shrinkage alone does not make squared loss robust. |
| Noise | Later trees can keep correcting noise left in residuals; measurement errors alter routing. Wrong labels affect classification-loss gradients. | Shallow trees, shrinkage and controlled round count restrict fitting. | Use validation to choose rounds/complexity; systematic noise remains. |
| Skew class distributions | Not applicable to the worked regression model; a classification variant can favour common classes through its loss. Sparse numeric-response regimes remain hard. | Alternative loss choices change emphasis; no particular balanced classification recipe is developed here. | Inspect class-specific or response-range errors; use compatible weights/resampling only within training. |
| Irrelevant features | Repeated residual searches can exploit chance structure, especially with many inputs. | Restrict tree depth and number of corrections. | Remove IDs/leakage; validate selection and stopping. |
| Different value ranges | Axis-aligned input splits do not require scaling; target scaling changes residual and loss magnitudes. | No input-scaling extension needed for ordinary base trees. | Reconsider loss-based tolerances after changing target units. |
| Different descriptive types | Base-tree feature handling determines support; residuals are numeric regardless of input types. | Threshold/categorical tests or suitable encoding apply. | Declare types correctly and avoid arbitrary nominal order. |

Redundant features offer interchangeable correction splits; high dimension and little data increase accidental residual fits. Shrinkage and shallower trees reduce flexibility but cannot supply absent feature-space coverage.

#### Important choices and variants

Choose loss, learning rate $\eta$, number of trees $B$, depth and minimum leaf support. The individual trees and their corrections are learned. Smaller learning rates and simpler trees often need larger $B$; tune these jointly. A classification loss changes the target of each correction; sign-of-error versus magnitude correction changes robustness and the objective. The textbook's note that shallow trees are common does not mean every random forest also uses shallow trees.

#### Strengths, limitations, and practical use

Useful for fitting complex tabular response functions through controlled incremental improvements. Loss flexibility is a strength, while sequential training, tuning burden and reduced interpretability are costs. Prediction/storage grow with the number and size of trees; training rounds depend on earlier fits. Use held-out task-appropriate loss and metrics to select settings, with a separate final test set. The textbook's comparison of ensembles on datasets with more or fewer than 4,000 features is an empirical result from cited experiments, **not a universal model-selection threshold** (§4.5, printed p. 170).

### Shared validation rule

Fit imputers, encoders, scaling, feature selection and any resampling **only within the training split or current cross-validation training fold**. Use validation data for pruning, stopping and hyperparameter selection, then estimate final performance on untouched data. A pruning set is no longer an untouched test set. For scarce classes, stratification improves representation but cannot make a tiny minority sample statistically reliable. Evaluation additions such as class-specific precision/recall and MAE above are practical guidance, not extra prediction models taught in the chapter.

## 4. Exam and comparison points

The personal summary questions are answered by §1 (main idea), §3.1 (standard induction), the variant sections, and each seven-row data-quality table. The following distinctions resolve the main assessment traps.

1. **Decision tree is the umbrella term.** Classification, regression and model trees differ in their leaf predictions; oblique trees change internal tests (quiz Q21). Classification supports more than two classes (Q11). A numeric-target tree does not use class entropy merely because classification trees do (Q18).
2. **Entropy measures proportions, not sample count alone.** In Q17's [basket image](quiz-04-information-based-learning-images/entropy.png), C has equally many red and blue items, hence the greatest binary entropy. For fixed $K$, base-2 entropy lies in $[0,\log_2K]$, not an unbounded range. Slides 15–19 uses base $K$ to normalise the maximum to one; base 2 retains a finite maximum for fixed $K$. Use a fixed class universe/base throughout a calculation; define a pure node's entropy as zero without evaluating $\log_1$.
3. **There is a substantive gain-ratio formula conflict.** Slide 23 divides by split entropy using base $O$, the number of outcomes, while its class entropy uses base $K$. The textbook equation 4.8 uses base 2 consistently. These are not interchangeable rankings when candidate tests have different numbers of outcomes: normalising split entropy by $\log_2O$ changes the penalty. With equally sized branches, slide 23's denominator is one regardless of $O$, undermining the intended many-values penalty. Use the textbook's consistent-base gain ratio for the standard mechanism; identify the convention explicitly if a question supplies another formula.
4. **Gini bounds depend on class count.** $G=0$ for purity and $G_{\max}=1-1/K$ for $K$ equally likely classes. Slide 24's $0.5$ is the balanced **binary** value, and one is a limiting value as $K$ grows, not the maximum for any fixed finite $K$.
5. **Gain ratio does not make identifier removal unnecessary.** Q12 records **True**, reasoning that identifiers will be penalised and later pruned. Qualified answer: with $n$ unique identifiers, $IG=H(D)$ and split information $=\log_2 n$, giving a positive ratio that can still win if alternatives are weak. Pruning is also not guaranteed to remove it. Q1 correctly rejects the claim that information gain eliminates the need for prior feature selection. Remove meaningless identifiers and leakage; distinguish useful entity information from row IDs.
6. **“Robust to missing values” is variant-dependent.** Q1 leaves that statement unselected as false; slides 35–38 assumes a missing-capable induction procedure. Basic ID3 and the local implementation do not thereby acquire that capability. Q24 correctly distinguishes gain discounting/fractional routing from median imputation; a missing feature can still be selected early if sufficiently informative.
7. **Pruning helps with noise; it does not guarantee immunity.** Q5 correctly rejects mandatory removal of all outliers before classification-tree induction; Q7 correctly rejects mean-leaf regression robustness to extreme targets. Slides 31/34 and the personal notes assume noise/outliers are isolated minority leaves. They may instead influence earlier splits or form systematic patterns. A pruned parent predicts the majority of **cases**, not the majority of its child labels. Pure leaves on noisy data can overfit (Q10), but tree induction is not inherently required to overfit under every stopping policy (Q1's lecture convention).
8. **Imbalance is a cost and evidence problem.** Q15 records **False** and says balancing preprocessing “has to” be done for proportions $0.46,0.48,0.06$. The skew warrants assessment of minority errors and pruning, but does not logically require resampling: suitable weighting, costs, validated thresholds or an already adequate model may suffice. Q22 records **False** because class weighting can be used; minority-class rarity is not itself label noise.
9. **Oblique simplicity is conditional.** Q2 records **False** and Q4 **True**, explaining that oblique trees produce simpler models/fewer rules. This is plausible for a sloping boundary, not a guarantee over all datasets. More coefficients per node and richer searches can overfit even with fewer leaves. Q25 correctly warns that nonlinear internal functions can overfit quickly. “Model tree for classification” there describes nonlinear decision functions, whereas slides 50–56 uses model tree for numeric regression at leaves.
10. **Stopping overlaps but the target criterion changes.** Q23 records **False**, contrasting numeric MSE tolerance with class homogeneity. Both types can use the same depth and sample-size limits. Numeric singleton leaves can have zero MSE; classification with contradictory labels need not reach purity. In the slides, “SDR small enough” should mean insufficient *further reduction* if used as a gain stopping rule; low node dispersion is a different condition.
11. **One leaf, one directly extracted path rule.** Q8's rule count is the number of leaves. Q20's grouped categorical splits can reduce branching, but a binary split alone does not guarantee fewer total leaves. The same tree can explain and predict (Q14). Q6's white-box rationale supports inspection in medicine; interpretability alone does not demonstrate predictive quality.
12. **Best local test is not globally optimal tree induction.** Q3 identifies selection of non-terminal conditions as central; greedy scoring still lacks look-ahead. In Q13 select C because its gain $0.49$ is largest. A feature's gain must be recomputed on each node's subset; an initially weak feature may become useful conditionally. Post-pruning can retain useful interactions that early stopping would never grow.
13. **Keep the ensemble mechanisms separate.** Bagging uses varied samples independently; random forests add varied feature sets; reweighting boosting emphasises mistakes; gradient boosting fits error corrections. Classification voting, mean/median numeric aggregation and additive residual correction are different prediction rules. Confidence $\alpha$ in boosting increases as weighted error decreases; its round error is not a held-out validation estimate.
14. **Paper-specific claims remain unverified.** Q26's recorded answer says GPMCC is not better than Cubist/NeuroLinear in **speed** and **generalisation**, while implying advantages for the other listed traits. The local materials do not supply the study/results needed to verify or generalise that comparison. Preserve it as the recorded assessment answer, not an established performance ranking. Q19's warning that complex local models fit noise is consistent with the model-tree mechanism.

## 5. Sources and unresolved questions

**Reference convention:** slide numbers equal PDF page numbers in the 59-page slide file. In the supplied 56-page textbook excerpt, PDF page 1 corresponds to printed p. 117, so **PDF page = printed page − 116**. For example, printed p. 149 is PDF p. 33. Textbook references above use checked section names and printed page numbers.

Local sources used:

- [Lecture slides: Topic 4](04informationBasedLearningTopic4.pdf): slides 4–41 classification trees, criteria, pruning, missing inputs and biases; 42–49 regression trees; 50–56 model trees; 57–58 oblique trees; 59 implementation pointers. Diagram-only examples on slides 26, 32, 44–46, 51–52 and 57 were also inspected visually.
- [Textbook chapter 4 excerpt](fundamentals_of_ml_2ed_ch.pdf): §§4.2–4.3 information measures and ID3; §4.4.1 alternative criteria; §4.4.2 numeric inputs; §4.4.3 numeric targets; §4.4.4 pruning; §4.4.5 ensembles; §4.5 summary and qualifications.
- [Personal notes and Personal Summary Questions](README.md): mechanism, metrics, pruning and ensemble notes; unanswered data-quality questions integrated above. The skipped regression section and blank gradient-boosting section are supplied from the textbook.
- [Saved quiz and Moodle explanations](quiz-04-information-based-learning.md): Q1–Q26 assessment emphasis, with conflicts qualified in §4; [entropy diagram](quiz-04-information-based-learning-images/entropy.png) for Q17.
- [Notebook](notebooks/ch04_notebook.ipynb): Markdown, code and saved entropy outputs inspected without execution. It illustrates categorical ID3, not the full set of lecture extensions.
- [Local ID3 implementation](src/ch04/id3.py): categorical/explicit numeric tests, majority defaults and prediction; [local pruning implementation](src/ch04/reduced_error_pruning.py): bottom-up validation pruning and documented complete-input requirements. These are learning implementations, not evidence that every C4.5/CART feature is present. In particular, the notebook's empty-data branch is placed after a purity check that would try to index an empty array; the overview uses the conceptual empty-partition rule, not that code ordering.

Material gaps and conflicts:

- The papers named on slide 3 (`Quinlan.pdf`, `Potgieter_etal.pdf`, `Wang_etal.pdf`, `Wickramarachchi_etal.pdf`) and the C4.5 book are absent. Detailed model-tree/evolutionary optimisers and the Q26 GPMCC/Cubist/NeuroLinear comparison cannot be verified from their original sources.
- Probabilistic decision trees are named only, so no unsupported mechanism is invented. Nonlinear oblique tests appear in the quiz without a complete induction method.
- The supplied sources conflict on gain-ratio logarithm conventions and differ on regression sample variance versus MSE/SDR; see §§3.2 and 4. Slide 49's all-surrogates-missing fallback remains underspecified. Broad robustness, balancing and oblique-complexity claims are qualified where used.
- No external sources were needed for this overview. Practical qualifications derived from the stated mechanisms are separated from recorded course answers; missing paper-specific evidence remains a gap.
