# Chapter 5 / Topic 3: Similarity-Based Learning — Course Overview

## 1. Topic at a glance

**Similarity-based learning (SBL)**—also called **instance-based learning**, **memory-based learning**, or **case-based reasoning (CBR)**—is built on the human intuition that the best way to make a prediction for a new situation is to search memory for similar past situations and predict what happened then ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), Section 5.1, pp. 181–182 / PDF pp. 1–2; [03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slides 4, 6).

- **Learning setting:** Supervised, **non-parametric**, **lazy learning** (with a direct extension to **feature imputation**). As a non-parametric approach, SBL makes no assumption about a global functional form (such as a linear boundary or polynomial curve) mapping descriptive features to the target ([03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slide 4). As a lazy learner, it delays abstracting from the training data until a query instance must be predicted ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), Section 5.5, p. 232 / PDF p. 52; [03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slide 6).
- **Inductive bias:** Instances with similar descriptive feature values (i.e., lying in the same small local neighbourhood in feature space) tend to have the same or similar target feature values ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), Section 5.3, p. 191 / PDF p. 11; [03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slide 4; [README.md](README.md), lines 20–23).
- **Core components:** Every similarity-based model requires two ingredients ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), Section 5.2, pp. 182–184 / PDF pp. 2–4; [03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slide 5; [README.md](README.md), lines 27–30):
  1. A **feature space** holding a memorised set of labelled input–target mappings (training instances).
  2. A **similarity or distance measure** defined over the descriptive feature space to quantify proximity between instances.

### Distinction between models, similarity measures, indexing structures, and preprocessing

To avoid conflating components in exams, distinguish the following categories covered in this topic:

1. **Prediction and imputation models (and substantive variants):**
   - **Nearest Neighbour ($1$-NN) Classifier:** The standard textbook baseline ($k = 1$) that assigns a query the target class of the single closest training instance ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), Section 5.3, pp. 187–191 / PDF pp. 7–11).
   - **$k$-Nearest Neighbour ($k$-NN) Classifier:** Generalises $1$-NN to take a majority vote across the $k$ closest training instances to mitigate noise ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), Section 5.4.1, pp. 191–193 / PDF pp. 11–13; [03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slides 7–9).
   - **Distance-Weighted $k$-NN (Shepard's Method):** Weights each neighbour's contribution by a function of inverse distance ($1/d$ or $1/d^2$) so closer neighbours outweigh distant ones ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), Section 5.4.1, pp. 194–196 / PDF pp. 14–16; [03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slides 7, 21).
   - **$k$-Nearest Neighbour ($k$-NN) Regressor:** Adapts $k$-NN (unweighted mean, outlier-resistant median, or distance-weighted average) to predict continuous numeric targets ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), Section 5.4.4, pp. 208–211 / PDF pp. 28–31; [03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slides 25–27).
   - **$k$-Nearest Neighbour ($k$-NN) Imputer:** Uses weighted partial-distance $k$-NN to fill in missing descriptive feature values using the mean/median (numeric) or mode (categorical) of the $k$ nearest neighbours ([03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slides 5, 18, 29).
2. **Similarity indices and distance metrics (not models themselves):**
   - *Numeric metrics/indices:* **Euclidean ($L_2$)**, **Manhattan ($L_1$)**, **Minkowski ($L_p$)**, **Chebyshev ($L_\infty$)**, **Cosine similarity**, and **Mahalanobis distance** ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), Sections 5.2.2, 5.4.5.2–5.4.5.3; [03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slide 11).
   - *Binary and categorical measures:* **Russel-Rao**, **Sokal-Michener (simple matching)**, **Jaccard**, **Hamming**, **Overlap**, and **Levenshtein (edit) distance** ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), Section 5.4.5.1; [03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slides 12–13).
   - *Mixed and missing-data measures:* **Gower's similarity** and **missing-value rescaled Euclidean distance** ([03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slides 14–15, 17).
   - *Non-tabular domain measures:* Image similarity measures, **Dynamic Time Warping (DTW)** and **Complexity-Invariant Distance (CID)** for time series, and **TF-IDF vectorisation** for text ([03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slide 28).
3. **Memory indexing structures:**
   - **$k$-d Trees ($k$-dimensional trees):** Balanced binary search trees that partition the feature space with axis-aligned hyperplanes to accelerate nearest-neighbour retrieval without changing the underlying $k$-NN prediction rule ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), Section 5.4.2, pp. 196–204 / PDF pp. 16–24).
4. **Preprocessing, resampling, and feature selection methods:**
   - *Scaling:* **Range normalisation (min-max scaling)** and **Z-score standardisation** ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), Section 5.4.3; [03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slide 24).
   - *Class-imbalance resampling:* **Tomek links** (undersampling) and **SMOTE** (oversampling) ([03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slides 22–23).
   - *Feature selection:* **Rank-and-prune filters** and **greedy local search wrappers** (**Forward Sequential Selection** and **Backward Sequential Selection**) to combat the **curse of dimensionality** ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), Section 5.4.6, pp. 223–230 / PDF pp. 43–50).

---

## 2. Quick comparison

| Model / Variant | Task / target type | Descriptive feature types and preparation | Main strength | Main limitation |
| --- | --- | --- | --- | --- |
| **Standard Nearest Neighbour ($1$-NN)** | Binary or multiclass classification (or regression) | Continuous/discrete numeric (normalised), binary, categorical, or mixed (via matching distance/similarity measure) | Zero training cost; captures arbitrary local decision boundaries; low bias | Highly sensitive to noise and mislabelled instances (high variance / overfitting) |
| **$k$-Nearest Neighbour ($k$-NN) Classifier** | Binary or multiclass classification | Same as $1$-NN; numeric features require normalisation so large-range features do not dominate distance | Smooths out local label/measurement noise via majority vote over $k$ neighbours | Vulnerable to class imbalance at large $k$; high prediction/storage cost; curse of dimensionality |
| **Distance-Weighted $k$-NN (Shepard's Method)** | Classification or regression | Same as $k$-NN; requires well-scaled distances to compute inverse-distance weights ($1/d$ or $1/d^2$) | Reduces influence of distant/majority-class neighbours even when $k$ is large (up to $n$) | Still affected by severe class imbalance or nearby noisy instances; expensive if $k = n$ |
| **$k$-NN with $k$-d Tree Index** | Classification or regression (accelerates retrieval) | Requires a true **distance metric** (must satisfy triangle inequality) on coordinate axes | Speeds up average query retrieval by pruning subtrees outside the target hypersphere | Degrades to exhaustive search in high dimensions (needs roughly $n \gtrsim 2^m$ instances); tree becomes unbalanced with insertions |
| **$k$-Nearest Neighbour ($k$-NN) Regressor** | Continuous numeric regression | Same as $k$-NN; descriptive features should be normalised before computing distances | Non-parametric local interpolation (mean, median, or distance-weighted mean) without global functional assumptions | Sensitive to target outliers when using the mean; cannot extrapolate beyond training target range |
| **$k$-Nearest Neighbour ($k$-NN) Imputer** | Missing-value imputation (preprocessing) | Incomplete numeric and/or categorical descriptive features using missing-aware weighted distances | Imputes missing values using local multivariate context (mean/median for numeric, mode for categorical) | Ineffective when too many features are missing per instance; computationally costly on large datasets |

---

## 3. Shared foundations: Feature space and similarity measures

Because all $k$-NN variants rely on the same geometric and distance foundations, this section consolidates the similarity measures and distance metrics covered in [fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf) (Sections 5.2.2 and 5.4.5) and [03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf) (Slides 10–15, 17, 28).

### 3.0.1 Distance metrics versus similarity indices

- **Distance vs. similarity:** A **distance measure** quantifies how *different* (far apart) two instances are (smaller values = more similar), whereas a **similarity measure** quantifies how *alike* two instances are (larger values = more similar; generally the inverse of distance) ([03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slide 10; [fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), Section 5.4.5, p. 211 / PDF p. 31).
- **Metric vs. index:** A mathematical **distance metric** $\text{metric}(\mathbf{a}, \mathbf{b})$ must satisfy four axioms for all instances $\mathbf{a}, \mathbf{b}, \mathbf{c}$ ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), Section 5.2.2, p. 184 / PDF p. 4):
  1. **Non-negativity:** $\text{metric}(\mathbf{a}, \mathbf{b}) \ge 0$
  2. **Identity (of indiscernibles):** $\text{metric}(\mathbf{a}, \mathbf{b}) = 0 \iff \mathbf{a} = \mathbf{b}$
  3. **Symmetry:** $\text{metric}(\mathbf{a}, \mathbf{b}) = \text{metric}(\mathbf{b}, \mathbf{a})$
  4. **Triangle inequality:** $\text{metric}(\mathbf{a}, \mathbf{b}) \le \text{metric}(\mathbf{a}, \mathbf{c}) + \text{metric}(\mathbf{b}, \mathbf{c})$
  
  Measures that violate one or more of these axioms are called **similarity indices** ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), Section 5.4.5, pp. 211–212 / PDF pp. 31–32). Standard brute-force $k$-NN can use either metrics or indices, but **$k$-d trees strictly require a true metric** (specifically the triangle inequality) to safely prune branches ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), p. 212 / PDF p. 32).

### 3.0.2 Measures for numeric descriptive features

Let $\mathbf{a}, \mathbf{b} \in \mathbb{R}^m$ be two instances across $m$ (denoted $N$ in the slides) numeric descriptive features:

| Measure | Formula | Key properties and when to use |
| --- | --- | --- |
| **Euclidean distance** ($L_2$, Minkowski $p=2$) | $$d_E(\mathbf{a}, \mathbf{b}) = \sqrt{\sum_{i=1}^m (\mathbf{a}[i] - \mathbf{b}[i])^2}$$ | Straight-line distance in Euclidean coordinate space; default metric in $k$-NN. Squaring differences weights a single large feature difference more heavily than many small differences ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), Eq. 5.1, pp. 185–187 / PDF pp. 5–7; [03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slide 11). |
| **Manhattan distance** ($L_1$, taxicab, Minkowski $p=1$) | $$d_{MH}(\mathbf{a}, \mathbf{b}) = \sum_{i=1}^m |\mathbf{a}[i] - \mathbf{b}[i]|$$ | Sum of absolute axis-aligned differences. Less influenced by a single large difference in one feature than Euclidean distance, and slightly cheaper computationally (no squaring or square root) ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), Eq. 5.2, pp. 185–187 / PDF pp. 5–7; [03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slide 11). |
| **Minkowski distance** ($L_p$) | $$d_M(\mathbf{a}, \mathbf{b}) = \left(\sum_{i=1}^m |\mathbf{a}[i] - \mathbf{b}[i]|^p\right)^{\frac{1}{p}}, \quad p > 0$$ | General family of $L_p$ metrics ($p=1$ is Manhattan; $p=2$ is Euclidean). Larger $p$ places greater emphasis on the largest single-feature difference ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), Eq. 5.3, p. 186 / PDF p. 6; [03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slide 11). |
| **Chebyshev distance** ($L_\infty$, chessboard) | $$d_\infty(\mathbf{a}, \mathbf{b}) = \lim_{p \to \infty} d_M(\mathbf{a}, \mathbf{b}) = \max_{i=1,\dots,m} |\mathbf{a}[i] - \mathbf{b}[i]|$$ | Extreme case of Minkowski distance as $p \to \infty$; returns only the maximum difference across any single feature ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), p. 186 n. 3 / PDF p. 6). |
| **Cosine similarity** (similarity index) | $$\text{sim}_{\text{COSINE}}(\mathbf{a}, \mathbf{b}) = \cos(\theta) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\|_2 \|\mathbf{b}\|_2} = \frac{\sum_{i=1}^m (\mathbf{a}[i]\times\mathbf{b}[i])}{\sqrt{\sum_{i=1}^m \mathbf{a}[i]^2}\sqrt{\sum_{i=1}^m \mathbf{b}[i]^2}}$$ | Measures the cosine of the inner angle $\theta$ between vectors from the origin. Normalises within each instance to the unit hypersphere, capturing **relative feature proportions (mix)** rather than vector magnitude, and ignoring $0 \times 0 = 0$ co-absences in **sparse non-binary data** (e.g., bag-of-words / TF-IDF). Range is $[0, 1]$ for non-negative features or $[-1, 1]$ generally ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), Eqs. 5.12–5.15, pp. 216–217 / PDF pp. 36–37). *(See [Section 4.3](#43-disagreements-and-corrections-in-course-materials) for a formula discrepancy on Slide 11.)* |
| **Mahalanobis distance** (metric) | $$\text{Mahalanobis}(\mathbf{a}, \mathbf{b}) = \sqrt{(\mathbf{a} - \mathbf{b})^T \boldsymbol{\Sigma}^{-1} (\mathbf{a} - \mathbf{b})}$$ | Uses the dataset's **inverse covariance matrix** $\boldsymbol{\Sigma}^{-1}$ to rotate and scale the coordinate system so features have unit variance and zero pairwise covariance. Downweights directions of large spread/variance and correlated (redundant) features; reduces to Euclidean distance when $\boldsymbol{\Sigma} = \mathbb{I}$ ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), Eq. 5.16, pp. 217–223 / PDF pp. 37–43). |

### 3.0.3 Measures for binary, categorical, mixed, missing, and non-tabular data

For two binary feature vectors $\mathbf{q}$ and $\mathbf{d}$ of length $|\mathbf{q}| = m$, define the $2 \times 2$ contingency counts ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), Table 5.12, pp. 213–214 / PDF pp. 33–34; [03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slide 12):
- $CP(\mathbf{q}, \mathbf{d})$ (or $a$ on Slide 12): **co-presence** — number of features where both are `true` ($1, 1$).
- $PA(\mathbf{q}, \mathbf{d})$ (or $b$ on Slide 12): **presence–absence** — number of features where $\mathbf{q}$ is `true` ($1$) and $\mathbf{d}$ is `false` ($0$).
- $AP(\mathbf{q}, \mathbf{d})$ (or $c$ on Slide 12): **absence–presence** — number of features where $\mathbf{q}$ is `false` ($0$) and $\mathbf{d}$ is `true` ($1$).
- $CA(\mathbf{q}, \mathbf{d})$: **co-absence** — number of features where both are `false` ($0, 0$), with $|\mathbf{q}| = CP + PA + AP + CA$.

| Data type | Measure | Formula | Key properties and when to use |
| --- | --- | --- | --- |
| **Binary** | **Russel-Rao similarity** | $$\text{sim}_{RR}(\mathbf{q}, \mathbf{d}) = \frac{CP(\mathbf{q}, \mathbf{d})}{|\mathbf{q}|}$$ | Ratio of co-presences to total binary features; focuses purely on shared `true` attributes ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), Eq. 5.9, p. 214 / PDF p. 34). |
| **Binary** | **Sokal-Michener similarity** | $$\text{sim}_{SM}(\mathbf{q}, \mathbf{d}) = \frac{CP(\mathbf{q}, \mathbf{d}) + CA(\mathbf{q}, \mathbf{d})}{|\mathbf{q}|}$$ | Counts both co-presences and co-absences equally. Suitable when shared absence is meaningful (e.g., medical symptom absence) ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), Eq. 5.10, p. 214 / PDF p. 34). |
| **Binary** | **Jaccard similarity** | $$\text{sim}_{J}(\mathbf{q}, \mathbf{d}) = \frac{CP(\mathbf{q}, \mathbf{d})}{CP(\mathbf{q}, \mathbf{d}) + PA(\mathbf{q}, \mathbf{d}) + AP(\mathbf{q}, \mathbf{d})} = \frac{a}{a + b + c}$$ | Ignores co-absences ($CA$). Ideal for **sparse binary data** (e.g., retail item purchases where most items are unbought). Undefined ($0/0$) if $CP = PA = AP = 0$ ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), Eq. 5.11, p. 215 / PDF p. 35; [03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slide 12). |
| **Binary** | **Hamming / Manhattan / Euclidean on binary** | $$d_{\text{Hamming}} = d_{MH} = b + c, \quad d_E = \sqrt{b + c}$$ | Counts mismatching binary features ($PA + AP$) ([03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slide 12). |
| **Categorical (discrete)** | **Overlap similarity** | $$\text{Overlap}(\mathbf{x}, \mathbf{y}) = \sum_{n=1}^N \delta(x_n, y_n)$$ | Counts the number of categorical features with the exact same value ([03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slide 13). |
| **Categorical (sets/discrete)** | **Set Jaccard similarity** | $$\text{sim}_J(\mathbf{x}, \mathbf{y}) = \frac{|\mathbf{x} \cap \mathbf{y}|}{|\mathbf{x} \cup \mathbf{y}|}$$ | Ratio of shared attribute values to total distinct attribute values across the two instances ([03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slide 13). |
| **Categorical / sequences** | **Levenshtein distance** | Minimum single-value edits (insertions, deletions, substitutions) to transform $\mathbf{x}$ into $\mathbf{y}$ | Measures sequence/string edit distance between discrete representations ([03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slide 13). |
| **Mixed (numeric + categorical)** | **Gower's similarity** | $$d_G(\mathbf{x}_1, \mathbf{x}_2) = \frac{\sum_{n=1}^{n_t} w_n s_n}{\sum_{n=1}^{n_t} w_n}, \quad s_n^{\text{num}} = 1 - \frac{|x_{1n} - x_{2n}|}{r_n}$$ | Combines numeric features ($r_n$ = range of feature $n$, so $s_n \in [0, 1]$) and categorical features (any categorical similarity normalised to $[0, 1]$) using non-negative weights $w_n$ ([03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slides 14–15). |
| **Incomplete (missing values)** | **Missing-value rescaled Euclidean distance** | Slide 17: $d'_E(\mathbf{x}_1, \mathbf{x}_2) = \sqrt{w \times d_E(\mathbf{x}_1, \mathbf{x}_2)}$ (standard convention: $\sqrt{w \sum_{n \in \text{obs}}(x_{1n} - x_{2n})^2}$) with $$w = \frac{\text{Total number of descriptive features}}{\text{Number of descriptive features without missing values}}$$ | Ignores features where either instance has a missing value and scales up the remaining distance by $w$. Only effective when few values are missing ([03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slide 17). *(See [Section 4.3](#43-disagreements-and-corrections-in-course-materials) for the square-root discrepancy.)* |
| **Non-tabular** | **Image, Time-Series, and Text measures** | *Images:* domain image similarity measures.<br>*Time series:* **Dynamic Time Warping (DTW)** or **Complexity-Invariant Distance (CID)**.<br>*Text:* clean text (strip punctuation, lowercase, remove stop words) $\to$ **TF-IDF vectorisation** ($\text{TF-IDF} = \text{TF} \times \text{IDF}$) $\to$ numeric $k$-NN (e.g., with cosine similarity). | Allows $k$-NN to operate on non-tabular data either by using a domain-specific distance metric directly or by vectorising into a numeric feature space ([03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slide 28). |

---

## 3.1 Nearest Neighbour ($1$-NN) and $k$-Nearest Neighbour ($k$-NN) Classifier (with Distance-Weighted $k$-NN and $k$-d Trees)

### Main idea and assumptions

The **Nearest Neighbour ($1$-NN)** algorithm is the standard baseline of similarity-based learning: it classifies a query instance by assigning it the target class of the single closest training instance in the feature space ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), Section 5.3, p. 187 / PDF p. 7). Geometrically, under Euclidean distance, $1$-NN partitions the feature space into a **Voronoi tessellation**, where each training instance defines a local **Voronoi region** containing all points closer to it than to any other training instance; aggregating neighbouring Voronoi regions of the same class yields an implicit global **decision boundary** ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), Section 5.3.1, pp. 189–190 / PDF pp. 9–10).

Because $1$-NN relies on single-instance local models, any noisy or mislabelled instance creates an erroneous region. The **$k$-Nearest Neighbour ($k$-NN)** algorithm generalises $1$-NN by expanding the local neighbourhood to the $k$ closest training instances and predicting the majority class, smoothing the decision boundary ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), Section 5.4.1, pp. 192–193 / PDF pp. 12–13; [03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slides 7–9). **Distance-weighted $k$-NN (Shepard's method)** further refines voting so closer neighbours receive higher weight than distant ones ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), pp. 194–195 / PDF pp. 14–15; [03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slide 21).

**Assumptions:**
- *To fit and predict:*
  1. **Local smoothness / cluster assumption (inductive bias):** Points that are close under the chosen distance/similarity measure belong to the same class with high probability ([03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slide 4).
  2. **Meaningful feature space and metric:** Coordinate differences (or similarity scores) reflect true domain similarity across all included features.
  3. **Sufficient sampling density:** The training set populates the feature space densely enough that a query's nearest neighbours are genuinely nearby; in a unit hypercube of dimension $m$ containing $k$ instances, sampling density is $\text{density} = k^{(1/m)}$, which drops exponentially as $m$ grows unless sample size grows exponentially ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), Eq. 5.17, pp. 224–225 / PDF pp. 44–45).
  4. **Stationarity and closed class set:** The target distribution does not introduce unseen classes at test time—a classifier trained on `{lion, frog, duck}` can never predict `platypus` ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), Section 5.7, p. 234 / PDF p. 54).
- *For statistical inference / theoretical guarantees:* Standard $k$-NN makes no parametric distributional assumptions (e.g., Gaussianity) to fit or predict; asymptotic consistency guarantees (covered in outside statistical theory referenced in Section 5.6, p. 233 / PDF p. 53) require $k \to \infty$ and $k/n \to 0$ as training size $n \to \infty$.

### How it learns and predicts

#### Standard $1$-NN and Unweighted $k$-NN

- **Training phase (lazy learning):**
  1. Apply any required preprocessing (e.g., range normalisation) to the training dataset $D = \{(\mathbf{x}_1, t_1), \dots, (\mathbf{x}_n, t_n)\}$ and store the transformation parameters.
  2. Store all training instances $D$ in memory (in the standard algorithm, as a simple list) ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), Section 5.3, p. 187 / PDF p. 7; [03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slide 6).
- **Prediction phase (given query $\mathbf{q}$ and hyperparameter $k$):**
  1. Apply the exact same preprocessing/normalisation parameters from the training set to $\mathbf{q}$ ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), Section 5.4.3, p. 207 / PDF p. 27).
  2. For every training instance $\mathbf{d}_i \in D$, compute the distance $\text{dist}(\mathbf{q}, \mathbf{d}_i)$ (or similarity $\text{sim}(\mathbf{q}, \mathbf{d}_i)$).
  3. Rank the instances by increasing distance (or decreasing similarity) and select the set $S$ of the $k$ closest instances $\mathbf{d}_1, \dots, \mathbf{d}_k$ ([03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slide 8; [fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), Algorithm 2, p. 188 / PDF p. 8).
  4. Return the majority target level in $S$ ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), Eq. 5.4, p. 192 / PDF p. 12):
     $$\mathbb{M}_k(\mathbf{q}) = \underset{l \in \text{levels}(t)}{\arg\max} \sum_{i=1}^k \delta(t_i, l)$$
     where:
     - $\mathbb{M}_k(\mathbf{q})$ is the predicted target class for query $\mathbf{q}$ using neighbourhood size $k$,
     - $\text{levels}(t)$ is the set of distinct class labels in the target domain and $l$ is a candidate class level,
     - $i \in \{1, \dots, k\}$ indexes the training instances $\mathbf{d}_i$ ordered by increasing distance from $\mathbf{q}$,
     - $t_i$ is the target class of neighbour $\mathbf{d}_i$, and
     - $\delta(t_i, l)$ is the **Kronecker delta function** ($\delta(t_i, l) = 1$ if $t_i = l$, and $0$ otherwise).
  5. **Tie-breaking:** If two or more classes tie for the maximum vote count, either break ties randomly or choose $k$ (e.g., odd $k$ in binary classification) or distance weighting to avoid ties ([03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slide 7).

#### Variant 1: Distance-Weighted $k$-NN (Shepard's Method)

Instead of giving every neighbour in $S$ an equal vote of $1$, each neighbour $\mathbf{d}_i$ receives a weight inversely proportional to its distance from $\mathbf{q}$:
- **Slide formulation (Shepard's method, inverse distance)** ([03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slide 21; [notebooks/ch05_notebook.ipynb](notebooks/ch05_notebook.ipynb), lines 197–216):
  $$w_i = \frac{1}{d(\mathbf{d}_i, \mathbf{q})}$$
- **Textbook formulation (inverse squared distance)** ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), Eqs. 5.5–5.6, pp. 194–195 / PDF pp. 14–15):
  $$w_i = \frac{1}{\text{dist}(\mathbf{q}, \mathbf{d}_i)^2}, \qquad \mathbb{M}_k(\mathbf{q}) = \underset{l \in \text{levels}(t)}{\arg\max} \sum_{i=1}^k \frac{1}{\text{dist}(\mathbf{q}, \mathbf{d}_i)^2} \times \delta(t_i, l)$$
- **Zero-distance edge case:** If $\text{dist}(\mathbf{q}, \mathbf{d}_i) = 0$ (an exact match), $w_i$ is undefined ($\infty$); the model handles this by returning the target class of the exact-matching training instance $\mathbf{d}_i$ ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), p. 194 n. 6 / PDF p. 14).

#### Variant 2: $k$-d Tree Accelerated Nearest Neighbour

A **$k$-d tree** (where $k$ denotes the dimensionality of the tree, *not* the number of neighbours) invests one-off preprocessing during training to index instances into a balanced binary tree so queries do not require exhaustive search ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), Section 5.4.2, pp. 196–203 / PDF pp. 16–23):
- **Tree construction (training):**
  1. Fix an ordering over the descriptive features (e.g., `SPEED`, then `AGILITY`, cycling back when exhausted).
  2. At the current node, select the next feature in the ordering and find the instance with the **median** value of that feature (using the median rather than the mean keeps the tree balanced even with outliers).
  3. Index that median instance at the current node—defining an axis-aligned **hyperplane** boundary along that feature's median value—and partition the remaining instances into the left branch ($<$ median) and right branch ($\ge$ median).
  4. Recurse depth-first on both branches until a partition contains fewer than two instances.
- **Tree search (prediction — Algorithm 3, [fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), p. 200 / PDF p. 20):**
  1. Descend from the root to a leaf node by following the branch matching $\mathbf{q}$'s value at each split (`descendTree(kdtree, q)`).
  2. Initialise `best` to that leaf node and `best-distance` to $\text{distance}(\mathbf{q}, \text{node})$ (defining a **target hypersphere** of radius `best-distance` around $\mathbf{q}$).
  3. Ascend and backtrack through the tree (`while node != NULL`):
     - If $\text{distance}(\mathbf{q}, \text{node}) < \text{best-distance}$, update `best` and `best-distance`.
     - Test whether the axis-aligned splitting hyperplane at `node` intersects the target hypersphere: `boundaryDist(q, node) < best-distance`, where `boundaryDist` is simply the 1D absolute difference $|\mathbf{q}[f] - \text{node}[f]|$ along the node's splitting feature $f$.
     - If `boundaryDist < best-distance`, a closer instance might exist on the other side of the hyperplane, so descend down the other branch (`descendTree(node, q)`). Otherwise, **prune** that entire subtree and move up to `parent(node)`.
  4. Return `best` when the root is exited. (To retrieve $k > 1$ nearest neighbours, track the $k$ closest nodes found so far and set `best-distance` to the distance of the $k$-th closest instance.)

#### Worked example (Athlete Draft Prediction)

From [fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf) (Tables 5.2–5.3, pp. 183, 189 / PDF pp. 3, 9), consider query $\mathbf{q} = (\text{SPEED}=6.75, \text{AGILITY}=3.00)$ and the three closest training instances under Euclidean distance:
- $\mathbf{d}_{18} = (7.00, 4.25)$, $\text{DRAFT} = \text{yes}$: $d_E(\mathbf{q}, \mathbf{d}_{18}) = \sqrt{(6.75 - 7.00)^2 + (3.00 - 4.25)^2} = \sqrt{0.0625 + 1.5625} = 1.2748$.
- $\mathbf{d}_{12} = (5.00, 2.50)$, $\text{DRAFT} = \text{no}$: $d_E(\mathbf{q}, \mathbf{d}_{12}) = \sqrt{(6.75 - 5.00)^2 + (3.00 - 2.50)^2} = \sqrt{3.0625 + 0.25} = 1.8200$.
- $\mathbf{d}_{10} = (4.25, 3.75)$, $\text{DRAFT} = \text{no}$: $d_E(\mathbf{q}, \mathbf{d}_{10}) = \sqrt{(6.75 - 4.25)^2 + (3.00 - 3.75)^2} = \sqrt{6.25 + 0.5625} = 2.6101$.

Predictions for $\mathbf{q}$:
- **$1$-NN ($k = 1$):** Nearest neighbour is $\mathbf{d}_{18}$ ($\text{yes}$), so $\mathbb{M}_1(\mathbf{q}) = \text{yes}$.
- **Unweighted $3$-NN ($k = 3$):** Neighbours are $\{\mathbf{d}_{18}(\text{yes}), \mathbf{d}_{12}(\text{no}), \mathbf{d}_{10}(\text{no})\}$. Vote counts: $\text{no} = 2$, $\text{yes} = 1 \implies \mathbb{M}_3(\mathbf{q}) = \text{no}$.
- **Distance-weighted $3$-NN ($w_i = 1/\text{dist}^2$):**
  - Score for $\text{yes}$: $1 / 1.2748^2 = 1 / 1.625 = 0.6154$.
  - Score for $\text{no}$: $1 / 1.8200^2 + 1 / 2.6101^2 = 1 / 3.3125 + 1 / 6.8125 = 0.3019 + 0.1468 = 0.4487$.
  - Because $0.6154 > 0.4487$, weighted $3$-NN predicts $\mathbb{M}_3(\mathbf{q}) = \text{yes}$.

### Supported data and tasks

- **Descriptive features:**
  - *Continuous and discrete numeric:* Supported natively via Minkowski ($L_1, L_2, L_p$), Cosine, or Mahalanobis metrics. Requires **normalisation** (e.g., range normalisation to $[0, 1]$) when feature scales differ ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), Section 5.4.3, pp. 204–208 / PDF pp. 24–28; [quiz-03-similarity-based-learning.md](quiz-03-similarity-based-learning.md), Q1).
  - *Binary and categorical (nominal/ordinal):* Supported natively **without** converting to continuous representations if an appropriate discrete similarity/distance measure is used (e.g., Russel-Rao, Sokal-Michener, Jaccard, Hamming, Overlap, Levenshtein) ([quiz-03-similarity-based-learning.md](quiz-03-similarity-based-learning.md), Q13; [03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slides 12–13). Warning: arbitrary numeric category codes should not be fed into Euclidean distance for nominal categories because the numeric differences impose a false ordering and magnitude.
  - *Mixed data:* Supported either by encoding categorical features into numeric form and normalising, or by combining type-specific similarities via a weighted sum such as **Gower's similarity** ([03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slides 14–15; [fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), Section 5.5, p. 231 / PDF p. 51).
  - *Non-tabular data:* Supported via domain similarity measures (images), DTW / CID (time series), or TF-IDF vectorisation (text) ([03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slide 28).
- **Target features:** Categorical—supports both **binary** and **multiclass** classification natively via majority (or plurality) voting across $\text{levels}(t)$ ([quiz-03-similarity-based-learning.md](quiz-03-similarity-based-learning.md), Q4; [fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), Eq. 5.4, p. 192 / PDF p. 12). Continuous targets require the $k$-NN regression variant ([Section 3.2](#32-k-nearest-neighbour-k-nn-regressor)).
- **Outputs:** Predicted class label $\mathbb{M}_k(\mathbf{q}) \in \text{levels}(t)$, vote counts $\sum_{i=1}^k \delta(t_i, l)$, or unnormalised distance-weighted class scores $\sum_{i=1}^k w_i \delta(t_i, l)$. Note that raw vote counts and inverse-distance sums are **scores**, not calibrated probabilities (though vote proportions can be used as naive local frequency estimates).

### Data quality and preprocessing

| Issue | Effect on the standard model and why | Relevant extension and how it helps | Preprocessing / mitigation and tradeoffs |
| --- | --- | --- | --- |
| **1. Missing values** | *Descriptive:* Standard $k$-NN uses **all** descriptive features in distance calculations (unlike decision trees, which only test path features), so a missing value in $\mathbf{q}$ or $\mathbf{d}_i$ prevents standard pairwise distance evaluation ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), Section 5.5, p. 231 / PDF p. 51).<br>*Target:* Unlabelled instances cannot vote in supervised $k$-NN. | **Missing-value rescaled distance** ignores missing features for a pair and scales non-missing feature distances by $w = \frac{\text{total features}}{\text{non-missing features}}$ ([03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slide 17). Thus, imputation **is not** always required to use $k$-NN ([quiz-03-similarity-based-learning.md](quiz-03-similarity-based-learning.md), Q7). | **Impute missing values** (e.g., via [$k$-NN Imputer](#33-k-nearest-neighbour-k-nn-imputer), mean/median/mode) fitted on the training split only. *Tradeoff:* Rescaled partial distance degrades rapidly when many values are missing ([03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slide 17), while imputation can introduce artificial similarity if imputed values are inaccurate. Drop training rows whose targets are missing. |
| **2. Outliers** | *Feature-based outliers:* Distance metrics are sensitive to extreme feature values, placing training outliers far from normal data. For **small $k$**, they are rarely selected as neighbours and have **little/no influence** on predictions. For **very large $k$**, they may enter the $k$-neighbourhood, but **majority voting limits their impact** to a single vote (though large $k$ itself causes underfitting) ([03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slide 16; [quiz-03-similarity-based-learning.md](quiz-03-similarity-based-learning.md), Q9).<br>*Target-based outliers:* Act as label noise (see row 3) ([03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slide 16). | **Distance-weighted $k$-NN (Shepard's method)** assigns tiny weights ($1/d$ or $1/d^2$) to distant feature-based outliers even if $k$ is large ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), p. 195 / PDF p. 15). **Manhattan ($L_1$) distance** is also less amplified by a single extreme feature difference than Euclidean ($L_2$) ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), p. 186 / PDF p. 6). | Note that **feature outliers distort range normalisation** ($\min/\max$) by compressing non-outlying values into a narrow sub-interval. Inspect data quality reports, clip/remove erroneous outliers in the training split, or keep $k$ moderate. |
| **3. Noise** | Both descriptive measurement noise (placing an instance in the wrong region) and target label noise corrupt local Voronoi cells. For **$1$-NN or small $k$**, queries near a noisy instance copy its wrong label (**high variance / overfitting**) ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), Section 5.4.1, pp. 191–192 / PDF pp. 11–12; [03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slide 19). | **Increasing $k$ in $k$-NN** dilutes individual noisy instances via majority voting, regularising the decision boundary ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), p. 192 / PDF p. 12; [quiz-03-similarity-based-learning.md](quiz-03-similarity-based-learning.md), Q5, Q10). **Distance-weighted $k$-NN** shrinks the error region around a noisy instance compared to $1$-NN, though a query landing right next to the noisy instance can still be misclassified ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), p. 195 / PDF p. 15). | Tune $k$ via cross-validation, or remove noisy/borderline training instances (e.g., **Tomek links** or **Condensed Nearest Neighbour**) within the training split ([03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slide 22; [fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), p. 233 / PDF p. 53). *Tradeoff:* Increasing $k$ too much causes underfitting and, on imbalanced data, majority-class dominance ([quiz-03-similarity-based-learning.md](quiz-03-similarity-based-learning.md), Q12). |
| **4. Skew class distributions** | Sensitive depending on $k$: as **$k$ increases**, the neighbourhood expands and the **majority class dominates** the vote (when $k$ exceeds twice the minority class size, the majority class can dominate the entire feature space). **Smaller $k$** is more robust to class imbalance ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), Section 5.4.1, p. 193 / PDF p. 13; [03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slide 20). | **Distance-weighted $k$-NN (Shepard's method)** ($w_n = 1/d$ or $1/d^2$) counteracts large-$k$ majority dominance by giving nearby minority neighbours much greater weight than distant majority neighbours ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), pp. 194–195 / PDF pp. 14–15; [03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slide 21; [quiz-03-similarity-based-learning.md](quiz-03-similarity-based-learning.md), Q3). However, under severe imbalance, even weighted $k$-NN can still be dominated ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), p. 196 / PDF p. 16). | Apply training-split resampling ([03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slides 21–23; [quiz-03-similarity-based-learning.md](quiz-03-similarity-based-learning.md), Q3):<br>• **Tomek links** (undersampling): identify mutual nearest-neighbour pairs of opposite classes and remove the majority-class instance from each pair, increasing class separation.<br>• **SMOTE** (oversampling): for each minority instance, find its $k$ nearest minority neighbours and generate synthetic instances along the line segments connecting them.<br>• **Tune $k$** (limited success if noisy). *Tradeoff:* Undersampling discards majority data; SMOTE can synthesise points inside majority regions if classes overlap. |
| **5. Irrelevant features** | Standard distance metrics weight **all features equally** ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), p. 231 / PDF p. 51). Irrelevant features add random coordinate differences to every distance calculation, pushing truly similar instances apart and lowering sampling density (**curse of dimensionality**, $\text{density} = k^{1/m}$) ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), Section 5.4.6, pp. 223–227 / PDF pp. 43–47). Standard $k$-NN **cannot ignore** them automatically. | No native mechanism in unweighted $k$-NN or Mahalanobis distance (Mahalanobis scales by variance/covariance of descriptive features without using target relevance). Weighted feature metrics require external weights. | Apply **feature selection** on the training split ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), Section 5.4.6, pp. 227–230 / PDF pp. 47–50):<br>• **Rank and prune (filters):** rank each feature independently (e.g., by information gain) and keep top $X\%$. Fast, but misses **interacting features** and keeps **redundant features**.<br>• **Wrapper greedy local search:** **Forward Sequential Selection (FSS)** (starts empty, adds features; fast when many features are irrelevant, but can miss interacting features) or **Backward Sequential Selection (BSS)** (starts full, removes features; preserves interacting features at higher computational cost). |
| **6. Features with different value ranges** | Highly sensitive: a feature with a large numeric range/variance (e.g., `SALARY` $\in [44\,200, 73\,200]$ vs. `AGE` $\in [26, 60]$) completely dominates squared or absolute differences, causing $k$-NN to virtually ignore small-range features ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), Section 5.4.3, pp. 204–206 / PDF pp. 24–26; [03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slide 24; [quiz-03-similarity-based-learning.md](quiz-03-similarity-based-learning.md), Q1). | **Mahalanobis distance** rescales all numeric features to unit variance using $\boldsymbol{\Sigma}^{-1}$ ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), p. 220 / PDF p. 40). **Gower's similarity** divides each numeric feature difference by its range $r_n$ ([03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slide 15). **Cosine similarity** normalises each instance vector to unit length, though within-instance normalisation still depends on relative feature scales. | Apply **range normalisation** to $[low, high]$ (typically $[0, 1]$): $$a'_i = \frac{a_i - \min(a)}{\max(a) - \min(a)} \times (high - low) + low$$ or **Z-score normalisation** (Slide 24 notes Z-score can be used provided the original distribution is normal). **Crucial:** Fit $\min(a), \max(a)$ (or $\mu, \sigma$) on the **training set only** and apply those exact parameters to query/test instances to prevent data leakage ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), pp. 206–207 / PDF pp. 26–27). |
| **7. Different descriptive feature data types** | Standard Euclidean/Manhattan metrics require numeric coordinates; applying them to arbitrary integer codes of nominal categories introduces artificial distances. | Substitute type-appropriate measures: **Russel-Rao, Sokal-Michener, Jaccard, Hamming** (binary); **Overlap, Jaccard, Levenshtein** (categorical); **Gower's similarity** or weighted custom metrics (mixed); **DTW/CID** (time series) ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), Section 5.4.5, p. 231; [03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slides 12–15, 28; [quiz-03-similarity-based-learning.md](quiz-03-similarity-based-learning.md), Q13). | Alternatively, encode categorical features into numeric features (e.g., binary one-hot indicators) or vectorise text via **TF-IDF**, then normalise and apply numeric similarity measures ([03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slides 14, 28). *Tradeoff:* One-hot encoding increases dimensionality $m$ and sparsity. |

#### Additional data-quality considerations

- **Redundant features and multicollinearity:** Unlike linear regression, collinearity does not cause matrix inversion failure in standard Euclidean $k$-NN, *but* redundant (strongly correlated) features are counted multiple times in Euclidean/Manhattan distance, overweighting the underlying factor they share and inflating dimensionality ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), Section 5.4.6, p. 227 / PDF p. 47). **Mitigations:** Use **Mahalanobis distance**, which multiplies by $\boldsymbol{\Sigma}^{-1}$ to remove pairwise covariance between features (provided $\boldsymbol{\Sigma}$ is invertible, i.e., no exact multicollinearity) ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), p. 220 / PDF p. 40), or use **wrapper feature selection (FSS/BSS)** to prune redundant features ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), pp. 227–229 / PDF pp. 47–49).
- **High dimensionality and sparse data vs. sparsely populated spaces:**
  - *Curse of dimensionality (sparsely populated feature space):* As dimension $m$ grows for fixed $n$, unit-hypercube sampling density $k^{1/m}$ collapses, leaving most of the space empty so "nearest" neighbours are far away ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), pp. 224–226 / PDF pp. 44–46). Real data partially mitigates this because instances cluster on lower effective dimensions and vary smoothly locally ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), p. 225 / PDF p. 45), and **distance-weighted $k$-NN** helps when feature space is sparsely populated ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), p. 211 / PDF p. 31), but **feature selection** remains essential.
  - *Sparse data (mostly zero feature values):* Distinct from a sparsely populated space ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), p. 225 n. 27 / PDF p. 45). In high-dimensional document or retail datasets where most entries are $0$, co-absences ($0, 0$) dominate Euclidean/Sokal-Michener calculations; use **Jaccard similarity** (binary) or **Cosine similarity** (non-binary) to ignore co-absences ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), pp. 215, 217, 223 / PDF pp. 35, 37, 43).

### Important choices and variants

- **Learned parameters vs. hyperparameters:** Because $k$-NN is a non-parametric lazy learner, it has **no learned parametric weights** fitted by an optimisation algorithm; the stored training set $D$ (and optional $k$-d tree structure or sample covariance matrix $\boldsymbol{\Sigma}$ for Mahalanobis distance) acts as the model memory.
- **Key hyperparameters:**
  1. **Number of neighbours $k$:** Controls the bias–variance tradeoff ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), Section 5.4.1, pp. 192–193 / PDF pp. 12–13; [03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slide 9):
     - **Small $k$ (e.g., $k = 1$):** **Low bias, high variance.** Creates jagged, locally complex decision boundaries that tightly track training instances; unstable and sensitive to training-set changes and noise (**overfitting**).
     - **Large $k$:** **High bias, low variance.** Smooths the decision boundary into a more stable model, but setting $k$ too large loses the true local pattern and biases toward the majority class (**underfitting**).
     - **Choosing $k$:** Best chosen via **hyperparameter tuning / evaluation experiments** (e.g., cross-validation). Heuristics such as $k = \sqrt{N}$ (Slide 9 defines $N$ as the number of descriptive features, whereas literature often uses sample size) should **not** be treated as standard practice ([03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slide 9).
  2. **Similarity / distance measure and Minkowski $p$:** Larger $p$ emphasises single-feature maximum differences; $p = 1$ (Manhattan) aggregates absolute differences linearly; Cosine captures vector angle/mix; Mahalanobis accounts for feature covariance; Jaccard/Russel-Rao/Sokal-Michener control whether binary co-absences count.
  3. **Voting scheme (unweighted vs. distance-weighted):** Distance weighting ($1/d$ or $1/d^2$) reduces sensitivity to the choice of $k$—even allowing $k = n$ (all training instances) while keeping decision boundaries smooth—though tuning $k < n$ can still better suppress noise ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), p. 195 / PDF p. 15).
  4. **Search index (brute-force list vs. $k$-d tree):** Affects retrieval speed, not exact $1$-NN/$k$-NN predictions (provided the distance is a true metric).

### Strengths, limitations, and practical use

- **When it is a good choice:**
  - Datasets with low-to-moderate dimensionality (or after feature selection) where classes form local clusters or complex non-linear boundaries.
  - Dynamic environments subject to **concept drift** (where the relationship between descriptive features and target changes over time, e.g., spam filtering): unlike eager learners (decision trees, neural networks) that require costly retraining, a lazy $k$-NN model can be updated immediately by adding verified new labelled instances to memory (and retiring old ones) ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), Section 5.5, pp. 232–233 / PDF pp. 52–53).
  - Domains where **interpretability** via case-based reasoning ("this instance was classified as $X$ because it is most similar to historical cases $A, B, C$") builds stakeholder trust ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), p. 231 / PDF p. 51).
- **When it is a poor choice:**
  - High-dimensional datasets with many irrelevant or redundant features (unless paired with aggressive feature selection or appropriate sparse/non-tabular similarity measures).
  - Large datasets with strict **low-latency prediction** or tight memory constraints.
- **Computational and storage costs** (for $n$ training instances and $m$ descriptive features):
  - *Storage:* $\mathcal{O}(nm)$ to keep the entire training set in memory.
  - *Training time:* $\mathcal{O}(1)$ for standard list storage (or $\mathcal{O}(nm)$ if normalising); constructing a $k$-d tree adds one-off indexing overhead.
  - *Prediction time per query:* Brute-force distance calculation across $n$ instances takes $\mathcal{O}(nm)$ (plus selecting the $k$ smallest distances, e.g., via a max-heap of size $k$ as explored in [src/ch05/heap.py](src/ch05/heap.py) and [notebooks/ch05_notebook.ipynb](notebooks/ch05_notebook.ipynb)). A balanced **$k$-d tree** reduces average query comparisons significantly when $n \gg 2^m$ (rule of thumb: requires roughly $n \approx 2^m$ instances for $m$ features), but degrades toward exhaustive search when $n \ll 2^m$ or when repeated dynamic insertions unbalance the tree (requiring periodic rebuilding) ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), Section 5.4.2, pp. 203–204 / PDF pp. 23–24).

---

## 3.2 $k$-Nearest Neighbour ($k$-NN) Regressor

### Main idea and assumptions

The **$k$-NN Regressor** adapts similarity-based learning to continuous numeric targets by replacing majority voting with local target interpolation—predicting the **average** (or **median**, or **distance-weighted average**) of the target values of the $k$ nearest training instances ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), Section 5.4.4, pp. 208–211 / PDF pp. 28–31; [03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slides 25–27).

**Assumptions:**
- Shares the feature-space and sampling-density assumptions of [$k$-NN Classification](#31-nearest-neighbour-1-nn-and-k-nearest-neighbour-k-nn-classifier-with-distance-weighted-k-nn-and-k-d-trees).
- **Local target continuity (smoothness):** Small changes in descriptive feature values correspond to small, smooth changes in the continuous target value within a local neighbourhood ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), Section 5.4.6, p. 225 / PDF p. 45).
- Zero-mean random target noise ($\mathbb{E}[\epsilon_i] = 0$) is assumed when relying on simple averaging over $k$ neighbours to smooth out target noise ([quiz-03-similarity-based-learning.md](quiz-03-similarity-based-learning.md), Q11).

### How it learns and predicts

- **Training phase:** Identical to classification—normalise descriptive features (not necessarily the target) on the training set and store the training instances $D = \{(\mathbf{x}_1, t_1), \dots, (\mathbf{x}_n, t_n)\}$ with $t_i \in \mathbb{R}$.
- **Prediction phase (given query $\mathbf{q}$ and hyperparameter $k$):**
  1. Normalise $\mathbf{q}$'s descriptive features using the training-set normalisation parameters.
  2. Compute distances $\text{dist}(\mathbf{q}, \mathbf{d}_i)$ to all training instances and identify the $k$ nearest neighbours $i = 1, \dots, k$.
  3. Compute the predicted target value $\hat{y} = \mathbb{M}_k(\mathbf{q})$ using one of three aggregation rules:
     - **Standard unweighted mean** ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), Eq. 5.7, p. 208 / PDF p. 28; [03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slide 25):
       $$\hat{y} = \mathbb{M}_k(\mathbf{q}) = \frac{1}{k} \sum_{i=1}^k t_i$$
     - **Outlier-resistant median variant** (used when target-based outliers are present) ([03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slide 25):
       $$\hat{y} = \text{median}(t_1, t_2, \dots, t_k)$$
     - **Distance-weighted $k$-NN regression** ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), Eq. 5.8, p. 210 / PDF p. 30):
       $$\mathbb{M}_k(\mathbf{q}) = \frac{\sum_{i=1}^k \frac{1}{\text{dist}(\mathbf{q}, \mathbf{d}_i)^2} \times t_i}{\sum_{i=1}^k \frac{1}{\text{dist}(\mathbf{q}, \mathbf{d}_i)^2}}$$
       where $\text{dist}(\mathbf{q}, \mathbf{d}_i)$ is the distance between query $\mathbf{q}$ and its $i$-th nearest neighbour $\mathbf{d}_i$, and $t_i \in \mathbb{R}$ is that neighbour's continuous target value. Notice that the denominator normalises the weights so they sum to $1$.

#### Worked example (Whiskey Reserve Price Prediction)

From [fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf) (Section 5.4.4, Tables 5.8–5.10, pp. 209–212 / PDF pp. 29–32), we predict `PRICE` from `AGE` $\in [0, 30]$ and `RATING` $\in [1, 5]$ for a query bottle with `AGE` $= 2$, `RATING` $= 5$:
1. **Range-normalise query to $[0, 1]$:**
   $$\text{AGE}' = \frac{2 - 0}{30 - 0} = 0.0667, \qquad \text{RATING}' = \frac{5 - 1}{5 - 1} = 1.0000$$
2. **Find $k = 3$ nearest neighbours (Euclidean distance on normalised features):**
   - $\mathbf{d}_{12} = (0.2000, 0.88)$, $t_{12} = 200.00$, $\text{dist} = 0.1828$, $w_{12} = 1/0.1828^2 = 29.9376$
   - $\mathbf{d}_{16} = (0.2667, 0.88)$, $t_{16} = 250.00$, $\text{dist} = 0.2358$, $w_{16} = 1/0.2358^2 = 17.9775$
   - $\mathbf{d}_{3} = (0.3333, 0.75)$, $t_{3} = 55.00$, $\text{dist} = 0.3655$, $w_{3} = 1/0.3655^2 = 7.4844$
3. **Predict:**
   - *Unweighted $3$-NN mean:* $\mathbb{M}_3(\mathbf{q}) = \frac{200.00 + 250.00 + 55.00}{3} = 168.33$.
   - *Median $3$-NN:* $\text{median}(55.00, 200.00, 250.00) = 200.00$.
   - *Distance-weighted $3$-NN:* $\frac{29.9376(200) + 17.9775(250) + 7.4844(55)}{29.9376 + 17.9775 + 7.4844} = \frac{5987.52 + 4494.38 + 411.64}{55.3995} = 196.64$ (or across all $k = 20$ instances in Table 5.10: $\frac{16\,249.85}{99.2604} = 163.71$).

### Supported data and tasks

- **Descriptive features:** Same as [$k$-NN Classification](#supported-data-and-tasks) (numeric, binary, categorical, mixed, or non-tabular with appropriate similarity/distance measures and feature normalisation).
- **Target features:** Continuous (or discrete numeric) target $t \in \mathbb{R}$ for **regression**.
- **Outputs:** A single continuous numeric prediction $\hat{y} \in \mathbb{R}$ bounded within $[\min_i t_i, \max_i t_i]$ of the training targets (cannot extrapolate outside the observed target range).

### Data quality and preprocessing

All descriptive-feature issues apply as in [$k$-NN Classification](#data-quality-and-preprocessing), with critical differences on **outliers**, **noise**, and **class skew** ([03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slide 27; [quiz-03-similarity-based-learning.md](quiz-03-similarity-based-learning.md), Q2, Q5, Q6, Q11):

| Issue | Effect on the standard model and why | Relevant extension and how it helps | Preprocessing / mitigation and tradeoffs |
| --- | --- | --- | --- |
| **1. Missing values** | *Descriptive:* Same as classification—missing feature values block standard distance calculation.<br>*Target:* Training instances with missing target $t_i$ cannot contribute to $\hat{y}$. | **Missing-value rescaled distance** ($w$-scaled Euclidean distance over non-missing features) ([03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slide 17). | Impute missing descriptive features on the training split (e.g., via [$k$-NN Imputer](#33-k-nearest-neighbour-k-nn-imputer)); drop training instances with missing targets. |
| **2. Outliers** | **Sensitive to both types in regression** ([quiz-03-similarity-based-learning.md](quiz-03-similarity-based-learning.md), Q2; [03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slide 27):<br>• *Feature-based outliers:* Ignored for small $k$, but **large $k$** pulls feature outliers into the neighbourhood, skewing the arithmetic average and causing **underfitting**.<br>• *Target-based outliers:* If any neighbour in the $k$-neighbourhood has an extreme target value $t_i$, the arithmetic mean $\frac{1}{k}\sum_{i=1}^k t_i$ is **biased directly toward the outlier** even for small/moderate $k$. | • For *target-based outliers*, replace the arithmetic mean with the **median** over the $k$ neighbours' target values ([03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slide 25).<br>• For *feature-based outliers* at larger $k$, **distance-weighted $k$-NN** downweights distant instances by $1/d^2$ ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), Eq. 5.8, p. 210 / PDF p. 30). | **Do not simply increase $k$ to fix target outliers in regression** ([quiz-03-similarity-based-learning.md](quiz-03-similarity-based-learning.md), Q6)—increasing $k$ includes distant instances (underfitting) while the mean remains sensitive to extreme values. Instead, use the **median** target rule or detect/remove erroneous target outliers during training data cleaning. |
| **3. Noise** | Zero-mean random target noise ($\mathbb{E}[\epsilon_i] = 0$) is smoothed out in expectation by averaging over the $k$ neighbours, and the model becomes less sensitive to target noise as $k$ increases ([quiz-03-similarity-based-learning.md](quiz-03-similarity-based-learning.md), Q5, Q10, Q11). *(See [Section 4.3](#43-disagreements-and-corrections-in-course-materials) for finite-$k$ variance qualification on Q11.)* Descriptive feature noise perturbs neighbour rankings, especially at very small $k$. | Increasing $k$ reduces the variance of the sample mean ($\sigma^2/k$); **distance-weighted $k$-NN** smooths predictions continuously across feature space. | Tune $k$ via cross-validation to balance noise smoothing against over-smoothing (underfitting). |
| **4. Skew class distributions** | **Not applicable:** Regression targets are continuous rather than discrete classes ([03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slide 27). *(Related regression issue: sparsely populated regions of the feature space or skewed target distributions, where distant neighbours pull the unweighted mean toward denser regions.)* | **Distance-weighted $k$-NN** improves accuracy when the feature space is **sparsely populated**, because distant among the $k$ neighbours receive much smaller weights ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), Section 5.4.4, p. 211 / PDF p. 31). | Use distance weighting or target transformations if the target distribution is heavily skewed. |
| **5. Irrelevant features** | Same as classification: corrupts pairwise distances and reduces sampling density $k^{1/m}$ via the curse of dimensionality ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), Section 5.4.6, pp. 223–227 / PDF pp. 43–47). | None native to standard $k$-NN regression. | Apply **filter** or **wrapper (FSS/BSS) feature selection** on the training split ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), pp. 227–230 / PDF pp. 47–50). |
| **6. Features with different value ranges** | Same as classification: unnormalised large-range descriptive features dominate distance rankings ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), Table 5.8–5.9, pp. 209–210 / PDF pp. 29–30). | **Mahalanobis distance** or **Gower's similarity** scales descriptive features automatically. | **Normalise descriptive features** (e.g., range normalisation to $[0, 1]$) using training-split parameters before computing distances ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), p. 209 / PDF p. 29). Target feature $t$ does not need normalisation for distance calculation. |
| **7. Different descriptive feature data types** | Same as classification: standard Euclidean distance requires continuous/numeric descriptive features. | Use discrete/categorical similarity indices, **Gower's similarity** for mixed features, or domain measures (DTW, TF-IDF) ([03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slides 12–15, 28). | Encode categorical descriptive features or use mixed similarity measures. |

### Important choices and variants

- **Number of neighbours $k$:** Small $k$ produces a step-wise, high-variance regression surface that interpolates local training points closely (overfitting); large $k$ averages over wide regions, flattening predictions toward the global mean (underfitting) ([03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slides 9, 26–27).
- **Aggregation function:**
  - *Arithmetic mean:* Default; optimal for smoothing zero-mean random noise ([quiz-03-similarity-based-learning.md](quiz-03-similarity-based-learning.md), Q5, Q11), but vulnerable to target outliers ([quiz-03-similarity-based-learning.md](quiz-03-similarity-based-learning.md), Q2, Q6).
  - *Median:* Robust to target-based outliers in the $k$-neighbourhood ([03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slide 25).
  - *Distance-weighted average:* Essential when the feature space is sparsely populated so distant neighbours in the top $k$ do not bias the prediction ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), p. 211 / PDF p. 31).

### Strengths, limitations, and practical use

- **Strengths:** Captures arbitrary non-linear continuous relationships locally without assuming a global functional form; easy to understand and explain via comparable historical cases (e.g., pricing a whiskey bottle or house from the $k$ most similar recent sales) ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), pp. 208–209 / PDF pp. 28–29). Implemented in `scikit-learn` as `KNeighborsRegressor` ([03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slide 29).
- **Limitations:** Cannot extrapolate trends beyond the minimum and maximum target values in the training set; suffers from the curse of dimensionality, high query-time cost $\mathcal{O}(nm)$, and sensitivity to target outliers under mean aggregation.

---

## 3.3 $k$-Nearest Neighbour ($k$-NN) Imputer

### Main idea and assumptions

Beyond predicting a designated target feature, similarity-based learning can be used as a **preprocessing imputer** (`KNNImputer` in `scikit-learn`) to estimate missing descriptive feature values ([03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slides 5, 18, 29; [README.md](README.md), line 25). It assumes that if two instances are similar across their *observed* descriptive features, they will also have similar values on the feature that is missing.

### How it learns and predicts

Given a dataset with missing descriptive entries ([03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slides 17–18):
1. **Identify** the missing values in an instance $\mathbf{x}$.
2. **Compute distances** between $\mathbf{x}$ and candidate donor instances using the **missing-value weighted distance measure** over the shared non-missing descriptive features ([03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slide 17):
   - Ignore any descriptive feature that has a missing value in either instance, and scale up the weight of the non-missing descriptive features by:
     $$w = \frac{\text{Number of descriptive features}}{\text{Number of descriptive features without missing values}}$$
3. **Find the $k$ nearest neighbours** that have observed (non-missing) values for the feature being imputed.
4. **Impute the missing value** ([03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slide 18):
   - For a **numerical descriptive feature**: compute the **average (mean) or median** over the corresponding feature values of the $k$ nearest neighbours.
   - For a **categorical descriptive feature** (referred to as "numerical categorical features" on Slide 18): compute the **mode** (most frequent category) over the corresponding feature values of the $k$ nearest neighbours.

### Supported data and tasks

- **Task:** Preprocessing / missing-data imputation (unsupervised with respect to the downstream target, or treating each incomplete descriptive feature in turn as a temporary target).
- **Descriptive features:** Supports both numeric descriptive features (imputed via mean/median) and categorical descriptive features (imputed via mode) ([03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slide 18).
- **Outputs:** A completed feature matrix with missing entries replaced by their locally imputed values.

### Data quality and preprocessing

| Issue | Effect on the standard model and why | Relevant extension and how it helps | Preprocessing / mitigation and tradeoffs |
| --- | --- | --- | --- |
| **1. Missing values** | Designed specifically to handle missing descriptive values via partial-distance scaling ($w$). However, if an instance has **too many missing values**, $w$ becomes large and distance estimates based on very few remaining features become unreliable ([03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slide 17). | Weighted distance scaling $w$ allows pairwise comparison without prior imputation. | Drop instances or features with excessive missingness before running $k$-NN imputation. Fit the imputer on the **training split only** to avoid data leakage. |
| **2. Outliers** | *Feature outliers* on observed features distort partial distances; *extreme values* on the feature being imputed skew the imputed value if the mean is used. | Use the **median** of the $k$ neighbours (for numeric features) or distance-weighted averaging to resist extreme neighbour values ([03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slide 18). | Inspect and clean extreme errors prior to imputation, and prefer median aggregation for skewed/outlier-prone numeric features. |
| **3. Noise** | Measurement noise on observed features perturbs neighbour selection; noise on the imputed feature is smoothed by averaging/mode over $k$ neighbours. | Choosing $k > 1$ smooths random noise on the imputed feature. | Tune $k$ to balance local specificity against noise smoothing. |
| **4. Skew class distributions** | For categorical features being imputed via the **mode**, a globally dominant category will dominate the mode if $k$ is set too large. | Use smaller $k$ or distance-weighted mode voting. | Do not include the downstream supervised target $y$ as a feature inside the imputer (which would leak test-set label information or fail at test time). |
| **5. Irrelevant features** | Uninformative features dilute the partial distance calculation across observed features. | Feature selection prior to or alongside imputation. | Remove known ID columns or irrelevant features before computing neighbour distances. |
| **6. Features with different value ranges** | Unnormalised large-range features dominate the partial Euclidean distance, so neighbours are chosen based only on large-scale features. | Normalise features before computing $d'_E$, or use range-normalised Gower similarity. | Apply range or Z-score normalisation (fitted on training data) before running `KNNImputer`. |
| **7. Different descriptive feature data types** | Numeric features require mean/median imputation; categorical features require mode imputation and an appropriate categorical/mixed distance ([03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slide 18). | Combine Gower's similarity for distance finding with mean/median (numeric) and mode (categorical) aggregation. | Separate numeric and categorical aggregation rules as specified on Slide 18. |

### Important choices and variants

- **Hyperparameters:** Number of donor neighbours $k$, aggregation statistic (mean vs. median for numeric; mode for categorical), and uniform vs. distance-weighted neighbour contributions.
- **Leakage prevention:** When used inside a supervised learning pipeline, the donor pool and feature normalisation parameters for `KNNImputer` must be restricted strictly to the **training split** (or current cross-validation training fold).

### Strengths, limitations, and practical use

- **Strengths:** Preserves multivariate relationships between features (unlike global column mean/median imputation, which ignores instance context).
- **Limitations:** Computationally expensive ($\mathcal{O}(n^2 m)$ to impute a full dataset), sensitive to feature scaling and high missingness rates ([03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slide 17).

---

## 4. Exam and comparison points

### 4.1 Answers to Personal Summary Questions (`README.md`)

1. **What is the main idea behind how these algorithms work? ([README.md](README.md), line 120)**
   - SBL models predict for a query instance by searching a memorised set of labelled training instances in a **feature space** using a **similarity or distance measure**, assuming that instances close together in descriptive feature space share the same or similar target values ([Section 1](#1-topic-at-a-glance)).
2. **Describe the standard algorithm for this prediction approach. ([README.md](README.md), line 122)**
   - In **training**, store all labelled training instances in a list (lazy learning). In **prediction**, compute the Euclidean distance from the query $\mathbf{q}$ to every stored training instance, find the single closest instance ($1$-NN) or $k$ closest instances ($k$-NN), and return its target class (or the majority class across the $k$ neighbours) ([Section 3.1](#how-it-learns-and-predicts)).
3. **What are important variations of this algorithm and what do those do? ([README.md](README.md), line 124)**
   - **$k$-NN ($k > 1$):** Takes a majority vote over $k$ neighbours to smooth out noise ([Section 3.1](#how-it-learns-and-predicts)).
   - **Distance-weighted $k$-NN (Shepard's method):** Weights each neighbour's vote/value by $1/d$ or $1/d^2$ to reduce the influence of distant neighbours and mitigate class imbalance or sparse feature-space issues ([Section 3.1](#variant-1-distance-weighted-k-nn-shepards-method)).
   - **$k$-d Trees:** Index training instances into a balanced binary tree split at feature medians to prune subtrees and speed up nearest-neighbour retrieval ([Section 3.1](#variant-2-k-d-tree-accelerated-nearest-neighbour)).
   - **$k$-NN for Regression:** Predicts the mean, median, or distance-weighted mean of the $k$ neighbours' continuous targets ([Section 3.2](#32-k-nearest-neighbour-k-nn-regressor)).
   - **$k$-NN Imputer:** Imputes missing descriptive values from the $k$ nearest neighbours using missing-rescaled distances ([Section 3.3](#33-k-nearest-neighbour-k-nn-imputer)).
   - **Alternative similarity/distance measures:** Manhattan, Minkowski, Cosine, Mahalanobis, Russel-Rao, Sokal-Michener, Jaccard, Gower's similarity, Levenshtein, DTW/CID ([Section 3.0](#3-shared-foundations-feature-space-and-similarity-measures)).
4. **How does the standard algorithm handle data quality issues, and can they be addressed? ([README.md](README.md), lines 126–132)**
   - **3.a. Imbalanced classes:** Small $k$ is relatively robust; large $k$ lets the majority class dominate. Addressed via **Shepard's method (distance weighting)**, **Tomek links** (undersampling), **SMOTE** (oversampling), or tuning $k$ ([Section 3.1 Table, Issue 4](#data-quality-and-preprocessing)).
   - **3.b. Missing values:** Standard distance fails on missing values, but imputation is *not* mandatory—distances can ignore missing features and rescale by $w = \frac{\text{total features}}{\text{non-missing features}}$, or missing values can be imputed first (including via $k$-NN imputation) ([Section 3.1 Table, Issue 1](#data-quality-and-preprocessing)).
   - **3.c. Outliers:** Distinguish **feature-based outliers** (far away in feature space; ignored at small $k$, only affect predictions at large $k$ where majority voting limits classification impact but mean aggregation skews regression) from **target-based outliers** (act as label noise in classification, mitigated by larger $k$; bias the mean in regression even at small $k$, mitigated by predicting the **median** target rather than increasing $k$) ([Section 3.1 & 3.2 Tables, Issue 2](#data-quality-and-preprocessing)).
   - **3.d. Noise:** Small $k$ overfits to noisy instances; increasing $k$ smooths out target/label noise via majority voting or averaging ([Section 3.1 & 3.2 Tables, Issue 3](#data-quality-and-preprocessing)).
   - **3.e. Continuous/Categorical descriptive features:** Continuous features use Minkowski/Cosine/Mahalanobis (requiring normalisation); categorical/binary features do *not* need conversion to continuous values if discrete measures (Jaccard, Sokal-Michener, Russel-Rao, Overlap, Hamming) are used; mixed features use Gower's similarity or encoding ([Section 3.0](#3-shared-foundations-feature-space-and-similarity-measures)).
   - **3.f. Continuous/Categorical target features:** Categorical targets (binary and multiclass) use majority/weighted voting; continuous targets use mean, median, or distance-weighted averaging ([Sections 3.1 and 3.2](#31-nearest-neighbour-1-nn-and-k-nearest-neighbour-k-nn-classifier-with-distance-weighted-k-nn-and-k-d-trees)).

### 4.2 High-yield exam traps and quiz-tested distinctions

- **Trap 1: "If a dataset has a lot of noise, larger values of $k$ *necessarily* result in better performance." $\to$ FALSE** ([quiz-03-similarity-based-learning.md](quiz-03-similarity-based-learning.md), Q12).
  - *Why:* While larger $k$ reduces sensitivity to noise (Q5, Q10), setting $k$ too large causes **underfitting** and, on **imbalanced datasets**, causes the majority class to dominate predictions ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), p. 193 / PDF p. 13; [03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slide 20).
- **Trap 2: "Increasing $k$ is a good way to handle target outliers in $k$-NN regression." $\to$ FALSE** ([quiz-03-similarity-based-learning.md](quiz-03-similarity-based-learning.md), Q6).
  - *Why:* In classification, target outliers (mislabelled instances) are outvoted when $k$ increases. In regression, however, an extreme target value $t_i$ still pulls the arithmetic mean $\frac{1}{k}\sum t_i$ toward the outlier, and making $k$ large pulls in feature-based outliers and causes underfitting (Q2). The course-taught fix for target-based outliers in regression is to take the **median** of the $k$ neighbours' targets ([03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slide 25).
- **Trap 3: "Feature-based outliers strongly affect $k$-NN classification at small $k$." $\to$ FALSE** ([quiz-03-similarity-based-learning.md](quiz-03-similarity-based-learning.md), Q9; [03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slide 16).
  - *Why:* Because distance metrics yield a large distance when even a single descriptive feature has an outlying value, a feature-based outlier sits far from normal query points and is not selected among the $k$ nearest neighbours for small $k$. Only at **large $k$** is it included, where **majority voting** still limits its influence in classification.
- **Trap 4: "Discrete/categorical descriptive features must be transformed into continuous representations to use $k$-NN." $\to$ FALSE** ([quiz-03-similarity-based-learning.md](quiz-03-similarity-based-learning.md), Q13).
  - *Why:* Dedicated discrete similarity/distance measures exist (Overlap, Jaccard, Hamming, Sokal-Michener, Russel-Rao, Levenshtein), so no continuous transformation is required.
- **Trap 5: "Missing values must always be imputed before applying $k$-NN." $\to$ FALSE** ([quiz-03-similarity-based-learning.md](quiz-03-similarity-based-learning.md), Q7; [03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slide 17).
  - *Why:* Distances can be computed directly over non-missing features and scaled by $w = \frac{\text{total features}}{\text{non-missing features}}$.
- **Trap 6: "The $k$ in $k$-d tree is the same $k$ as in $k$-NN." $\to$ FALSE** ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), p. 196 n. 8 / PDF p. 16).
  - *Why:* In $k$-NN, $k$ is the number of nearest neighbours retrieved; in $k$-d tree, $k$ refers to the dimensionality/levels of the spatial index.
- **Trap 7: "Sparse data and a sparsely populated feature space mean the same thing." $\to$ FALSE** ([fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), p. 225 n. 27 / PDF p. 45).
  - *Why:* **Sparse data** means most feature values in each instance vector are zero (mitigated by Jaccard or Cosine similarity, which ignore $0$–$0$ co-absences). A **sparsely populated feature space** means there are too few training instances relative to the dimensionality $m$ of the space (the curse of dimensionality, mitigated by feature selection and distance-weighted $k$-NN).
- **Trap 8: Image similarity in retrieval vs. Multidimensional Scaling (MDS) visualisation** ([quiz-03-similarity-based-learning.md](quiz-03-similarity-based-learning.md), Q8; [03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slide 28).
  - *Quiz point:* An image similarity measure that performs better than other measures at image retrieval will likely perform **similarly** to other measures when visualising image similarity using multidimensional scaling (MDS) (from the assigned reading `Rodden_etal.pdf`).

### 4.3 Disagreements and corrections in course materials

Where the slides, quiz key, and textbook differ or use imprecise formulas, keep both the **recorded course version** (for Moodle/exam alignment) and the **technical qualification** in mind:

1. **Random target noise in $k$-NN regression ([quiz-03-similarity-based-learning.md](quiz-03-similarity-based-learning.md), Q11):**
   - *Recorded course answer:* **False** to the statement *"Random noise in the target value of a dataset will affect the prediction value of kNN when applied to a regression problem"*, with Moodle explanation: *"Random noise is zero mean and will be smoothed out by the average in the prediction."*
   - *Technical qualification:* Zero-mean random noise $\epsilon_i$ has expectation $\mathbb{E}[\frac{1}{k}\sum_{i=1}^k \epsilon_i] = 0$, so averaging is **unbiased in expectation** and shrinks noise variance as $\sigma_\epsilon^2 / k$ as $k$ grows (which is why Q5 and Q10 state that $k$-NN becomes *less sensitive* to noise for larger $k$). For any finite $k$ (and especially small $k$ like $k=1$), the realised sample average of $k$ noisy values is not zero and *does* perturb individual predictions. In Moodle quizzes, follow the zero-mean smoothing interpretation (**False**).
2. **Cosine and Jaccard formulas on Slides 11–12 ([03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slides 11–12 vs. [fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), pp. 215–217 / PDF pp. 35–37):**
   - *Slide 11:* Writes *"Cosine distance: $d_C(\mathbf{x}_1, \mathbf{x}_2) = \|\mathbf{x}_1\|_2 \|\mathbf{x}_2\|_2 \cos \theta$"*. Technically, $\|\mathbf{x}_1\|_2 \|\mathbf{x}_2\|_2 \cos \theta = \mathbf{x}_1 \cdot \mathbf{x}_2$ is the **unnormalised dot product** (Textbook Eq. 5.13). **Cosine similarity** divides the dot product by the vector lengths: $\cos \theta = \frac{\mathbf{x}_1 \cdot \mathbf{x}_2}{\|\mathbf{x}_1\|_2 \|\mathbf{x}_2\|_2}$ (Textbook Eq. 5.14–5.15), and **cosine distance** is $1 - \cos \theta$.
   - *Slide 12:* Lists *"Jaccard distance: $\frac{a}{a+b+c}$"* and *"Cosine distance: $\frac{a}{(a+b)(a+c)}$"*. Technically, $\frac{a}{a+b+c}$ is **Jaccard similarity** (Textbook Eq. 5.11; Jaccard *distance* is $1 - \frac{a}{a+b+c} = \frac{b+c}{a+b+c}$), and binary **cosine similarity** has a square root in the denominator: $\frac{a}{\sqrt{a+b}\sqrt{a+c}}$.
3. **Missing-value rescaled Euclidean distance ([03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slide 17):**
   - *Slide 17:* Writes $d'_E(\mathbf{x}_1, \mathbf{x}_2) = \sqrt{w \times d_E(\mathbf{x}_1, \mathbf{x}_2)}$.
   - *Technical qualification:* Because $d_E(\mathbf{x}_1, \mathbf{x}_2) = \sqrt{\sum (x_{1n} - x_{2n})^2}$ already contains a square root, placing $d_E$ inside $\sqrt{w \times d_E}$ takes the fourth root of the squared differences. Standard missing-value Euclidean distance (as in `scikit-learn`'s `KNNImputer` / `nan_euclidean_distances`) multiplies the *squared* Euclidean distance over present features by $w$ inside the square root: $d'_E(\mathbf{x}_1, \mathbf{x}_2) = \sqrt{w \sum_{n \in \text{obs}} (x_{1n} - x_{2n})^2} = \sqrt{w} \times d_{E,\text{obs}}(\mathbf{x}_1, \mathbf{x}_2)$.
4. **Distance-weighting exponent ([03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slide 21 vs. [fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf), Eqs. 5.5, 5.6, 5.8):**
   - Slide 21 (Shepard's method) and [notebooks/ch05_notebook.ipynb](notebooks/ch05_notebook.ipynb) use inverse distance $w_n = \frac{1}{d(\mathbf{x}_n, \mathbf{x})}$, whereas the textbook uses inverse *squared* distance $w_i = \frac{1}{\text{dist}(\mathbf{q}, \mathbf{d}_i)^2}$. Both are valid members of Shepard's (1968) inverse-distance family $w_i = 1 / d^p$; state which exponent you are using in worked calculations.
5. **Z-score normalisation normality requirement ([03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slide 24):**
   - *Slide 24:* States that Z-score normalisation can be used *"provided that the original data distribution was normal"*.
   - *Technical qualification:* Mathematically, Z-score standardisation ($z = \frac{x - \mu}{\sigma}$) can be applied to any non-constant feature to achieve zero mean and unit variance, though on non-normal (skewed or heavy-tailed) distributions $\mu$ and $\sigma$ are sensitive to outliers and the resulting min–max ranges across features will not match.

---

## 5. Sources and unresolved questions

### Local sources used

- [03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf) — Lecture slides (*Machine Learning 441/741, Topic 3: Similarity-based Learning*, Prof. Andries Engelbrecht, Stellenbosch University, 2026, Slides 1–29).
- [fundamentals_of_ml_2ed_ch.pdf](fundamentals_of_ml_2ed_ch.pdf) — Textbook chapter (Kelleher, Mac Namee, and D'Arcy, *Fundamentals of Machine Learning for Predictive Data Analytics*, Chapter 5: "Similarity-Based Learning", printed pp. 181–236 / PDF pp. 1–56).
- [quiz-03-similarity-based-learning.md](quiz-03-similarity-based-learning.md) — Moodle Quiz 3 questions, recorded answer key, and explanations (Questions 1–13).
- [README.md](README.md) — Personal course notes, Python pseudocode snippets, chapter outline, and Personal Summary Questions.
- [notebooks/ch05_notebook.ipynb](notebooks/ch05_notebook.ipynb) and [notebooks/iris.ipynb](notebooks/iris.ipynb) — Worked Python notebooks implementing Euclidean/Manhattan metrics, $1$-NN, $k$-NN, and distance-weighted $k$-NN on the textbook athlete dataset.
- [src/ch05/distance.py](src/ch05/distance.py), [src/ch05/neartest_neighbour.py](src/ch05/neartest_neighbour.py), and [src/ch05/heap.py](src/ch05/heap.py) — Local Python module implementations of Euclidean, Manhattan, and Minkowski distances, $1$-NN/$k$-NN search, and a binary max-heap.

### Missing sources and gaps

- **External articles referenced on [03similarityBasedLearningTopic3.pdf](03similarityBasedLearningTopic3.pdf), Slide 28 (`Rodden_etal.pdf` and `Batista_etal.pdf`):** Slide 28 refers to two uploaded STEMlearn articles—`Rodden_etal.pdf` (for image similarity measures, tested in [quiz-03-similarity-based-learning.md](quiz-03-similarity-based-learning.md), Question 8) and `Batista_etal.pdf` (for temporal distance measures such as dynamic time warping and complexity-invariant distance)—and [quiz-03-similarity-based-learning.md](quiz-03-similarity-based-learning.md), Question 3 mentions **Boruta feature selection** as a distractor option. Neither `Rodden_etal.pdf` nor `Batista_etal.pdf` is present in this directory, so coverage of those papers is limited to the points explicitly stated on Slide 28 and in Quiz Question 8.
