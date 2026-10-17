# Chapter 4: Information-Based Learning
These notes are based on a university course, and draws from

- Chapter 4 of "Fundamentals of Machine Learning for Predictive Data Analytics"
  by Aoife D'Arcy, Brian Mac Namee, and John D. Kelleher.
- Course slides, and
- the Internet.

## 4.1 Big Idea
The big idea behind information based learning is that when you are trying to
predict something about a query instance, you want to ask questions about that
query that are maximally informative with respect to the feature you are trying
to predict.

The textbook puts it as follows: Information-based learning "algorithms
determine which descriptive features provide the most information about a target
feature and make predictions by sequentially testing the features in order of
their informativeness."

## 4.2 Fundamentals
Decision trees + Entropy + Measure of information gain.

### 4.2.1 Decision Trees
Decision trees carry out a series of tests on the values of the descriptive
features describing a query instance and use the answers to these tests to
determine the prediction.

```
                 Root
                /    \
           Criterion  Criterion
            /   \      /   \
          Leaf Leaf  Leaf  Leaf
```

- Root and internal nodes: tests
- Leaf nodes: predictions

Preference bias of decision trees: choose the tree that has fewer test nodes/
fewer levels.

The way we determine which descriptive features give us the most information
(lead us to the correct target values soonest) is through Shannon's entropy
model.

### 4.2.2 Shannon's Entropy Model

Shannon's entropy model is a measure of the impurity of elements in a set: How
uncertain are you in guessing the result of a random selection from the set?

Uncertainty is inversely related to probability:

- High probability → Low uncertainty
- Low probability → High uncertainty

The negative logarithm performs the probability-to-information conversion:

$$
I(x)=-\log_s(P(x))
$$

The entropy of a target feature is:

$$
H(t)=-\sum_{i=1}^{l}P(t=i)\log_s(P(t=i))
$$

Where:

- $P(t=i) \approx$ Probability that a randomly selected element is of type $i$.
- $l \approx$ Number of different types.
- $s \approx$ Arbitrary logarithmic base.

For a set containing only one target class:

$$
H(t)=-(1\cdot\log_s(1))=0
$$

Entropy is a measure of the heterogeneity of a set.

### 4.2.3 Information Gain

The informativeness of a feature is how much it tells us about the final label
of an instance. If a feature perfectly predicts an instance's target value, it
is maximally informative.

The measure of informativeness that we will use is information gain, which
measures the overall reduction in entropy of a set of instances by testing/
splitting on a descriptive feature.

There are three steps to calculate information gain:

1. Compute the entropy of the original dataset with respect to the target
   feature. This measures the uncertainty in the target feature.
2. For each descriptive feature, partition the dataset according to the values
   of the feature. Calculate the entropy with respect to the target feature for
   each partition.
3. Calculate the information gain by subtracting the weighted remaining entropy
   from the original entropy.

Step 1: Original entropy

$$
H(t,D)=-\sum_l P(t=l)\log_2(P(t=l))
$$

This measures uncertainty in the target feature.

Step 2: Remaining entropy

$$
\operatorname{rem}(d,D)= \sum_{l\in\operatorname{levels}(d)} \frac{|D\_{d=l}|}{|D|} \times H(t,D\_{d=l})
$$

This measures the remaining uncertainty after partitioning the dataset.

Step 3: Information gain

$$
IG(d,D)=H(t,D)-\operatorname{rem}(d,D)
$$

If the information gain equals the original dataset's entropy, you have
discovered a perfectly discriminative feature.

We now know how to select a feature for splitting a dataset. Next: How to grow a
tree.

## 4.3 Standard Approach: The ID3 Algorithm

ID3 = Dichotomiser 3.

It is a recursive algorithm that starts by creating a root node and works
towards leaf nodes.

1. Compute the information gain of each feature.
2. Partition the data according to the feature that had the highest information
   gain.
3. Add a node and label it with the feature.
4. Add a branch for each of the resulting partitions.
5. Remove the selected feature from the dataset.
6. For each branch, repeat the process unless all instances have the same target
   value, in which case a leaf node is added to the branch.

## 4.4 Extensions and Variations

1. Alternative feature selection and impurity metrics.
2. Handling continuous descriptive features.
3. Predicting continuous targets.
4. Tree pruning.
5. Model ensembles.

### 4.4.1 Alternative Feature Selection and Impurity Metrics

#### Gini Index

$$
\operatorname{Gini}(t,D)= 1-\sum_{l\in\operatorname{levels}(t)}P(t=l)^2
$$

Where:

- $t \approx$ Target feature of the query instance.
- $D \approx$ Dataset.
- $\operatorname{levels}(t) \approx$ Levels or domain of the target feature.

$\operatorname{Gini}(t,D)$ calculates how often target levels of instances in a
dataset would be misclassified if predictions were made by randomly sampling
labels according to the distribution of target levels in the dataset.

For one class:

$$
\begin{aligned}
\operatorname{Gini}(t,D) &=1-\sum_{l\in\operatorname{levels}(t)}P(t=l)^2\\\
&=1-1^2\\\
&=0
\end{aligned}
$$

0% of the time you would be wrong.

For $k$ equally likely classes:

$$
\begin{aligned}
\operatorname{Gini}(t,D) &=1-\sum_{i=1}^{k}\left(\frac{1}{k}\right)^2\\\
&=1-k\left(\frac{1}{k^2}\right)\\\
&=1-\frac{1}{k}
\end{aligned}
$$

__Derivation of the Gini Index:__

Let there be $n$ classes:

$$
C_1,C_2,\ldots,C_n
$$

Then the probability of making an error when guessing an item's class by
independently sampling from the class distribution is:

$$
\begin{aligned} P(\text{error}) &=P(C_1)P(C_1^c)+\cdots+P(C_n)P(C_n^c)\\\ &=\sum_{i=1}^{n}P(C_i)(1-P(C_i))\\\ &=\sum_{i=1}^{n}\left(P(C_i)-P(C_i)^2\right)\\\ &=\sum_{i=1}^{n}P(C_i)-\sum_{i=1}^{n}P(C_i)^2\\\ &=1-\sum_{i=1}^{n}P(C_i)^2 \end{aligned}
$$

#### Gain Ratio

$$
GR(d,D)= \frac{H(t,D)-\operatorname{rem}(d,D)} {-\sum_l P(d=l)\log_2(P(d=l))}
$$

Equivalently:

$$
GR(d,D)=\frac{IG(d,D)}{H(d,D)}
$$

The denominator measures the entropy of the descriptive feature used for
splitting.

### 4.4.2 Handling Continuous Descriptive Features

A categorical split makes a branch for each level. A continuous split instead
tests a threshold, making two branches: `feature < threshold` and
`feature >= threshold`.

At each node:

1. Sort the continuous feature's values.
2. Find adjacent distinct values whose target labels differ and use their
   midpoint as a candidate threshold. Equal values are grouped first; boundaries
   next to a mixed-label group are also candidates.
3. Calculate the information gain of each binary split:

   ```text
   IG(D, t) = H(D) - (|D_left| / |D|) H(D_left)
                  - (|D_right| / |D|) H(D_right)
   ```

4. Compare the continuous feature's best gain with the categorical features'
   gains, then split on the best feature/test.
5. Recurse. Remove a selected categorical feature, but keep a selected
   continuous feature so descendants can split it again at another threshold.

1. Sort instances by the value of the descriptive feature.
2. For each point where the target feature changes, consider a possible split.
3. Let \\(x\\) be the average of the two adjacent descriptive feature values.
4. Split the dataset according to whether an instance's descriptive feature value is $\leq x$.
5. Calculate the entropy with respect to the target feature based on that split.
6. The split that results in the most information gained is the best value to split on.

Features are explicitly declared continuous: numeric category codes can still
be categorical. The target column must be last, and identifiers should be
dropped.

### 4.4.3 Predicting Continuous Targets
(skipped for now)

### 4.4.4 Tree Pruning
Decision trees overfit because they recrsively split the training data until
leaf nodes are homogenous, which often results in noisy instances receiving
their own leaf nodes. The likelihood of overfitting increases as a tree gets
deeper because the resulting predictions are based on smaller and smaller
subsets as the dataset is partitioned after each test in the path.

There are two approaches to prune decision trees: 1) pre-pruning/early stopping,
and 2) post-pruning

__Early stopping__

- Stopping conditions that can be used:
   - stop creating subtrees when the number of instances in a partition falls
     below a threshold
   - when the information gain measured at a node is not deemed to be sufficient
     to make partitioning the data worthwhile
   - when the depth of the tree goes beyond a predefined limit
   - use statistical significance tests to determine the importance of subtrees,
     for example, chi-squared pruning.
- Advantages:
   - is computationally feasible
- Disadvantages:
   - may lead to useful subtrees never being created (underfitting)


__Post-pruning__

- A tree is grown until it perfectly learns the training set, then subtrees that
  likely capture noise of the training set are pruned.
- Pruning conditions that can be used:
   - Have a validation set and if the error rate (number of incorrect
     predictions) for the validation set is the same with our without the
     subtree, then it should be pruned.
   - If the number of instances in a node is below a threshold, remove it.

### 4.4.5 Model Ensembles
#### 4.4.5.1 Bagging
Bootstrap sampling is when $n$ instances are randomly selected with replacement
from a dataset that has $n$ instances.

Bagging is when multiple models are trained using different subsets of the
training dataset, e.g. bootstrap sampling, and the mode or median of predictions
is used as the ensemble's prediction.

Sub-bagging is when proper subsets of the training set is used to train models
in a bagging ensemble.

#### 4.4.5.2 Boosting
Boosting iteratively trains models where each subsequent model is biased to
correctly predict instances which previous models predicted incorrectly.

The dataset used to train the models is weighted, where the weight associated
with instance $i$ is $\mathbf{w}_i \geq 0$. These weights determine the
distribution over which the dataset is sampled. Initially, all weights are set
to $\frac{1}{n}$.

Models are iteratively added to the ensemble until some number of models have
been added. During each iteration, the following is done:

1. Induce a model using the weighted training dataset and calculate the total
   error $\epsilon$ as the sum of weights of incorrectly predicted instances.
   The training set is used as this 'validation set'.
2. Increase the weights of the incorrectly predicted instances:
   $$
   \mathbf{w}_i \leftarrow
   \mathbf{w}_i \times \left( \frac{1}{2 \times \epsilon} \right)
   $$
   and decrease the weights of the correctly predicted instances:
   $$
   \mathbf{w}_i \leftarrow
   \mathbf{w}_i \times \left( \frac{1}{2 \times (1-\epsilon)} \right)
   $$
3. Calculate a confidence factor, $\alpha$, for the model such that $\alpha$
  increases as $\epsilon$, e.g.:
   $$
   \alpha = \frac{1}{2} \times
   \log_{e} \left( \frac{1 - \epsilon}{\epsilon} \right)
   $$


#### 4.4.5.3 Gradient Boosting

# Personal Summary Questions
## 1. What is the main idea behind how these algorithms work?

## 2. Describe the standard algorithm for this prediction approach.

## 3. What are important variations of this algorithm and what do those do?

## 3. How does the standard algorithm take the following data quality issues, and can they be addressed?
### 3.a. Imbalanced classes
Decision trees are not robust to skew class distributions. If a leaf is induced
for a minority class, the leaf may be pruned if there aren't instances of that
leaf in the validation set.

### 3.b. Missing Values

### 3.c. Outliers
During tree induction, splits are determined by thresholds and class counts, not
distances. So classification trees will be relatively robust to outliers.
If noisy instances do induce separate branches, they will be pruned if their
added complexity does not exceed their predictive ability.

### 3.d. Noise
If a classification tree makes use of pruning, it will be robust to noise. Noisy
patterns will be alone on leaf nodes. When the leaf node is pruned, the parent
node will take the majority class of its child nodes, which will be the non-
noisy class.

### 3.e. Continuous/Categorical Descriptive features
Decision trees can work with both continuous and categorical descriptive
features.

### 3.f. Continuous/Categorical Target features
Decision trees can work with both continuous and categorical target features.

