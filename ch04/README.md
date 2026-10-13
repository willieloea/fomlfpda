# Chapter 4: Information-Based Learning

## 4.4 Extensions and variations
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

Features are explicitly declared continuous: numeric category codes can still
be categorical. The target column must be last, and identifiers should be
dropped.

### 4.4.3 Predicting Continuous Targets

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

# Personal Summary Questions
## 1. What is the main idea behind how these algorithms work?

## 2. Describe the standard algorithm for this prediction approach.

## 3. What are important variations of this algorithm and what do those do?

## 3. How does the standard algorithm take the following data quality issues, and can they be addressed?
### 3.a. Imbalanced classes
### 3.b. Missing Values
### 3.c. Outliers
### 3.d. Noise
### 3.e. Continuous/Categorical Descriptive features
### 3.f. Continuous/Categorical Target features

