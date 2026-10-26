# 9. Evaluation
These notes are based on a university course, and draws from

- Chapter 9 of "Fundamentals of Machine Learning for Predictive Data Analytics"
  by Aoife D'Arcy, Brian Mac Namee, and John D. Kelleher.
- Course slides, and
- the Internet.

## 9.1 Big Idea
The point of evaluation is to give an accurate sense of how the model will
perform when deployed in the real world.

The data used to evaluate a model should not be the same as the data used to
train a model.

## 9.2 Fundamentals
Evaluation should help understand the trade-offs between different models, the
strengts/weaknesses, and what would be best in deployment.

## 9.3 Standard Approach: Misclassification Rate on a Hold-Out Test Set
## 9.4 Extensions and Variations
### 9.4.1 Designing Evaluation Experiments
### 9.4.2 Performance Measures: Categorical Targets
#### 9.4.2.1 Confusion matrix-based performance measures
True positive rate:
$$
TPR = \frac{TP}{(TP + FN)}
$$

True negative rate (specificity)
$$
TNR = \frac{TN}{(TN + FP)}
$$

False negative rate
$$
FPR = \frac{FP}{(TN + FP)}
$$
FNR = 1 - TPR

False positive rate
$$
FNR = \frac{FN}{(TP + FN)}
$$
FPR = 1 - TNR

#### 9.4.2.2 Precision, recall, and F1 measure
precission
$$
precission = \frac{TP}{(TP + FP)}
$$

recall
$$
recall = \frac{TP}{(TP + FN)}
$$

F1, or harmonic mean of precission of precission and recall
$$
F_1 = 2 \times \frac{precission \times recall}{precission + recall}
$$

#### 9.4.2.3 Average class accuracy
$$
average\ class\ accuracy =
\frac{1}{|levels(t)|} \sum_{l \in levels(t)} recall_l
$$

### 9.4.3 Performance Measures: Prediction Scores
### 9.4.4 Performance Measures: Multinomial Targets
### 9.4.5 Performance Measures: Continuous Targets
### 9.4.6 Evaluating Models after Deployment
## 9.5 Summary
Cohen's kappa $\kappa$

$$
\kappa = \frac{\text{Accuracy} - p_e}{1 - p_e} \in (-\infty, 1]
$$

where

$$
p_e =
\frac{(TP + FN)(TP + FP) + (TN + FP)(TN + FN)}
{(TP + TN + FP + FN)^2}
$$

- Compares how well the binary classifier performs against randomized accuracy (or expected agreement), $p_e$.
- $p_e$ is the accuracy you would expect by chance, assuming predictions and actual labels are random but based on their observed frequencies.
- Measures agreement between the predicted and real classes.
- Measures how much better the model is than random guessing.

Matthews' correlation coefficient
$$
\mathrm{MCC}
=
\frac{TN \times TP - FN \times FP}
{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}
\in [-1, 1]
$$

- Measures the correlation between the target and predicted values of the instances.
- Flows from Pearson’s correlation coefficient.
- Is not fooled by imbalanced data.
