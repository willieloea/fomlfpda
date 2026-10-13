# 5. Similarity Based Learning
These notes are based on a university course, and draws from

- Chapter 5 of "Fundamentals of Machine Learning for Predictive Data Analytics"
  by Aoife D'Arcy, Brian Mac Namee, and John D. Kelleher.
- Course slides, and
- the Internet.

The following concepts were covered:

- Inductive bias of similarity based learning (SBL)
- The core components of a SBL model
- The nearest neighbour algorithm
- Lazy learning
- The $k$-nearest neighbour algorithm ($k$NN)
- Similarity measures for $k$NN
- $k$NN for regression
- Data quality issues

What is the inductive bias of similarity based learning?  
The inductive bias of similarity based learning (SBL) is that instances that
have similar descriptive feature values should have similar target feature
values. SBL is a form of supervised learning.

How can SBL be applied to classification, regression, and imputation?  

What are the core components of a SBL model?  
Any similarity based model has to have a set of memorised descriptive-target
feature mappings, and a similarity measure or metric defined on the descriptive
feature space.

What is the nearest neighbour algorithm?  
The nearest neighbour algorithm predicts the target value of a query instance by
assigning it the target value of the training instance that is most similar,
according to the similarity metric defined on the descriptive features.

```py
import math

class Instance:
    def __init__(self, descriptive, target):
        self.descriptive = descriptive
        self.target = target

    def __str__(self):
        return f"{self.descriptive} -> {self.target}"

def nearest_neighbour_prediction(training, metric, query):
  nn_dist = math.inf
  nn_prediction = None

  for t_inst in training:
    dist = metric(t_inst.descriptive, query.descriptive)
    if dist < nn_dist:
      nn_dist = dist
      nn_prediction = t_inst.target

  return nn_prediction
```

What is lazy learning?  
Lazy learning is when a model delays processing training data until a prediction
is needed. The nearest neighbour algorithm is an example of lazy learning
because it only uses training data when actual measurements have to be made with
the query instance.

How does the $k$-nearest neighbour algorithm ($k$NN) work?  
The $k$NN algorithm finds the $k$ training instances that are closest to the
query instance, and use them to predict the target value for the query instance.

```py
import math

def k_nearest_neighbour_prediction(training, metric, query, k):
  k_nn = []

  for t_inst in training:
    dist = metric(t_inst.descriptive, query.descriptive)
    if len(k_nn) < k:
      k_nn.append((dist, t_inst))
    else:
      farthest_nn = max(k_nn, key=lambda x : x[0])
      if dist < farthest_nn[0]:
        k_nn.remove(farthest_nn)
        k_nn.append((dist, t_inst))

  return select(query, k_nn)

def majority(query, k_nn):
  counts = {}

  for n in k_nn:
    if n.target not in counts:
      counts[n.target] = 0
    counts[n.target] += 1

  return max(counts, key=counts.get)
```

- Similarity measures for $k$NN
- $k$NN for regression
- Data quality issues

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


