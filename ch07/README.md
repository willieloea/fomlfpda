# 7. Error-Based Learning
These notes are based on a university course, and draws from

- Chapter 7 of "Fundamentals of Machine Learning for Predictive Data Analytics"
  by Aoife D'Arcy, Brian Mac Namee, and John D. Kelleher.
- Course slides, and
- the Internet.

## 7.1 Big Idea
You have a model with randomly initialized parameters. You make a prediction,
and update parameters based on the magnitude of the error.

## 7.2 Fundamentals
### 7.2.1 Simple Linear Regression
### 7.2.2 Measuring Error
### 7.2.3 Error Surfaces
## 7.3 Standard Approach: Multivariable Linear Regression with Gradient
## Descent
### 7.3.1 Multivariable Linear Regression
### 7.3.2 Gradient Descent
why can't we just take the partial derivative of everything to find the point
where all error will be a minimum?

How do things actually work?
The model is a linear equation. You have a vector of feature/variable values: d,
and you have a vector of model weights, w. The dot product of these two vectors
is what the model predicts. M_w(d) = w*d. (Oversimplification.) Error based
learning works by measuring the magnitude of a model's error, and adjusting the
model according to that error. A common error function is the sum of squared
errors:
$$
\frac{1}{2} \Sum^n_{i=1}(t_i - M_w(d_i))^2
$$

What is the derivative with respect to?
The partial derivatives are with respect to the model weights. We're looking for
the point where the gradient is zero - where the error will be as small as
possible.

### 7.3.3 Choosing Learning Rates and Initial Weights
### 7.3.4 A Worked Example
## 7.4 Extensions and Variations
### 7.4.1 Interpreting Multivariable Linear Regression Models
To determine the importance of a feature: use a statistical significance test.

We want to know if there is enough evidence to accept or reject a null
hypothesis

Steps:
1. State a null hypothesis
2. Compute a test statistic
3. Calculate the probability that a value equal to or greater than the test
   statistic is due to chance - the 'p-value'
4. Compare the p-value to a significance threshold. If the p-value is <= the
   threshold, the null hypothesis should be rejected.


1. "This feature does not have a significant impact on a model's predictions"
2. Compute the t-statistic
3. Lookup the p-value in a t-statistic table
4. Reject or accept the null hypothesis.

### 7.4.2 Setting the Learning Rate Using Weight Decay
Decay with
$$
\alpha_\tau = \alpha_0 \frac{c}{c+\tau}
$$
where typically:

- $\alpha_0$ = 1.0
- $c$ = 100
- $\tau$ = current iteration of gradient descent algorithm

### 7.4.3 Handling Categorical Descriptive Features
You can do one-hot encoding: each level of the categorical feature is
represented by a binary feature (1 = level was present, 0 = not).

### 7.4.4 Handling Categorical Target Features: Logistic Regression
#### 7.4.4.1 Predicting categorical targets using linear regression
If you have a linear regression model that produces an error surface, you can
have a threshold function like this:
$$
M_\mathbf{w}(\mathbf{d}) = 1 \text{if} \mathbf{w}\cdot\mathbf{d} \geq 0
M_\mathbf{w}(\mathbf{d}) = 0 \text{otherwise}
$$

But this means we won't be able to use gradient descent for two reasons:
1. The threshold function is discontinuous, so it is not differentiable
2. The model's predictions are not nuanced, they are very stark (0 or 1)

we will solve both with the logistic function

#### 7.4.4.2 Logistic regression

$$
M_\mathbf{w}(\mathbf{d}) = \frac{1}{1 + e^{-\mathbf{w}\cdot\mathbf{d}}}
$$


#### 7.4.4.3 A worked example of multivariable logistic regression
You should normalise descriptive features for logistic regression because it:
- speeds up and stabilizes gradient descent optimization,
- ensures fair regularization penalties in models like Lasso or Ridge, and
- prevents features with large numeric ranges from dominating the cost function
  or causing numerical overflow

### 7.4.5 Modeling Non-Linear Relationships
### 7.4.6 Multinomial Logistic Regression
### 7.4.7 Support Vector Machines
## 7.5 Summary

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

