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
### 7.4.2 Setting the Learning Rate Using Weight Decay
### 7.4.3 Handling Categorical Descriptive Features
### 7.4.4 Handling Categorical Target Features: Logistic Regression
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
