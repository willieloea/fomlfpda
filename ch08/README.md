# 8. Deep Learning
These notes are based on a university course, and draws from

- Chapter 8 of "Fundamentals of Machine Learning for Predictive Data Analytics"
  by Aoife D'Arcy, Brian Mac Namee, and John D. Kelleher.
- Course slides, and
- the Internet.

## 8.1 Big Idea
## 8.2 Fundamentals
### 8.2.1 Artificial Neurons
### 8.2.2 Artificial Neural Networks
### 8.2.3 Neural Networks as Matrix Operations
### 8.2.4 Why Are Non-Linear Activation Functions Necessary?
### 8.2.5 Why Is Network Depth Important?
## 8.3 Standard Approach: Backpropagation and Gradient Descent
### 8.3.1 Backpropagation: The General Structure of the Algorithm
### 8.3.2 Backpropagation: Backpropagating the Error Gradients
### 8.3.3 Backpropagation: Updating the Weights in a Network
### 8.3.4 Backpropagation: The Algorithm
### 8.3.5 A Worked Example: Using Backpropagation to Train a Feedforward Network for a Regression Task
## 8.4 Extensions and Variations
### 8.4.1 Vanishing Gradients and ReLUs
### 8.4.2 Weight Initialization and Unstable Gradients
### 8.4.3 Handling Categorical Target Features: Softmax Output Layers and Cross-Entropy Loss Functions
### 8.4.4 Early Stopping and Dropout: Preventing Overfitting
### 8.4.5 Convolutional Neural Networks
### 8.4.6 Sequential Models: Recurrent Neural Networks and Long Short-Term Memory Networks
## 8.5 Summary

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

# Practice questions
1. sigmoid activation function in hidden and output layers of NN, and gradient descent.
   1. Should you scale target features?
      - if the target values fall outside the sigmoid's output range, yes.
   2. Should you scale input features?
      - Yes.
      - Large input magnitudes can produce large weighted sums, pushing sigmoid neurons into **saturation**, where their derivatives are close to zero. This leads to small gradients and slow learning.
      - Features with very different scales can produce poorly conditioned optimization: gradient descent may oscillate in some directions while progressing slowly in others. Scaling makes updates more balanced and generally improves convergence.
  3. Why shouldn't you initialize weights and biases with large values?
      - can cause sigmoid saturation from the start. For
        $$
        \sigma(z)=\frac{1}{1+e^{-z}},
        \qquad
        \sigma'(z)=\sigma(z)\bigl(1-\sigma(z)\bigr),
        $$
        large positive or negative $z$ gives $\sigma'(z)\approx 0$. Gradients propagated through these neurons become very small, so gradient descent changes their parameters very slowly and training may effectively stall.


2. Why is momentum needed for stochastic gradient descent?
  - In stochastic learning, weights are updated using individual randomly selected training examples. These updates provide noisy estimates of the full gradient, so successive updates may point in conflicting directions, causing oscillation. Momentum incorporates the previous update into the current one:
    $$
    \Delta w_t=-\eta\nabla E_t(w_t)+\alpha\Delta w_{t-1}.
    $$
    It smooths fluctuations, reduces oscillation, and builds speed in directions where gradients consistently agree, often accelerating convergence.

3. Can a single artificial neuron only seperate linearly separable classes?
  - True for a standard artificial neuron that computes a weighted sum followed by a threshold or sigmoid activation. Its decision is determined by
    $$
    w^\top x+b \geq c,
    $$
    so its decision boundary is the hyperplane
    $$
    w^\top x+b=c.
    $$
    It can therefore perfectly separate two classes only if they are **linearly separable** in its input space. It cannot perfectly classify XOR using the original inputs; that requires additional neurons or a nonlinear feature transformation.
