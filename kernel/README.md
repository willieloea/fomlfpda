#### Inductive bias of linear SVMs

Linear SVMs favour a linear separating hyperplane with the largest margin between the two classes. In soft-margin SVMs, this preference is balanced against penalties for margin violations. fundamentals_of_ml_2ed_ch

#### Rationale for the maximum-margin approach

Many hyperplanes may correctly separate the training data. The maximum-margin approach chooses the one with the greatest perpendicular clearance from the closest points in either class.

This provides a buffer against small changes in feature values: a point can move farther before crossing the decision boundary. The resulting preference for a less sensitive boundary is intended to improve generalisation to unseen data.

#### Computational cost

Standard kernel SVMs generally scale poorly with the number of training instances.

- Training involves a constrained optimisation problem whose size increases with the dataset.
- A full kernel matrix contains $n^2$ entries, making its storage and computation expensive for large $n$.
- Prediction can also become expensive when there are many support vectors, because each query requires kernel evaluations against them.

Linear SVMs are an important exception: specialised solvers can train them efficiently on large, sparse datasets without constructing a kernel matrix. Thus, distinguish standard kernel SVM training from scalable linear SVM implementations.

#### Kernels for linearly non-separable problems

A nonlinear mapping $\phi(\mathbf x)$ represents inputs in another, often higher-dimensional, feature space where a linear separator may exist.

A kernel computes inner products in that space directly:

$$
K(\mathbf x,\mathbf z)
=\phi(\mathbf x)^\top\phi(\mathbf z),
$$

without explicitly constructing the mapped vectors. A linear hyperplane in the transformed space corresponds to a nonlinear decision boundary in the original input space.
### Data quality issues
#### Noise
SVR is generally robust to moderate noise, provided its parameters are chosen appropriately:

- Deviations within the $\varepsilon$-tube incur zero loss, so the model does not need to fit every small fluctuation.
- Deviations outside the tube receive a linear penalty.
- Regularisation, controlled through $C$, discourages excessive fitting to noisy observations.

A very small $\varepsilon$, a large $C$, or an overly flexible kernel can reduce this robustness.
A hard-margin SVM is sensitive to noise:

- Noisy points can shift or rotate the boundary substantially and reduce the margin.
- Noise can make the classes overlap, so no feasible hard-margin solution exists.

Adapting to noise:

Use a soft-margin SVM, introducing slack variables $\xi_i\geq0$ that allow margin violations and misclassifications:

$$
y_i(\mathbf w^\top\mathbf x_i+b)\geq1-\xi_i.
$$

Training minimises

$$
\frac12\|\mathbf w\|^2+C\sum_i\xi_i,
$$

balancing a wide margin against the penalty for violations.

Advantages:

- They allow training when noisy or overlapping classes cannot be perfectly separated.
- They can improve generalisation by accepting some violations instead of distorting the boundary to accommodate every point.

Disadvantages:

- The parameter $C$ must be tuned: too large can make the model sensitive to noise; too small can allow excessive violations and underfitting.
- Robustness is limited: influential noisy points can still distort the boundary, and a linear model still cannot capture an inherently nonlinear class boundary. 06kernelBasedLearningTopic6

#### Outliers
SVR is relatively robust compared with ordinary least squares, but it is not immune to target outliers.

Standard SVR uses the $\varepsilon$-insensitive loss:

$$
L_\varepsilon(y,\hat y)
=\max\left(0,\lvert y-\hat y\rvert-\varepsilon\right).
$$

Errors outside the tube are penalised linearly, whereas least squares penalises them quadratically. This reduces outliers’ influence. Nevertheless, outliers outside the tube become support vectors and can pull the fitted function towards them, especially when $C$ is large.
