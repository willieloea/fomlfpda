import csv
import random
from pathlib import Path


def predict(features, weights):
    return sum(w * x for w, x in zip(weights, features))


def error_delta(X, y, weights, j):
    return sum(
        (target - predict(features, weights)) * features[j]
        for features, target in zip(X, y)
    )


def gradient_descent(
    X, y, alpha=0.001, tol=1e-8, max_iter=10_000, seed=None
):
    if not X or not X[0]:
        raise ValueError("X must contain instances with at least one feature.")

    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of instances.")

    n_features = len(X[0])

    if any(len(row) != n_features for row in X):
        raise ValueError("All instances must have the same number of features.")

    rng = random.Random(seed)
    weights = [rng.uniform(-0.01, 0.01) for _ in range(n_features)]

    for _ in range(max_iter):
        previous_weights = weights.copy()

        # Update weights sequentially, as in the pseudocode.
        for j in range(n_features):
            weights[j] += alpha * error_delta(X, y, weights, j)

        largest_change = max(
            abs(w - previous)
            for w, previous in zip(weights, previous_weights)
        )

        if largest_change < tol:
            return weights

    raise RuntimeError("Did not converge within max_iter iterations.")

if __name__ == '__main__':
    path = Path("~/Laboratory/datasets/foml_datasets/tab7_1.csv").expanduser()

    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))

    # Scale size to keep gradient descent numerically manageable.
    scale = 1000.0

    X = [
        [1.0, float(row["SIZE"]) / scale]
        for row in rows
    ]
    y = [float(row["RENTAL PRICE"]) for row in rows]

    weights = gradient_descent(
        X,
        y,
        alpha=0.1,
        max_iter=100_000,
        seed=0,
    )

    # Convert the slope back to the original size units.
    intercept = weights[0]
    slope = weights[1] / scale

    print(f"price = {intercept:.4f} + {slope:.4f} * size")


    def predict_price(size):
        return predict([1.0, size / scale], weights)


    print(f"Predicted rental price for size 750: {predict_price(750):.2f}")
