"""Various distance metrics

This module contains a variety of distance metrics, as discussed in chapter 5 of
Fundamentals of Machine Learning for Predictive Data Analytics by Kelleher et
al.
"""
import math
from collections.abc import Iterable


def euclidean(x, y):
    """Euclidean distance between two vectors"""
    assert isinstance(x, Iterable)
    assert isinstance(y, Iterable)

    return math.sqrt(sum((a - b) ** 2 for a, b in zip(x, y)))


def manhattan(x, y):
    """Manhattan distance between two vectors"""
    assert isinstance(x, Iterable)
    assert isinstance(y, Iterable)

    return sum(abs(a - b) for a, b in zip(x, y))


def minkowski(x, y, p=1):
    """Minkowski distance between two vectors"""
    assert isinstance(x, Iterable)
    assert isinstance(y, Iterable)

    return (sum(abs(xi-yi)**p for xi, yi in zip(x,y)))**(1/p)

if __name__ == '__main__':
    x = [43, 94]
    y = [98, 4]

    print(f"The Euclidean distance between {x} and {y} is {euclidean(x, y):.2f}")
    print(f"The Manhattan distance between {x} and {y} is {manhattan(x, y)}")
    print(f"The Minkowski distance between {x} and {y} is {minkowski(x, y)}")
