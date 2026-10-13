"""
An implementation of the neartest neighbour algorithm.

Pseudocode description of the nearest neighbor algorithm.

Require: a set of training instances
Require: a query instance

1: Iterate across the instances in memory to find the nearest neighbor - this is
   the instance with the shortest distance across the feature space to the query
   instance.
2: Make a prediction for the query instance that is equal to the value of the
   target feature of the nearest neighbor.
"""
import math

from . import distance

def find_nearest_n(space, metric, query):
    """
    Find the nearest neighbour of the query in the space, according to the
    metric.
    """
    nn_dist = math.inf
    nn = None

    # Iterate accross the instances
    for instance in space:
        distance = metric(instance['features'], query['features'])

        # Find the instance with the shortest distance
        if distance < nn_dist:
            nn_dist = distance
            nn = instance

    return nn

def find_k_nearest_n(space, metric, query, k):
    """
    Find the k nearest neighbours of the query in the space, according to the
    metric.
    """
    k_nearest_n = []
    for instance in space:
        dist = metric(instance['features'], query['features'])
        # insert in sorted_space, preserving order
        if len(k_nearest_n) < k:
            k_nearest_n.append((dist, instance))
        else:
            farthest = max(k_nearest_n, key=lambda x: x[0])
            if dist < farthest[0]:
                k_nearest_n.remove(farthest)
                k_nearest_n.append((dist, instance))
    
    return [instance for dist, instance in k_nearest_n]

if __name__ == '__main__':
    space = [
        {'features': [0, 0], 'target': 'zero'},
        {'features': [0, 1], 'target': 'one'},
        {'features': [0, 2], 'target': 'two'},
        {'features': [0, 3], 'target': 'three'},
    ]
    query = {'features': [0, 0], 'target': None}
    nearest_n = find_nearest_n(space, distance.euclidean, query)
    print(nearest_n['target'])
