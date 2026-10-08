# file: info_metrics.py

import pandas as pd
import entropy # type: ignore
import gini # type: ignore

def information_gains(df):
    # 1. Calcualte the entropy of the original dataset wrt the target feature
    target = df.iloc[:, -1]  # last column
    probabilities = target.value_counts(normalize=True)
    initial_entropy = entropy.entropy(probabilities.tolist())

    # 2. For each descriptive feature, partition the dataset according to the
    #    value of that descriptive feature, and calcualte the entropy of the
    #    feature.
    features = list(df.columns)
    descriptive_features = features[:-1]

    feature_remaining_entropies = dict()

    for descriptive_feature in descriptive_features:
        feature_remaining_entropy = 0
        for _, partition in df.groupby(descriptive_feature):
            partition_probs = partition[features[-1]].value_counts(normalize=True)
            partition_entropy = entropy.entropy(partition_probs.tolist())
            weight = len(partition) / len(df)
            feature_remaining_entropy += weight * partition_entropy

        feature_remaining_entropies[descriptive_feature] = \
            feature_remaining_entropy

    # 3. Subtract the entropy measure of step 2 from the entropy measure of step
    #    1 to give the information gain.
    info_gains = dict()
    for descriptive_feature in descriptive_features:
        info_gains[descriptive_feature] = \
            initial_entropy - feature_remaining_entropies[descriptive_feature]

    return info_gains


def gini_information_gains(df):
    # 1. Calcualte the gini of the original dataset wrt the target feature
    target = df.iloc[:, -1]  # last column
    probabilities = target.value_counts(normalize=True)
    initial_gini = gini.gini_index(probabilities.tolist())

    # 2. For each descriptive feature, partition the dataset according to the
    #    value of that descriptive feature, and calcualte the gini of the
    #    feature.
    features = list(df.columns)
    descriptive_features = features[:-1]

    feature_remaining_entropies = dict()

    for descriptive_feature in descriptive_features:
        feature_remaining_gini = 0
        for _, partition in df.groupby(descriptive_feature):
            partition_probs = partition[features[-1]].value_counts(normalize=True)
            partition_gini = gini.gini_index(partition_probs.tolist())
            weight = len(partition) / len(df)
            feature_remaining_gini += weight * partition_gini

        feature_remaining_entropies[descriptive_feature] = \
            feature_remaining_gini

    # 3. Subtract the gini measure of step 2 from the gini measure of step
    #    1 to give the information gain.
    info_gains = dict()
    for descriptive_feature in descriptive_features:
        info_gains[descriptive_feature] = \
            initial_gini - feature_remaining_entropies[descriptive_feature]

    return info_gains

def gain_ratios(df):
    """Calculates Gain Ratio for each descriptive feature."""
    # 1. Get the standard Information Gains (the numerator)
    info_gains = information_gains(df)

    gain_ratios = dict()

    # 2. For each feature, calculate the Split Information (the denominator)
    for feature, ig in info_gains.items():
        # Probabilities of the feature's own values: P(d = l)
        feature_probs = df[feature].value_counts(normalize=True).tolist()

        # Denominator: entropy of the descriptive feature
        split_info = entropy.entropy(feature_probs)

        # 3. Handle edge-case: avoid ZeroDivisionError if a feature has only 1 unique value
        if split_info == 0:
            gain_ratios[feature] = 0.0
        else:
            gain_ratios[feature] = ig / split_info

    return gain_ratios


if __name__ == '__main__':
    # Read the dataset
    df = pd.read_csv("~/Laboratory/datasets/foml_datasets/tab4_2.csv")
    info_gains = information_gains(df)
    for info_gain in info_gains:
        print(f"{info_gain} IG: {info_gains[info_gain]}")

    gini_info_gains = gini_information_gains(df)
    for gini_info_gain in gini_info_gains:
        print(f"{gini_info_gain} IG: {gini_info_gains[gini_info_gain]}")
