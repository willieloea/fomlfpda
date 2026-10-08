# file: id3.py

# Iterative Dichotomizer 3
import pandas as pd
import info_metrics # type: ignore

def id3(df: pd.DataFrame, info_metric, domains=None, parent_majority=None):
    """
    Iteratively partitions the dataset based on which feature is most informative
    and grows branches for each of the descriptive feature levels.
    """
    if domains is None:
        domains = {col: df[col].unique().tolist() for col in df.columns[:-1]}

    if df.empty:
        # a decision tree consisting of a leaf node with the label of the
        # majority target level of the dataset of the immediate parent node
        return parent_majority

    target_feature = list(df.columns)[-1]
    descriptive_features = list(df.columns)[:-1]

    current_majority = df[target_feature].mode()[0]

    if len(df[target_feature].unique()) == 1:
        # a decision tree consisting of a leaf node with label C
        # return df[target_feature].unique()
        return df[target_feature].iloc[0]  # returns 'conifer', not an array


    elif len(descriptive_features) == 0:
        # a decision tree consisting of a leaf node with the label of the
        # majority target level in D
        return df[target_feature].mode()[0]

    else:
        # 5. find the most informative descriptive feature
        data = info_metric(df)
        best_feature = max(data, key=data.get)

        # 6. make a new node and label it with the best descriptive feature
        tree = {best_feature: {}}

        # 7. partition D based on the best descriptive feature
        # vals = [val for val, _ in df.groupby(best_feature)]
        for val in domains[best_feature]:
            partition = df[df[best_feature] == val]
            # 8. remove the best descriptive feature from the dataset
            reduced_partition = partition.drop(columns=[best_feature])

            subtree = id3(
                df=reduced_partition,
                info_metric=info_metric,
                domains=domains,
                parent_majority=current_majority,
            )
            tree[best_feature][val] = subtree

        return tree

if __name__ == '__main__':
    # Read the dataset
    df = pd.read_csv("~/Laboratory/datasets/foml_datasets/tab4_3.csv")
    tree = id3(df.iloc[:, 1:], info_metrics.information_gains) # dropping the ID column
    # {feature_name: {value_1: subtree_or_leaf, value_2: subtree_or_leaf}}.
