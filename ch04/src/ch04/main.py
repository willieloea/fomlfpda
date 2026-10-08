import pandas as pd
import entropy # type: ignore

df = pd.read_csv("~/Laboratory/datasets/foml_datasets/tab4_2.csv")

target = df.iloc[:, -1]  # last column
probabilities = target.value_counts(normalize=True)
ps = probabilities.tolist()

entrop = entropy.entropy(ps)

print(entrop)
