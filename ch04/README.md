# Chapter 4: Information-Based Learning

## Decision trees with continuous features (Section 4.4.2)

A categorical split makes a branch for each level. A continuous split instead
tests a threshold, making two branches: `feature < threshold` and
`feature >= threshold`.

At **each node**:

1. Sort the continuous feature's values.
2. Find adjacent distinct values whose target labels differ and use their
   midpoint as a candidate threshold. Equal values are grouped first; boundaries
   next to a mixed-label group are also candidates.
3. Calculate the information gain of each binary split:

   ```text
   IG(D, t) = H(D) - (|D_left| / |D|) H(D_left)
                  - (|D_right| / |D|) H(D_right)
   ```

4. Compare the continuous feature's best gain with the categorical features'
   gains, then split on the best feature/test.
5. Recurse. Remove a selected categorical feature, but keep a selected continuous
   feature so descendants can split it again at another threshold.

Features are explicitly declared continuous: numeric category codes can still
be categorical. The target column must be last, and identifiers should be dropped.

### Run the textbook example

From `ch04/`:

```bash
uv run python -m ch04.continuous_example
```

This reads `~/Laboratory/datasets/foml_datasets/tab4_8.csv`, shows the threshold
scores and tree, and predicts the seven training rows.

The candidate ELEVATION thresholds reproduce Table 4.10:

| Threshold (feet) | Information gain (bits) |
| ---: | ---: |
| 750 | 0.3060 |
| 1350 | 0.1839 |
| 2250 | 0.5917 |
| 4175 | 0.8631 |

Since STREAM scores `0.3060` and SLOPE scores `0.5774`, the root uses
**ELEVATION >= 4175**. Its upper branch is a pure `conifer` leaf. The lower
branch is split recursively; ELEVATION remains available there.

### Use the implementation

```python
import pandas as pd
from ch04.id3 import id3, predict
from ch04.info_metrics import information_gains, continuous_information_gains
from ch04.draw_tree import print_ascii_tree

df = pd.read_csv("~/Laboratory/datasets/foml_datasets/tab4_8.csv")
data = df.drop(columns="ID")

scores = continuous_information_gains(data, "ELEVATION")
tree = id3(data, information_gains, continuous_features={"ELEVATION"})
print_ascii_tree(tree)

label = predict(tree, {"STREAM": True, "SLOPE": "steep", "ELEVATION": 4500})
print(label)  # conifer
```

Continuous nodes store the feature and numeric threshold as a tuple key:

```python
{("ELEVATION", 4175.0): {"<": lower_subtree, ">=": "conifer"}}
```

Categorical nodes retain the existing `{feature: {value: subtree}}` structure.
Both ASCII and Graphviz displays support these nodes. Existing categorical-only
calls to `id3(data, information_gains)` still work. The threshold scorer also
accepts `gini_information_gains` or `gain_ratios` as its `info_metric` argument.
