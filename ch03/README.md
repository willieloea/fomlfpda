# 3. Data Exploration
These notes are based on a university course, and draws from

- Chapter 3 of "Fundamentals of Machine Learning for Predictive Data Analytics"
  by Aoife D'Arcy, Brian Mac Namee, and John D. Kelleher.
- Course slides, and
- the Internet.

## 3.1 The Data Quality Report
The data quality report is an important tool in getting to know the data we are
studying.

A data quality report must include:

- two tabular reports:
   - one for continuous features and one for categorical features
   - columns for both features:
      - feature
      - count
      - % missing
      - cardinality
   - columns (statistical properties) for continuous features:
      - 1st quartile
      - mean
      - median
      - 3rd quartile
      - maximum
      - Standard deviation
   - columns (statistical properties) for categorical features:
      - mode
      - mode frequency
      - mode %
      - 2nd mode
      - 2nd mode frequency
      - 2nd mode %
   - each row represents a specific feature
- visualizations that illustrate the distribution of the values in each feature
   - histogram for each continuous feature
   - bar plot for each categorical feature


| Feature | Count | % Missing | Cardinality | 1st Quartile | Mean | Median | 3rd Quartile | Maximum | Standard deviation |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |

: Table 1: Columns (statistical properties) for Continuous Features


| Feature | Count | % Missing | Cardinality | Mode | Mode Frequency | Mode % | 2nd Mode | 2nd Mode Frequency | 2nd Mode % |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |

: Table 2: Columns (statistical properties) for Categorical Features


## 3.2 Getting to Know the Data
These histograms and bar plots can show us if a dataset is skew

During data exploration, we simply have to identify which plots match with which
distributions. Later on we can use this information to decide which ML model to
use.

distributions and what they might mean:
- uniform     -> perhaps IDs
- normal      -> natrually occuring event
- left skew   -> (`__^`) -> values tend to be high
- right skew  -> (`^__`) -> values tend to be low
- exponential -> outliers are likely, and high values are not uncommon.
- multimodal  -> frequent if a measurement is made between two groups.
              -> note that the mean is no longer 'representative'.
              -> seperate peaks could associate with target feature.

## 3.3 Identifying Data Quality Issues
A data quality issue is loosely defined as anything unusual about the data.

Common data quality issues include:

- missing values
- irregular cardinality
- outliers

data quality issues can be due to:

- valid data
   - not necessarily correct, take note of issues that may influence the model
   - make a 'data quality plan' = Feature + Data Quality Issue + Potential Handling Strategies
- invalid data
   - errors in generating the dataset, usually with calculating derived features
   - these errors should be addressed
      - regenerate the analytics base table
      - re-create the data quality report

### 3.3.1 Missing Values
Why might there be missing values?

- mistakes in the data collection or derivation process
- some information would only be collected after a certain date
- people may choose not to provide certain information (e.g. age/gender)

if >= 60% of a feature is missing, it's best to remove it.

### 3.3.2 Irregular Cardinality
Cases:

1. Card = 1: if not due to error, remove
2. Low card: check if continuous feature should be categorical
3. Higher than expected: possibly because multiple values are used to represent
   the same thing, e.g. male, m, M, etc.
4. High card: not a mistake, but a problem for certain algos; to be noted in
   data quality plan

### 3.3.3 Outliers
There are two types of outliers:

- invalid outliers
   - often due to entry errors
- valid outliers

Ways of identifying outliers:

1. ensure that the min/max is within the range you'd expect (e.g. >0 for age)
2. compare the gap between mean, quartiles, and max/min. large dif -> outlier
3. Use box plots or histograms to notice outliers

Some ML algos don't work well with outliers, so this should be noted in the data
quality plan.

## 3.4 Handling Data Quality Issues
### 3.4.1 Handling Missing Values
Approaches for handling missing values:

1. Drop the feature with missing values
  - (-) (often needless) loss of data
2. Derive a binary feature to indicate missing values and discard original
   feature
  - (+) missingness has relationship to target value
3. Complete case analysis
  - remove instances with missing values.
  - (-) may remove too many instances or introduce bias
4. Impute missing values
  - Replaces missing values with plausible estimated values
    - Continuous: media or mean (or a model)
    - Categorical: mode (or a model)
  - (-) risks biasing dataset (avoid imputation if >30% of values are missing)
  - (+) avoids data loss

### 3.4.2 Handling Outliers
Approaches for handling outliers:
1. Clamping values exceeding upper and lower thresholds to those thresholds
  - Thresholds can be manually set, or calculated from data. E.g.
    - quartiles, plus/minus 1.5 times inter-quartile range
    - 2 standard deviations from the mean (if data is normally distributed)

## 3.5 Advanced Data Exploration
### 3.5.1 Visualizing Relationships between Features
### 3.5.2 Measuring Covariance and Correlation

## 3.6 Data Preparation
### 3.6.1 Normalization
### 3.6.2 Binning
Used to convert a categorical feature into a continuous one.
Define a series of ranges and if instances fall in that range, increase the
count for that bin.

Few bins:  
- loss of information, wipes out information of trends in the data.
- (advantage) large number of instances in each bin.

Many bins:  
- can capture the noise of the sample/training data.
- (advantage) more closely aligns with or reflects trends in data

__Equal width binning__:  
- bin instances based on the width of continuous feature bin ranges.
- Advantages:
  - intuitive: Easy to understand and explain.
  - preserves scale relationships: Keeps a direct, linear mapping to the
    original physical scale (e.g., age, income).
- Disadvantages:
  - Sensitive to outliers
  - Risk of sparse/empty bins


Equal frequency binning:
- sort instances, and bin instances according to number of instances divided by
  the number of bins.
- Advantages:
  - Prevents data sparsity
  - Resilient to outliers
  - Maximizes categorical entropy
- Disadvantages:
  - Distorts distances: Compresses regions with high data density and stretches
    regions with low data density.


### 3.6.3 Sampling
The sampling approach should not introduce bias - the sample used to train the
model should accurately reflect the domain that is being modeled.

Sampling approaches:
- Top sampling: Take the top s% of instances
- Random sampling: Select s% instances randomly
- Stratified sampling: Ensure that the sample maintains the relative frequency
  of different strata.
- Under/Over-sampling: Under/Over sample different strata to ensure (more) equal
  frequencies of different strata.

Sampling with replacement means you do not remove an instances from your dataset
after it has been sampled, so that it may be sampled again.

## 3.7 Summary

