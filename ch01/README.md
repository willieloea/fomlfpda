# Introduction to Machine Learning
These notes are based on a university course, and draws from

- Chapter 1 & 2 of "Fundamentals of Machine Learning for Predictive Data
  Analytics" by Aoife D'Arcy et al.,  
- Course slides, and  
- the Internet.

A variety of introductory concepts were covered, including:  

- What is ML? Algorithms + Data (programming 2.0)  
- ML Pipeline: Data gathering, data preparation (feature engineering), model
  training, model deployment  
- Descriptive features vs target features  
- Confounding features  
- Training vs Validation vs Test set  
- ML Paradigms: Supervised vs Unsupervised vs Reinforcement learning  
- Underfitting vs Overfitting  
- Bias-variance trade-off in models  
- Inductive bias, restriction bias, and preference bias in models  
- Ill-posed problems  
- $k$-fold cross validation  
- Data missing completely at random vs at random vs not at random  
- Data quality issues:  
    - Data leakage: Test-train contamination + Target leakage  
    - Concept drift  
    - Sampling bias  
- The analytics base table


One thing that Andries Engelbrecht emphasised is that not every problem requires
the most sophisticated or computationally expensive model - some problems are
best solved with simple solutions.

What is machine learning?  
Machine learning is the use of algorithms that can learn or change their
behaviour based on data that is provided to them. The end result of providing an
algorithm with data is that it has developed a 'model' - a representation of the
data that it was exposed to.

What is involved in each stage of the ML pipeline?  
The ML pipeline is the process followed to build and deploy a model. First, data
has to be gathered - this could involve scraping, querying databases, or
deploying sensors in the real world to collect information. Then data
preparation involves going through the collected data to identify data quality
issues, addressing those issues, and building new features out of the data you
have (i.e. feature engineering) - this is a discipline in itself and will be
covered as a later topic. After this, an algorithm has to be selected, and a
model trained - again a whole discipline and will be covered in later topics.
Finally, the model has to be deployed in the real world - built into an app or
integrated into an API.

What are descriptive and target features?  
Features are pieces of data, e.g. the height of a person, temperature at a time
of day, etc. Descriptive features are features that will be used when a model is
deployed to make a prediction.

What are confounding features?  
Confounding features are features that are not not relevant to what the model is
trying to predict. Holding on to this data will just slow you down.

What are training, validation, and test sets?  
When building a model ('model training' step of the ML pipeline), the dataset
(constructed during 'data preparation') is split into three disjoint sets,
namely training, validation, and test set, to train, validate, and test a
model.
During training, data fed to the algorithm is used to mold and shape the model
- it's the process of narrowing down the space of all possible models to just
one. A handful of different models (that use different algorithms) can be
trained. In the validation step, the performance of each trained model is
assessed to see how they perform on data that was not in the training set. The
results of this assessment can be used to filter out models that don't
generalize beyond the training set. After this, testing is where a model is
assessed again to ensure it didn't just perform well on the training and
validation sets, and to get a sense of how well it will perform in deployment.

What is supervised, unsupervised, and reinforcement learning?  
Supervised learning is when instances have descriptive feature values and target
feature values or 'labels'. The goal is to develop a model that correctly maps
descriptive feature values to target feature values. E.g. medical diagnosis
based on a common set of health measurements.  
Unsupervised learning is when instances only have descriptive feature values.
E.g. clustering or dimensionality reduction.  
Reinforcement learning algorithms are ones that are able to explore their
environment and learn through trial and error. E.g. an agent learns how to
navigate by being rewarded for successful navigation.  

What does it mean for a model to be underfit or overfit?  
Underfitting is when a model is when a machine learning algorithm has not been
given sufficient data to develop an accurate model.  
Overfitting is when a model is too complex for its domain, and learns the
anomalies or peculiarities of the data that it was trained on.

What is the bias variance trade-off in model development?  
Bias is indicative of wrong assumptions made by a model. A model with high bias
will, on average, make predictions that are far from the true target value. Bias
is often a result of underfitting.  
Variance in indicative of a model's sensitivity. A model with high variance will
make very different predictions as a result of small changes in descriptive
feature values. This is often a result of overfitting.  
The trade-off or tension between bias and variance is that reducing one could
lead to increasing the other. If you train a model more to avoid bias, you might
end up with an overfit or high variance model. The point in training that
minimizes bias and variance optimally has to be determined during model
development.

What is inductive, restriction, and preference bias in ML algorithms?  
The inductive bias of a machine learning algorithm is the set of assumptions
that it makes about the world. The restriction bias defines how the models an
algorithm develops is restricted, e.g. linear models assume that points lie on a
straight line. The preference bias defines how different models are compared or
preferred, e.g. the model with smaller errors is selected over others.

What are ill-posed problems?  
An ill-posed problem is one where a unique solution cannot be determined using
only the information that is available. This applies to machine learning because
most ML problems are ill-posed, and multiple models can be developed to solve
them.

What is $k$-fold cross validation?  
$k$-fold cross validation is the process of partitioning a dataset into $k$
equal sized subsets, and running $k$ training experiments where, for each
experiment, one data-subset is used as a test set, and the other subsets are
used for training and validation. The performance of an algorithm is then
measured as the average of the $k$ experiments. This technique is often used
when there is little data.

What is the difference between data missing completely at random, at random, and
not at random?  
Missing completely at random (MCAR) is when there is no link between the missing
values and the actual data. E.g. power went out and caused a sensor to stop
recording data.  
Missing at random (MAR) is when data for a particular feature is missing, not
because of the values of that feature, but because of some related (potentially
recorded) feature. E.g. historically men are less likely to answer surveys on
mental health.  
Missing not at random (MNAR) is when there is a direct relationship between the
actual value of a feature, and whether or not those values are missing. E.g.
severely overweight people may not have a reported weight because it could not
be measured.

What is data leakage, test-train contamination, and target leakage?  
Data leakage is when information appears in a data or training set that does not
reflect the world as it would look when a model is deployed.  
Test-train contamination is when information in the test set has influenced the
values in the training set. E.g. test set values are used to impute missing
values in the training set, thereby contaminating it.  
Target leakage is when the training set contains values that aren't available or
set as they are during deployment. E.g. in a time-series context (like
administering medicine to patients) a descriptive feature might change after a
prediction has been made, and one risks leaking values that arose after a
model's prediction into the data that is used to train it for before the
prediction.

What is concept drift?  
Concept drift is when the values of a target feature changes over time. E.g.
inflation of asset prices would lead to a model trained on old data to be
highly biased.

What is sampling bias?  
Sampling bias is when the data used to train a model does not accurately reflect
the domain that has to be modeled.

In order for predictions to be relevant, you have to make sure that they make
useful predictions in the context where the predictive model is used. This often
involves understanding a business's needs, domain, and what sort of predictions
would be relevant to the business. Don't just use machine learning because it's
sexy - find out what the actual problems is and reason about whether a machine
learning solution is a good fit, and then which approach is best.

Other ideas were discussed but they were fairly trivial.  

What is the analytics base table?  
It is the structured data from which predictive models learn. It is the result
of collecting and organising data from within an organisation.

