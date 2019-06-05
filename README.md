# Introduction

A project to build a machine learning model that can recognize handwritten letters.


# Project Setup
We use Conda for package / dependency management.  You can get a copy of conda for your platform here: https://conda.io/en/latest/miniconda.html

One can then create a suitable environment as follows:

```
$ conda env create -f environment.yml
```

You can then activate the environment with:

```
$ source activate cs229aProject
```

To pull in any updated dependencies, one can execute
```
$ conda env update
```


# Models

## Baseline - Logisitc Regression
This one attained ~65% dev set accuracy.

## SVM
Of the SVMs tested, polynomial kernel worked best and attained 73% accuracy on dev set.
The other kernels performed poorly.

```
Classification report:               precision    recall  f1-score   support
macro avg       0.75      0.73      0.73     20800
weighted avg       0.75      0.73      0.73     20800
accuracy                           0.73     20800
```
