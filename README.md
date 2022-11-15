# Predict Customer Churn

- Project **Predict Customer Churn** is one of the projects of DevOps Engineer Nanodegree Udacity

## Project Description
In this project, Jupyter notebook with machine learning project was provided. The goal was to reorganize the code into a python file respecting coding standards. The additional task was to develop tests that verify correct implementation of machine learning project.

## Files and data description
Overview of the files and data present in the root directory:

* churn_library.py - a library of functions to find customers who are likely to churn
* churn_script_logging_and_tests.py - tests to verify that functions to find custmers who are likely to churn work correctly
* ./data/bank_data.csv - data of the bank customers used for churn prediction. The file can be replaced with current customer data.

## Running Files
The customer churn prediction is started by following command:

```
$ python3 churn_library.py
```

The results are stored in the images folder with following structure:
* eda - EDA results
* results - models result
* models - models

Running the tests

The tests can be started by following command:

```
$ pytest churn_script_logging_and_tests.py
```
