'''
Tests of simple ML model to predict bank customer churn using Random Forest Classifer
and Logistic Regression

Author: Lukas Lengyel

Date: 15.11.2022
'''

import logging
import os
from math import ceil
import pytest
import churn_library as cls

logging.basicConfig(
    filename='./logs/churn_library_tests.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


@pytest.fixture(scope="module")
def path():
    '''
    providing path to tests
    '''
    return "./data/bank_data.csv"


def test_import(path):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        data_frame = cls.import_data(path)
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert data_frame.shape[0] > 0
        assert data_frame.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(path):
    '''
    test if eda can be executed
    '''
    customer_churn_histogram_filename = "customer_churn_histogram.png"
    customer_age_histogram_filename = "customer_age_histogram.png"
    marital_status_bar_chart_filename = "marital_status_bar_chart.png"
    transaction_density_histogram_filename = "transaction_density_histogram.png"
    heatmap_filename = "heatmap.png"

    dataframe = cls.import_data(path)
    try:
        cls.perform_eda(dataframe=dataframe)
        logging.info("EDA could be executed")
    except KeyError as error:
        logging.error('"%s" was not found', error.args[0])
        raise error

    # was `customer_churn_histogram` created
    try:
        assert os.path.isfile("./images/eda/" +
                              customer_churn_histogram_filename) is True
        logging.info('%s was found', customer_churn_histogram_filename)
    except AssertionError as error:
        logging.error('"%s" was not found.', customer_churn_histogram_filename)
        raise error

    # was `customer_age_histogram.png` created
    try:
        assert os.path.isfile(
            "./images/eda/" +
            customer_age_histogram_filename) is True
        logging.info('%s was found', customer_age_histogram_filename)
    except AssertionError as error:
        logging.error('"%s" was not found.', customer_age_histogram_filename)
        raise error

    # was `marital_status_distribution.png` created
    try:
        assert os.path.isfile("./images/eda/" +
                              marital_status_bar_chart_filename) is True
        logging.info('%s was found', marital_status_bar_chart_filename)
    except AssertionError as error:
        logging.error('"%s" was not found.', marital_status_bar_chart_filename)
        raise error

    # was `transaction_density_histogram_filename.png` created
    try:
        assert os.path.isfile("./images/eda/" +
                              transaction_density_histogram_filename) is True
        logging.info(
            'File %s was found',
            transaction_density_histogram_filename)
    except AssertionError as error:
        logging.error('"%s" was not found.',
                      transaction_density_histogram_filename)
        raise error

    # was `heatmap.png` is created
    try:
        assert os.path.isfile("./images/eda/" + heatmap_filename) is True
        logging.info('File %s was found', 'heatmap.png')
    except AssertionError as error:
        logging.error('"%s" was not found.', heatmap_filename)
        raise error


def test_encoder_helper(path):
    '''
    test encoder helper
    '''
    # import Dataframe
    dataframe = cls.import_data(path)

    # create Churn feature
    dataframe['Churn'] = dataframe['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    # categorical features
    category_list = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

    try:
        encoded_dataframe = cls.encoder_helper(
            dataframe=dataframe,
            category_list=[],
            response=None)
        assert encoded_dataframe.equals(dataframe) is True
        logging.info(
            "Testing encoder_helper(data_frame, category_list=[]): SUCCESS")
    except AssertionError as error:
        logging.error(
            "Testing encoder_helper(data_frame, category_list=[]): ERROR")
        raise error

    try:
        encoded_dataframe = cls.encoder_helper(
            dataframe=dataframe,
            category_list=category_list,
            response=None)

    # Columns are the same
        assert encoded_dataframe.columns.equals(dataframe.columns) is True

    # Dataframe are different
        assert encoded_dataframe.equals(dataframe) is False
        logging.info(
            "Test encoder_helper(data_frame,category_list=category_list,response=None): SUCCESS")
    except AssertionError as error:
        logging.error(
            "Test encoder_helper(data_frame,category_list=category_list,response=None): ERROR")
        raise error

    try:
        encoded_dataframe = cls.encoder_helper(
            dataframe=dataframe,
            category_list=category_list,
            response='Churn')

    # Columns are the same
        assert encoded_dataframe.columns.equals(dataframe.columns) is False

    # Dataframe are different
        assert encoded_dataframe.equals(dataframe) is False

    # Encoded columns are sum of columns in Dataframe and new columns from
    # categorical columns
        assert len(
            encoded_dataframe.columns) == len(
            dataframe.columns) + len(category_list)
        logging.info(
            "Testing encoder_helper(data_frame, category_list=category_list," +
            " response='Churn'): SUCCESS")
    except AssertionError as error:
        logging.error(
            "Testing encoder_helper(data_frame, category_list=category_list," +
            " response='Churn'): ERROR")
        raise error


def test_perform_feature_engineering(path):
    '''
    test perform_feature_engineering
    '''
    # Load the DataFrame
    dataframe = cls.import_data(path)

    # Create Churn feature
    dataframe['Churn'] = dataframe['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    try:
        (_, x_test, _, _) = cls.perform_feature_engineering(
            dataframe=dataframe,
            response='Churn')

    # Churn is in Dataframe
        assert 'Churn' in dataframe.columns
        logging.info(
            "Testing perform_feature_engineering. `Churn` column is present: SUCCESS")
    except KeyError as error:
        logging.error(
            'The `Churn` column is not present in the DataFrame: ERROR')
        raise error

    try:
        # x_test size is 30%
        assert (x_test.shape[0] == ceil(dataframe.shape[0] * 0.3)) is True
        logging.info(
            'Testing perform_feature_engineering. DataFrame sizes are consistent: SUCCESS')
    except AssertionError as error:
        logging.error(
            'Testing perform_feature_engineering. DataFrame sizes are not correct: ERROR')
        raise error


def test_train_models(path):
    '''
    test train_models
    '''
    logistic_model_filename = 'logistic_model.pkl'
    rfc_model_filename = 'rfc_model.pkl'
    roc_curve_filename = 'roc_curve_result.png'
    random_forest_results_filename = 'random_forest_results.png'
    logistic_results_filename = 'logistic_results.png'
    feature_importance_filename = 'feature_importance.png'

    # import data
    dataframe = cls.import_data(path)

    # create Churn feature
    dataframe['Churn'] = dataframe['Attrition_Flag'].\
        apply(lambda val: 0 if val == "Existing Customer" else 1)

    # Feature engineering
    (x_train, x_test, y_train, y_test) = cls.perform_feature_engineering(
        dataframe=dataframe,
        response='Churn')

    # was logistic model created
    try:
        cls.train_models(x_train, x_test, y_train, y_test)
        assert os.path.isfile("./models/" + logistic_model_filename) is True
        logging.info('"%s" was found', logistic_model_filename)
    except AssertionError as error:
        logging.error('"%s" was found', logistic_model_filename)
        raise error

    # was random forest classifier model created
    try:
        assert os.path.isfile("./models/" + rfc_model_filename) is True
        logging.info('"%s" was found', rfc_model_filename)
    except AssertionError as error:
        logging.error('"%s" was found', rfc_model_filename)
        raise error

    # was roc curve result created
    try:
        assert os.path.isfile('./images/results/' + roc_curve_filename) is True
        logging.info('"%s" was found', roc_curve_filename)
    except AssertionError as error:
        logging.error('"%s" was not found', roc_curve_filename)
        raise error

    # was random forest lassifier results created
    try:
        assert os.path.isfile(
            './images/results/' +
            random_forest_results_filename) is True
        logging.info('"%s" was found', random_forest_results_filename)
    except AssertionError as err:
        logging.error('"%s" was not found', random_forest_results_filename)
        raise err

    # was logistic results created
    try:
        assert os.path.isfile(
            './images/results/' +
            logistic_results_filename) is True
        logging.info('"%s" was found', logistic_results_filename)
    except AssertionError as error:
        logging.error('"%s" was not found', logistic_results_filename)
        raise error

    # was feature importance created
    try:
        assert os.path.isfile(
            './images/results/' +
            feature_importance_filename) is True
        logging.info('"%s" was found', feature_importance_filename)
    except AssertionError as error:
        logging.error('"%s" was not found', feature_importance_filename)
        raise error


if __name__ == "__main__":
    test_import()
    test_eda()
    test_encoder_helper()
    test_perform_feature_engineering()
    test_train_models()
