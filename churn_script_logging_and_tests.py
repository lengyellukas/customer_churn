from math import ceil
import os
import logging
import churn_library as cls

logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def test_import(import_data):
	'''
	test data import - this example is completed for you to assist with the other test functions
	'''
	try:
		df = import_data("./data/bank_data.csv")
		logging.info("Testing import_data: SUCCESS")
	except FileNotFoundError as err:
		logging.error("Testing import_eda: The file wasn't found")
		raise err

	try:
		assert df.shape[0] > 0
		assert df.shape[1] > 0
	except AssertionError as err:
		logging.error("Testing import_data: The file doesn't appear to have rows and columns")
		raise err


def test_eda(perform_eda):
	'''
	test if eda can be executed
	'''
	customer_churn_histogram_filename = "customer_churn_histogram.png"
	customer_age_histogram_filename = "customer_age_histogram.png"
	marital_status_bar_chart_filename = "marital_status_bar_chart.png"
	transaction_density_histogram_filename = "transaction_density_histogram.png"
	heatmap_filename = "heatmap.png"


	dataframe = cls.import_data("./data/bank_data.csv")
	try:
		cls.perform_eda(dataframe=dataframe)
		logging.info("EDA could be executed")
	except KeyError as error:
		logging.error('"%s" was not found', error.args[0])
		raise error

    # was `customer_churn_histogram` created
	try:
		assert os.path.isfile("./images/eda/" + customer_churn_histogram_filename) is True
		logging.info('%s was found', customer_churn_histogram_filename)
	except AssertionError as error:
		logging.error('"%s" was not found.', customer_churn_histogram_filename)
		raise error

    # was `customer_age_histogram.png` created
	try:
		assert os.path.isfile("./images/eda/" + customer_age_histogram_filename) is True
		logging.info('%s was found', customer_age_histogram_filename)
	except AssertionError as error:
		logging.error('"%s" was not found.', customer_age_histogram_filename)
		raise error

    # was `marital_status_distribution.png` created
	try:
		assert os.path.isfile("./images/eda/" + marital_status_bar_chart_filename) is True
		logging.info('%s was found', marital_status_bar_chart_filename)
	except AssertionError as error:
		logging.error('"%s" was not found.', marital_status_bar_chart_filename)
		raise error

    # was `transaction_density_histogram_filename.png` created
	try:
		assert os.path.isfile("./images/eda/" + transaction_density_histogram_filename) is True
		logging.info('File %s was found', transaction_density_histogram_filename)
	except AssertionError as error:
		logging.error('"%s" was not found.', transaction_density_histogram_filename)
		raise error

    # was `heatmap.png` is created
	try:
		assert os.path.isfile("./images/eda/" + heatmap_filename) is True
		logging.info('File %s was found', 'heatmap.png')
	except AssertionError as error:
		logging.error('"%s" was not found.', heatmap_filename)
		raise error


def test_encoder_helper(encoder_helper):
	'''
	test encoder helper
	'''
	# Load DataFrame
	dataframe = cls.import_data("./data/bank_data.csv")

    # Create `Churn` feature
	dataframe['Churn'] = dataframe['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    # Categorical Features
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
		logging.info("Testing encoder_helper(data_frame, category_list=[]): SUCCESS")
	except AssertionError as error:
		logging.error("Testing encoder_helper(data_frame, category_list=[]): ERROR")
		raise error
		
	try:
		encoded_dataframe = cls.encoder_helper(
            dataframe=dataframe,
            category_list=category_list,
            response=None)

        # Column names should be same
		assert encoded_dataframe.columns.equals(dataframe.columns) is True

        # Data should be different
		assert encoded_dataframe.equals(dataframe) is False
		logging.info(
            "Testing encoder_helper(data_frame, category_list=category_list, response=None): SUCCESS")
	except AssertionError as error:
		logging.error("Testing encoder_helper(data_frame, category_list=category_list, response=None): ERROR")
		raise error
		
	try:
		encoded_dataframe = cls.encoder_helper(
            dataframe=dataframe,
            category_list=category_list,
            response='Churn')

        # Columns names should be different
		assert encoded_dataframe.columns.equals(dataframe.columns) is False

        # Data should be different
		assert encoded_dataframe.equals(dataframe) is False

        # Number of columns in encoded_df is the sum of columns in data_frame
        # and the newly created columns from cat_columns
		assert len(
            encoded_dataframe.columns) == len(
            dataframe.columns) + len(category_list)
		logging.info(
            "Testing encoder_helper(data_frame, category_list=category_list, response='Churn'): SUCCESS")
	except AssertionError as error:
		logging.error(
            "Testing encoder_helper(data_frame, category_list=category_list, response='Churn'): ERROR")
		raise error


def test_perform_feature_engineering(perform_feature_engineering):
	'''
	test perform_feature_engineering
	'''
	# Load the DataFrame
	dataframe = cls.import_data("./data/bank_data.csv")

    # Churn feature
	dataframe['Churn'] = dataframe['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

	try:
		(_, X_test, _, _) = cls.perform_feature_engineering(
            dataframe=dataframe,
            response='Churn')

        # `Churn` must be present in `data_frame`
		assert 'Churn' in dataframe.columns
		logging.info("Testing perform_feature_engineering. `Churn` column is present: SUCCESS")
	except KeyError as error:
		logging.error('The `Churn` column is not present in the DataFrame: ERROR')
		raise error

	try:
        # X_test size should be 30% of `data_frame`
		assert (X_test.shape[0] == ceil(dataframe.shape[0] *0.3)) is True   # pylint: disable=E1101
		logging.info('Testing perform_feature_engineering. DataFrame sizes are consistent: SUCCESS')
	except AssertionError as error:
		logging.error('Testing perform_feature_engineering. DataFrame sizes are not correct: ERROR')
		raise error


def test_train_models(train_models):
	'''
	test train_models
	'''
	logistic_model_filename = 'logistic_model.pkl'
	rfc_model_filename = 'rfc_model.pkl'

	# Load the DataFrame
	dataframe = cls.import_data("./data/bank_data.csv")

    # Churn feature
	dataframe['Churn'] = dataframe['Attrition_Flag'].\
        apply(lambda val: 0 if val == "Existing Customer" else 1)

    # Feature engineering
	(X_train, X_test, y_train, y_test) = cls.perform_feature_engineering(
        dataframe=dataframe,
        response='Churn')

    # Assert if `logistic_model.pkl` file is present
	try:
		cls.train_models(X_train, X_test, y_train, y_test)
		assert os.path.isfile("./models/" + logistic_model_filename) is True
		logging.info('"%s" was found', logistic_model_filename)
	except AssertionError as error:
		logging.error('"%s" was found', logistic_model_filename)
		raise error

    # Assert if `rfc_model.pkl` file is present
	try:
		assert os.path.isfile("./models/" + rfc_model_filename) is True
		logging.info('"%s" was found', rfc_model_filename)
	except AssertionError as error:
		logging.error('"%s" was found', rfc_model_filename)
		raise error

    # Assert if `roc_curve_result.png` file is present
	try:
		assert os.path.isfile('./images/results/roc_curve_result.png') is True
		logging.info('"%s" was found', 'roc_curve_result.png')
	except AssertionError as error:
		logging.error('"%s" was not found', 'roc_curve_result.png')
		raise error

    # Assert if `rfc_results.png` file is present
	try:
		assert os.path.isfile('./images/results/rf_results.png') is True
		logging.info('"%s" was found', 'rf_results.png')
	except AssertionError as err:
		logging.error('"%s" was not found', 'rf_results.png')
		raise err

    # Assert if `logistic_results.png` file is present
	try:
		assert os.path.isfile('./images/results/logistic_results.png') is True
		logging.info('"%s" was found', 'logistic_results.png')
	except AssertionError as error:
		logging.error('"%s" was not found', 'logistic_results.png')
		raise error

    # Assert if `feature_importances.png` file is present
	try:
		assert os.path.isfile('./images/results/feature_importances.png') is True
		logging.info('"%s" was found', 'feature_importances.png')
	except AssertionError as error:
		logging.error('"%s" was not found', 'feature_importances.png')
		raise error

if __name__ == "__main__":
	test_import()
	test_eda()
	test_encoder_helper()
	test_perform_feature_engineering()
	test_train_models()