# library doc string
""" This is a simple ML model to predict bank customer churn using Random Forest Classifer
and Logistic Regression """

# import libraries
import os
import joblib
import logging
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


os.environ['QT_QPA_PLATFORM'] = 'offscreen'
matplotlib.use('Agg')
sns.set()

#basic logging configuration
logging.basicConfig(
    filename='./results.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            dataframe: pandas dataframe
    '''
    read_data_frame = pd.read_csv(pth)
    logging.info("csv file has been successfully read")
    return read_data_frame


def perform_eda(dataframe):
    '''
    perform eda on dataframe and save figures to images folder
    input:
            dataframe: pandas dataframe

    output
            dataframe_eda: pandas dataframe with Churn
    '''
    # create copy of dataframe
    dataframe_eda = dataframe.copy(deep=True)

    # categorize the column Attrition_Flag
    # existing customer is marked as 0, and the attired customer is marked as 1
    dataframe_eda['Churn'] = dataframe_eda['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    # create histogram of customer churn
    plt.figure(figsize=(20,10))
    dataframe_eda['Churn'].hist()
    logging.info("Customer Churn histogram created")
    plt.savefig(fname='./images/eda/customer_churn_histogram.png')
    plt.close()

    # create histogram of customer age
    plt.figure(figsize=(20,10))
    dataframe_eda['Customer_Age'].hist()
    logging.info("Customer Age histogram created")
    plt.savefig(fname='./images/eda/customer_age_histogram.png')
    plt.close()

    # create bar chart of customer marital status
    plt.figure(figsize=(20,10))
    dataframe_eda.Marital_Status.value_counts('normalize').plot(kind='bar')
    logging.info("Maritual Status Bar chart created")
    plt.savefig(fname='./images/eda/marital_status_bar_chart.png')
    plt.close()

    # create transaction density histoplot
    plt.figure(figsize=(20,10))
    sns.histplot(dataframe_eda['Total_Trans_Ct'], stat='density', kde=True)
    logging.info("Transaction Density Histogram created")
    plt.savefig(fname='./images/eda/transaction_density_histogram.png')
    plt.close()

    # create fancy heatmap
    plt.figure(figsize=(20,10))
    sns.heatmap(dataframe_eda.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    logging.info("Heatmap created")
    plt.savefig(fname='./images/eda/heatmap.png')
    plt.close()

    return dataframe_eda

def encoder_helper(dataframe, category_list, response):
    '''
    helper function to turn each categorical column into a new column with propotion
    of churn for each category - associated with cell 15 from the notebook

    input:
            dataframe: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used
            for naming variables or index y column]

    output:
            dataframe: pandas dataframe with new columns for
    '''
    # create copy of dataframe
    dataframe_encode = dataframe.copy(deep=True)

    logging.info("turning each categorical column into a new column with propotion of churn")
    for category in category_list:
        temp_list = []
        temp_groups = dataframe.groupby(category).mean()['Churn']
        
        for cat in dataframe[category]:
            logging.info("Converting categorical column %s", cat)    
            temp_list.append(temp_groups.loc[cat])

        if response:
            new_column = category + "_" + response
            logging.info("Creating new column %s", new_column)
            dataframe_encode[new_column] = temp_list
        
        else:
            dataframe_encode[category] = temp_list

    logging.info("Returning dataframe with new columns")
    return dataframe_encode


def perform_feature_engineering(dataframe, response):
    '''
    input:
            dataframe: pandas dataframe
            response: string of response name [optional argument that could be used
            for naming variables or index y column]

    output:
            X_train: X training data
            X_test: X testing data
            y_train: y training data
            y_test: y testing data
    '''
    # encode categorical data put it into the relation with Credit_Limit
    category_list = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']
    dataframe_encoded = encoder_helper(dataframe, category_list, response)

    # divide date into training and testing data
    # get the dependent variable from dataframe => Customer Churn
    y = dataframe_encoded['Churn']
    # create empty dataframe
    X = pd.DataFrame()

    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    X[keep_cols] = dataframe_encoded[keep_cols]

    # split into training set and test set
    logging.info("Split data into training set and test set")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    logging.info("Creating classification reports and storing them into files")
    #Random Forest Classifier
    plt.rc('figure', figsize=(6, 6))
    plt.text(0.01, 1.25,
            str('Random Forest Train'),
            {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05,
            str(classification_report(y_test, y_test_preds_rf)),
            {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6,
            str('Random Forest Test'),
            {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7,
            str(classification_report(y_train, y_train_preds_rf)),
            {'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig(fname='./images/results/random_forest_results.png')
    logging.info("The result has been stored in the file %s", "random_forest_results.png")
    plt.close()

    # Logistic Regression
    plt.rc('figure', figsize=(6, 6))
    plt.text(0.01, 1.25,
            str('Logistic Regression Train'),
            {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05,
            str(classification_report(y_train, y_train_preds_lr)),
            {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6,
    str('Logistic Regression Test'),
            {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7,
            str(classification_report(y_test, y_test_preds_lr)),
            {'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig(fname='./images/results/logistic_results.png')
    logging.info("The result has been stored in the file %s", "logistic_results.png")
    plt.close()


def feature_importance_plot(model, X_data, output_path):
    '''
    creates and stores the feature importances in path
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_path: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    logging.info("Starting to calculate feature importances")
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(25, 20))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    # store the feature importance
    file_name_suffix = 'png'
    feature_file_name = 'feature_importance'
    full_path = os.path.join(
        output_path,
        feature_file_name +
        '.' +
        file_name_suffix)
    plt.savefig(full_path)
    logging.info("Feature importances were stored in the file %s.%s", 
                                feature_file_name, file_name_suffix)


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    random_forest_classifier = RandomForestClassifier(random_state=42, n_jobs=-1)
    logistic_regression = LogisticRegression(max_iter=1000, n_jobs=-1)

    # grid search
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    logging.info("Calculating cross validation") 
    cross_validation_rfc = GridSearchCV(
        estimator=random_forest_classifier,
        param_grid=param_grid,
        cv=5)
    cross_validation_rfc.fit(X_train, y_train)

    logistic_regression.fit(X_train, y_train)

    #store model
    logging.info("Storing the models")
    joblib.dump(cross_validation_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(logistic_regression, './models/logistic_model.pkl')
    logging.info("The models were stored")

    # predict using random forest classifier using feature set
    logging.info("Predicting using random forrest classifier using feature set")
    y_train_prediction_rf = cross_validation_rfc.best_estimator_.predict(X_train)
    y_test_prediction_rf = cross_validation_rfc.best_estimator_.predict(X_test)

    # predict using logistic regression using feature set
    logging.info("Predicting using logistic regression")
    y_train_prediction_lr = logistic_regression.predict(X_train)
    y_test_prediction_lr = logistic_regression.predict(X_test)

    # roc curve
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    logging.info("Calculating roc curve of logistic regression")
    lrc_plot = plot_roc_curve(logistic_regression, X_test, y_test, ax=ax, alpha=0.8)
    rfc_disp = plot_roc_curve(cross_validation_rfc.best_estimator_, 
    X_test, y_test, ax=ax, alpha=0.8)
    plt.savefig(fname='./images/results/roc_curve_result.png')
    plt.close()

    # store model results
    classification_report_image(y_train,
                                y_test,
                                y_train_prediction_lr,
                                y_train_prediction_rf,
                                y_test_prediction_lr,
                                y_test_prediction_rf)

    feature_importance_plot(
        model=cross_validation_rfc,
        X_data=X_test,
        output_path='./images/results/')


if __name__ == "__main__":
    data_frame = import_data("./data/bank_data.csv")
    dataframe_eda = perform_eda(data_frame)
    train_data_X, test_data_X, train_data_y, test_data_y = perform_feature_engineering(
        dataframe=dataframe_eda, response='Churn')
    train_models(train_data_X, test_data_X, train_data_y, test_data_y)
