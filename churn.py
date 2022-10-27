# library doc string


# import libraries
import os
from re import X
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve, classification_report

os.environ['QT_QPA_PLATFORM']='offscreen'

def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    return pd.read_csv(pth)


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output
            None
    '''
    #get some basic statistic values
    df.describe()
    
    #describe columns if they are quantitative or categorical
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'                
    ]

    quant_columns = [
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
        'Avg_Utilization_Ratio'
    ]
    
    #categorize the column Attrition_Flag
    #existing customer is marked as 0, and the attired customer is marked as 1
    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
    
    #create histogram of customer churn
    #plt.figure(figsize=(20,10))
    customer_churn_histogram = df['Churn'].hist()
    save_eda_image(customer_churn_histogram, 'customer_churn_histogram')
    
    #create histogram of customer age
    #plt.figure(figsize=(20,10))
    customer_age_histogram = df['Customer_Age'].hist()
    save_eda_image(customer_age_histogram, 'customer_age_histogram')
    
    #create bar chart of customer marital status
    #plt.figure(figsize=(20,10))
    marital_status_bar_chart = df.Marital_Status.value_counts('normalize').plot(kind='bar')
    save_eda_image(marital_status_bar_chart, 'marital_status_bar_chart')
    
    #create transaction density histoplot
    #plt.figure(figsize=(20,10))
    transaction_density_histogram = sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
    save_eda_image(transaction_density_histogram, 'transaction_density_histogram')
    
    #create another fancy thing
    #plt.figure(figsize=(20,10))
    heatmap = sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
    save_eda_image(heatmap, 'heatmap')
    
def save_eda_image(plot, file_name):
    image_path = './images/eda/'
    file_name_suffix = 'png'
    full_path = os.path.join(image_path, file_name + '.' +  file_name_suffix)
    plt.figure(figsize=(20,10))
    plot.figure.savefig(full_path)

def encoder_helper(df, category_list, response=None):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    for category in category_list:
        temp_list = []
        temp_groups = df.groupby(category).mean()['Churn']
        
        for val in df[category]:
            temp_list.append(temp_groups.loc[val])
            
        new_col = category + "_Churn"
        df[new_col] = temp_list
        
    return df
    

def perform_feature_engineering(df, response=None):
    '''
    input:
            df: pandas dataframe
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            X_train: X training data
            X_test: X testing data
            y_train: y training data
            y_test: y testing data
    '''
    #divide date into training and testing data
    #get the dependent variable from dataframe => Customer Churn
    y = df['Churn']
    #create empty dataframe 
    X = pd.DataFrame()
    
    #encode categorical data put it into the relation with Credit_Limit
    category_list = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
    df_with_new_cols = encoder_helper(df, category_list, response)

    keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
             'Total_Relationship_Count', 'Months_Inactive_12_mon',
             'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
             'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
             'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
             'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn', 
             'Income_Category_Churn', 'Card_Category_Churn']

    X[keep_cols] = df_with_new_cols[keep_cols]
    
    #split into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=42)
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
    save_results(y_test, y_test_preds_rf, 'Random Forest Test', 'random_forest_test')
    save_results(y_train, y_train_preds_rf, 'Random Forest Train', 'random_forest_train')
    save_results(y_test, y_test_preds_lr, 'Logistic Regression Test', 'logistic_regression_test')
    save_results(y_train, y_train_preds_lr, 'Logistic Regression Train', 'logistic_regression_train')

def save_results(y, y_preds, report_name, file_name):
    image_path = './images/results/'
    file_name_suffix = 'png'
    full_path = os.path.join(image_path, file_name + '.' +  file_name_suffix)
    font = 'monospace'
    font_size = 10
    
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 0.6, str(report_name), {'fontsize': font_size}, fontproperties = font)
    plt.text(0.01, 0.7, str(classification_report(y, y_preds)), {'fontsize': font_size}, fontproperties = font)
    plt.axis('off');

    plt.figure.savefig(full_path)


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20,5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90);
    
    # store the feature importance
    file_name_suffix = 'png'
    full_path = os.path.join(output_pth, 'feature_importance' + '.' +  file_name_suffix)
    plt.figure.savefig(full_path)
    

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
    # grid search
    random_forest_classifier = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    logistic_regression = LogisticRegression(solver='lbfgs', max_iter=3000)
    
    param_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt'],
    'max_depth' : [4,5,100],
    'criterion' :['gini', 'entropy']
    }


    cross_validation_rfc = GridSearchCV(estimator=random_forest_classifier, param_grid=param_grid, cv=5)
    cross_validation_rfc.fit(X_train, y_train)
    
    logistic_regression.fit(X_train, y_train)

    #predict using random forest classifier using feature sat
    y_train_preds_rf = cross_validation_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cross_validation_rfc.best_estimator_.predict(X_test)

    #predict using logistic regression using feature set
    y_train_preds_lr = logistic_regression.predict(X_train)
    y_test_preds_lr = logistic_regression.predict(X_test)

    #store model results
    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)

    X = X_train.merge(X_test)

    feature_importance_plot(cross_validation_rfc.best_estimator_, X, './images/results/')

if __name__ == "__main__":
    df = import_data("./data/bank_data.csv")
    perform_eda(df)
    X_train, X_test, y_train, y_test = perform_feature_engineering(df)
    train_models(X_train, X_test, y_train, y_test)