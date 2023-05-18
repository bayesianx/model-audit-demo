import pandas as pd
import numpy as np
from .preprocessing_tools import *
import joblib
from model_auditing.risks import calculate_residuals
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OneHotEncoder
import xgboost as xgb

def load_data():
    """ Function to load data from file and preprocess it"""
    # load data from file
    X_train = pd.read_csv('./regression_data/X_train.csv', sep=';', index_col=0)
    y_train = pd.read_csv('./regression_data/y_train.csv', sep=';', index_col=0)
    X_test = pd.read_csv('./regression_data/X_test.csv', sep=';', index_col=0)
    y_test = pd.read_csv('./regression_data/y_test.csv', sep=';', index_col=0)
    df = pd.read_csv('./regression_data/all_data.csv', sep=';', index_col=0)
    model1 = joblib.load('./regression_data/lr_model1.joblib')

    # preprocess data
    y_name = 'mpg'
    df.rename(columns = {y_name:'Y'}, inplace = True)
    to_category(df, 'model year', 'origin', 'cylinders')
    column_encoder(X_train, df)# Apply same encoding to test and training data
    column_encoder(X_test, X_train)

    remove_weird(df, '?')
    remove_weird(X_train, '?')
    remove_weird(X_test, '?')

    df.dropna(inplace=True)
    X_train.dropna(inplace=True)
    X_test.dropna(inplace=True)

    y_train = y_train.loc[X_train.index,:]
    y_test = y_test.loc[X_test.index, :]

    calculate_residuals(df, df['Y'], df['model_pred'], task='R')
    
    return df, X_train, X_test, y_train, y_test, model1

def fit_encoder(X_train):
    """
    Fit encoder used for encoding categorical and scaling numerical variables
    """
    ## Inputs of the model. Change accordingly to perform variable selection
    inputs_num = X_train.select_dtypes(include=['int64','float64']).columns.tolist()
    inputs_cat = X_train.select_dtypes(include=['category']).columns.tolist()
    inputs = inputs_num + inputs_cat
    numeric_transformer = RobustScaler()
    categorical_transformer = OneHotEncoder(sparse_output=False,
                                            drop='if_binary',
                                            handle_unknown='ignore') 
    preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_transformer, inputs_num),
            ('cat', categorical_transformer, inputs_cat)],
            verbose_feature_names_out=False
            ).set_output(transform="pandas")

    preprocessor.fit(X_train)

    return preprocessor

def fit_model(df, X_train):
    """Function to fit model to data
    
    Parameters:
    df (pd.DataFrame): dataframe with all data
    X_train (pd.DataFrame): dataframe with training data
    
    Returns:
    X_train_proc (pd.DataFrame): dataframe with training data after preprocessing
    y_train_res (pd.DataFrame): dataframe with residuals of training data
    BSTR (xgb.XGBRegressor): fitted model
    
    """
    # Define input and output matrices
    INPUTS = X_train.columns.tolist()
    OUTPUT = 'res'

    # We have categorical inputs with many classes. We will create dummy variables automatically after
    X_res = df[INPUTS]
    y_res = df[OUTPUT]

    # Split
    X_train_res, X_test_res, y_train_res, y_test_res = train_test_split(X_res, y_res,
                                                        random_state=2023) #seed for replication

    # fit encoder
    enc = fit_encoder(X_train_res)
    X_train_proc = enc.transform(X_train_res)


    BSTR = xgb.XGBRegressor(random_state=2022,
                            objective='reg:squarederror',
                            max_depth=5,
                            booster='gbtree',
                            n_estimators=200,
                            learning_rate=0.05)


    BSTR.fit(X_train_proc, y_train_res)

    return X_train_proc, y_train_res, BSTR

