from datetime import date
import numpy as np 
import pandas as pd
import sys
import os
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import normaltest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix
from sympy import N
import statsmodels.formula.api as smf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname('__file__'), '..')))

def get_data():
    return pd.read_csv("C:/Users/akarwoski/Downloads/NY_full_dataset.csv")

def split_data(features, labels, test_size = 0.25, random_state = 42):
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)
    print('Training Features Shape:', train_features.shape)
    print('Training Labels Shape:', train_labels.shape)
    print('Testing Features Shape:', test_features.shape)
    print('Testing Labels Shape:', test_labels.shape)
    return train_features, test_features, train_labels, test_labels

#Cleaning Data
def add_features(df):
    df['maturity_dt'] = pd.to_datetime(df['maturity_dt'], errors = 'coerce')
    df['orig_dt'] = pd.to_datetime(df['orig_dt'], errors = 'coerce')
    df['bk_file_dt'] = pd.to_datetime(df['bk_file_dt'], errors = 'coerce')
    df['months_to_maturity'] = df.apply(lambda loan: lf.age(date.today(), loan['maturity_dt']) if not pd.isna(loan['maturity_dt']) and loan['maturity_dt'] is not None else None, axis = 1)
    df['loan_age'] = df.apply(lambda loan: lf.age(loan['orig_dt'], date.today()) if not pd.isna(loan['orig_dt']) and loan['orig_dt'] is not None else None, axis = 1)
    df['months_since_bk'] = df.apply(lambda loan: lf.age(loan['bk_file_dt'], date.today()) if not pd.isna(loan['bk_file_dt']) and loan['bk_file_dt'] is not None  else None, axis = 1)
    df = df.drop(columns = ['bk_file_dt', 'orig_dt', 'maturity_dt'])
    return df

def fill_numeric_nulls(df):
    for col in df._get_numeric_data():
        df[col].fillna(df[col].mean())

def remove_unique_strings(df, threshold = .05):
    print("_______________")
    for col in [col for col in df.select_dtypes(exclude = np.number)]:
        print(col)
        print(len(df[col].unique()) / len(df[col]))
        if(len(df[col].unique()) / len(df[col])) > threshold:
            print(f"Dropping Column {col}")
            df.drop(columns = [col], inplace = True)

def fill_common_string(df):
    for col in [col for col in df.select_dtypes(exclude = np.number)]:
        df[col].fillna(df[col].mode())

def drop_na_heavy_columns(df, threshold = .95):
    threshold_value = int(threshold * len(df))
    df.dropna(axis = 1, thresh = threshold_value, inplace = True)

def drop_non_unique_cols(df):
    for col in df.columns:
        if len(df[col].unique()) == 1:
            df.drop(columns = col, inplace = True)   

def drop_date_columns(df):
    df.drop(columns = [col for col in df.select_dtypes(include=['datetime64'])], inplace = True)
    for col in df.columns:
        try:
            if 'dt' in col:
                df.drop(columns = [col], inplace = True)
        except:
            df.drop(columns = [col], inplace = True)

def d_ag_normal_test(feature_series, alpha = .05):
    stat, p = normaltest(feature_series)
    guassian = True if p > alpha else False
    return guassian, p

def remove_outliers(df, std_cutoff = 3, iqr_multiple = 1.5, min_unique_values = 1000, plot_features = False):
    for col in df._get_numeric_data():
        if len(df[col].unique()) > min_unique_values:
            feature_series = df[col]
            print(feature_series)
            if plot_features:
                plot_feature(feature_series = feature_series, close_time = 5)
            guassian, p = d_ag_normal_test(feature_series)
            try:
                if guassian:
                    df = remove_normal_outliers(df, feature_series, std_cutoff = 3)
                else:
                    df = remove_outliers_quantile(df, feature_series, iqr_multiple)      
            except:
                pass
    return df

def plot_feature(df = None, feature = None, feature_series = None, close_time = None):
    feature_series = feature_series if feature_series is not None else df[feature]
    fig_1 = plt.figure(figsize=(16,5))
    plt.subplot(1,2,1)
    sns.boxplot(feature_series)
    plt.subplot(1,2,2)
    sns.distplot(feature_series)
    if close_time:
        timer = fig_1.canvas.new_timer(interval = 10000) #creating a timer object and setting an interval of 3000 milliseconds
        timer.add_callback(close_plot)
        timer.start()
    plt.show()

def scatter_plot(x , y):
    plt.scatter(x, y, alpha=0.5)
    plt.show()

def plot_all_features(df):
    for col in df.select_dtypes(include = np.number):
        plot_feature(df, col)

def close_plot():
    plt.close()


def remove_outliers_quantile(df, feature_series, iqr_multiple = 1.5):
    percentile_25 = feature_series.quantile(.25)
    percentile_75 = feature_series.quantile(.75)
    iqr = percentile_75 - percentile_25
    upper_limit = percentile_75 + iqr_multiple * iqr
    lower_limit = percentile_25 - iqr_multiple * iqr
    return df.loc[(feature_series > lower_limit) & (feature_series < upper_limit)]

def remove_normal_outliers(df, feature_series, std_cutoff):
    try:
        min_value = feature_series.mean() - feature_series.std() * std_cutoff
        max_value = feature_series.mean() + feature_series.std() * std_cutoff
        return df.loc[(feature_series > min_value) & (feature_series < max_value)]
    except:
        return df

#Transofrming Data
def convert_categorical_to_numerical(df, col_to_convert = None):
    col_to_convert = col_to_convert if col_to_convert is not None else df.columns
    for col in [col for col in df.select_dtypes(exclude = np.number) if col in col_to_convert]:
        try:
            df[col] = df[col].astype(str)
            le = LabelEncoder().fit(df[col])
            X_cat_transformed = le.transform(df[col])
            X_train_trans_df = pd.DataFrame(index = df.index, columns = [col], data = X_cat_transformed)
            df[col] = X_train_trans_df[col]
        except:
            df.drop(columns = [col], inplace = True)
            print("{} failed conversion and will be omitted from the model.".format(col))
    return df

def fill_na_values(data):
    for col in data.columns:
        try:
            data[col] = data[col].fillna(data[col].median())
        except:
            data[col] = data[col].fillna(data[col].mode())


def data_clean_process(data):
    return data

def manual_corrections(data):
    data['lot_flag'] = (data['bed'].isna()) & (data['bath'].isna()) & (data['house_size'].isna())
    data['lot_flag'] = data['lot_flag'].astype(bool)
    data = data.sort_values(by = ['population'])
    data = data.drop_duplicates(subset = ['full_address'], keep = 'last')
    return data

def clean_housing_data():
    data = get_data()
    data = manual_corrections(data)
    remove_unique_strings(data, threshold = .1)
    # data = convert_categorical_to_numerical(data)
    fill_na_values(data)
    drop_non_unique_cols(data)
    for col in data.columns:
        data = remove_normal_outliers(data, data[col], 2)
    print(data)
    data.to_csv("housing_clean.csv")
