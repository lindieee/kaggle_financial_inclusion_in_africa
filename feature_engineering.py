import pandas as pd
import numpy as np


# one-hot-encoding for truth labels
def convert_truth_labels(y_data):
    y_data = y_data.map({"Yes": 1, "No": 0})
    return y_data


# train only on selected features
def data_filtering_for_features(X_data):
    cat_features = ['country', 'location_type', 'cellphone_access', 
                'gender_of_respondent', 'relationship_with_head', 
                'marital_status', 'education_level', 'job_type']
    num_features = ['household_size', 'age_of_respondent']
    features = cat_features + num_features
    return X_data[features]


# one-hot-encoding for categorical features
def one_hot_encoding(X_data):
    return pd.get_dummies(X_data, drop_first=True)













