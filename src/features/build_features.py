import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_feature_data(data):
    columns_to_drop = ['CHAS', 'ZN', 'RAD', 'INDUS']
    X = data.drop(columns_to_drop, axis=1)
    X = X.drop('MEDV', axis=1)
    y = data.iloc[:, -1].values
    return train_test_split(X, y, test_size=0.2, random_state=42)


def remove_low_variance_features(data, threshold=0.0):
    selector = VarianceThreshold(threshold)
    return selector.fit_transform(data)


def standardize_features(data):
    scaler = StandardScaler()
    numerical_features = data.select_dtypes(include=['float64', 'int64']).columns
    data[numerical_features] = scaler.fit_transform(data[numerical_features])
    return data
