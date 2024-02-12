import pandas as pd
from sklearn.impute import SimpleImputer


def preprocess_data(output_path, df):
    imputer = SimpleImputer(strategy='mean')
    df[['CRIM', 'ZN', 'INDUS', 'CHAS', 'AGE', 'LSTAT']] = imputer.fit_transform(
        df[['CRIM', 'ZN', 'INDUS', 'CHAS', 'AGE', 'LSTAT']])
    df.to_csv(output_path, index=False)


def load_data(data_path):
    df = pd.read_csv(data_path)
    return df
