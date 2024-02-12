import pandas as pd
from sklearn.impute import SimpleImputer


def preprocess_data(input_path, output_path):
    df = pd.read_csv(input_path)
    print(df.isnull().sum())
    imputer = SimpleImputer(strategy='mean')
    df[['CRIM', 'ZN', 'INDUS', 'CHAS', 'AGE', 'LSTAT']] = imputer.fit_transform(
        df[['CRIM', 'ZN', 'INDUS', 'CHAS', 'AGE', 'LSTAT']])
    df.to_csv(output_path, index=False)


if __name__ == '__main__':
    preprocess_data('data/raw/HousingData.csv', 'data/processed/Clean_HousingData.csv')
