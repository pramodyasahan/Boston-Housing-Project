import argparse
import pandas as pd
from src.data.make_dataset import preprocess_data, load_data
from src.models.train_model import train_and_save_model
from src.models.predict_model import load_model, make_predictions
from src.models.evaluate import evaluate, print_result
from src.features.build_features import load_feature_data
from src.visualization.visualize import visualize_residuals


def predict(kwargs):
    model = load_model(kwargs.model_path)
    predictions = make_predictions(model, X_test)
    evaluate(predictions=predictions, y_test=y_test)
    return predictions


def train(kwargs):
    train_and_save_model(X_train, y_train, kwargs.model_type, kwargs.model_path)
    print("Model trained and saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Boston Housing Price Prediction Model")
    parser.add_argument('--data_path', type=str, default='data/raw/HousingData.csv')
    parser.add_argument('--data_processed_path', type=str, default='data/processed/Clean_HousingData.csv')
    parser.add_argument('--model_path', type=str, default='linear_regression.joblib',
                        help='Path to save the trained model')
    parser.add_argument('--model_type', type=str, required=True,
                        help='Type of model to train (e.g., linear_regression, logistic_regression, svm, '
                             'random_forest_regression, decision_tree_regression)')

    args = parser.parse_args()
    raw_data = load_data(args.data_path)
    preprocess_data(df=raw_data, output_path=args.data_processed_path)

    data = pd.read_csv(args.data_processed_path)
    X_train, X_test, y_train, y_test = load_feature_data(data)
    if args.model_path:
        train(kwargs=args)
    y_pred = predict(kwargs=args)
    print_result(predictions=y_pred, y_test=y_test)
    visualize_residuals(y_true=y_test, y_pred=y_pred)