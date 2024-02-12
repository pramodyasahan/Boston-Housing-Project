import pandas as pd
import joblib


def load_model(model_path):
    model = joblib.load(model_path)
    return model


def make_predictions(model, X):
    predictions = model.predict(X)
    return predictions
