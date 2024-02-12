from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import joblib


def train_and_save_model(X_train, y_train, model_type, model_path):
    if model_type == 'linear_regression':
        model = LinearRegression()
    elif model_type == 'logistic_regression':
        model = LogisticRegression(max_iter=1000)  # Increased max_iter for convergence
    elif model_type == 'svm':
        model = SVC()
    elif model_type == 'random_forest_regression':
        model = RandomForestRegressor()
    elif model_type == 'random_forest_classification':
        model = RandomForestClassifier()
    elif model_type == 'decision_tree_regression':
        model = DecisionTreeRegressor()
    elif model_type == 'decision_tree_classification':
        model = DecisionTreeClassifier()
    else:
        raise ValueError("Unsupported model type provided.")

    # Train the model
    model.fit(X_train, y_train)
    print("Model has trained successfully!")

    # Save the trained model
    joblib.dump(model, model_path)

    print(f"{model_type} model trained and saved to {model_path}")
