from sklearn.metrics import mean_squared_error, r2_score


def evaluate(predictions, y_test):
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f"Mean Squared Error: {mse}, R^2 Score: {r2}")


def print_result(predictions, y_test):
    for i in range(10):
        print(f"Actual: {y_test[i]}, Predicted: {predictions[i]}")
