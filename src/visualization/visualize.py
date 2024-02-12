import matplotlib.pyplot as plt
import seaborn as sns


def visualize_residuals(y_true, y_pred):
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.xlabel('Prediction Error')
    plt.ylabel('Density')
    plt.title('Distribution of Prediction Errors')
    plt.show()
