
---

# Boston Housing Price Prediction Project

## Project Overview
This project aims to predict housing prices in the Boston area using various machine learning models. It leverages the Boston Housing dataset, applying preprocessing techniques, feature engineering, and model evaluation to understand and predict housing prices effectively.

## Getting Started

### Prerequisites
Before running this project, ensure you have the following installed:
- Python 3.8 or later
- pip (Python package manager)

### Installation
1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/pramodyasahan/Boston-Housing-Project.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Boston-Housing-Project
   ```
3. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
To run the project and train the machine learning model, execute the `main.py` script with the necessary command-line arguments. For example:
```bash
python main.py --model_type linear_regression
```

### Command-Line Arguments
- `--data_path`: Path to the raw dataset. Default is `data/raw/HousingData.csv`.
- `--data_processed_path`: Path to save the processed dataset. Default is `data/processed/Clean_HousingData.csv`.
- `--model_path`: Path to save the trained model. Default is `models/linear_regression.joblib`.
- `--model_type`: Type of model to train. Supported values include `linear_regression`, `logistic_regression`, `svm`, `random_forest_regression`, and `decision_tree_regression`.

## Project Structure
```
Boston-Housing-Project/
├── data/
│   ├── raw/              # Original dataset
│   └── processed/        # Preprocessed dataset
├── src/
│   ├── data/             # Data loading and preprocessing scripts
│   ├── features/         # Feature engineering scripts
│   ├── models/           # Model training, prediction, and evaluation scripts
├── requirements.txt      # Python package dependencies
└── main.py               # Main script to run the project
```

## Models
This project includes several machine learning models for predicting housing prices:
- Linear Regression
- Logistic Regression (for classification tasks related to housing)
- Support Vector Machine (SVM)
- Random Forest
- Decision Tree

## Contributing
Contributions to improve the project are welcome. Please follow these steps to contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a pull request.

## License
Distributed under the MIT License. See `LICENSE` for more information.

## Contact

Project Link: [https://github.com/pramodyasahan/Boston-Housing-Project](https://github.com/yourusername/Boston-Housing-Project)

---