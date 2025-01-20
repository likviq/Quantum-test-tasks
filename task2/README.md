# Model Training and Inference

This project provides model training using H2O and validation for predictions. It consists of two main scripts:
- **train.py** — for model training.
- **predict.py** — for validation and making predictions.

## Description

- **train.py**: Uses the dataset for training a model. It includes data preprocessing (normalization, anomaly removal) and model training using the H2O library.
- **predict.py**: Uses a saved model to make predictions on new data.

## Dependencies

Before running the project, you need to set up a virtual environment and install the dependencies from the `requirements.txt` file.

### 1. Create a virtual environment:

```bash
python -m venv venv
```

### 2. Activate the virtual environment:
- On Windows:

```bash
venv\Scripts\activate
```
- On Linux/macOS:

```bash
source venv/bin/activate
```

### 3. Install the dependencies:
```bash
pip install -r requirements.txt
```

## Running the Model Training

To run the model training, follow these steps:

1. Make sure you have a dataset file for training (CSV format).
2. Configure the parameters in the command line or use the default values.

### Run Training:

```bash
python train.py --data_path path_to_training_data.csv --target target_column --features feature1 feature2 --cv_column fold_column --model_output_path path_to_save_model
```

#### Parameters:

- --data_path: Path to the training data CSV file.
- --target: The name of the column you want to predict.
- --features: List of feature columns to be used for training.
- --cv_column: The column used for cross-validation (folds).
- --model_output_path: The directory where the trained model will be saved.


## Running Inference (Predictions)
To perform predictions with a saved model, use the predict.py script.

### Run Inference:

```bash
python predict.py --model_path path_to_trained_model --input_csv path_to_input_data.csv --output_csv path_to_save_predictions.csv
```

#### Parameters:

- --model_path: Path to the saved H2O model.
- --input_csv: Path to the input CSV file with new data for predictions.
- --output_csv: Path to save the prediction results as a CSV file.

# Exploratory Data Analysis (EDA)
The eda.ipynb file contains the exploratory data analysis (EDA) steps performed on the dataset. Key observations include:

- A notable correlation was identified between the feature column 6 and the target variable.
- Visualizations, correlation matrices, and statistical analysis were performed to understand relationships between features.
- Potential anomalies and outliers were observed, and strategies for handling them were discussed.
The insights from the EDA were instrumental in selecting relevant features and preparing the dataset for training.


# Project Structure

```bash
.
├── data/
│   ├── models/                    # h2o models weights
│   ├── train.csv                  # Training data
│   ├── hidden_test.csv            # Test feature data
│   └── hidden_test_predicted.csv  # Test data with predicted h2o and base y
├── src/
│   ├── data_preprocessor.py       # Data preprocessing module
│   ├── inference.py               # Inference (prediction) module
│   └── model_trainer.py           # Model training module
├── config.py                      # Default configuration values
├── eda.ipynb                      # Exploratory data analysis of data
├── predict.py                     # Inference script for predictions
├── README.md                      # This file
├── requirements.txt               # List of dependencies
└── train.py                       # Model training script
```

# Configuration
The parameters for running can be configured in the config.py file:
```bash
DEFAULTS = {
    "data_path": "path/to/train.csv",
    "target": "target_column",
    "features": ["feature1", "feature2", "feature3"],
    "cv_column": "fold_column",
    "model_output_path": "path/to/save_model",
    "model_path": "path/to/saved_model",
    "input_csv": "path/to/input_data.csv",
    "output_csv": "path/to/output_predictions.csv"
}
```

You can also pass these parameters through the command line when running the scripts, and they will override the default values.

# Results

The `hidden_test_predicted.csv` file contains the predictions from two models:

- `y_pred_h2o`: Predictions from a custom trained model.

- `y_pred_baseline`: Predictions from a baseline model, calculated using the formula:
`y_pred_baseline` = (feature "6" * feature "6") + feature "7".
