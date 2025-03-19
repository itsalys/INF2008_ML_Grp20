# Resale Price Prediction Using XGBoost and Feature Engineering

## Overview
This project implements an **XGBoost** regression model to predict **resale prices of flats**. The model utilizes **hyperparameter tuning** for improved accuracy. It also includes an interactive tool for users to input flat attributes and obtain resale price predictions.

## Dependencies
The notebook requires the following Python libraries:
```bash
pip install pandas numpy xgboost scikit-learn scipy joblib matplotlib ipywidgets
```

## Dataset
The script reads data from CSV files:
- **Train dataset:** `../../DataCollection/Data/cleaned_train.csv`
- **Test dataset:** `../../DataCollection/Data/cleaned_test.csv`

## Notebook Structure
### 1 Installing dependencies 
- Installs dependencies listed above

### 2 Data Loading & Preprocessing
- Reads the cleaned train and test datasets from CSV files.
- Displays dataset shape and first few rows for verification.
- Defines features (X) and target (y) for model training.
- Ensures train and test datasets have the same features by aligning them.
- Converts data to NumPy arrays for efficient model processing.

### 3 Model Tuning, Training, and Saving

#### Hyperparameter Tuning
- Uses RandomizedSearchCV to find the best XGBoost hyperparameters.
- Optimizes parameters like n_estimators, learning_rate, max_depth, subsample, and regularization factors.
#### Training & Saving
- If a pre-trained model exists, it is loaded using joblib.
- Otherwise, the model is trained from scratch with the best hyperparameters.
- The trained model is saved to xgb_model.pkl for future predictions.

### 4 Model Evaluation
- Predicts resale prices on the test dataset.
- Calculates key performance metrics:
- Displays Train and Test R² Scores to assess model performance.

### 5 Feature Importance Analysis
- Uses XGBoost’s built-in feature importance function to identify the most influential factors affecting resale prices.
- Generates a visualization of feature importance using matplotlib.

### 6 Interactive Price Prediction Tool

#### Functionality
- Allows users to input flat details via an interactive UI.
- Predicts resale price in real-time based on the trained model.