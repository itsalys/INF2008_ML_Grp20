# Random Forest Model with PCA

## Overview
This project implements a Random Forest Regressor to predict resale prices of flats based on various property features. Principal Component Analysis (PCA) is applied to reduce dimensionality and improve the model's performance. The dataset includes multiple property attributes and geographical features, which are used to train and test the model.

## Dependencies
The notebook requires the following Python libraries:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Dataset
The script reads data from CSV files:
- **Train dataset:** `../../DataCollection/Data/cleaned_train.csv`
- **Test dataset:** `../../DataCollection/Data/cleaned_test.csv`

## Notebook Structure
### 1 Data Loading
- Loads the training and testing datasets using pandas. 
- Displays the first few rows of the datasets to confirm data loading.

### 2 Feature Selection
- Defines a list of features (columns) to be used in the model. These include various property attributes like month, year, town_LE, price_per_sqm, and others.
- The features are extracted from both the train and test datasets.

### 3 Data Exploration and Visualization
- Displaying summary statistics of the training and test datasets.
- Visualizes the correlation between features using a correlation heatmap.
- Plots the distribution of price per square meter to observe the spread of property prices in the training data.

### 4 Random Forest Model
- Defines and trains the RandomForestRegressor model using the selected features. Hyperparameters such as n_estimators, max_depth, and others are set to prevent overfitting.
- The model is then trained on the training data (X_train, y_train).

### 5 Model Evaluation
- The model predicts resale prices for the test dataset (y_pred).
    Evaluation metrics are calculated:
    Mean Absolute Error (MAE)
    Mean Squared Error (MSE)
    Root Mean Squared Error (RMSE)
    R² Score
    Mean Absolute Percentage Error (MAPE)

### 6 Hyperparameter Tuning
- Optimize parameters like n_estimators, max_depth, min_samples_split, min_samples_leaf, and max_features.

## Usage
1. Ensure all dependencies are installed.
2. Update the dataset paths if necessary.
3. Run all cells in the file.
4. The final model evaluation results, including MAE, MAPE, MSE, RMSE, and R² Score, will be displayed at the end.