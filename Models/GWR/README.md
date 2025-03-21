# Geographically Weighted Regression (GWR) with PCA and Clustering

## Overview
This Jupyter Notebook implements a **Geographically Weighted Regression (GWR)** model to predict **resale prices of flats**, incorporating **Principal Component Analysis (PCA)** and **spatial clustering** to improve the model's performance. The dataset includes various property attributes and geographical features.


## Dependencies
The notebook requires the following Python libraries:
```bash
pip install geopandas mgwr pandas numpy matplotlib scikit-learn
```

## Dataset
The script reads data from CSV files:
- **Train dataset:** `../../DataCollection/Data/cleaned_train.csv`
- **Test dataset:** `../../DataCollection/Data/cleaned_test.csv`

## Notebook Structure
### Cell 1: Installing dependencies 
- Installs dependencies listed above

### Cell 2: PCA Test
- Produce a graph to visualises cumulative explained variance against the number of PCA components, helping to determine how many principal components retain sufficient variance in the dataset.

### Cell 3: GWR Model
- Trains and evaluates the gwr model. 

## Usage
1. Ensure all dependencies are installed.
2. Update the dataset paths if necessary.
3. Run all cells in the Jupyter Notebook.
4. The final model evaluation results, including **MAE, MAPE, MSE, RMSE, and R² Score**, will be displayed at the end.


