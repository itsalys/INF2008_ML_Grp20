# INF20008_ML_Grp20 - HDB Resale Price Prediction

## Project Description
This project aims to predict HDB resale prices using various machine learning techniques. 
The dataset includes transaction details, property features, and key geographical factors influencing property prices. 
Models tested and evaluated are Random Forest, Geographically Weighted Regression (GWR) and XGBoost.

---

# File Structure

**DataCollection**: This folder contains scripts and datasets used for processing and preparing the data for analysis and modeling.
- **Data**: Contains preprocessed and raw data files.
  - `Data/`: Cleaned Train and Test datasets are stored here.
  - `Data_Coordinates/`: Stores spatial coordinate data related to locations of various amenities, such as MRT Locations, Malls.
  - `Data_Raw/`: Holds raw datasets before preprocessing.
  - `Geocoding/`: Scripts and data for geocoding locations.
- **Scripts**:
  - `data_processing_hdbresale.ipynb`: Jupyter notebook for processing HDB resale data.
  - `data_processing_locations.ipynb`: Jupyter notebook for processing datasets of various amenities, such as MRT Locations, Malls.
  - `feature_engineering-complex_coordinates.py`: Python script for feature engineering for datasets where 1 location has multiple coordinates.
  - `feature_engineering.py`: Python script for feature engineering for datasets where 1 location has 1 coordinate. 
- **Other Files**:
  - `label_encoders.json`: JSON file storing label encoding mappings used in preprocessing.

---

**Models**: This folder contains models and results from different machine learning approaches.
- **GWR**:
  - `gwr.ipynb`: Jupyter notebook for running GWR models.
- **XGBoost**:
  - `xgb_model.pkl`: Pickle file storing a trained XGBoost model.
  - `xgb_model1.pkl`: Another version of a trained XGBoost model.
  - `XGBoost.ipynb`: Jupyter notebook for training and evaluating XGBoost models.
- **Random Forest**:
  - 
- **Other Files**:
  - `singapore_clustered_avg_hover.html`: Interactive HTML visualization of clustered data.

---

# Notes

- For running specific models, refer to the README within each model folder.

