import pandas as pd
import numpy as np
import ast
from geopy.distance import geodesic
from joblib import Parallel, delayed
from tqdm import tqdm

# Constants
RADIUS_KM = 1  # 1 km radius
HDB_FILENAME = "Data_Coordinates/HDBResale_with_coordinates.csv"
NPARKS_FILENAME = "Data_Coordinates/NParks.csv"

# Load main HDB dataset
df = pd.read_csv(HDB_FILENAME)

# df = df.head(10).copy()  # Process only 10 rows for testing

df_coords = df[['Latitude', 'Longitude']].to_numpy()  # Convert to NumPy array for fast processing

# Load NParks dataset
nparks_df = pd.read_csv(NPARKS_FILENAME)

if "coordinates" not in nparks_df.columns:
    raise ValueError(f"Invalid format in '{NPARKS_FILENAME}': Must contain 'coordinates' column.")

# Preprocess NParks coordinates
nparks_df["coordinates"] = nparks_df["coordinates"].apply(ast.literal_eval)  # Convert to lists
nparks_coords = [np.array(coords) for coords in nparks_df["coordinates"]]  # Convert to NumPy arrays

# Define function for computing distances efficiently
def compute_nparks_distances(hdb_index, hdb_coord):
    min_distance = float("inf")
    count_within_radius = 0

    for park_coords in nparks_coords:
        # Convert park coordinates to NumPy array
        park_coords = np.array(park_coords)

        # Compute distance using vectorised operations (Haversine formula approximation)
        lat1, lon1 = np.radians(hdb_coord)  # Convert HDB property coordinates to radians
        lat2, lon2 = np.radians(park_coords[:, 1]), np.radians(park_coords[:, 0])  # Convert park coordinates to radians

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        distances_km = 6371 * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))  # Haversine formula

        # Check if any point in this park is within RADIUS_KM
        if np.any(distances_km <= RADIUS_KM):
            count_within_radius += 1

        # Update minimum distance
        min_distance = min(min_distance, np.min(distances_km))

    return count_within_radius, (min_distance if min_distance != float("inf") else None)

# Parallel processing
results = Parallel(n_jobs=8, batch_size=500)(
    delayed(compute_nparks_distances)(idx, hdb_coord) for idx, hdb_coord in tqdm(enumerate(df_coords), total=len(df_coords))
)

# Convert results to DataFrame and merge with original df
results_df = pd.DataFrame(results, columns=["NParks_within_1km", "NParks_nearest"])
df = df.reset_index(drop=True)
df["NParks_within_1km"] = results_df["NParks_within_1km"]
df["NParks_nearest"] = results_df["NParks_nearest"]

# Save final dataset
df.to_csv("Data/data_complex_NParks.csv", index=False)
print(f"Updated dataset saved as 'data_complex_NParks.csv'.")
