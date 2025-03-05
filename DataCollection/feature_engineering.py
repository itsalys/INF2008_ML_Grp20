import pandas as pd
import numpy as np
from geopy.distance import geodesic
from scipy.spatial import cKDTree
import ast
import os
from joblib import Parallel, delayed
from tqdm import tqdm

# Set variables
RADIUS_KM = 1  # Change as needed
df_filename = "Data_Coordinates/HDBResale_with_coordinates.csv"
locations_filenames = [
    # "Data_Coordinates/LTAMRTStation.csv",
    "Data_Coordinates/MallCoordinates.csv",
    "Data_Coordinates/Hawker.csv",
    "Data_Coordinates/PreSchool.csv",
    "Data_Coordinates/Primary.csv",
    "Data_Coordinates/Secondary.csv",
    "Data_Coordinates/JuniorCollege.csv",
    "Data_Coordinates/MixedLevel.csv",
    "Data_Coordinates/Sports.csv",
    # "Data_Coordinates/NParks.csv"
]

# Load main data
df = pd.read_csv(df_filename)

# Convert latitude and longitude to radians for faster calculations
df[['Latitude_rad', 'Longitude_rad']] = np.radians(df[['Latitude', 'Longitude']])

def process_location_file(locations_filename, df):
    """Processes a single location file and updates the DataFrame with results."""
    
    print(f"Processing locations from '{locations_filename}'")
    locations_df = pd.read_csv(locations_filename)
    
    # Extract base name
    locations_csv_name = os.path.splitext(os.path.basename(locations_filename))[0]
    
    # Generate column names
    within_col_name = f"{locations_csv_name}_within_{RADIUS_KM}km"
    nearest_col_name = f"{locations_csv_name}_nearest"
    
    if "Latitude" in locations_df.columns and "Longitude" in locations_df.columns:
        print(f"Detected simple location format in '{locations_filename}'")

        # Convert to radians for fast spatial processing
        locations_df[['Latitude_rad', 'Longitude_rad']] = np.radians(locations_df[['Latitude', 'Longitude']])

        # Build KD-Tree for fast nearest neighbour search
        kd_tree = cKDTree(locations_df[['Latitude_rad', 'Longitude_rad']].values)

        def compute_distances(row):
            distances, _ = kd_tree.query([row["Latitude_rad"], row["Longitude_rad"]], k=len(locations_df))
            distances_km = distances * 6371  # Convert to km using Earth's radius
            
            # Count locations within radius
            count_within_radius = np.sum(distances_km <= RADIUS_KM)
            
            # Get the nearest distance
            nearest_distance = distances_km[0] if distances_km.size > 0 else None
            
            return count_within_radius, nearest_distance

        # Compute distances using parallel processing
        results = Parallel(n_jobs=8, batch_size=100)(
            delayed(compute_distances)(row) for _, row in tqdm(df.iterrows(), total=len(df))
        )
        
        # Convert results to DataFrame and assign properly
        results_df = pd.DataFrame(results, columns=[within_col_name, nearest_col_name])
        df = df.reset_index(drop=True)  # Ensure alignment
        df[within_col_name] = results_df[within_col_name]
        df[nearest_col_name] = results_df[nearest_col_name]

    elif "coordinates" in locations_df.columns:
        print(f"Detected complex location format in '{locations_filename}'")

        # Convert coordinates column to NumPy arrays for efficient processing
        locations_df["coordinates"] = locations_df["coordinates"].apply(ast.literal_eval)
        locations_np = locations_df["coordinates"].to_numpy()

        def compute_complex_distances(row):
            min_distance = float("inf")
            count_within_radius = 0  # Track number of locations (not points) within radius
            row_coords = np.array([row["Latitude"], row["Longitude"]])

            for location_coordinates in locations_np:
                distances = np.array([
                    geodesic((lat, lon), row_coords).km for lon, lat in location_coordinates
                ])

                # ✅ If at least one coordinate is within RADIUS_KM, count this location as 1
                if np.any(distances <= RADIUS_KM):
                    count_within_radius += 1

                # ✅ Find the nearest distance across all locations
                min_distance = min(min_distance, np.min(distances))

            return count_within_radius, (min_distance if min_distance != float("inf") else None)

        # Use parallel processing with optimised batch size
        results = Parallel(n_jobs=8, batch_size=100)(
            delayed(compute_complex_distances)(row) for _, row in tqdm(df.iterrows(), total=len(df))
        )

        # Convert results to DataFrame and assign properly
        results_df = pd.DataFrame(results, columns=[within_col_name, nearest_col_name])
        df = df.reset_index(drop=True)  # Ensure alignment
        df[within_col_name] = results_df[within_col_name]
        df[nearest_col_name] = results_df[nearest_col_name]

    else:
        raise ValueError(f"Invalid format in '{locations_filename}': Must contain either 'Latitude' & 'Longitude' OR 'coordinates' column.")
    
    return df

# Process all location files and update df correctly
for file in locations_filenames:
    df = process_location_file(file, df)

# Remove temporary columns
df.drop(columns=['Latitude_rad', 'Longitude_rad'], inplace=True)

# Save updated dataset
df.to_csv("data_simple.csv", index=False)
print(f"Updated test CSV saved as 'data_simple.csv'.")
