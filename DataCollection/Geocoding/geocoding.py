import googlemaps
import pandas as pd


# Google Maps API Key (Replace with your own)
API_KEY = "Google Maps API Key"

# Initialize Google Maps Client
gmaps = googlemaps.Client(key=API_KEY)

# Function to get longitude and latitude from address
def get_coordinates(address):
    print(f"Geocoding - {address}")
    try:
        result = gmaps.geocode(address)
        if result:
            location = result[0]['geometry']['location']
            return location['lng'], location['lat']  # Return as (Longitude, Latitude)
    except Exception as e:
        print(f"Error fetching {address}: {e}")
    return None, None  # Return None if not found

if __name__ == "__main__":
    print(get_coordinates("252 Houngang Ave 3"))