print("Script is running")
import spacy
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from collections import Counter
import time
import os
import googlemaps
# Load spaCy model for Named Entity Recognition (NER)
nlp = spacy.load("en_core_web_sm")

# Initialize geolocator
geolocator = Nominatim(user_agent="my_app_name_or_email@example.com")

#  Function to extract location names using spaCy NER
def extract_locations(text):
    """Identifies location mentions in a text using spaCy."""
    if not isinstance(text, str) or text.strip() == "":  # Handle missing or invalid text
        return []  # Return empty list instead of crashing

    doc = nlp(text)
    locations = [ent.text for ent in doc.ents if ent.label_ == "GPE"]  # Extract geographical entities
    return locations

DATA_PATH = "../Dataset/cleaned_healthcare_reddit_data_praw.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f" ERROR: Dataset was not found at: {DATA_PATH}")
else:
    print(f" Dataset found at: {DATA_PATH}")

#  Load dataset
df = pd.read_csv(DATA_PATH, encoding="utf-8")

print(f" Dataset Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(df.head())  # Show first few rows to confirm data is present


# Apply function safely to `Cleaned_Text`
df["Extracted_Locations"] = df["Cleaned_Text"].astype(str).fillna("").apply(extract_locations)

#  Flatten location list & count occurrences
all_locations = [loc for sublist in df["Extracted_Locations"].dropna().tolist() for loc in sublist]
location_counts = Counter(all_locations)

# Convert to DataFrame
location_df = pd.DataFrame(location_counts.items(), columns=["Location", "Mentions"])

manual_coordinates = {
    "alabama": (32.806671, -86.791130),
    "alaska": (61.370716, -152.404419),
    "arizona": (33.729759, -111.431221),
    "arkansas": (34.969704, -92.373123),
    "california": (36.116203, -119.681564),
    "colorado": (39.059811, -105.311104),
    "connecticut": (41.597782, -72.755371),
    "delaware": (39.318523, -75.507141),
    "florida": (27.766279, -81.686783),
    "georgia": (33.040619, -83.643074),
    "hawaii": (21.094318, -157.498337),
    "idaho": (44.240459, -114.478828),
    "illinois": (40.349457, -88.986137),
    "indiana": (39.849426, -86.258278),
    "iowa": (42.011539, -93.210526),
    "kansas": (38.526600, -96.726486),
    "kentucky": (37.668140, -84.670067),
    "louisiana": (31.169546, -91.867805),
    "maine": (44.693947, -69.381927),
    "maryland": (39.063946, -76.802101),
    "massachusetts": (42.230171, -71.530106),
    "michigan": (43.326618, -84.536095),
    "minnesota": (45.694454, -93.900192),
    "mississippi": (32.741646, -89.678696),
    "missouri": (38.456085, -92.288368),
    "montana": (46.921925, -110.454353),
    "nebraska": (41.125370, -98.268082),
    "nevada": (38.313515, -117.055374),
    "new hampshire": (43.452492, -71.563896),
    "new jersey": (40.298904, -74.521011),
    "new mexico": (34.840515, -106.248482),
    "new york": (42.165726, -74.948051),
    "north carolina": (35.630066, -79.806419),
    "north dakota": (47.528912, -99.784012),
    "ohio": (40.388783, -82.764915),
    "oklahoma": (35.565342, -96.928917),
    "oregon": (44.572021, -122.070938),
    "pennsylvania": (40.590752, -77.209755),
    "rhode island": (41.680893, -71.511780),
    "south carolina": (33.856892, -80.945007),
    "south dakota": (44.299782, -99.438828),
    "tennessee": (35.747845, -86.692345),
    "texas": (31.054487, -97.563461),
    "utah": (40.150032, -111.862434),
    "vermont": (44.045876, -72.710686),
    "virginia": (37.769337, -78.169968),
    "washington": (47.400902, -121.490494),
    "west virginia": (38.491226, -80.954456),
    "wisconsin": (44.268543, -89.616508),
    "wyoming": (42.755966, -107.302490),
    "washington d.c.": (38.9072, -77.0369)
}


#Function to get coordinates
gmaps = googlemaps.Client(key="AIzaSyDuh-5DpZGHmiuwS5Ga7cC8qVaQyN7R6t0")

def get_coordinates(location):
    """Uses Google Maps API to get coordinates."""
    try:
        geocode_result = gmaps.geocode(location)
        if geocode_result:
            lat = geocode_result[0]["geometry"]["location"]["lat"]
            lon = geocode_result[0]["geometry"]["location"]["lng"]
            return lat, lon
        return None, Noneß
    except Exception as e:
        print(f" Google API Geocoding failed for {location}: {e}")
        return None, None
# Get latitude & longitude for each location
location_df["Coordinates"] = location_df["Location"].apply(get_coordinates)
location_df["Latitude"] = location_df["Coordinates"].apply(lambda x: x[0])
location_df["Longitude"] = location_df["Coordinates"].apply(lambda x: x[1])
location_df.drop(columns=["Coordinates"], inplace=True)

# Save the processed geospatial data
location_df.to_csv("../Dataset/geospatial_insights.csv", index=False)

print(" Geospatial data extraction complete!")
