import pandas as pd
import os
import requests


# --- Configuration ---
RAW_DATA_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"
RAW_DATA_FILE = "data/raw_yellow_tripdata_2024-01.parquet"
PROCESSED_DATA_FILE = "data/processed_yellow_tripdata_2024-01.parquet"

def extract_data(url, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    if not os.path.exists(file_path):
        print(f"Downloading data from {url}...")
        try:
            response = requests.get(url)
            response.raise_for_status()
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Successfully downloaded to {file_path}")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading data: {e}")
            return None
    else:
        print(f'{file_path} already exists')
    return file_path

def transform_data(file_path):
    print("Starting data transformation...")
    data = pd.read_parquet(file_path)
    # --- 1. Data Cleaning ---
    # Convert pickup and dropoff columns to datetime objects
    data['tpep_pickup_datetime'] = pd.to_datetime(data['tpep_pickup_datetime'])
    data['tpep_dropoff_datetime'] = pd.to_datetime(data['tpep_dropoff_datetime'])

    # Filter out trips with no passengers or no distance
    # These are often data errors and not useful for analysis
    initial_rows = len(data)
    data = data[(data['passenger_count']) & (data['trip_distance']>0)]
    print(f"Removed {initial_rows - len(data)} rows with zero passengers or distance.")

    # --- 2. Feature Engineering ---
    # Calculate trip duration in minutes
    data['trip_duration'] = (data['tpep_dropoff_datetime'] - data['tpep_pickup_datetime']).dt.total_seconds() / 60

    # Filter out trips with unreasonable durations (e.g., less than 1 min or more than 2 hours)

    initial_rows = len(df)
    df = df[(df['trip_duration'] >= 1) & (df['trip_duration'] <= 120)]
    print(f"Removed {initial_rows - len(df)} rows with unreasonable trip durations.")