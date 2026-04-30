import pandas as pd
import os
import requests


# --- Configuration ---
RAW_DATA_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-01.parquet"
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
    initial_rows = len(data)
    data = data[(data['passenger_count']>0) & (data['trip_distance']>0)]
    print(f"Removed {initial_rows - len(data)} rows with zero passengers or distance.")

    # --- 2. Feature Engineering ---
    # Calculate trip duration in minutes
    data['trip_duration'] = (data['tpep_dropoff_datetime'] - data['tpep_pickup_datetime']).dt.total_seconds() / 60

    # Filter out trips with unreasonable durations (e.g., less than 1 min or more than 2 hours)

    initial_rows = len(data)
    data = data[(data['trip_duration'] >= 1) & (data['trip_duration'] <= 120)]
    print(f"Removed {initial_rows - len(data)} rows with unreasonable trip durations.")

    # Calculate average speed in miles per hour. 
    data['average_speed'] = data['trip_distance'] / (data['trip_duration'] / 60)

    # Extract time-based features
    data['pickup_hour'] = data['tpep_pickup_datetime'].dt.hour
    data['pickup_day_of_week'] = data['tpep_pickup_datetime'].dt.day_name()

    print("Transformation complete. New features created: 'trip_duration', 'average_speed', 'pickup_hour', 'pickup_day_of_week'")
    
    return data

def load_data(df, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    try:
        df.to_parquet(file_path, index=False)
        print(f"Processed data successfully saved to {file_path}")
    except Exception as e:
        print(f"Error saving data: {e}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run ETL for a specific month or the default dataset.")
    parser.add_argument("--year", type=int, help="Year (e.g. 2024)")
    parser.add_argument("--month", type=int, help="Month number (e.g. 3)")
    args = parser.parse_args()

    print("--- Starting ETL Process ---")
    if args.year and args.month:
        url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{args.year}-{args.month:02d}.parquet"
        raw_path = f"data/raw_yellow_tripdata_{args.year}-{args.month:02d}.parquet"
        processed_path = f"data/processed_yellow_tripdata_{args.year}-{args.month:02d}.parquet"
    else:
        url, raw_path, processed_path = RAW_DATA_URL, RAW_DATA_FILE, PROCESSED_DATA_FILE

    raw_file_path = extract_data(url, raw_path)
    if raw_file_path:
        transformed_df = transform_data(raw_file_path)
        load_data(transformed_df, processed_path)
    print("--- ETL Process Completed ---")

if __name__ == '__main__':
    main()