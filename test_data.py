import pandas as pd
import requests
import os


url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-01.parquet"

file_name = "yellow_tripdata_2024-01.parquet"

def load_data(url, file_name):

    if not os.path.exists(file_name):
        print('Downloading data...')
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(file_name, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print('successfully loaded the data')
        except requests.exceptions.RequestException as e:
            print(f"Error downloading data: {e}")
            return False
    else:
        print(f"{file_name} already exists.")
    return True


if __name__ == '__main__':
    if load_data(url, file_name):
        print('loading data into datafrane')
        data = pd.read_parquet(file_name)

        print("\n first 5 rows of the dataset")
        print(data.head)

        print("\nDataset Information (schema, data types, nulls):")
        data.info()