import pandas as pd

processed_data = "data/processed_yellow_tripdata_2024-01.parquet"

def calculate_kpi(df):

    total_revenue = df['total_amount'].sum()

    total_trips = len(df)

    avg_fare = df['total_amount'].mean()

    avg_distance = df['trip_distance'].mean()

    avg_duration = df['trip_duration'].mean()

    kpis = {
        "Total Revenue": f'${total_revenue:,.2f}',
        "total trips": f'{total_trips:,}',
        "Average Fare per Trip": f'{avg_fare:.2f}',
        "Average trip duration": f'{avg_duration:.2f}',
        "Average Trip Duration (minutes)": f"{avg_duration:.2f}"
    }

    return kpis

def main():
    
    try:
        df = pd.read_parquet(processed_data)
        print('Calculating kpis...')
        kpis = calculate_kpi(df)
        print('\n--- Business metrics summary---')
        for key, value in kpis.items():
                print(f"{key}: {value}")
        print("---------------------------------")
        
    except FileNotFoundError:
        print(f"Error: Processed data file not found at '{processed_data}'.")
        print("Please ensure you have run the etl_pipeline.py script successfully.")


if __name__ == '__main__':
    main()