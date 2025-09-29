import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

processed_data_file = "data/processed_yellow_tripdata_2024-01.parquet"


def create_label_feature(df):
    # --- 1. Label Creation (Heuristics) ---
    # We define "fraud" based on a combination of suspicious metrics.
    # Rule 1: Impossibly high average speed (e.g., > 100 mph)
    # Rule 2: Very short trip duration (< 1 min) but a significant fare ( > $10)
    # Rule 3: Very long trip distance (e.g., > 100 miles)

    fraud_conditions = (
        (df['average_speed']>100) |
        ((df['trip_duration'] < 1) & (df['total_amount'] > 10)) |
        (df['trip_distance']>100)
    )

    df['is_fraud'] = fraud_conditions.astype(int)

    fraud_count = df['is_fraud'].sum()
    total_count = len(df)
    print(f'\nIdentifided {fraud_count} total fraud cases out of {total_count} trips = {fraud_count/total_count:.4f}%')

    print("This is a highly imbalanced dataset, which is typical for fraud detection.")

    # --- 2. Feature Selection & Engineering ---
    # We select features that could predict our fraud labels.
    # We must avoid features that were used to create the label directly (like average_speed).
    # This avoids "data leakage," where the model cheats by looking at the answer.
    
    features = [
        'trip_distance', 
        'fare_amount', 
        'passenger_count',
        'pickup_hour',
        'pickup_day_of_week',
        'PULocationID',
        'DOLocationID'
    ]

    target = 'is_fraud'

    df_encoded = pd.get_dummies(df[features], columns=['pickup_day_of_week'], drop_first=True)

    X = df_encoded
    y = df[target]

    return X, y, df

def train_evaluate_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    random_forest = RandomForestClassifier(n_estimators = 100, random_state=42, class_weight='balanced',n_jobs=-1)

    random_forest.fit(X_train, y_train)

    # y_pred = random_forest.predict(X_test)

    # print(classification_report(y_test, y_pred, target_names=['Not Fraud (0)', 'Fraud (1)']))
    
    # print("--- How to Read This Report ---")
    # print("Precision (for Fraud): Of all trips we flagged as fraud, this percentage was correct.")
    # print("Recall (for Fraud): Of all actual fraudulent trips, this is the percentage we caught.")
    # print("F1-Score: A combined score that balances Precision and Recall.")

    return random_forest


def main():
    df = pd.read_parquet(processed_data_file)
    X_main, y_main, _ = create_label_feature(df)
    train_evaluate_model(X_main, y_main)

if __name__ == '__main__':
    main()