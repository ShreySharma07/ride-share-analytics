import pandas as pd
from scipy import stats


processed_data = "data/processed_yellow_tripdata_2024-01.parquet"

def test_payment_hypothesis(df):

    card_payments = df[df['payment_type']==1]['total_amount']
    cash_payments = df[df['payment_type']==2]['total_amount']

    t_stat, p_value = stats.ttest_ind(card_payments, cash_payments, equal_var = False)

    print(f"Credit Card Average Fare: ${card_payments.mean():.2f}")
    print(f"Cash Average Fare: ${cash_payments.mean():.2f}")
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_value}")

    alpha = 0.05
    conclusion = ""
    if p_value < 0.05:
        conclusion = "The simulated surcharge resulted in a statistically significant increase in the average fare."
    else:
        conclusion = "The change did not result in a statistically significant difference."

    return {
        "card_mean": card_payments.mean(),
        "cash_mean": cash_payments.mean(),
        "p_value": p_value,
        "conclusion": conclusion
    }


def simulate_ab_test(df):
        print("\n--- Simulating A/B Test for Peak Hour Surcharge ---")

        peak_hours = [17,18,19]

        # Group A is the entire dataset's average fare
        control_mean_fare = df['total_amount'].mean()

        # Create Group B (Treatment)
        df_treatment = df.copy()
        peak_hr_trip_mask = df['pickup_hour'].isin(peak_hours)

        df_treatment.loc[peak_hr_trip_mask, 'total_amount']+=2.00

        treatment_mean_fare = df_treatment['total_amount'].mean()

        # Perform a t-test to see if the difference is statistically significant
        t_stat, p_value = stats.ttest_ind(df['total_amount'], df_treatment['total_amount'], equal_var=False)

        print(f"Control Group (A) Average Fare: ${control_mean_fare:.2f}")
        print(f"Treatment Group (B) Average Fare: ${treatment_mean_fare:.2f}")
        print(f"P-value: {p_value}")

        conclusion = ""
        if p_value < 0.05:
            conclusion = "The simulated surcharge resulted in a statistically significant increase in the average fare."
        else:
            conclusion = "The change did not result in a statistically significant difference."
            
        # This return statement is crucial
        return {
            "control_mean": control_mean_fare,
            "treatment_mean": treatment_mean_fare,
            "p_value": p_value,
            "conclusion": conclusion
        }


def main():
    try:
          df = pd.read_parquet(processed_data)

          test_payment_hypothesis(df)

          simulate_ab_test(df)

    except FileNotFoundError:
        print(f"Error: Processed data file not found at '{processed_data}'.")
        print("Please run the etl_pipeline.py script first.")

if __name__ == "__main__":
    main()
