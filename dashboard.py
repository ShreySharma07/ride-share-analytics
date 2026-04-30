import os
import pandas as pd
import streamlit as st

from business_metrics import calculate_kpi
from statistical_analysis import test_payment_hypothesis, simulate_ab_test
from fraud_detection_model import create_label_feature, train_evaluate_model
from tlc_pipeline import load_state, run_pipeline, get_processed_path, scan_existing_files

# --- Page Configuration ---
st.set_page_config(page_title="Ride-Share Analytics", page_icon="🚕", layout="wide")


# --- Helpers ---
@st.cache_data
def load_months(months: tuple):
    dfs = []
    for m in months:
        year, month = m.split("-")
        path = get_processed_path(int(year), int(month))
        if os.path.exists(path):
            dfs.append(pd.read_parquet(path))
    return pd.concat(dfs, ignore_index=True) if dfs else None


@st.cache_resource
def get_model(X, y):
    return train_evaluate_model(X, y)


# --- Sidebar: Pipeline Status & Controls ---
with st.sidebar:
    st.header("Data Pipeline")

    state = scan_existing_files(load_state())
    available_months = state.get("processed_months", [])

    last_checked = state.get("last_checked")
    if last_checked:
        st.caption(f"Last checked: {last_checked[:10]}")
    else:
        st.caption("Pipeline has never been run.")

    st.metric("Processed Months", len(available_months))
    if available_months:
        st.caption(f"Latest: **{available_months[-1]}**")

    st.divider()

    if st.button("Check for New Data", use_container_width=True):
        log_lines = []

        def capture_log(msg):
            log_lines.append(msg)

        with st.spinner("Running pipeline..."):
            new_months = run_pipeline(log=capture_log)

        with st.expander("Pipeline log", expanded=bool(new_months)):
            st.code("\n".join(log_lines))

        if new_months:
            st.success(f"New data added: {', '.join(new_months)}")
            st.cache_data.clear()
            st.rerun()
        else:
            st.info("No new data found.")


# --- Main Title ---
st.title("Ride-Share Business Analytics Dashboard")
st.markdown("NYC Yellow Taxi KPIs, fraud detection, and statistical analyses.")

# --- Month Selector ---
state = scan_existing_files(load_state())
available_months = state.get("processed_months", [])

if not available_months:
    st.warning(
        "No processed data found. Click **Check for New Data** in the sidebar to download "
        "and process TLC trip records."
    )
    st.stop()

selected_months = st.multiselect(
    "Select month(s) to analyse",
    options=available_months,
    default=[available_months[-1]],
    help="Hold Ctrl/Cmd to select multiple months. Data is combined across selections.",
)

if not selected_months:
    st.info("Select at least one month above to load the dashboard.")
    st.stop()

df_raw = load_months(tuple(sorted(selected_months)))

if df_raw is None:
    st.error("Could not load data for the selected month(s).")
    st.stop()

st.caption(f"Showing **{len(df_raw):,}** trips across {len(selected_months)} month(s): {', '.join(selected_months)}")

st.markdown("<hr/>", unsafe_allow_html=True)

# --- ML Model ---
X, y, df_with_labels = create_label_feature(df_raw.copy())
model = get_model(X, y)

# --- KPI Section ---
st.header("Executive KPI Summary")
kpis = calculate_kpi(df_raw)
kpi_cols = st.columns(5)
kpi_cols[0].metric("Total Revenue", kpis["Total Revenue"])
kpi_cols[1].metric("Total Trips", kpis["Total Trips"])
kpi_cols[2].metric("Avg. Fare per Trip", kpis["Average Fare per Trip"])
kpi_cols[3].metric("Avg. Trip Distance (miles)", kpis["Average Trip Distance (miles)"])
kpi_cols[4].metric("Avg. Trip Duration (mins)", kpis["Average Trip Duration (minutes)"])

st.markdown("<hr/>", unsafe_allow_html=True)

# --- Statistical Analysis Section ---
st.header("Statistical Analysis & A/B Testing")
analysis_cols = st.columns(2)
with analysis_cols[0]:
    st.subheader("Hypothesis Test: Payment Type vs. Fare")
    hyp_results = test_payment_hypothesis(df_raw)
    col1, col2 = st.columns(2)
    col1.metric("Avg. Card Fare", f"${hyp_results['card_mean']:.2f}")
    col2.metric("Avg. Cash Fare", f"${hyp_results['cash_mean']:.2f}")
    st.write(f"**P-value:** `{hyp_results['p_value']:.4g}`")
    st.success(f"**Conclusion:** {hyp_results['conclusion']}")
with analysis_cols[1]:
    st.subheader("A/B Test: Peak Hour Surcharge")
    ab_results = simulate_ab_test(df_raw)
    diff = ab_results['treatment_mean'] - ab_results['control_mean']
    col1, col2 = st.columns(2)
    col1.metric("Control Avg. Fare (A)", f"${ab_results['control_mean']:.2f}")
    col2.metric("Treatment Avg. Fare (B)", f"${ab_results['treatment_mean']:.2f}", delta=f"${diff:.2f}")
    st.write(f"**P-value:** `{ab_results['p_value']:.4f}`")
    st.info(f"**Conclusion:** {ab_results['conclusion']}")

st.markdown("<hr/>", unsafe_allow_html=True)

# --- Fraud Detection Section ---
st.header("Fraud Detection Alerts")

predictions = model.predict(X)
probabilities = model.predict_proba(X)[:, 1]

predictions_df = pd.DataFrame(
    {"is_fraud_prediction": predictions, "fraud_probability": probabilities},
    index=X.index,
)
df_with_predictions = df_with_labels.join(predictions_df)
flagged_trips = df_with_predictions[df_with_predictions["is_fraud_prediction"] == 1]

st.metric("Total Trips Flagged as Potential Fraud", len(flagged_trips))

prob_threshold = st.slider(
    "Set Fraud Probability Threshold",
    min_value=0.5, max_value=1.0, value=0.75, step=0.05,
)

st.dataframe(
    flagged_trips[flagged_trips["fraud_probability"] >= prob_threshold][
        ["trip_distance", "fare_amount", "total_amount", "average_speed", "fraud_probability"]
    ].sort_values(by="fraud_probability", ascending=False),
    use_container_width=True,
)
