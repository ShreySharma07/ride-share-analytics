import os
import json
import requests
import pandas as pd
from datetime import datetime, date

from etl_pipeline import extract_data, transform_data, load_data

TLC_BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"
DATA_DIR = "data"
STATE_FILE = "data/pipeline_state.json"
START_YEAR = 2024
START_MONTH = 1


def get_file_url(year, month):
    return f"{TLC_BASE_URL}/yellow_tripdata_{year}-{month:02d}.parquet"


def get_raw_path(year, month):
    return f"{DATA_DIR}/raw_yellow_tripdata_{year}-{month:02d}.parquet"


def get_processed_path(year, month):
    return f"{DATA_DIR}/processed_yellow_tripdata_{year}-{month:02d}.parquet"


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"last_checked": None, "processed_months": [], "latest_month": None}


def save_state(state):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def _add_month(year, month):
    if month == 12:
        return year + 1, 1
    return year, month + 1


def get_months_to_check():
    """Return (year, month) tuples from START through 2 months before today."""
    today = date.today()
    cutoff_month = today.month - 2
    cutoff_year = today.year
    if cutoff_month <= 0:
        cutoff_year -= 1
        cutoff_month += 12

    months = []
    year, month = START_YEAR, START_MONTH
    while (year, month) <= (cutoff_year, cutoff_month):
        months.append((year, month))
        year, month = _add_month(year, month)
    return months


def check_month_available(year, month):
    url = get_file_url(year, month)
    try:
        resp = requests.head(url, timeout=15)
        return resp.status_code == 200
    except requests.RequestException:
        return False


def scan_existing_files(state):
    """Detect already-processed files on disk and add them to state (backwards compat)."""
    processed = set(state.get("processed_months", []))
    for year, month in get_months_to_check():
        key = f"{year}-{month:02d}"
        if key not in processed and os.path.exists(get_processed_path(year, month)):
            processed.add(key)
    state["processed_months"] = sorted(list(processed))
    if state["processed_months"]:
        state["latest_month"] = state["processed_months"][-1]
    return state


def run_pipeline(force=False, log=print):
    """
    Check TLC for new monthly files, download and process any that are missing.

    Args:
        force: Reprocess months even if already in state.
        log: Callable for status messages (default: print).

    Returns:
        List of newly processed month strings, e.g. ["2024-03"].
    """
    state = scan_existing_files(load_state())
    processed = set(state.get("processed_months", []))
    new_months = []

    log(f"Checking for new TLC data ({len(get_months_to_check())} months in range)...")

    for year, month in get_months_to_check():
        key = f"{year}-{month:02d}"
        processed_path = get_processed_path(year, month)

        if key in processed and os.path.exists(processed_path) and not force:
            log(f"  {key} already processed, skipping.")
            continue

        log(f"  Probing TLC for {key}...")
        if not check_month_available(year, month):
            log(f"  {key} not yet available on TLC.")
            continue

        log(f"  Downloading {key}...")
        raw_path = get_raw_path(year, month)
        downloaded = extract_data(get_file_url(year, month), raw_path)
        if not downloaded:
            log(f"  Failed to download {key}.")
            continue

        log(f"  Transforming {key}...")
        df = transform_data(raw_path)
        load_data(df, processed_path)

        processed.add(key)
        new_months.append(key)
        log(f"  {key} complete.")

    state["last_checked"] = datetime.now().isoformat()
    state["processed_months"] = sorted(list(processed))
    state["latest_month"] = state["processed_months"][-1] if state["processed_months"] else None
    save_state(state)

    log(f"Pipeline done. {len(new_months)} new month(s): {new_months or 'none'}")
    return new_months


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TLC Automated Data Pipeline")
    parser.add_argument("--force", action="store_true", help="Reprocess all months")
    args = parser.parse_args()
    run_pipeline(force=args.force)
