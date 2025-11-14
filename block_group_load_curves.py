import pandas as pd
import json
import numpy as np
from pathlib import Path

print('='*10 + 'Start block group file' + '='*10)

# -------------------------------------------------------------------
# Define the base data directory (relative to this script's location)
# -------------------------------------------------------------------
DATA_DIR = Path.cwd() / "data"
file_path = DATA_DIR / "ev_load_curves.csv"

DELTA_T = 0.25

def load_ev_load_curves() -> pd.DataFrame:
    """Read the main output file that contains the EV load curves
    for each location and each charger type"""
    df = pd.read_csv(
        file_path,
        dtype={"load_curve": str},
    )

    # convert load curve from string to list
    df["load_curve"] = df["load_curve"].apply(
        lambda x: json.loads(x.replace("'", '"')) if isinstance(x, str) else x
    )

    df["charger_type"] = df["charger_type"].apply(lambda x: x.split(".")[-1])

    # We want to explode the load_curve column so that each element in the list becomes a separate row
    df = df.explode("load_curve", ignore_index=True).astype({"load_curve": float})

    # Add a column for the timestep
    df["hour"] = df.groupby(["geoid", "charger_type", "number_of_sessions"]).cumcount()
    df["hour"] = pd.to_timedelta(df["hour"] * DELTA_T, unit="h")

    # Rearrange the DataFrame to have the charger types as columns.
    df = df.pivot(
        index=["geoid", "hour"],
        columns="charger_type",
        values=["load_curve", "number_of_sessions"],
    ).fillna(0)

    # df has now multi-level columns

    df = df.reset_index()

    df["geoid_str"] = df["geoid"]

    # sort the columns to improve performance later
    df = df.sort_index(axis=1)
    return df

df = load_ev_load_curves()

print('='*10 + 'Finish block group file' + '='*10)