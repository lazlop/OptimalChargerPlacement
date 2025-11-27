import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path

DATA_DIR = Path.cwd() / "data"  
file_path = DATA_DIR / "public_charger_data/alt_fuel_stations.csv"

def load_charger_data() -> pd.DataFrame:

    df = pd.read_csv(
        file_path
    )

    df = df[['ID', 'Access Days Time', 'EV Level1 EVSE Num', 'EV Level2 EVSE Num', 'EV DC Fast Count', 'Latitude', 'Longitude']]

    return df

charger_df = load_charger_data()

charger_gdf = gpd.GeoDataFrame(
    charger_df, 
    geometry=gpd.points_from_xy(charger_df.Longitude, charger_df.Latitude)
)

charger_df.to_parquet(DATA_DIR / 'network_analysis/charger_df.parquet')
charger_gdf.to_parquet(DATA_DIR / 'network_analysis/charger_gdf.parquet')

# print(charger_gdf.head())
# print(len(charger_df))
# print(len(charger_gdf))
# print(charger_df.head())