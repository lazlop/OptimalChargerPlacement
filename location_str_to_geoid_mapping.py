import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path
# from block_group_load_curves import *

print('='*10 + 'Start location file' + '='*10)
print('(Takes ~30s to run)')

DATA_DIR = Path.cwd() / "data" 
file_path = DATA_DIR / "geographyFiles/location_str_to_geoid_mapping.shp"

pd.set_option('display.max_colwidth', None)

def load_geoid_locations() -> gpd.GeoDataFrame:
    """Read the main output file that contains the lat and lon
    for each census block centroid"""
    gdf = gpd.read_file(
        file_path 
    )

    return gdf

gdf = load_geoid_locations()

# Add list of neighbours to each node
gdf['NEIGHBOURS'] = None

# Perform a spatial join to find touching geometries
touching = gpd.sjoin(gdf, gdf, how="left", predicate="touches")

# Remove self-matches
touching = touching[touching["GEOID_STR_left"] != touching["GEOID_STR_right"]]

# Group by origin and aggregate neighbors
neigh_dict = touching.groupby("GEOID_STR_left")["GEOID_STR_right"].apply(list).to_dict()

# Assign to GeoDataFrame
gdf["NEIGHBOURS"] = gdf["GEOID_STR"].map(neigh_dict)

gdf_subset = gdf[['GEOID_STR', 'INTPTLAT', 'INTPTLON', 'NEIGHBOURS']]

# Match block group load curve name convention for df merge
gdf_subset = gdf_subset.rename(columns={'GEOID_STR': 'geoid_str_'})

gdf.to_parquet(DATA_DIR / 'network_analysis/gdf.parquet')
gdf_subset.to_parquet(DATA_DIR / 'network_analysis/gdf_subset.parquet')
gdf = gdf.drop(columns=['NEIGHBOURS'])
gdf.to_file(DATA_DIR / 'network_analysis/gdf.geojson', driver='GeoJSON')

print('='*10 + 'Finish location file' + '='*10)