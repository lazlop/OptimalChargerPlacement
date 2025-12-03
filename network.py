# from block_group_load_curves import *
# from location_str_to_geoid_mapping import *
# from public_chargers import *
from sklearn.metrics.pairwise import haversine_distances
import folium
from folium.features import GeoJsonTooltip
import pandas as pd
from pathlib import Path
import geopandas as gpd
import numpy as np
import json
import osmnx as ox

print('='*10 + 'Start network file' + '='*10)

# =========================================================
# Find origin and destination energy demand and distances
# =========================================================

DATA_DIR = Path.cwd() / "data"

df = pd.read_parquet(DATA_DIR / 'network_analysis/df.parquet')
gdf_subset = pd.read_parquet(DATA_DIR / 'network_analysis/gdf_subset.parquet')
charger_df = pd.read_parquet(DATA_DIR / 'network_analysis/charger_df.parquet')
charger_gdf = gpd.read_parquet(DATA_DIR / 'network_analysis/charger_gdf.parquet')
gdf = gpd.read_parquet(DATA_DIR / 'network_analysis/gdf.parquet')
gdf_for_json = gpd.read_file(DATA_DIR / 'network_analysis/gdf.geojson')

with open((DATA_DIR / 'Geography_Files/location_str_to_geoid_mapping.json'), 'r') as f:
    mapping = json.load(f)

# Flatten column MultiIndex
df.columns = [
    '_'.join(col).strip() if isinstance(col, tuple) else col
    for col in df.columns.values
]

# Merge lat and lon data from GeoPandas df with block group data and map to census track
demand_df = df.merge(gdf_subset, on='geoid_str_', how='left')

# Filter df for data at only 6 pm
demand_df_filtered = demand_df[demand_df['hour_'] == pd.to_timedelta('18:00:00')]

# Drop locations with invalid coordinates
demand_df_filtered = demand_df_filtered.dropna(subset=['INTPTLAT', 'INTPTLON'])

# Take only locations from Alameda County
demand_df_filtered = demand_df_filtered[demand_df_filtered['geoid_str_'].str.contains('Alameda', case=False, na=False)]

# Compute mass
demand_df_filtered['mass'] = demand_df_filtered.loc[:, 'load_curve_MFH_LD_L2':'load_curve_Work_LD_L2'].sum(axis=1)

# Convert to radians for haversine calculation
coords = np.radians(demand_df_filtered[['INTPTLAT', 'INTPTLON']])
dist_matrix = haversine_distances(coords) * 6371

# Change dist matrix from array to pandas df
df_dist = pd.DataFrame(
    dist_matrix,
    columns=demand_df_filtered['geoid_str_'].values
).assign(geoid_str_=demand_df_filtered['geoid_str_'].values)[
    ['geoid_str_'] + demand_df_filtered['geoid_str_'].values.tolist()
]
# Set geo ids as index
df_dist = df_dist.set_index('geoid_str_')

# Map index values to geo id values
index = pd.Series(range(len(df_dist)), index=df_dist.index)

# =========================================================
# Nearest neighbours
# =========================================================

# Take nearest neighbours data from location_str_to_geoid_mapping.py
network_df = gdf_subset.explode('NEIGHBOURS')

# Spatial join of charger location points and census track polygons to categorize each charger
# with its respective census track
charger_loc_gdf = gpd.sjoin(
    charger_gdf,
    gdf[["GEOID_STR", "geometry"]],
    how="left",
    predicate="within"
)
charger_loc_gdf = charger_loc_gdf.drop(columns=["index_right"])
charger_loc_gdf.rename(columns={"GEOID_STR": "geoid_str_"}, inplace=True)

# group chargers by geoid_str_ and aggregate into a list
chargers_per_tract = (
    charger_loc_gdf.groupby("geoid_str_")["ID"]
    .apply(list)
    .reset_index()  # convert back to DataFrame for merging
)

# First merge: origin chargers
network_df = network_df.merge(
    chargers_per_tract,
    how="left",
    left_on="geoid_str_",
    right_on="geoid_str_"
).rename(columns={"ID": "origin_chargers"}) 

# Second merge: destination chargers
network_df = network_df.merge(
    chargers_per_tract,
    how="left",
    left_on="NEIGHBOURS",
    right_on="geoid_str_"
).rename(columns={"ID": "destination_chargers"})

# fix column names
network_df = network_df.drop(columns=["geoid_str__y"])
network_df = network_df.rename(columns={"geoid_str__x": "geoid_str_"})

# =========================================================
# Simplify to Alameda County
# =========================================================

# Take only locations from Alameda County
network_df = network_df[network_df['geoid_str_'].str.contains('Alameda', case=False, na=False)]
network_df = network_df[network_df['NEIGHBOURS'].str.contains('Alameda', case=False, na=False)]

# Get demand for each census block
network_df = network_df.merge(demand_df_filtered[['geoid_str_', 'mass']], how='left', on='geoid_str_')
network_df = network_df.rename(columns={'mass':'origin_demand_(kW)'})

# Add destination mass here with df.merge
network_df = pd.merge(network_df, demand_df_filtered[['geoid_str_', 'mass']], left_on='NEIGHBOURS', right_on='geoid_str_', how='left')
network_df = network_df.rename(columns={'mass':'destination_demand_(kW)'})
network_df = network_df.rename(columns={'geoid_str__x': 'geoid_str_'}).drop(columns=['geoid_str__y'])

# Begin Add Haversine distances
# Lookup matrix positions
row_idx = index[network_df['geoid_str_']].values
col_idx = index[network_df['NEIGHBOURS']].values

# Extract distances
network_df['distance_km'] = dist_matrix[row_idx, col_idx]

# Adding location str mapping ids 
network_df['geoid'] = network_df['geoid_str_'].map(mapping)
network_df['neighbor_geoid'] = network_df['NEIGHBOURS'].map(mapping)

network_df.to_parquet(DATA_DIR / 'network_analysis/network_df.parquet')

# =============================================
# End origin-destination analysis
# =============================================

# =============================================
# Begin plot of heat map
# =============================================

# Alameda plot

alameda_gdf = gdf[['GEOID_STR', 'INTPTLAT', 'INTPTLON', 'geometry', 'NEIGHBOURS']]
alameda_gdf = alameda_gdf.rename(columns={'GEOID_STR': 'geoid_str_'})
alameda_gdf = alameda_gdf[alameda_gdf['geoid_str_'].str.contains('Alameda', case=False, na=False)]
alameda_gdf['charging_demand'] = demand_df_filtered['mass'].values

# =============================================
# Begin POI analysis
# =============================================

# Load tag dictionaries
ox.settings.use_cache = True
ox.settings.log_console = False
tab_buildings = pd.read_html('https://wiki.openstreetmap.org/wiki/Key:building', match='Value')[0]
tab_amenities = pd.read_html('https://wiki.openstreetmap.org/wiki/Key:amenity', match='Value')[0]
tab_shop = pd.read_html('https://wiki.openstreetmap.org/wiki/Key:shop', match='Value')[0]
tab_leisure = pd.read_html('https://wiki.openstreetmap.org/wiki/Key:leisure', match='Value')[0]

desired_buildings = ['office']
desired_amenities = ['fitness_center', 'fast_food', 'bank']
desired_shops = ['supermarket', 'mall']
desired_leisure = ['park']

# Get bounding box / polygon for Alameda only
alameda_poly = alameda_gdf.unary_union  # union for bounding box only, polygons still separate later

all_pois = []

# Download each category separately
poi_layers = {
    "building": desired_buildings,
    "amenity": desired_amenities,
    "shop": desired_shops,
    "leisure": desired_leisure
}

for key, values in poi_layers.items():
    pois = ox.geometries_from_polygon(
        alameda_poly,
        tags={key: values}
    )
    pois["poi_type"] = key
    all_pois.append(pois)

# Merge into one GeoDataFrame
pois_gdf = pd.concat(all_pois, ignore_index=True)
pois_gdf = pois_gdf.to_crs(alameda_gdf.crs)  # ensure same CRS

pois_joined = gpd.sjoin(
    pois_gdf,
    alameda_gdf[["geoid_str_", "geometry"]],
    how="left",
    predicate="within"
)

poi_counts = (
    pois_joined.groupby(["geoid_str_", "poi_type"])
    .size()
    .unstack(fill_value=0)
    .reset_index()
)

network_df = network_df.merge(
    poi_counts,
    how="left",
    on="geoid_str_"
)

# Replace NaN with zeros for blocks with no POIs
for c in ["amenity", "building", "shop", "leisure"]:
    if c in network_df.columns:
        network_df[c] = network_df[c].fillna(0)

# =============================================
# End POI analysis
# =============================================

# Simple heat map, keep for now
# fig, ax = plt.subplots(figsize=(10, 10))

# # Plot only the boundaries
# alameda_gdf.plot(
#     column = 'charging_demand',
#     cmap = 'Reds',
#     legend=True,
#     ax=ax, 
#     edgecolor='black', 
#     linewidth=0.8
# )

# # Optional: plot the geometries filled lightly
# # alameda_gdf.plot(ax=ax, color='lightblue', alpha=0.3, edgecolor='black')

# # Add titles and axes
# ax.set_title("EV Demand Heatmap - Alameda County", fontsize=14)
# ax.set_xlabel("Longitude")
# ax.set_ylabel("Latitude")
# plt.show()

alameda_chargers = charger_loc_gdf.copy()
alameda_chargers = alameda_chargers[alameda_chargers['geoid_str_'].str.contains('Alameda', case=False, na=False)]

alameda_gdf = alameda_gdf.drop(columns=['NEIGHBOURS'])

# --- Create interactive map ---
m = folium.Map(
    location=[alameda_gdf['INTPTLAT'].mean(), alameda_gdf['INTPTLON'].mean()],
    zoom_start=10,
    tiles='cartodbpositron'
)

# Add choropleth for charging demand
folium.Choropleth(
    geo_data=alameda_gdf,
    name="Charging Demand",
    data=alameda_gdf,
    columns=["geoid_str_", "charging_demand"],
    key_on="feature.properties.geoid_str_",
    fill_color="Reds",
    fill_opacity=0.7,
    line_opacity=0.5,
    legend_name="EV Charging Demand (kW)"
).add_to(m)

# Add tract tooltips
folium.GeoJson(
    alameda_gdf,
    style_function=lambda x: {"fillColor": "transparent", "color": "black", "weight": 0.8},
    tooltip=GeoJsonTooltip(
        fields=["geoid_str_", "charging_demand"],
        aliases=["Tract:", "Charging demand:"],
        localize=True
    )
).add_to(m)

# Add chargers as points
for _, row in alameda_chargers.iterrows():
    folium.CircleMarker(
        location=[row.geometry.y, row.geometry.x],
        radius=4,
        color='green',
        fill=True,
        fill_color='green',
        fill_opacity=0.7,
        popup=f"Charger ID: {row['ID']}"  # optional popup
    ).add_to(m)

    # Add census tract centroids as points
for _, row in alameda_gdf.iterrows():
    folium.CircleMarker(
        location=[row.geometry.centroid.y, row.geometry.centroid.x],
        radius=1,
        color='blue',
        fill=True,
        fill_color='blue',
        fill_opacity=0.7,
        popup=f"Census Track ID: {row['geoid_str_']}"  # optional popup
    ).add_to(m)

# Convert all geometries to points for plotting
pois_gdf["plot_geom"] = pois_gdf.geometry.apply(
    lambda geom: geom.centroid if geom.geom_type != "Point" else geom
)
# Add POIs as points
for _, row in pois_gdf.iterrows():
    pt = row.plot_geom
    folium.CircleMarker(
        location=[pt.y, pt.x],
        radius=1,
        color='purple',
        fill=True,
        fill_color='purple',
        fill_opacity=0.6
    ).add_to(m)

# --- Add Custom HTML Legend for Markers ---
legend_html = """
     <div style="position: fixed; 
     bottom: 50px; left: 50px; width: 200px; height: 90px; 
     border:2px solid grey; z-index:9999; font-size:14px;
     background-color:white; opacity: 0.9;">
       &nbsp; <b>Map Legend</b> <br>
       &nbsp; <i style="background:green; color:green; border-radius:50%; margin-top: 5px; width: 10px; height: 10px; display:inline-block;"></i> &nbsp; Charging Station<br>
       &nbsp; <i style="background:blue; color:blue; border-radius:50%; margin-top: 5px; width: 10px; height: 10px; display:inline-block;"></i> &nbsp; Census Block Centroid<br>
       &nbsp; <i style="background:purple; color:purple; border-radius:50%; margin-top: 5px; width: 10px; height: 10px; display:inline-block;"></i> &nbsp; POI<br>
     </div>
     """
m.get_root().html.add_child(folium.Element(legend_html))
# --- End Custom HTML Legend ---

# --- Save and open map ---
output_file = "alameda_charging_map.html"
m.save(output_file)
print(f"Interactive map saved as {output_file}. Open this file in a web browser to explore.")

# Alameda Charger Type Counts
# ================================================
print(alameda_chargers.head())
print(len(alameda_chargers))
print(charger_df[['EV Level1 EVSE Num', 'EV Level2 EVSE Num', 'EV DC Fast Count']].head())
print(f"Level 1 Charger Total: {alameda_chargers['EV Level1 EVSE Num'].sum()}")
print(f"Level 2 Charger Total: {alameda_chargers['EV Level2 EVSE Num'].sum()}")
print(f"DC Fast Charger Total: {alameda_chargers['EV DC Fast Count'].sum()}")
print(f"Level 1 Station Total: {alameda_chargers['EV Level1 EVSE Num'].count()}")
print(f"Level 2 Station Total: {alameda_chargers['EV Level2 EVSE Num'].count()}")
print(f"DC Fast Station Total: {alameda_chargers['EV DC Fast Count'].count()}")

# Uniqueness Counts
# ================================================
print(network_df['geoid_str_'].nunique())
print(len(network_df['geoid_str_']))
print(alameda_chargers.head())
print(len(alameda_chargers['geoid_str_']))
print(alameda_chargers['geoid_str_'].nunique())

# ================================================
# POI CALCS and PLOT
# ================================================

# Create weighted POI score for each node
# Duplicates are included, will be filtered out later

network_df['POI_SCORE'] = network_df['building']*3+network_df['shop']*2+network_df['leisure']*2+network_df['amenity']
network_df = network_df.drop(['building', 'shop', 'leisure', 'amenity'], axis=1)

# Left merge alameda_gdf with pois_gdf to assign census tract to each POI

alameda_gdf['centroid'] = alameda_gdf.geometry.centroid
pois_joined = gpd.sjoin(
    pois_gdf,
    alameda_gdf[['geoid_str_', 'centroid', 'geometry']],
    how='left',
    predicate='within'
)
pois_joined = pois_joined.dropna(subset=["centroid"])

# Calculate haversine distance from each POI to its census tract centroid

distances = []

for _, poi in pois_joined.iterrows():
    poi_rad = np.radians([poi.plot_geom.y, poi.plot_geom.x])
    centroid_rad = np.radians([poi.centroid.y, poi.centroid.x])

    d_rad = haversine_distances([poi_rad], [centroid_rad])[0][0]
    d_m = d_rad * 6371

    distances.append(d_m)

pois_joined["dist_to_centroid_km"] = distances

# Find avg distance from each centroid to its census tract POIs

avg_distances = pois_joined.groupby("geoid_str_")["dist_to_centroid_km"].mean().reset_index()
avg_distances = avg_distances.rename(columns={"dist_to_centroid_km": "avg_poi_dist_km"})

# Add avg distances to alameda_gdf

alameda_gdf = alameda_gdf.merge(avg_distances, how="left", on="geoid_str_")
alameda_gdf["avg_poi_dist_km"] = alameda_gdf["avg_poi_dist_km"].fillna(0)

# Create new heatmap to show where POIs are close and where they are far

# --- Create interactive map ---
m2 = folium.Map(
    location=[alameda_gdf['INTPTLAT'].mean(), alameda_gdf['INTPTLON'].mean()],
    zoom_start=10,
    tiles='cartodbpositron'
)

# Add choropleth for charging demand
folium.Choropleth(
    geo_data=alameda_gdf.drop(columns=['centroid']),
    name="Avg POI Distance to Centroid",
    data=alameda_gdf,
    columns=["geoid_str_", "avg_poi_dist_km"],
    key_on="feature.properties.geoid_str_",
    fill_color="Purples",
    fill_opacity=0.7,
    line_opacity=0.5,
    legend_name="Avg POI Distance (km)"
).add_to(m2)

# Add tract tooltips
folium.GeoJson(
    alameda_gdf.drop(columns=['centroid']),
    style_function=lambda x: {"fillColor": "transparent", "color": "black", "weight": 0.8},
    tooltip=GeoJsonTooltip(
        fields=["geoid_str_", "charging_demand"],
        aliases=["Tract:", "Charging demand:"],
        localize=True
    )
).add_to(m2)

    # Add census tract centroids as points
for _, row in alameda_gdf.iterrows():
    folium.CircleMarker(
        location=[row.geometry.centroid.y, row.geometry.centroid.x],
        radius=1,
        color='blue',
        fill=True,
        fill_color='blue',
        fill_opacity=0.7,
        popup=f"Census Track ID: {row['geoid_str_']}"  # optional popup
    ).add_to(m2)

# Convert all geometries to points for plotting
pois_gdf["plot_geom"] = pois_gdf.geometry.apply(
    lambda geom: geom.centroid if geom.geom_type != "Point" else geom
)
# Add POIs as points
for _, row in pois_gdf.iterrows():
    pt = row.plot_geom
    folium.CircleMarker(
        location=[pt.y, pt.x],
        radius=1,
        color='purple',
        fill=True,
        fill_color='purple',
        fill_opacity=0.6
    ).add_to(m2)

# --- Add Custom HTML Legend for Markers ---
legend_html = """
     <div style="position: fixed; 
     bottom: 50px; left: 50px; width: 200px; height: 90px; 
     border:2px solid grey; z-index:9999; font-size:14px;
     background-color:white; opacity: 0.9;">
       &nbsp; <b>Map Legend</b> <br>
       &nbsp; <i style="background:blue; color:blue; border-radius:50%; margin-top: 5px; width: 10px; height: 10px; display:inline-block;"></i> &nbsp; Census Block Centroid<br>
       &nbsp; <i style="background:purple; color:purple; border-radius:50%; margin-top: 5px; width: 10px; height: 10px; display:inline-block;"></i> &nbsp; POI<br>
     </div>
     """
m2.get_root().html.add_child(folium.Element(legend_html))
# --- End Custom HTML Legend ---

# --- Save and open map ---
output_file = "alameda_POI_map.html"
m2.save(output_file)
print(f"Interactive map saved as {output_file}. Open this file in a web browser to explore.")

# =============================================
# Start: Add travel time to network_df
# =============================================
import osmnx as ox
import networkx as nx

#create network for osmnx
G = ox.graph_from_place("Alameda County, California, USA", network_type="drive")
G = ox.speed.add_edge_speeds(G)
G = ox.speed.add_edge_travel_times(G)

#create travel time df of origin destination pairs and respective lat/lon 
tt_df = network_df[['geoid_str_', 'INTPTLAT', 'INTPTLON', 'NEIGHBOURS']].copy()
tt_df.columns = ["orig_geoid_str", "orig_lat", "orig_lon", "dest_geoid_str"]

tt_df = tt_df.merge(
    gdf_subset,
    left_on='dest_geoid_str',  # column in table1
    right_on='geoid_str_',       # column in table2
    how='left'
)

tt_df = tt_df.drop(columns = ["NEIGHBOURS", "geoid_str_"])
tt_df = tt_df.rename(columns={'INTPTLAT': 'dest_lat'})
tt_df = tt_df.rename(columns={'INTPTLON': 'dest_lon'})

# snap OD pair to node on G
tt_df["origin_node"] = ox.distance.nearest_nodes(
    G,
    tt_df["orig_lon"].values,
    tt_df["orig_lat"].values
)

tt_df["dest_node"] = ox.distance.nearest_nodes(
    G,
    tt_df["dest_lon"].values,
    tt_df["dest_lat"].values
)

def compute_tt(row):
    try:
        return nx.shortest_path_length(
            G,
            source=row["origin_node"],
            target=row["dest_node"],
            weight="travel_time"
        )
    except:
        return np.nan  # unreachable

tt_df["travel_time_sec"] = tt_df.apply(compute_tt, axis=1)
tt_df["travel_time_min"] = tt_df["travel_time_sec"] / 60
network_df = network_df.merge(
    tt_df[['orig_geoid_str', 'dest_geoid_str', "travel_time_min", "travel_time_sec"]],
    left_on=['geoid_str_', 'NEIGHBOURS'],  
    right_on=['orig_geoid_str', 'dest_geoid_str'], 
    how='left'                     
)

print(list(network_df.columns))
print('='*10 + 'Finish network file' + '='*10)








