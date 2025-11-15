from block_group_load_curves import *
from location_str_to_geoid_mapping import *
from public_chargers import *
from sklearn.metrics.pairwise import haversine_distances
from skmob.models.gravity import Gravity
import folium
from folium.features import GeoJsonTooltip

print('='*10 + 'Start network file' + '='*10)

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

# =============================================
# Full connection network ** DEPRECATED
# =============================================

# # Build origin-destination DataFrame
# origins = np.repeat(network_df_filtered['geoid_str_'].values, len(coords))
# destinations = np.tile(network_df_filtered['geoid_str_'].values, len(coords))
# distances = dist_matrix.flatten()
# mass_origins = np.repeat(network_df_filtered['mass'].values, len(coords))
# mass_destinations = np.tile(network_df_filtered['mass'].values, len(coords))

# od_df = pd.DataFrame({
#     'origin': origins,
#     'destination': destinations,
#     'distance': distances,
#     'mass_origin': mass_origins,
#     'mass_destination': mass_destinations
# })

# od_df_filtered = od_df[od_df['mass_origin']*10 < od_df['mass_destination']]

# =============================================
# Boundary Network
# =============================================

# Things to do:
# Explod df into edge table with columns of origin, destination, origin mass, dest mass, distance, origin chargers, dest chargers, origin cap, dest cap

# ------------
# To do by EOD 11/12/2025

# Step 1: Expode df

network_df = gdf_subset.explode('NEIGHBOURS')

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
 # drop only the right-hand duplicate

network_df = network_df.drop(columns=["geoid_str__y"])
network_df = network_df.rename(columns={"geoid_str__x": "geoid_str_"})


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

# Step 2: Add demand masses and haversine_distances

# -------------
# To do by EOD 11/13/2025

# Step 3: Get charger data

# Step 4: Sort chargers into census tracks and add to table




# -------------
# To do by EOW

# Step 5: Get OSM POIs


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
# alameda_gdf['charging_demand'] = np.random.randint(0, 100, size=len(alameda_gdf))
# print(f'alameda_gdf length: {len(alameda_gdf)}')
# print(f'network_filtered_df length {len(network_df_filtered)}')

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
    legend_name="EV Charging Demand"
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
        color='blue',
        fill=True,
        fill_color='blue',
        fill_opacity=0.7,
        popup=f"Charger ID: {row['ID']}"  # optional popup
    ).add_to(m)

# --- Save and open map ---
output_file = "alameda_charging_map.html"
m.save(output_file)
print(f"Interactive map saved as {output_file}. Open this file in a web browser to explore.")

print('='*10 + 'Finish network file' + '='*10)




