#%%
import geopandas as gpd
import pandas as pd
line_shp = "genx-notebooks\\interregional_connections_NYISO\\interregional_connections_NYISO_neha.shp"
transmission_expansion_data = "genx-scenarios\\And_No_IRA_results_summary\\transmission.csv"

# Replace 'lines.shp' with your actual filename (works for .geojson too)
gdf = gpd.read_file(line_shp)

# Check the first few rows to see data like Voltage or Owner
print(gdf.head())
transmission_df = pd.read_csv(transmission_expansion_data)

transmission_df[["start_region", "dest_region"]] = transmission_df["transmission_path_name"].str.split(
    pat="_to_",
    n=1,
    expand=True,
)


# Map `region_1`/`region_2` pairs to the matching transmission path rows
capacity_lookup = (
    transmission_df[["start_region", "dest_region", "New_Trans_Capacity"]]
    .rename(columns={"start_region": "region_1", "dest_region": "region_2"})
)
gdf = gdf.merge(capacity_lookup, on=["region_1", "region_2"], how="left")
gdf["value"] = gdf["New_Trans_Capacity"]

print(gdf)


#%%

# Check basic info
print(f"Features: {len(gdf)}")
print(f"CRS: {gdf.crs}")
print(gdf.columns)
# Check current CRS
# print(f"Original CRS: {gdf.crs}")
gdf_web = gdf.to_crs(epsg=3857)


#%%
import matplotlib.pyplot as plt
import contextily as ctx

# 1. Setup the figure size
fig, ax = plt.subplots(figsize=(12, 12))

# 2. Plot the transmission lines
# We color them by a column, e.g., 'VOLTAGE' (change this to a column in your data)
gdf_web.plot(ax=ax, column='value', cmap='viridis', legend=True, linewidth=2)

# 3. Add the basemap (OpenStreetMap style)
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

# 4. Clean up the plot
ax.set_axis_off()
plt.title("Regional Transmission Lines")
plt.show()

# %%
