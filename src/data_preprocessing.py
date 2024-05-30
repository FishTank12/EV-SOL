import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point

# Define file paths
file_paths = [
    "../data/alt_fuel_stations (May 30 2024) (1).csv",
    "../data/alt_fuel_stations (May 30 2024).csv",
    "../data/Electric_Vehicle_Charging_Stations.csv",
    "../data/EVChargingStationUsage.csv",
    "../data/light-duty-vehicles-2024-05-30 (1).csv",
    "../data/light-duty-vehicles-2024-05-30 (2).csv",
    "../data/light-duty-vehicles-2024-05-30.csv",
    "../data/medium-and-heavy-duty-vehicles-2024-05-30.csv",
    "../data/station_data_dataverse.csv"
]

# Load datasets
dfs = [pd.read_csv(file_path, low_memory=False) for file_path in file_paths]

# Define columns mapping
columns_mapping = {
    "Station Name": "Station_Name",
    "Street Address": "Street_Address",
    "City": "City",
    "State": "State",
    "ZIP": "ZIP",
    "Latitude": "Latitude",
    "Longitude": "Longitude",
    "Access Days Time": "Access_Days_Time",
    "EV Level1 EVSE Num": "EV_Level1_EVSE_Num",
    "EV Level2 EVSE Num": "EV_Level2_EVSE_Num",
    "EV DC Fast Count": "EV_DC_Fast_Count",
    "Total kWh": "Total_kWh",
    "Dollars Spent": "Dollars_Spent",
    "Start Date": "Timestamps",  # Assuming Start Date is used as Timestamps
    "Customer Usage History": "Customer_Usage_History",
    "Historical Charging Rates": "Historical_Charging_Rates"
}

# Clean and combine datasets
combined_df = pd.DataFrame()
for df in dfs:
    df = df.rename(columns=columns_mapping)
    common_columns = set(columns_mapping.values()).intersection(df.columns)
    df = df[list(common_columns)]
    combined_df = pd.concat([combined_df, df], ignore_index=True)

# Ensure necessary columns are in the final dataset
required_columns = ['City', 'Street_Address', 'Station_Name', 'Latitude', 'Longitude', 'Timestamps']

# Check for missing values
missing_values = combined_df.isnull().sum()
print("Missing values before any processing:")
print(missing_values[missing_values > 0])

# Fill missing numerical data with zeros
for col in ['EV_Level1_EVSE_Num', 'EV_Level2_EVSE_Num', 'EV_DC_Fast_Count', 'Total_kWh', 'Dollars_Spent']:
    if col in combined_df.columns:
        combined_df[col].fillna(0, inplace=True)

# Fill missing text data with a default value
combined_df['Access_Days_Time'].fillna('Unknown', inplace=True)

# Convert Timestamps to datetime and drop rows with missing critical data
combined_df['Timestamps'] = pd.to_datetime(combined_df['Timestamps'], errors='coerce')
cleaned_combined_df = combined_df.dropna(subset=['City', 'Street_Address', 'Station_Name', 'Latitude', 'Longitude'])

# Verify how many rows are left after dropping
print(f"Rows remaining after dropping rows with missing critical data: {len(cleaned_combined_df)}")

# Display the first few rows of the cleaned dataset
print(cleaned_combined_df.head())

# Plot distribution of charging stations by city
plt.figure(figsize=(12, 6))
sns.countplot(y=cleaned_combined_df['City'], order=cleaned_combined_df['City'].value_counts().index)
plt.title('Distribution of Charging Stations by City')
plt.xlabel('Number of Charging Stations')
plt.ylabel('City')
plt.savefig("../results/Distribution_of_Charging_Stations_by_City.png")

# Plot distribution of EV Level 1, Level 2, and DC Fast chargers
plt.figure(figsize=(12, 6))
sns.histplot(cleaned_combined_df['EV_Level1_EVSE_Num'], bins=20, label='Level 1', color='blue', kde=True)
sns.histplot(cleaned_combined_df['EV_Level2_EVSE_Num'], bins=20, label='Level 2', color='green', kde=True)
sns.histplot(cleaned_combined_df['EV_DC_Fast_Count'], bins=20, label='DC Fast', color='red', kde=True)
plt.title('Distribution of EVSE Numbers')
plt.xlabel('Number of EVSE')
plt.ylabel('Frequency')
plt.legend()
plt.savefig("../results/Distribution_of_EVSE_Numbers.png")

# Plot distribution of access days and times
plt.figure(figsize=(12, 6))
sns.countplot(y=cleaned_combined_df['Access_Days_Time'], order=cleaned_combined_df['Access_Days_Time'].value_counts().index)
plt.title('Distribution of Access Days and Times')
plt.xlabel('Number of Charging Stations')
plt.ylabel('Access Days and Times')
plt.savefig("../results/Distribution_of_Access_Days_and_Times.png")

# Plot correlation matrix
plt.figure(figsize=(12, 8))
corr_matrix = cleaned_combined_df.corr(numeric_only=True)
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.savefig("../results/Correlation_Matrix.png")

# Time Series Analysis
if 'Timestamps' in cleaned_combined_df.columns:
    cleaned_combined_df.set_index('Timestamps', inplace=True)

    # Plotting total kWh over time
    plt.figure(figsize=(12, 6))
    cleaned_combined_df['Total_kWh'].resample('M').sum().plot()
    plt.title('Total kWh Usage Over Time')
    plt.xlabel('Time')
    plt.ylabel('Total kWh')
    plt.savefig("../results/Total_kWh_Usage_Over_Time.png")

    # Plotting total dollars spent over time
    plt.figure(figsize=(12, 6))
    cleaned_combined_df['Dollars_Spent'].resample('M').sum().plot()
    plt.title('Total Dollars Spent Over Time')
    plt.xlabel('Time')
    plt.ylabel('Dollars Spent')
    plt.savefig("../results/Total_Dollars_Spent_Over_Time.png")

# Geospatial Analysis (requires geopandas)
if 'Latitude' in cleaned_combined_df.columns and 'Longitude' in cleaned_combined_df.columns:
    gdf = gpd.GeoDataFrame(
        cleaned_combined_df, geometry=gpd.points_from_xy(cleaned_combined_df.Longitude, cleaned_combined_df.Latitude))

    # Plotting charging stations
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    ax = world[world.name == "United States"].plot(color='white', edgecolor='black')
    gdf.plot(ax=ax, color='red', markersize=5)
    plt.title('Geographic Distribution of Charging Stations')
    plt.savefig("../results/Geographic_Distribution_of_Charging_Stations.png")
