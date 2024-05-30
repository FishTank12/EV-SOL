import pandas as pd

# Define file paths
file_paths = [
    "/mnt/data/alt_fuel_stations (May 30 2024).csv",
    "/mnt/data/Electric_Vehicle_Charging_Stations.csv",
    "/mnt/data/light-duty-vehicles-2024-05-30 (1).csv",
    "/mnt/data/light-duty-vehicles-2024-05-30 (2).csv",
    "/mnt/data/light-duty-vehicles-2024-05-30.csv",
    "/mnt/data/medium-and-heavy-duty-vehicles-2024-05-30.csv",
    "/mnt/data/station_data_dataverse.csv"
]

# Load datasets
dfs = [pd.read_csv(file_path) for file_path in file_paths]

# Select relevant columns and rename them for consistency
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
    "Timestamps": "Timestamps",
    "Customer Usage History": "Customer_Usage_History",
    "Historical Charging Rates": "Historical_Charging_Rates"
}

# Clean and combine datasets
combined_df = pd.DataFrame()
for df in dfs:
    df = df.rename(columns=columns_mapping)
    df = df[list(columns_mapping.values())]
    combined_df = pd.concat([combined_df, df], ignore_index=True)

# Display the first few rows of the combined and cleaned dataset
combined_df.head()
