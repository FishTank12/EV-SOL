import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define file paths
file_paths = [
    "../data/Electric_Vehicle_Charging_Stations.csv",
    "../data/light-duty-vehicles-2024-05-30 (1).csv",
    "../data/light-duty-vehicles-2024-05-30 (2).csv",
    "../data/light-duty-vehicles-2024-05-30.csv",
    "../data/medium-and-heavy-duty-vehicles-2024-05-30.csv",
    "../data/station_data_dataverse.csv"
]

# Load datasets
dfs = [pd.read_csv(file_path) for file_path in file_paths]

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
    "Timestamps": "Timestamps",
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

# Display the first few rows of the combined and cleaned dataset
print(combined_df.head())

# Check for missing values
missing_values = combined_df.isnull().sum()

# Display columns with missing values
print(missing_values[missing_values > 0])

# Remove rows with missing critical data
cleaned_combined_df = combined_df.dropna(subset=['City', 'Street_Address', 'Station_Name'])

# Fill missing numerical data with zeros
cleaned_combined_df['EV_Level1_EVSE_Num'].fillna(0, inplace=True)
