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
cleaned_combined_df['EV_Level2_EVSE_Num'].fillna(0, inplace=True)
cleaned_combined_df['EV_DC_Fast_Count'].fillna(0, inplace=True)

# Fill missing text data with a default value
cleaned_combined_df['Access_Days_Time'].fillna('Unknown', inplace=True)

# Save the cleaned dataset to a file
cleaned_combined_df.to_csv("../results/cleaned_combined_data.csv", index=False)

# Display the first few rows of the cleaned dataset
print(cleaned_combined_df.head())

# Plot distribution of charging stations by city
plt.figure(figsize=(12, 6))
sns.countplot(y=cleaned_combined_df['City'], order=cleaned_combined_df['City'].value_counts().index)
plt.title('Distribution of Charging Stations by City')
plt.xlabel('Number of Charging Stations')
plt.ylabel('City')
plt.savefig("../results/Distribution_of_Charging_Stations_by_City.png")
plt.close()

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
plt.close()

# Plot distribution of access days and times
plt.figure(figsize=(12, 6))
sns.countplot(y=cleaned_combined_df['Access_Days_Time'], order=cleaned_combined_df['Access_Days_Time'].value_counts().index)
plt.title('Distribution of Access Days and Times')
plt.xlabel('Number of Charging Stations')
plt.ylabel('Access Days and Times')
plt.savefig("../results/Distribution_of_Access_Days_and_Times.png")
plt.close()

# Save the first few rows of the cleaned dataset to a text file
with open("../results/head.txt", 'w') as f:
    f.write(cleaned_combined_df.head().to_string())
