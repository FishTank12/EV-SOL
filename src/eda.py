# src/eda.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load synthetic data
hourly_trends = pd.read_csv('../data/synthetic_hourly_trends.csv')
power_lines = pd.read_csv('../data/synthetic_power_lines.csv')
supplier_locations = pd.read_csv('../data/synthetic_supplier_locations.csv')
distributor_locations = pd.read_csv('../data/synthetic_distributor_locations.csv')

datasets = {
    "Distributor Locations": distributor_locations,
    "Hourly Trends": hourly_trends,
    "Power Lines": power_lines,
    "Supplier Locations": supplier_locations
}

for name, dataset in datasets.items():
    print(f"--- {name} ---")
    print(dataset.head(), "\n")
    print(dataset.info(), "\n")
    print(dataset.describe(), "\n")

# EDA on distributor locations
plt.figure(figsize=(10, 6))
sns.scatterplot(data=distributor_locations, x='Longitude', y='Latitude')
plt.title('Distributor Locations')
plt.savefig('../results/Distributor Locations.png')

# EDA on hourly trends
plt.figure(figsize=(10, 6))
sns.histplot(hourly_trends['Power_Demand_kWh'], bins=30, kde=True)
plt.title('Distribution of Hourly Power Demand')
plt.xlabel('Power Demand (kWh)')
plt.ylabel('Frequency')
plt.savefig('../results/Distribution of Hourly Power Demand.png')

# EDA on power lines with correct column name
plt.figure(figsize=(10, 6))
sns.histplot(power_lines['Max_Capacity_kWh'], bins=30, kde=True)
plt.title('Distribution of Power Line Capacity')
plt.xlabel('Capacity (kWh)')
plt.ylabel('Frequency')
plt.savefig('../results/Distribution of Power Line Capacity.png')

# Correlation analysis on numerical columns with correct column names
numeric_cols_corrected = ['Power_Demand_kWh', 'Max_Capacity_kWh', 'Max_Generation_Rate_kWh', 'Current_Generation_Rate_kWh']
combined_data_corrected = pd.concat([
    hourly_trends[['Power_Demand_kWh']], 
    power_lines[['Max_Capacity_kWh']], 
    supplier_locations[['Max_Generation_Rate_kWh', 'Current_Generation_Rate_kWh']]
], axis=1)

plt.figure(figsize=(10, 8))
sns.heatmap(combined_data_corrected.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.savefig('../results/Correlation Matrix.png')


# EDA on supplier locations
plt.figure(figsize=(10, 6))
sns.scatterplot(data=supplier_locations, x='Longitude', y='Latitude', size='Max_Generation_Rate_kWh', legend=False)
plt.title('Supplier Locations')
plt.savefig('../results/Supplier Locations.png')

# Display the columns of the power_lines dataset
print(power_lines.columns)

# Display the columns of the supplier_locations dataset to verify the correct column names
print(supplier_locations.columns)

