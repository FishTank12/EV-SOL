# src/eda.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load preprocessed data
final_data = pd.read_csv('../data/final_data.csv')
distributors = pd.read_csv('../data/distributors.csv')
suppliers = pd.read_csv('../data/suppliers.csv')

# Correlation matrix
correlation_matrix = final_data.corr(numeric_only=True)
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig('../results/Correlation_Matrix.png')

# Distribution of hourly power demand
plt.figure(figsize=(10, 6))
sns.histplot(final_data['Power_Demand_kWh'], kde=True)
plt.title('Distribution of Hourly Power Demand')
plt.xlabel('Power Demand (kWh)')
plt.ylabel('Frequency')
plt.savefig('../results/Distribution_of_Hourly_Power_Demand.png')

# Distribution of power line capacity
plt.figure(figsize=(10, 6))
sns.histplot(final_data['Max_Capacity_kWh'], kde=True)
plt.title('Distribution of Power Line Capacity')
plt.xlabel('Capacity (kWh)')
plt.ylabel('Frequency')
plt.savefig('../results/Distribution_of_Power_Line_Capacity.png')

# Distributor locations
plt.figure(figsize=(10, 6))
plt.scatter(distributors['Longitude'], distributors['Latitude'])
plt.title('Distributor Locations')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.savefig('../results/Distributor_Locations.png')

# Supplier locations
plt.figure(figsize=(10, 6))
plt.scatter(suppliers['Longitude'], suppliers['Latitude'])
plt.title('Supplier Locations')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.savefig('../results/Supplier_Locations.png')

print("EDA visualizations have been saved.")
