# src/preprocess.py
import pandas as pd
from sklearn.impute import SimpleImputer

# Load synthetic data
hourly_trends = pd.read_csv('../data/synthetic_hourly_trends.csv')
power_lines = pd.read_csv('../data/synthetic_power_lines.csv')
supplier_locations = pd.read_csv('../data/synthetic_supplier_locations.csv')
distributor_locations = pd.read_csv('../data/synthetic_distributor_locations.csv')

# Merge hourly trends with distributor locations
merged_data = hourly_trends.merge(distributor_locations, on='Distributor_ID')

# Merge with power lines data
merged_data = merged_data.merge(power_lines, left_on='Distributor_ID', right_on='Destination_ID')

# Merge with supplier locations
merged_data = merged_data.merge(supplier_locations, left_on='Source_ID', right_on='Supplier_ID')

# Select necessary columns
final_data = merged_data[['Timestamp', 'Distributor_ID', 'Power_Demand_kWh', 'Max_Capacity_kWh', 'Current_Load_kWh', 
                          'Max_Generation_Rate_kWh', 'Current_Generation_Rate_kWh', 'Latitude_x', 'Longitude_x', 
                          'Latitude_y', 'Longitude_y']]

# Rename columns for clarity
final_data.columns = ['Timestamp', 'Distributor_ID', 'Power_Demand_kWh', 'Max_Capacity_kWh', 'Current_Load_kWh', 
                      'Max_Generation_Rate_kWh', 'Current_Generation_Rate_kWh', 'Distributor_Latitude', 
                      'Distributor_Longitude', 'Supplier_Latitude', 'Supplier_Longitude']

# Handle missing values
imputer = SimpleImputer(strategy='mean')
final_data[['Max_Capacity_kWh', 'Current_Load_kWh', 'Max_Generation_Rate_kWh', 
            'Current_Generation_Rate_kWh', 'Distributor_Latitude', 'Distributor_Longitude', 
            'Supplier_Latitude', 'Supplier_Longitude']] = imputer.fit_transform(
                final_data[['Max_Capacity_kWh', 'Current_Load_kWh', 'Max_Generation_Rate_kWh', 
                            'Current_Generation_Rate_kWh', 'Distributor_Latitude', 'Distributor_Longitude', 
                            'Supplier_Latitude', 'Supplier_Longitude']])

# Save final data for model training
final_data.to_csv('../data/final_synthetic_data.csv', index=False)
