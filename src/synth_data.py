# src/synth_data.py
import pandas as pd
import numpy as np

# Number of records to generate
num_distributors = 50
num_suppliers = 10
num_power_lines = 100
num_hours = 720

# Generate random distributor data
distributors = pd.DataFrame({
    'Distributor_ID': [f'D{i+1}' for i in range(num_distributors)],
    'Latitude': np.random.uniform(-90, 90, num_distributors),
    'Longitude': np.random.uniform(-180, 180, num_distributors),
    'Connected_Lines': [','.join(np.random.choice([f'L{i+1}' for i in range(num_power_lines)], 
                                                  np.random.randint(1, 5))) for _ in range(num_distributors)]
})

# Generate random supplier data
suppliers = pd.DataFrame({
    'Supplier_ID': [f'S{i+1}' for i in range(num_suppliers)],
    'Latitude': np.random.uniform(-90, 90, num_suppliers),
    'Longitude': np.random.uniform(-180, 180, num_suppliers),
    'Max_Generation_Rate_kWh': np.random.uniform(500, 1000, num_suppliers),
    'Current_Generation_Rate_kWh': lambda x: x['Max_Generation_Rate_kWh'] * np.random.uniform(0.5, 0.8)
}).assign(Current_Generation_Rate_kWh=lambda x: x['Max_Generation_Rate_kWh'] * np.random.uniform(0.5, 0.8))

# Generate random power line data
power_lines = pd.DataFrame({
    'Power_Line_ID': [f'L{i+1}' for i in range(num_power_lines)],
    'Source_ID': np.random.choice(suppliers['Supplier_ID'], num_power_lines),
    'Destination_ID': np.random.choice(distributors['Distributor_ID'], num_power_lines),
    'Max_Capacity_kWh': np.random.uniform(100, 200, num_power_lines),
    'Current_Load_kWh': lambda x: x['Max_Capacity_kWh'] * np.random.uniform(0.6, 0.9)
}).assign(Current_Load_kWh=lambda x: x['Max_Capacity_kWh'] * np.random.uniform(0.6, 0.9))

# Generate random hourly trends data with more correlation
timestamps = pd.date_range('2024-05-01', periods=num_hours, freq='H')
base_demand = np.random.uniform(30, 50, num_distributors)  # Base demand per distributor
hourly_variation = np.sin(np.linspace(0, 2 * np.pi, num_hours)) * 10  # Sinusoidal variation

hourly_trends = pd.DataFrame({
    'Timestamp': np.tile(timestamps, num_distributors),
    'Distributor_ID': np.repeat(distributors['Distributor_ID'], num_hours),
    'Power_Demand_kWh': np.repeat(base_demand, num_hours) + np.tile(hourly_variation, num_distributors) + np.random.uniform(-5, 5, num_distributors * num_hours)
})

# Save synthetic data
distributors.to_csv('../data/distributors.csv', index=False)
suppliers.to_csv('../data/suppliers.csv', index=False)
power_lines.to_csv('../data/power_lines.csv', index=False)
hourly_trends.to_csv('../data/hourly_trends.csv', index=False)

print("Synthetic data has been saved to the 'data' directory.")
