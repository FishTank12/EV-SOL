# src/synth_data.py
import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta

# Initialize Faker
fake = Faker()

# Number of records to generate
num_distributors = 50
num_suppliers = 10
num_lines = 100
num_hours = 24 * 30  # 1 month of hourly data

# Generate synthetic data for hourly trends of each distributor
distributor_ids = [f"D{i+1}" for i in range(num_distributors)]
timestamps = [datetime(2024, 5, 1) + timedelta(hours=i) for i in range(num_hours)]

hourly_trends = pd.DataFrame({
    'Distributor_ID': np.random.choice(distributor_ids, num_hours),
    'Timestamp': [timestamps[i % len(timestamps)] for i in range(num_hours)],
    'Power_Demand_kWh': np.random.uniform(50, 200, num_hours)  # Increased demand range for more realistic values
})

# Generate synthetic data for power lines strength
line_ids = [f"L{i+1}" for i in range(num_lines)]
source_ids = [f"S{i+1}" for i in range(num_suppliers)]
destination_ids = np.random.choice(distributor_ids, num_lines)

power_lines = pd.DataFrame({
    'Power_Line_ID': line_ids,
    'Source_ID': np.random.choice(source_ids, num_lines),
    'Destination_ID': destination_ids,
    'Max_Capacity_kWh': np.random.uniform(150, 300, num_lines),  # Increased capacity range
    'Current_Load_kWh': np.random.uniform(50, 150, num_lines)  # Increased load range for realism
})

# Generate synthetic data for power supplier locations and generation rates
supplier_locations = pd.DataFrame({
    'Supplier_ID': source_ids,
    'Latitude': [fake.latitude() for _ in range(num_suppliers)],
    'Longitude': [fake.longitude() for _ in range(num_suppliers)],
    'Max_Generation_Rate_kWh': np.random.uniform(500, 1000, num_suppliers),  # Increased generation rates
    'Current_Generation_Rate_kWh': np.random.uniform(300, 800, num_suppliers)  # Increased current generation rates
})

# Generate synthetic data for power distributor locations and connections
distributor_locations = pd.DataFrame({
    'Distributor_ID': distributor_ids,
    'Latitude': [fake.latitude() for _ in range(num_distributors)],
    'Longitude': [fake.longitude() for _ in range(num_distributors)],
    'Connected_Lines': [",".join(np.random.choice(line_ids, np.random.randint(1, 5))) for _ in range(num_distributors)]
})

# Save synthetic data to CSV files
hourly_trends.to_csv('../data/synthetic_hourly_trends.csv', index=False)
power_lines.to_csv('../data/synthetic_power_lines.csv', index=False)
supplier_locations.to_csv('../data/synthetic_supplier_locations.csv', index=False)
distributor_locations.to_csv('../data/synthetic_distributor_locations.csv', index=False)

print("Synthetic data has been saved to the 'data' directory.")
