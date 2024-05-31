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

# Introduce seasonal and hourly patterns in power demand
def generate_power_demand(timestamps):
    power_demand = []
    for timestamp in timestamps:
        hour = timestamp.hour
        base_demand = 50 + 30 * np.sin(2 * np.pi * hour / 24)  # Daily pattern
        noise = np.random.normal(0, 5)
        demand = base_demand + noise
        power_demand.append(demand)
    return power_demand

hourly_trends = pd.DataFrame({
    'Distributor_ID': np.random.choice(distributor_ids, num_hours),
    'Timestamp': timestamps,
    'Power_Demand_kWh': generate_power_demand(timestamps)
})

# Generate synthetic data for power lines strength with realistic relationships
line_ids = [f"L{i+1}" for i in range(num_lines)]
source_ids = [f"S{i+1}" for i in range(num_suppliers)]
destination_ids = np.random.choice(distributor_ids, num_lines)

# Define correlated features
max_capacity = np.random.uniform(100, 200, num_lines)
current_load = max_capacity * np.random.uniform(0.7, 0.9)  # Correlated with max_capacity

power_lines = pd.DataFrame({
    'Power_Line_ID': line_ids,
    'Source_ID': np.random.choice(source_ids, num_lines),
    'Destination_ID': destination_ids,
    'Max_Capacity_kWh': max_capacity,
    'Current_Load_kWh': current_load
})

# Generate synthetic data for power supplier locations and generation rates with realistic ranges
max_generation_rate = np.random.uniform(500, 1000, num_suppliers)
current_generation_rate = max_generation_rate * np.random.uniform(0.65, 0.85)  # Correlated with max_generation_rate

supplier_locations = pd.DataFrame({
    'Supplier_ID': source_ids,
    'Latitude': [fake.latitude() for _ in range(num_suppliers)],
    'Longitude': [fake.longitude() for _ in range(num_suppliers)],
    'Max_Generation_Rate_kWh': max_generation_rate,
    'Current_Generation_Rate_kWh': current_generation_rate
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
