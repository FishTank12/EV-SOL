# src/eda.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ensure results directory exists
results_path = "../results/"
os.makedirs(results_path, exist_ok=True)

# Load datasets with specified dtypes to avoid DtypeWarnings
data_path = "../data/"

alt_fuel_stations_1 = pd.read_csv(os.path.join(data_path, "alt_fuel_stations (May 30 2024) (1).csv"), low_memory=False)
alt_fuel_stations_2 = pd.read_csv(os.path.join(data_path, "alt_fuel_stations (May 30 2024).csv"), low_memory=False)
ev_charging_stations = pd.read_csv(os.path.join(data_path, "Electric_Vehicle_Charging_Stations.csv"), dtype=str)
ev_population = pd.read_csv(os.path.join(data_path, "Electric_Vehicle_Population_Data.csv"), low_memory=False)
ev_usage = pd.read_csv(os.path.join(data_path, "EVChargingStationUsage.csv"), low_memory=False)
light_duty_vehicles_1 = pd.read_csv(os.path.join(data_path, "light-duty-vehicles-2024-05-30 (1).csv"), low_memory=False)
light_duty_vehicles_2 = pd.read_csv(os.path.join(data_path, "light-duty-vehicles-2024-05-30 (2).csv"), low_memory=False)
light_duty_vehicles_3 = pd.read_csv(os.path.join(data_path, "light-duty-vehicles-2024-05-30.csv"), low_memory=False)
medium_heavy_vehicles = pd.read_csv(os.path.join(data_path, "medium-and-heavy-duty-vehicles-2024-05-30.csv"), low_memory=False)
station_data = pd.read_csv(os.path.join(data_path, "station_data_dataverse.csv"), low_memory=False)

datasets = {
    "alt_fuel_stations_1": alt_fuel_stations_1,
    "alt_fuel_stations_2": alt_fuel_stations_2,
    "ev_charging_stations": ev_charging_stations,
    "ev_population": ev_population,
    "ev_usage": ev_usage,
    "light_duty_vehicles_1": light_duty_vehicles_1,
    "light_duty_vehicles_2": light_duty_vehicles_2,
    "light_duty_vehicles_3": light_duty_vehicles_3,
    "medium_heavy_vehicles": medium_heavy_vehicles,
    "station_data": station_data
}

for name, df in datasets.items():
    print(f"{name} shape: {df.shape}")
    print(f"{name} columns: {df.columns.tolist()}\n")
    print(f"{name} info:")
    print(df.info())
    print(f"{name} description:")
    print(df.describe(include='all'))
    print(f"{name} missing values:")
    print(df.isnull().sum())
    print("\n")

# Plot distributions for datasets with numerical columns
def plot_distributions(df, title, filename):
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if not numerical_cols.empty:
        df[numerical_cols].hist(figsize=(15, 10))
        plt.suptitle(title)
        plt.savefig(os.path.join(results_path, filename))
        plt.close()
    else:
        print(f"No numerical columns to plot for {title}")

def plot_correlations(df, title, filename):
    numerical_cols = df.select_dtypes(include=['float64', 'int64'])
    if not numerical_cols.empty:
        corr = numerical_cols.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title(title)
        plt.savefig(os.path.join(results_path, filename))
        plt.close()
    else:
        print(f"No numerical columns to correlate for {title}")

# Plot distributions and correlations for selected datasets
plot_distributions(alt_fuel_stations_1, "Alt Fuel Stations (1)", "alt_fuel_stations_1_distributions.png")
plot_distributions(ev_charging_stations, "EV Charging Stations", "ev_charging_stations_distributions.png")
plot_distributions(ev_population, "EV Population", "ev_population_distributions.png")

plot_correlations(alt_fuel_stations_1, "Alt Fuel Stations (1) Correlations", "alt_fuel_stations_1_correlations.png")
plot_correlations(ev_charging_stations, "EV Charging Stations Correlations", "ev_charging_stations_correlations.png")
plot_correlations(ev_population, "EV Population Correlations", "ev_population_correlations.png")

# Analyze EV Usage
ev_usage['Start Date'] = pd.to_datetime(ev_usage['Start Date'], errors='coerce')
ev_usage['End Date'] = pd.to_datetime(ev_usage['End Date'], errors='coerce')
ev_usage['Total Duration (minutes)'] = ev_usage['Total Duration (hh:mm:ss)'].apply(lambda x: int(x.split(':')[0])*60 + int(x.split(':')[1]))
ev_usage['Energy (kWh)'] = ev_usage['Energy (kWh)'].astype(float)

# Plot total energy consumption over time
ev_usage.set_index('Start Date')['Energy (kWh)'].resample('D').sum().plot(figsize=(15, 6), title='Total Energy Consumption Over Time')
plt.savefig(os.path.join(results_path, "total_energy_consumption_over_time.png"))
plt.close()

# Analyze EV Population
ev_population['Vehicle Location'] = ev_population['Vehicle Location'].apply(lambda x: eval(x.replace('POINT (', '').replace(')', '').replace(' ', ',')))

# Extract latitude and longitude
ev_population['Latitude'] = ev_population['Vehicle Location'].apply(lambda x: x[1])
ev_population['Longitude'] = ev_population['Vehicle Location'].apply(lambda x: x[0])

# Plot EV distribution
plt.figure(figsize=(10, 6))
sns.scatterplot(data=ev_population, x='Longitude', y='Latitude', hue='Electric Vehicle Type', palette='viridis')
plt.title('EV Ownership Distribution')
plt.savefig(os.path.join(results_path, "ev_ownership_distribution.png"))
plt.close()
