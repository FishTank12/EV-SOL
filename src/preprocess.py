# src/preprocess.py
import pandas as pd

# Load data
distributors = pd.read_csv('../data/distributors.csv')
suppliers = pd.read_csv('../data/suppliers.csv')
power_lines = pd.read_csv('../data/power_lines.csv')
hourly_trends = pd.read_csv('../data/hourly_trends.csv')

# Example preprocessing steps (customize as needed)
hourly_trends['Timestamp'] = pd.to_datetime(hourly_trends['Timestamp'])

# Merge datasets if needed
final_data = hourly_trends.merge(distributors, on='Distributor_ID', how='left')
final_data = final_data.merge(power_lines, left_on='Distributor_ID', right_on='Destination_ID', how='left')
final_data = final_data.merge(suppliers, left_on='Source_ID', right_on='Supplier_ID', how='left')

# Calculate additional features
final_data['Load_Ratio'] = final_data['Current_Load_kWh'] / final_data['Max_Capacity_kWh']
final_data['Generation_Ratio'] = final_data['Current_Generation_Rate_kWh'] / final_data['Max_Generation_Rate_kWh']

# Drop any unnecessary columns
final_data = final_data.drop(columns=['Connected_Lines', 'Source_ID', 'Destination_ID', 'Supplier_ID'])

# Drop rows with any NaN values
final_data = final_data.dropna()

# Save the preprocessed data
final_data.to_csv('../data/final_data.csv', index=False)

# Print the correlation matrix
correlation_matrix = final_data.corr(numeric_only=True)
print("Correlation Matrix:")
print(correlation_matrix)

print("Preprocessed data has been saved to the 'data' directory.")
