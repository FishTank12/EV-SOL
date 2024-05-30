# src/training.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# Load the final data
final_data = pd.read_csv('../data/final_synthetic_data.csv')

# Calculate distance between suppliers and distributors
final_data['Distance'] = np.sqrt((final_data['Distributor_Latitude'] - final_data['Supplier_Latitude'])**2 +
                                 (final_data['Distributor_Longitude'] - final_data['Supplier_Longitude'])**2)

# Calculate load ratio
final_data['Load_Ratio'] = final_data['Current_Load_kWh'] / final_data['Max_Capacity_kWh']

# Load the data with features
data = final_data

# Define features and target
X = data[['Max_Capacity_kWh', 'Current_Load_kWh', 'Max_Generation_Rate_kWh', 'Current_Generation_Rate_kWh', 'Distance', 'Load_Ratio']]
y = data['Power_Demand_kWh']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42)
}

# Train and evaluate each model
for name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Perform cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
    
    print(f'--- {name} ---')
    print(f'Mean Squared Error: {mse}')
    print(f'Mean Absolute Error: {mae}')
    print(f'R-squared: {r2}')
    print(f'Cross-Validation MAE Scores: {-cv_scores.mean()}')
    print("\n")

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the neural network model
def build_model():
    model = Sequential()
    model.add(Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

# Build and train the neural network
model = build_model()
model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, verbose=1)  # Reduce epochs for quicker evaluation

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'--- Neural Network ---')
print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')
print(f'R-squared: {r2}')

# Cross-Validation for Neural Network
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_mae_scores = []

for train_index, val_index in kf.split(X):
    X_train_cv, X_val_cv = X.iloc[train_index], X.iloc[val_index]
    y_train_cv, y_val_cv = y.iloc[train_index], y.iloc[val_index]
    
    # Standardize the data
    X_train_cv_scaled = scaler.fit_transform(X_train_cv)
    X_val_cv_scaled = scaler.transform(X_val_cv)
    
    # Build and train the neural network
    model_cv = build_model()
    model_cv.fit(X_train_cv_scaled, y_train_cv, epochs=20, batch_size=32, verbose=0)  # Reduce verbosity for cross-validation
    
    # Make predictions
    y_val_pred = model_cv.predict(X_val_cv_scaled)
    
    # Evaluate the model
    mae_cv = mean_absolute_error(y_val_cv, y_val_pred)
    cv_mae_scores.append(mae_cv)

print(f'Cross-Validation MAE Scores for Neural Network: {np.mean(cv_mae_scores)}')
