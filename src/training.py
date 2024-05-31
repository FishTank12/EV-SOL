# src/training.py
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.models import Sequential
from keras.layers import Dense

# Load preprocessed data
final_data = pd.read_csv('../data/final_data.csv')

# Define features and target variable
X = final_data[['Max_Capacity_kWh', 'Current_Load_kWh', 'Max_Generation_Rate_kWh', 'Current_Generation_Rate_kWh', 'Load_Ratio', 'Generation_Ratio']]
y = final_data['Power_Demand_kWh']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
print("\n--- Linear Regression ---")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred_lr)}")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred_lr)}")
print(f"R-squared: {r2_score(y_test, y_pred_lr)}")
print(f"Cross-Validation MAE Scores: {cross_val_score(lr_model, X, y, cv=5, scoring='neg_mean_absolute_error').mean()}")

# Random Forest
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print("\n--- Random Forest ---")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred_rf)}")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred_rf)}")
print(f"R-squared: {r2_score(y_test, y_pred_rf)}")
print(f"Cross-Validation MAE Scores: {cross_val_score(rf_model, X, y, cv=5, scoring='neg_mean_absolute_error').mean()}")

# Gradient Boosting
gb_model = GradientBoostingRegressor()
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)
print("\n--- Gradient Boosting ---")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred_gb)}")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred_gb)}")
print(f"R-squared: {r2_score(y_test, y_pred_gb)}")
print(f"Cross-Validation MAE Scores: {cross_val_score(gb_model, X, y, cv=5, scoring='neg_mean_absolute_error').mean()}")

# Neural Network
nn_model = Sequential()
nn_model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
nn_model.add(Dense(64, activation='relu'))
nn_model.add(Dense(1))

nn_model.compile(optimizer='adam', loss='mse')
nn_model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

y_pred_nn = nn_model.predict(X_test).flatten()
print("\n--- Neural Network ---")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred_nn)}")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred_nn)}")
print(f"R-squared: {r2_score(y_test, y_pred_nn)}")
print(f"Cross-Validation MAE Scores for Neural Network: {cross_val_score(nn_model, X, y, cv=5, scoring='neg_mean_absolute_error').mean()}")
