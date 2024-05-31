# src/training.py
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib
from keras.callbacks import History

# Load preprocessed data
final_data = pd.read_csv('../data/final_data.csv')

# Define features and target variable
X = final_data[['Max_Capacity_kWh', 'Current_Load_kWh', 'Max_Generation_Rate_kWh', 'Current_Generation_Rate_kWh', 'Load_Ratio', 'Generation_Ratio']]
y = final_data['Power_Demand_kWh']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# EDA on training data
# Correlation matrix
correlation_matrix = pd.concat([X_train, y_train], axis=1).corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix (Training Data)')
plt.savefig('../results/Correlation_Matrix_Training.png')

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
print("\n--- Linear Regression ---")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred_lr)}")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred_lr)}")
print(f"R-squared: {r2_score(y_test, y_pred_lr)}")
print(f"Cross-Validation MAE Scores: {cross_val_score(lr_model, X, y, cv=5, scoring='neg_mean_absolute_error').mean()}")
joblib.dump(lr_model, '../models/lr_model.pkl')

# Random Forest
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print("\n--- Random Forest ---")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred_rf)}")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred_rf)}")
print(f"R-squared: {r2_score(y_test, y_pred_rf)}")
print(f"Cross-Validation MAE Scores: {cross_val_score(rf_model, X, y, cv=5, scoring='neg_mean_absolute_error').mean()}")
joblib.dump(rf_model, '../models/rf_model.pkl')

# Gradient Boosting
gb_model = GradientBoostingRegressor()
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)
print("\n--- Gradient Boosting ---")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred_gb)}")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred_gb)}")
print(f"R-squared: {r2_score(y_test, y_pred_gb)}")
print(f"Cross-Validation MAE Scores: {cross_val_score(gb_model, X, y, cv=5, scoring='neg_mean_absolute_error').mean()}")
joblib.dump(gb_model, '../models/gb_model.pkl')

# Define a function to create the neural network model
def create_nn_model():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# Wrap the Keras model for use with scikit-learn
nn_model = KerasRegressor(build_fn=create_nn_model, epochs=10, batch_size=32, verbose=1)

# Train the neural network model
history = nn_model.fit(X_train, y_train)

y_pred_nn = nn_model.predict(X_test)
print("\n--- Neural Network ---")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred_nn)}")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred_nn)}")
print(f"R-squared: {r2_score(y_test, y_pred_nn)}")
joblib.dump(nn_model, '../models/nn_model.pkl')

# Cross-validation for the neural network model
from sklearn.model_selection import KFold
kf = KFold(n_splits=5)
cv_scores = cross_val_score(nn_model, X, y, cv=kf, scoring='neg_mean_absolute_error')
print(f"Cross-Validation MAE Scores for Neural Network: {np.mean(cv_scores)}")

# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Loss')
plt.title('Neural Network Training History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('../results/nn_training_history.png')

print("Model training history and models have been saved.")
