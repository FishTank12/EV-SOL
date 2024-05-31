import sys
import json
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('../models/rf_model.pkl')

# Define a function to preprocess the input data if necessary
def preprocess(data):
    data['Load_Ratio'] = data['Current_Load_kWh'] / data['Max_Capacity_kWh']
    data['Generation_Ratio'] = data['Current_Generation_Rate_kWh'] / data['Max_Generation_Rate_kWh']
    return data

def predict(data):
    features = pd.DataFrame([data])
    features = preprocess(features)  # Preprocess the features
    prediction = model.predict(features)
    return prediction[0]

if __name__ == '__main__':
    # Read the input data from the command line
    input_data = json.loads(sys.argv[1])
    result = predict(input_data)
    print(result)
