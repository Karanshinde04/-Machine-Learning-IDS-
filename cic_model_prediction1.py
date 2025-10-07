import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from joblib import load  # To load the saved model

# Load the saved objects
model = load('random_forest_model1.joblib')  # Random Forest Model
scaler = load('scaler1.joblib')              # StandardScaler
label_encoder = load('label_encoder1.joblib')  # LabelEncoder

# Load the captured network traffic data
captured_data = pd.read_csv('output.csv')  # Replace with your captured traffic CSV

# Ensure the features used for prediction match the ones during training
top_features = ['Avg Bwd Segment Size', 'Total Length of Bwd Packets',
                'Bwd Packet Length Max', 'Flow IAT Mean', 'Bwd Packet Length Mean',
                'Flow Packets/s', 'Bwd Packets/s', 'Fwd IAT Total', 'Flow IAT Max', 'Destination Port']

# Check if all required features are present
if not all(feature in captured_data.columns for feature in top_features):
    print("Error: Captured data does not have the required features.")
    exit()

# Select the features to predict
X_captured = captured_data[top_features].fillna(0)  # Fill missing values with 0

# Use the scaler that was used during training to scale the captured data
X_captured_scaled = scaler.transform(X_captured)

# Predict the labels for the captured data
predicted_labels = model.predict(X_captured_scaled)

# Map numeric labels back to original labels (use the LabelEncoder's inverse transform method)
predicted_labels_mapped = label_encoder.inverse_transform(predicted_labels)

# Add predictions to the original data
captured_data['Predicted Label'] = predicted_labels_mapped

# Save the results to a file
captured_data.to_csv('captured_traffic_with_predictions.csv', index=True)

# Display a few predictions
print(captured_data[['Flow (SRC IP, DST IP, SRC Port, DST Port)', 'Predicted Label']])
