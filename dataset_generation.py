import pandas as pd
import numpy as np

# Define the number of samples
num_samples = 10

# Simulate benign traffic
benign_traffic = {
    "Flow (SRC IP, DST IP, SRC Port, DST Port)": [
        ("192.168.1.100", "203.0.113.10", 1025, 80) for _ in range(num_samples // 2)
    ],
    "Avg Bwd Segment Size": np.random.randint(100, 1200, num_samples // 2),
    "Total Length of Bwd Packets": np.random.randint(1000, 15000, num_samples // 2),
    "Bwd Packet Length Max": np.random.randint(1000, 1500, num_samples // 2),
    "Flow IAT Mean": np.random.uniform(0.1, 1.0, num_samples // 2),
    "Bwd Packet Length Mean": np.random.randint(500, 1000, num_samples // 2),
    "Flow Packets/s": np.random.randint(10, 100, num_samples // 2),
    "Bwd Packets/s": np.random.randint(10, 90, num_samples // 2),
    "Fwd IAT Total": np.random.uniform(0.1, 1.0, num_samples // 2),
    "Flow IAT Max": np.random.uniform(0.2, 1.5, num_samples // 2),
    "Destination Port": np.random.choice([80, 443, 53], num_samples // 2)
}

# Simulate DDoS traffic
ddos_traffic = {
    "Flow (SRC IP, DST IP, SRC Port, DST Port)": [
        ("192.168.1.101", "203.0.113.10", 1026, 80) for _ in range(num_samples // 2)
    ],
    "Avg Bwd Segment Size": np.random.randint(1000, 3000, num_samples // 2),
    "Total Length of Bwd Packets": np.random.randint(15000, 50000, num_samples // 2),
    "Bwd Packet Length Max": np.random.randint(2000, 5000, num_samples // 2),
    "Flow IAT Mean": np.random.uniform(0.001, 0.01, num_samples // 2),
    "Bwd Packet Length Mean": np.random.randint(1500, 4000, num_samples // 2),
    "Flow Packets/s": np.random.randint(200, 1000, num_samples // 2),
    "Bwd Packets/s": np.random.randint(200, 900, num_samples // 2),
    "Fwd IAT Total": np.random.uniform(0.001, 0.01, num_samples // 2),
    "Flow IAT Max": np.random.uniform(0.01, 0.1, num_samples // 2),
    "Destination Port": np.random.choice([80, 443, 53], num_samples // 2)
}

# Combine both traffic types
benign_df = pd.DataFrame(benign_traffic)
ddos_df = pd.DataFrame(ddos_traffic)

# Add the 'Label' column
benign_df['Label'] = 'Benign'
ddos_df['Label'] = 'DDoS'

# Combine both dataframes
final_df = pd.concat([benign_df, ddos_df], ignore_index=True)

# Shuffle the dataset
final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the final dataset to a CSV file
final_df.to_csv('ddos_traffic_10.csv', index=True)

# Display a sample of the dataset
print(final_df.head())
