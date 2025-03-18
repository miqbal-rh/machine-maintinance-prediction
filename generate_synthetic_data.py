import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Create a directory to store the data
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

# Parameters for synthetic data generation
n_samples = 10000  # Number of data points
n_sensors = 14  # Number of sensors
n_units = 100  # Number of different units (engines)
max_time = 100  # Max time steps for each unit

# Function to simulate sensor data
def generate_sensor_data(n_samples, n_sensors):
    return np.random.uniform(0, 100, (n_samples, n_sensors))

# Function to simulate operational settings
def generate_operational_settings(n_samples):
    operational_setting_1 = np.random.uniform(0, 1, n_samples)
    operational_setting_2 = np.random.uniform(0, 1, n_samples)
    operational_setting_3 = np.random.uniform(0, 1, n_samples)
    return operational_setting_1, operational_setting_2, operational_setting_3

# Function to simulate Remaining Useful Life (RUL)
def generate_rul(n_samples):
    time = np.random.randint(1, max_time, n_samples)
    rul = max_time - time + np.random.uniform(0, 10, n_samples)
    return time, rul

# Generate synthetic data
units = np.random.randint(1, n_units + 1, n_samples)
time, rul = generate_rul(n_samples)
sensor_data = generate_sensor_data(n_samples, n_sensors)
operational_setting_1, operational_setting_2, operational_setting_3 = generate_operational_settings(n_samples)

# Create DataFrame
columns = ['unit', 'time', 'operational_setting_1', 'operational_setting_2', 'operational_setting_3'] + [f'sensor_{i}' for i in range(1, n_sensors + 1)] + ['RUL']
data = np.column_stack([units, time, operational_setting_1, operational_setting_2, operational_setting_3, sensor_data, rul])
df_synthetic = pd.DataFrame(data, columns=columns)

# Split into training, validation, and test sets
train_df, test_df = train_test_split(df_synthetic, test_size=0.2, shuffle=False)
train_df, val_df = train_test_split(train_df, test_size=0.2, shuffle=False)

# Save datasets in the "data" directory
train_df.to_csv(os.path.join(data_dir, 'train2.csv'), index=False)
val_df.to_csv(os.path.join(data_dir, 'validate2.csv'), index=False)
test_df.to_csv(os.path.join(data_dir, 'test2.csv'), index=False)

print(f"Synthetic data saved in '{data_dir}' directory:")
print(f"- {os.path.join(data_dir, 'train2.csv')}")
print(f"- {os.path.join(data_dir, 'validate2.csv')}")
