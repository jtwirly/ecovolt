import pandas as pd
import numpy as np

# Generate date range
date_rng = pd.date_range(start='2022-01-01', end='2022-12-31 23:00:00', freq='H')

# Initialize DataFrame
data = pd.DataFrame(date_rng, columns=['timestamp'])

# Generate synthetic consumption data
np.random.seed(42)
data['consumption'] = 1000 + 200 * np.sin(2 * np.pi * data.index / 24) + \
                      100 * np.random.randn(len(date_rng))

# Add temperature feature (synthetic)
data['temperature'] = 20 + 10 * np.sin(2 * np.pi * data.index / (24 * 365)) + \
                      5 * np.random.randn(len(date_rng))

# Add day_of_week feature
data['day_of_week'] = data['timestamp'].dt.dayofweek

# Add hour_of_day feature
data['hour_of_day'] = data['timestamp'].dt.hour

# **Add month feature**
data['month'] = data['timestamp'].dt.month

# Handle negative consumption values (if any)
data['consumption'] = data['consumption'].clip(lower=0)

# Display first few rows
print(data.head())

# Save to CSV
data.to_csv('energy_data.csv', index=False)
