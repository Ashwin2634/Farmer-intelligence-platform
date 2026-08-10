import pandas as pd
import numpy as np

# Define ideal parameters (Mean, Standard Deviation) for the Gaussian Bell Curve
# Format: 'Crop': {'N': (mean, std), 'P': (mean, std), ... }
# The 'mean' is the perfect target, the 'std' represents how much it naturally fluctuates
crop_profiles = {
    'lettuce': {
        'N': (100, 10), 'P': (40, 5), 'K': (100, 10),
        'temperature': (18.5, 1.5), 'humidity': (67.5, 3.5), 'ph': (6.1, 0.2)
    },
    'spinach': {
        'N': (110, 12), 'P': (50, 6), 'K': (110, 12),
        'temperature': (15.5, 2.0), 'humidity': (67.5, 3.5), 'ph': (6.4, 0.2)
    },
    'kale': {
        'N': (120, 12), 'P': (50, 5), 'K': (125, 12),
        'temperature': (18.0, 2.5), 'humidity': (70.0, 4.0), 'ph': (6.5, 0.3)
    }
}

rows_per_crop = 1000
data = []

# Set seed for reproducible results
np.random.seed(42)

for crop, params in crop_profiles.items():
    for _ in range(rows_per_crop):
        # 1. Generate base values using normal distribution (The Bell Curve)
        n = np.random.normal(params['N'][0], params['N'][1])
        p = np.random.normal(params['P'][0], params['P'][1])
        k = np.random.normal(params['K'][0], params['K'][1])
        t = np.random.normal(params['temperature'][0], params['temperature'][1])
        h = np.random.normal(params['humidity'][0], params['humidity'][1])
        ph = np.random.normal(params['ph'][0], params['ph'][1])
        
        # 2. Inject 5% random sensor noise/outliers (Simulating the "messy" real world)
        if np.random.rand() < 0.05:
            n += np.random.uniform(-20, 20)      # Farmer added too much/too little fertilizer
            t += np.random.uniform(-5, 5)        # Polyhouse cooling fan failed temporarily
            ph += np.random.uniform(-0.8, 0.8)   # pH sensor glitch

        # 3. Ensure no impossible values (e.g., negative humidity) and format
        row = {
            'N': max(0, int(n)),
            'P': max(0, int(p)),
            'K': max(0, int(k)),
            'temperature': round(max(0, t), 2),
            'humidity': round(max(0, min(100, h)), 2),
            'ph': round(max(0, min(14, ph)), 2),
            'label': crop
        }
        data.append(row)

# Create DataFrame and save to CSV
df = pd.DataFrame(data)
df.to_csv('leafy_greens_dataset.csv', index=False)

print(f"Success! Highly realistic dataset created with {len(df)} rows.")