import pandas as pd
import numpy as np

# Floriculture crops ke ideal ranges (Mean, Standard Deviation)
# Format: 'Crop': {'N': (mean, std), 'P': (mean, std), 'K': (mean, std), 'temp': (mean, std), 'hum': (mean, std), 'ph': (mean, std)}
flower_profiles = {
    'rose': {
        'N': (150, 12), 'P': (70, 6), 'K': (160, 12),
        'temperature': (22.0, 2.5), 'humidity': (70.0, 4.0), 'ph': (6.2, 0.25)
    },
    'gerbera': {
        'N': (135, 10), 'P': (70, 6), 'K': (175, 12),
        'temperature': (22.5, 2.0), 'humidity': (75.0, 4.0), 'ph': (6.0, 0.25)
    },
    'carnation': {
        'N': (145, 10), 'P': (60, 5), 'K': (155, 10),
        'temperature': (15.0, 2.0), 'humidity': (65.0, 4.0), 'ph': (6.2, 0.20)
    },
    'dutch_rose': {
        'N': (155, 12), 'P': (75, 6), 'K': (170, 12),
        'temperature': (21.0, 2.0), 'humidity': (70.0, 3.5), 'ph': (6.1, 0.20)
    },
    'lilium': {
        'N': (120, 10), 'P': (50, 5), 'K': (140, 10),
        'temperature': (18.0, 2.0), 'humidity': (80.0, 4.0), 'ph': (6.0, 0.30)
    },
    'orchid': {
        'N': (90, 10), 'P': (40, 5), 'K': (100, 10),
        'temperature': (24.0, 2.5), 'humidity': (75.0, 5.0), 'ph': (5.8, 0.30)
    },
    'chrysanthemum': {
        'N': (140, 10), 'P': (65, 6), 'K': (150, 10),
        'temperature': (20.0, 2.0), 'humidity': (70.0, 4.0), 'ph': (6.3, 0.25)
    }
}

rows_per_crop = 1000
data = []

# Random seed set karna taaki results consistent rahein
np.random.seed(42)

for crop, params in flower_profiles.items():
    for _ in range(rows_per_crop):
        # 1. Normal Distribution se base values generate karna
        n = np.random.normal(params['N'][0], params['N'][1])
        p = np.random.normal(params['P'][0], params['P'][1])
        k = np.random.normal(params['K'][0], params['K'][1])
        t = np.random.normal(params['temperature'][0], params['temperature'][1])
        h = np.random.normal(params['humidity'][0], params['humidity'][1])
        ph = np.random.normal(params['ph'][0], params['ph'][1])
        
        # 2. 5% Real-world sensor noise/outliers add karna
        if np.random.rand() < 0.05:
            n += np.random.uniform(-15, 15)
            t += np.random.uniform(-4, 4)
            ph += np.random.uniform(-0.5, 0.5)

        # 3. Proper formatting (Integers for NPK, 2 Decimals for Temp/Humidity/pH)
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

# DataFrame banana aur CSV file save karna
df = pd.DataFrame(data)
df.to_csv('floriculture_dataset.csv', index=False)

print(f"Success! 'floriculture_dataset.csv' created with {len(df)} total rows.")