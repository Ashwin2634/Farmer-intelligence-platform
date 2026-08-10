import pandas as pd
import numpy as np

# Herbs ki ideal ranges (Mean, Standard Deviation)
# Format: 'Crop': {'N': (mean, std), 'P': (mean, std), 'K': (mean, std), 'temp': (mean, std), 'hum': (mean, std), 'ph': (mean, std)}
herb_profiles = {
    'basil': {
        'N': (110, 10), 'P': (45, 5), 'K': (110, 10),
        'temperature': (25.0, 2.0), 'humidity': (65.0, 5.0), 'ph': (6.2, 0.2)
    },
    'parsley': {
        'N': (90, 10), 'P': (40, 5), 'K': (90, 10),
        'temperature': (18.0, 2.0), 'humidity': (70.0, 4.0), 'ph': (6.1, 0.2)
    },
    'celery': {
        'N': (120, 12), 'P': (50, 5), 'K': (130, 12),
        'temperature': (16.0, 2.0), 'humidity': (80.0, 4.0), 'ph': (6.4, 0.2)
    },
    'dill': {
        'N': (80, 10), 'P': (35, 5), 'K': (80, 10),
        'temperature': (20.0, 2.0), 'humidity': (60.0, 5.0), 'ph': (6.0, 0.2)
    },
    'chives': {
        'N': (95, 10), 'P': (45, 5), 'K': (95, 10),
        'temperature': (18.0, 2.0), 'humidity': (65.0, 4.0), 'ph': (6.3, 0.2)
    },
    'oregano': {
        'N': (60, 10), 'P': (30, 5), 'K': (70, 10),
        'temperature': (22.0, 2.0), 'humidity': (50.0, 5.0), 'ph': (6.8, 0.3)
    },
    'thyme': {
        'N': (50, 8), 'P': (30, 5), 'K': (60, 8),
        'temperature': (22.0, 2.0), 'humidity': (50.0, 5.0), 'ph': (7.0, 0.3)
    },
    'rosemary': {
        'N': (60, 10), 'P': (30, 5), 'K': (70, 10),
        'temperature': (20.0, 2.5), 'humidity': (55.0, 5.0), 'ph': (6.5, 0.3)
    }
}

rows_per_crop = 1000
data = []

np.random.seed(42)

for crop, params in herb_profiles.items():
    for _ in range(rows_per_crop):
        # 1. Normal Distribution se values nikalna
        n = np.random.normal(params['N'][0], params['N'][1])
        p = np.random.normal(params['P'][0], params['P'][1])
        k = np.random.normal(params['K'][0], params['K'][1])
        t = np.random.normal(params['temperature'][0], params['temperature'][1])
        h = np.random.normal(params['humidity'][0], params['humidity'][1])
        ph = np.random.normal(params['ph'][0], params['ph'][1])
        
        # 2. 5% Real-world sensor noise
        if np.random.rand() < 0.05:
            n += np.random.uniform(-15, 15)
            t += np.random.uniform(-4, 4)
            ph += np.random.uniform(-0.5, 0.5)

        # 3. Data clean aur format karna
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

# DataFrame aur CSV save
df = pd.DataFrame(data)
df.to_csv('herbs_dataset.csv', index=False)

print(f"Success! 'herbs_dataset.csv' ready hai. Total rows: {len(df)}")