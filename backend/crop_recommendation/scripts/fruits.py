import pandas as pd
import numpy as np

# Fruits ki ideal ranges (Mean, Standard Deviation)
# Format: 'Crop': {'N': (mean, std), 'P': (mean, std), 'K': (mean, std), 'temp': (mean, std), 'hum': (mean, std), 'ph': (mean, std)}
fruit_profiles = {
    'watermelon': {
        'N': (100, 10), 'P': (50, 5), 'K': (125, 12),
        'temperature': (30.0, 2.5), 'humidity': (70.0, 5.0), 'ph': (6.4, 0.2)
    },
    'dragon_fruit': {
        'N': (70, 10), 'P': (40, 5), 'K': (110, 10),
        'temperature': (25.0, 3.0), 'humidity': (60.0, 5.0), 'ph': (6.2, 0.3)
    },
    'kiwifruit': {
        'N': (130, 10), 'P': (50, 5), 'K': (140, 12),
        'temperature': (20.0, 2.5), 'humidity': (70.0, 5.0), 'ph': (6.0, 0.2)
    },
    'blackberry': {
        'N': (100, 10), 'P': (40, 5), 'K': (100, 10),
        'temperature': (21.0, 2.0), 'humidity': (68.0, 4.0), 'ph': (6.0, 0.2)
    }
}

rows_per_crop = 1000
data = []

# Random seed set karna taaki result consistent rahe
np.random.seed(42)

for crop, params in fruit_profiles.items():
    for _ in range(rows_per_crop):
        # 1. Normal Distribution se base values nikalna (Target practice math)
        n = np.random.normal(params['N'][0], params['N'][1])
        p = np.random.normal(params['P'][0], params['P'][1])
        k = np.random.normal(params['K'][0], params['K'][1])
        t = np.random.normal(params['temperature'][0], params['temperature'][1])
        h = np.random.normal(params['humidity'][0], params['humidity'][1])
        ph = np.random.normal(params['ph'][0], params['ph'][1])
        
        # 2. 5% Real-world sensor noise add karna (Khet ki galtiyan simulate karne ke liye)
        if np.random.rand() < 0.05:
            n += np.random.uniform(-15, 15)
            t += np.random.uniform(-4, 4)
            ph += np.random.uniform(-0.5, 0.5)

        # 3. Clean formatting: Integers for NPK, 2 decimals for baaki sab
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

# Table banakar CSV mein save karna
df = pd.DataFrame(data)
df.to_csv('fruits_dataset.csv', index=False)

print(f"Success! 'fruits_dataset.csv' ready hai. Total rows: {len(df)}")