#!/usr/bin/env python
"""Generate vehicle detection features for all raw images."""
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.vehicle_detector import detect_vehicles
raw_dir = ROOT / 'data' / 'images' / 'raw'
image_files = sorted([f for f in raw_dir.glob('*.*') if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}])

records = []
for idx, img_path in enumerate(image_files, 1):
    try:
        _, counts, total = detect_vehicles(img_path, model_name='yolov8n.pt', confidence=0.25, image_size=640)
        records.append({
            'image_path': str(img_path.name),
            'car_count': counts.get('car', 0),
            'bus_count': counts.get('bus', 0),
            'truck_count': counts.get('truck', 0),
            'motorcycle_count': counts.get('motorcycle', 0),
            'bicycle_count': counts.get('bicycle', 0),
            'total_vehicle_count': total
        })
        if idx % 100 == 0:
            print(f'Processed {idx}/{len(image_files)}')
    except Exception as e:
        print(f'Error on {img_path.name}: {e}')

df = pd.DataFrame(records)
output_path = ROOT / 'data' / 'images' / 'raw_detections.csv'
df.to_csv(output_path, index=False)
print(f'\nSaved {len(df)} detections to {output_path.name}')
print(f'\nVehicle count statistics:')
print(df['total_vehicle_count'].describe())
print(f'\nClass proportions if using thresholds (8, 15):')
print(f"  LOW (0-8): {(df['total_vehicle_count'] <= 8).sum()}")
print(f"  MEDIUM (9-15): {((df['total_vehicle_count'] > 8) & (df['total_vehicle_count'] <= 15)).sum()}")
print(f"  HIGH (16+): {(df['total_vehicle_count'] > 15).sum()}")
