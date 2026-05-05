import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------
# Configuration
# ------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSV_PATH = os.path.join(BASE_DIR, "data", "processed", "agriguard_real_training_dataset.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "images_real_preview")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_previews(limit=50):
    print(f"Reading dataset from: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    
    samples = df.head(limit)
    print(f"\nGenerating {len(samples)} preview images from real satellite data...")
    print("-" * 40)
    
    for idx, row in samples.iterrows():
        sample_id = row['sample_id']
        health_status = row['health_status']
        ndvi = row['NDVI']
        patch_path = os.path.join(BASE_DIR, row['patch_path'])
        
        if not os.path.exists(patch_path):
            print(f"❌ Missing file: {patch_path}")
            continue
            
        # Load the 4-band numpy array: [B2, B3, B4, B8]
        patch = np.load(patch_path)
        
        # We need RGB: Red (B4 -> index 2), Green (B3 -> index 1), Blue (B2 -> index 0)
        rgb_image = np.stack([patch[2], patch[1], patch[0]], axis=-1)
        
        # Normalize for visualization (enhance brightness)
        # Real satellite data tends to be dark, so we use a percentile clip or simple scaling
        p2, p98 = np.percentile(rgb_image, (2, 98))
        if p98 > p2:
            rgb_normalized = (rgb_image - p2) / (p98 - p2)
        else:
            rgb_normalized = np.clip(rgb_image, 0, 1)
        
        rgb_normalized = np.clip(rgb_normalized, 0, 1)
        
        # Plot and save
        plt.figure(figsize=(2, 2))
        plt.imshow(rgb_normalized)
        plt.title(f"{sample_id}\n{health_status.title()} (NDVI: {ndvi:.2f})", fontsize=8)
        plt.axis('off')
        
        filename = f"{idx+1:03d}_{sample_id}_{health_status}.png"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
    print("-" * 40)
    print(f"✅ Previews successfully saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    # Generate previews for the first 50 real images (adjust number as needed)
    generate_previews(50)