import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------
# Dynamic Path Configuration
# ------------------------------
# Assumes this script is inside 'AgriGuard/scripts/'
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSV_PATH = os.path.join(BASE_DIR, "data", "processed", "agriguard_training_dataset.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "sample_images")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_spatial_representation(ndvi_val):
    """Recreates the synthetic spatial patch from the dataloader."""
    patch_size = 32
    n_bands = 4
    # Base reflectance for Blue, Green, Red, NIR
    base_reflectance = torch.tensor([0.06, 0.10, 0.04, 0.50])
    
    health_factor = max(0.3, (ndvi_val + 1) / 2)
    patch = torch.zeros(n_bands, patch_size, patch_size)
    
    for band_idx, base_val in enumerate(base_reflectance):
        spatial_noise = torch.randn(patch_size, patch_size) * 0.02
        health_variation = base_val * health_factor
        patch[band_idx] = health_variation + spatial_noise
        patch[band_idx] = torch.clamp(patch[band_idx], 0.01, 0.95)
        
    return patch

def generate_samples(num_samples=10):
    print(f"Reading dataset from: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    
    # Grab the first 'num_samples' rows
    samples = df.head(num_samples)
    
    print(f"\nGenerating {num_samples} synthetic images...")
    print("-" * 40)
    
    for idx, row in samples.iterrows():
        sample_id = row['sample_id']
        health_status = row['health_status']
        ndvi = row['NDVI']
        
        # 1. Generate the 4-band tensor
        patch = create_spatial_representation(ndvi)
        
        # 2. Extract Red, Green, Blue bands (Indices 2, 1, 0 respectively)
        # We use stack to create an image matrix of shape (32, 32, 3)
        rgb_image = torch.stack([patch[2], patch[1], patch[0]], dim=-1).numpy()
        
        # 3. Normalize the pixel values to [0, 255] for standard PNG viewing
        rgb_normalized = (rgb_image / np.max(rgb_image) * 255).astype(np.uint8)
        
        # 4. Plot and save the image
        plt.figure(figsize=(2, 2))
        plt.imshow(rgb_normalized)
        plt.title(f"{health_status.title()}\nNDVI: {ndvi:.2f}", fontsize=8)
        plt.axis('off')
        
        # Save to disk
        filename = f"{idx+1:02d}_{sample_id}_{health_status}.png"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close() # Free up memory
        
        print(f"✅ Saved: {filename}")
        
    print("-" * 40)
    print(f"All images successfully saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_samples(10)