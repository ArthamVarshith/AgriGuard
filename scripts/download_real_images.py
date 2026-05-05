# scripts/download_real_images.py

import ee
import pandas as pd
import numpy as np
import os
import time

# --------------------------------------------------
# Initialize Google Earth Engine
# --------------------------------------------------
try:
    ee.Initialize(project='ee-arthamvarshith')
    print("✅ Google Earth Engine Initialized")
except Exception:
    print("🔐 Authenticating Google Earth Engine...")
    ee.Authenticate()
    ee.Initialize(project='ee-arthamvarshith')
    print("✅ Google Earth Engine Authenticated and Initialized")

# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSV_PATH = os.path.join(BASE_DIR, "data", "processed", "agriguard_training_dataset.csv")
IMAGE_DIR = os.path.join(BASE_DIR, "data", "images")

os.makedirs(IMAGE_DIR, exist_ok=True)

print("Dataset:", CSV_PATH)
print("Image directory:", IMAGE_DIR)

# --------------------------------------------------
# Download Sentinel-2 patches
# --------------------------------------------------
def download_patches(limit=1000):

    df = pd.read_csv(CSV_PATH)
    samples = df.head(limit)

    print(f"\nStarting download of {len(samples)} real satellite patches...\n")

    for idx, row in samples.iterrows():

        sample_id = row['sample_id']
        lat = row['lat']
        lon = row['lon']
        date_str = row['date']

        out_file = os.path.join(IMAGE_DIR, f"{sample_id}.npy")

        if os.path.exists(out_file):
            print(f"[{idx+1}/{limit}] ⏭ Skipping {sample_id} (already exists)")
            continue

        try:

            # --------------------------------------------------
            # Define region (320m square ≈ 32x32 pixels)
            # --------------------------------------------------
            point = ee.Geometry.Point([lon, lat])
            region = point.buffer(160).bounds()

            # --------------------------------------------------
            # Date window
            # --------------------------------------------------
            start_date = pd.to_datetime(date_str) - pd.Timedelta(days=15)
            end_date = pd.to_datetime(date_str) + pd.Timedelta(days=15)

            # --------------------------------------------------
            # Sentinel-2 Image Collection
            # --------------------------------------------------
            collection = (
                ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                .filterBounds(region)
                .filterDate(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60))
            )

            # --------------------------------------------------
            # Check if images exist
            # --------------------------------------------------
            count = collection.size().getInfo()

            if count == 0:
                print(f"[{idx+1}/{limit}] ❌ No Sentinel-2 images for {sample_id}")
                continue

            # --------------------------------------------------
            # Use median composite
            # --------------------------------------------------
            image = collection.median().select(["B2", "B3", "B4", "B8"])

            # --------------------------------------------------
            # Sample region
            # --------------------------------------------------
            band_arrays = image.sampleRectangle(region=region)

            info = band_arrays.getInfo()

            if not info or "properties" not in info:
                print(f"[{idx+1}/{limit}] ❌ No valid pixels for {sample_id}")
                continue

            # --------------------------------------------------
            # Extract bands
            # --------------------------------------------------
            b2 = np.array(info["properties"]["B2"])
            b3 = np.array(info["properties"]["B3"])
            b4 = np.array(info["properties"]["B4"])
            b8 = np.array(info["properties"]["B8"])

            patch = np.stack([b2, b3, b4, b8], axis=0)

            # --------------------------------------------------
            # Ensure patch size 32x32
            # --------------------------------------------------
            patch = patch[:, :32, :32]

            if patch.shape[1] < 32 or patch.shape[2] < 32:

                pad_h = 32 - patch.shape[1]
                pad_w = 32 - patch.shape[2]

                patch = np.pad(
                    patch,
                    ((0, 0), (0, pad_h), (0, pad_w)),
                    mode="reflect"
                )

            # --------------------------------------------------
            # Save patch
            # --------------------------------------------------
            np.save(out_file, patch)

            print(f"[{idx+1}/{limit}] ✅ Downloaded {sample_id}")

            # Avoid API rate limits
            time.sleep(1)

        except Exception as e:

            print(f"[{idx+1}/{limit}] ❌ Failed on {sample_id}: {str(e)}")


# --------------------------------------------------
# Run
# --------------------------------------------------
if __name__ == "__main__":

    download_patches(limit=1000)