import argparse
import os
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import ee
import numpy as np
import pandas as pd
import requests


@dataclass
class RegionConfig:
    name: str
    bbox: tuple[float, float, float, float]  # min_lon, min_lat, max_lon, max_lat
    crops: tuple[str, ...]


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = DATA_DIR / "images_real"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_CSV = PROCESSED_DIR / "agriguard_real_training_dataset.csv"

NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
DEFAULT_EE_PROJECT = os.getenv("EE_PROJECT", "ee-arthamvarshith")

REGIONS = [
    RegionConfig(
        name="amaravati",
        bbox=(80.20, 16.30, 80.80, 16.80),
        crops=("cotton", "chilli", "paddy", "maize"),
    ),
]


def initialize_earth_engine(project: str | None = None) -> None:
    active_project = project or DEFAULT_EE_PROJECT

    try:
        if active_project:
            ee.Initialize(project=active_project)
        else:
            ee.Initialize()
    except Exception:
        print("Earth Engine not initialized. Starting authentication flow...")
        ee.Authenticate()
        if active_project:
            ee.Initialize(project=active_project)
        else:
            ee.Initialize()
    print(f"Earth Engine initialized. Project: {active_project}")


def seasonal_name(month: int) -> str:
    if month in (6, 7, 8, 9):
        return "Monsoon"
    if month in (10, 11):
        return "Post-Monsoon"
    if month in (12, 1, 2):
        return "Winter"
    return "Summer"


def random_date(start: datetime, end: datetime) -> datetime:
    delta_days = (end - start).days
    offset = random.randint(0, max(0, delta_days))
    return start + timedelta(days=offset)


def build_point(region: RegionConfig) -> tuple[float, float]:
    min_lon, min_lat, max_lon, max_lat = region.bbox
    lon = random.uniform(min_lon, max_lon)
    lat = random.uniform(min_lat, max_lat)
    return lat, lon


def get_weather_features(lat: float, lon: float, date_obj: datetime) -> dict | None:
    day = date_obj.strftime("%Y%m%d")
    params = {
        "parameters": "T2M_MAX,T2M_MIN,RH2M,PRECTOTCORR",
        "community": "AG",
        "longitude": f"{lon:.5f}",
        "latitude": f"{lat:.5f}",
        "start": day,
        "end": day,
        "format": "JSON",
    }

    response = requests.get(NASA_POWER_URL, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()

    try:
        data = payload["properties"]["parameter"]
        tmax = float(data["T2M_MAX"][day])
        tmin = float(data["T2M_MIN"][day])
        rh = float(data["RH2M"][day])
        rain = float(data["PRECTOTCORR"][day])
    except Exception:
        return None

    if any(val < -900 for val in (tmax, tmin, rh, rain)):
        return None

    return {
        "temp_max": tmax,
        "temp_min": tmin,
        "humidity": max(0.0, min(100.0, rh)),
        "rainfall": max(0.0, rain),
    }


def compute_indices_from_patch(patch_4_band: np.ndarray) -> dict:
    # patch order: B2, B3, B4, B8
    b2 = patch_4_band[0].astype(np.float32)
    b4 = patch_4_band[2].astype(np.float32)
    b8 = patch_4_band[3].astype(np.float32)

    eps = 1e-6
    ndvi = ((b8 - b4) / (b8 + b4 + eps)).mean()
    evi = (2.5 * ((b8 - b4) / (b8 + 6.0 * b4 - 7.5 * b2 + 1.0 + eps))).mean()
    savi = (1.5 * (b8 - b4) / (b8 + b4 + 0.5 + eps)).mean()

    # REP is approximated from red-edge trend proxy due band-limited patch in this dataset.
    rep = 700.0 + (40.0 * float(np.clip((ndvi + 1.0) / 2.0, 0.0, 1.0)))

    return {
        "NDVI": float(np.clip(ndvi, -1.0, 1.0)),
        "EVI": float(np.clip(evi, -1.0, 1.0)),
        "SAVI": float(np.clip(savi, -1.0, 1.0)),
        "REP": float(np.clip(rep, 680.0, 750.0)),
    }


def patch_texture_score(patch_4_band: np.ndarray) -> float:
    # Mean of per-band spatial std-dev; low values indicate flat/low-information tiles.
    return float(np.mean(np.std(patch_4_band.astype(np.float32), axis=(1, 2))))


def derive_proxy_label(indices: dict, weather: dict) -> tuple[str, float]:
    ndvi = indices["NDVI"]
    evi = indices["EVI"]
    humidity = weather["humidity"]
    rainfall = weather["rainfall"]
    temp_max = weather["temp_max"]
    temp_min = weather["temp_min"]

    score = 0.0
    if ndvi < 0.30:
        score += 0.35
    elif ndvi < 0.40:
        score += 0.20

    if evi < 0.20:
        score += 0.20
    elif evi < 0.30:
        score += 0.10

    if humidity > 85:
        score += 0.15
    elif humidity > 75:
        score += 0.08

    if rainfall > 12:
        score += 0.20
    elif rainfall > 5:
        score += 0.10

    if temp_max > 36 or temp_min < 12:
        score += 0.10

    score = float(np.clip(score, 0.0, 1.0))

    if score >= 0.60:
        return "diseased", score
    if score >= 0.35:
        return "stressed", score
    return "healthy", score


def extract_patch(
    lat: float,
    lon: float,
    date_obj: datetime,
    cloud_threshold: float = 30.0,
    texture_threshold: float = 0.006,
) -> np.ndarray | None:
    point = ee.Geometry.Point([lon, lat])
    region = point.buffer(160).bounds()  # around 32x32 at 10m

    start_date = (date_obj - timedelta(days=12)).strftime("%Y-%m-%d")
    end_date = (date_obj + timedelta(days=12)).strftime("%Y-%m-%d")

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_threshold))
    )

    if collection.size().getInfo() == 0:
        return None

    # Important: collection reductions can lose native projection and default to coarse grid.
    # Force projection from a real Sentinel-2 band so sampleRectangle returns true spatial tiles.
    try:
        reference_image = ee.Image(collection.first())
        reference_projection = reference_image.select("B2").projection()
    except Exception:
        return None

    image = (
        collection.median()
        .select(["B2", "B3", "B4", "B8"])
        .divide(10000)
        .setDefaultProjection(reference_projection)
        .resample("bilinear")
        .unmask(0)
    )

    try:
        sample = image.sampleRectangle(region=region, defaultValue=0).getInfo()
    except Exception:
        return None
    props = sample.get("properties", {})

    if not props:
        return None

    try:
        b2 = np.array(props["B2"], dtype=np.float32)
        b3 = np.array(props["B3"], dtype=np.float32)
        b4 = np.array(props["B4"], dtype=np.float32)
        b8 = np.array(props["B8"], dtype=np.float32)
    except KeyError:
        return None

    patch = np.stack([b2, b3, b4, b8], axis=0)
    patch = patch[:, :32, :32]

    if patch.shape[1] < 32 or patch.shape[2] < 32:
        pad_h = max(0, 32 - patch.shape[1])
        pad_w = max(0, 32 - patch.shape[2])
        patch = np.pad(patch, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")

    patch = np.nan_to_num(patch, nan=0.0, posinf=1.0, neginf=0.0)
    patch = np.clip(patch, 0.0, 1.0)
    if patch_texture_score(patch) < texture_threshold:
        return None
    return patch.astype(np.float32)


def build_real_dataset(
    samples_per_region: int,
    start_date: str,
    end_date: str,
    seed: int,
    cloud_threshold: float,
    texture_threshold: float,
    max_attempts_multiplier: int,
) -> pd.DataFrame:
    random.seed(seed)
    np.random.seed(seed)

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    dt_start = datetime.strptime(start_date, "%Y-%m-%d")
    dt_end = datetime.strptime(end_date, "%Y-%m-%d")

    rows = []
    sample_counter = 1

    for region in REGIONS:
        print(f"\nCollecting for region: {region.name}")

        attempts = 0
        collected = 0
        max_attempts = samples_per_region * max_attempts_multiplier

        while collected < samples_per_region and attempts < max_attempts:
            attempts += 1
            lat, lon = build_point(region)
            obs_date = random_date(dt_start, dt_end)
            crop = random.choice(region.crops)

            weather = get_weather_features(lat, lon, obs_date)
            if weather is None:
                continue

            patch = extract_patch(
                lat,
                lon,
                obs_date,
                cloud_threshold=cloud_threshold,
                texture_threshold=texture_threshold,
            )
            if patch is None:
                continue

            indices = compute_indices_from_patch(patch)
            label, risk_score = derive_proxy_label(indices, weather)

            sample_id = f"REAL_{sample_counter:05d}"
            patch_path = IMAGES_DIR / f"{sample_id}.npy"
            np.save(patch_path, patch)

            row = {
                "sample_id": sample_id,
                "date": obs_date.strftime("%Y-%m-%d"),
                "month": obs_date.month,
                "season": seasonal_name(obs_date.month),
                "crop": crop,
                "region": region.name,
                "lat": round(lat, 6),
                "lon": round(lon, 6),
                "temp_max": round(weather["temp_max"], 3),
                "temp_min": round(weather["temp_min"], 3),
                "humidity": round(weather["humidity"], 3),
                "rainfall": round(weather["rainfall"], 3),
                "health_status": label,
                "disease_type": "proxy_risk_label",
                "severity": round(risk_score, 4),
                "disease_probability": round(risk_score, 4),
                "NDVI": round(indices["NDVI"], 4),
                "EVI": round(indices["EVI"], 4),
                "SAVI": round(indices["SAVI"], 4),
                "REP": round(indices["REP"], 2),
                "patch_path": str(patch_path.relative_to(BASE_DIR)).replace("\\", "/"),
                "label_source": "auto_proxy_satellite_weather",
            }
            rows.append(row)
            sample_counter += 1
            collected += 1

            if collected % 10 == 0:
                print(f"  collected {collected}/{samples_per_region}")

        print(f"Completed {region.name}: {collected} samples")

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(
            "No samples were collected. Check Earth Engine auth, internet, date range, or cloud threshold."
        )

    df.to_csv(OUTPUT_CSV, index=False)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a real satellite-weather training dataset.")
    parser.add_argument("--samples-per-region", type=int, default=1000)
    parser.add_argument("--start-date", type=str, default="2024-01-01")
    parser.add_argument("--end-date", type=str, default="2025-12-31")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ee-project", type=str, default=DEFAULT_EE_PROJECT)
    parser.add_argument("--cloud-threshold", type=float, default=25.0)
    parser.add_argument("--texture-threshold", type=float, default=0.008)
    parser.add_argument("--max-attempts-multiplier", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    initialize_earth_engine(project=args.ee_project)

    print("Starting real dataset generation")
    print(f"Output CSV: {OUTPUT_CSV}")
    print(f"Output patches: {IMAGES_DIR}")

    df = build_real_dataset(
        samples_per_region=args.samples_per_region,
        start_date=args.start_date,
        end_date=args.end_date,
        seed=args.seed,
        cloud_threshold=args.cloud_threshold,
        texture_threshold=args.texture_threshold,
        max_attempts_multiplier=args.max_attempts_multiplier,
    )

    print("\nDataset build complete.")
    print(f"Total samples: {len(df)}")
    print(df["health_status"].value_counts())
    print(f"Saved CSV: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
