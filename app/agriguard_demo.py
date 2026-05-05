import os
from pathlib import Path

import ee
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


SPECTRAL_FEATURES = ["NDVI", "EVI", "SAVI", "REP"]
WEATHER_FEATURES = ["temp_max", "temp_min", "humidity", "rainfall"]
ALL_FEATURES = SPECTRAL_FEATURES + WEATHER_FEATURES

BASE_DIR = Path(__file__).resolve().parents[1]
REAL_CSV_PATH = BASE_DIR / "data" / "processed" / "agriguard_real_training_dataset.csv"
PATCH_REAL_CSV_PATH = BASE_DIR / "data" / "processed" / "agriguard_real_from_patches_dataset.csv"
SYNTHETIC_CSV_PATH = BASE_DIR / "data" / "processed" / "agriguard_training_dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "agriguard_model.pth"


def resolve_dataset_path() -> Path:
    for path in [REAL_CSV_PATH, PATCH_REAL_CSV_PATH, SYNTHETIC_CSV_PATH]:
        if path.exists():
            return path
    return SYNTHETIC_CSV_PATH


CSV_PATH = resolve_dataset_path()
NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
DEFAULT_EE_PROJECT = os.getenv("EE_PROJECT", "ee-arthamvarshith")
REAL_IMAGES_DIR = BASE_DIR / "data" / "images_real"
LEGACY_IMAGES_DIR = BASE_DIR / "data" / "images"


def apply_clean_theme() -> None:
    st.markdown(
        """
        <style>
        .stats-badge {
            background: rgba(255, 255, 255, 0.15);
            border: 1px solid rgba(255, 255, 255, 0.25);
            border-radius: 8px;
            padding: 0.35rem 0.8rem;
            font-size: 0.82rem;
            font-weight: 600;
            color: #ffffff !important;
            display: inline-block;
        }
        .hero-wrap {
            background: linear-gradient(135deg, #103426 0%, #1f5d45 70%, #277b58 100%);
            border-radius: 18px;
            padding: 1.2rem 1.35rem;
            margin-bottom: 1rem;
        }
        .hero-title {
            margin: 0;
            color: #ffffff !important;
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.7rem;
            font-weight: 700;
        }
        .hero-subtitle {
            margin: 0.35rem 0 0 0;
            color: #d7f7e8 !important;
            font-size: 0.98rem;
        }
        .section-card {
            border: 1px solid rgba(128, 128, 128, 0.2);
            border-radius: 16px;
            padding: 1rem;
            margin-bottom: 1rem;
            background-color: rgba(128, 128, 128, 0.05);
        }
        .tiny-muted {
            font-size: 0.86rem;
            opacity: 0.7;
        }
        h1, h2, h3 {
            font-family: 'Space Grotesk', sans-serif !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


class MultiModalCNN(nn.Module):
    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()

        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.spectral_encoder = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )

        self.weather_encoder = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
        )

        self.fusion = nn.Sequential(
            nn.Linear(128 + 64 + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(
        self, spatial: torch.Tensor, spectral: torch.Tensor, weather: torch.Tensor
    ) -> torch.Tensor:
        spatial_features = self.spatial_encoder(spatial)
        spectral_features = self.spectral_encoder(spectral)
        weather_features = self.weather_encoder(weather)
        combined = torch.cat([spatial_features, spectral_features, weather_features], dim=1)
        return self.fusion(combined)


@st.cache_resource
def load_model_and_scaler():
    if not MODEL_PATH.exists():
        st.error(f"Model file not found at {MODEL_PATH}. Please train the model first.")
        st.stop()

    if not CSV_PATH.exists():
        st.error(f"Dataset file not found at {CSV_PATH}.")
        st.stop()

    df = pd.read_csv(CSV_PATH)
    scaler = StandardScaler()
    scaler.fit(df[ALL_FEATURES])

    try:
        checkpoint = torch.load(
            MODEL_PATH, map_location=torch.device("cpu"), weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
    label_encoder = checkpoint["label_encoder"]

    model = MultiModalCNN(num_classes=len(label_encoder.classes_))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, scaler, label_encoder


def create_spatial_representation(spectral_values: torch.Tensor) -> torch.Tensor:
    patch_size = 32
    base_reflectance = torch.tensor([0.06, 0.10, 0.04, 0.50], dtype=torch.float32)

    ndvi_val = float(spectral_values[0].item())
    health_factor = max(0.3, (ndvi_val + 1.0) / 2.0)

    x = torch.linspace(-1.0, 1.0, patch_size)
    y = torch.linspace(-1.0, 1.0, patch_size)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    pattern = (torch.sin(grid_x * np.pi) + torch.cos(grid_y * np.pi * 1.5)) * 0.01

    patch = torch.zeros(4, patch_size, patch_size, dtype=torch.float32)
    for band_idx, base_val in enumerate(base_reflectance):
        band = (base_val * health_factor) + pattern + (band_idx * 0.005)
        patch[band_idx] = torch.clamp(band, 0.01, 0.95)

    return patch


@st.cache_resource
def initialize_earth_engine() -> tuple[bool, str]:
    try:
        ee.Initialize(project=DEFAULT_EE_PROJECT)
        return True, DEFAULT_EE_PROJECT
    except Exception:
        try:
            ee.Initialize()
            return True, "default"
        except Exception as exc:
            return False, str(exc)


def fetch_weather_features(lat: float, lon: float, date_text: str) -> dict | None:
    date_key = pd.Timestamp(date_text).strftime("%Y%m%d")
    params = {
        "parameters": "T2M_MAX,T2M_MIN,RH2M,PRECTOTCORR",
        "community": "AG",
        "longitude": f"{lon:.6f}",
        "latitude": f"{lat:.6f}",
        "start": date_key,
        "end": date_key,
        "format": "JSON",
    }

    try:
        response = requests.get(NASA_POWER_URL, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
        weather = payload["properties"]["parameter"]

        temp_max = float(weather["T2M_MAX"][date_key])
        temp_min = float(weather["T2M_MIN"][date_key])
        humidity = float(weather["RH2M"][date_key])
        rainfall = float(weather["PRECTOTCORR"][date_key])
    except Exception:
        return None

    if any(value < -900 for value in [temp_max, temp_min, humidity, rainfall]):
        return None

    return {
        "temp_max": temp_max,
        "temp_min": temp_min,
        "humidity": max(0.0, min(100.0, humidity)),
        "rainfall": max(0.0, rainfall),
    }


def _pad_and_crop_patch(patch: np.ndarray, size: int = 32) -> np.ndarray:
    patch = patch[:, :size, :size]
    if patch.shape[1] < size or patch.shape[2] < size:
        pad_h = max(0, size - patch.shape[1])
        pad_w = max(0, size - patch.shape[2])
        patch = np.pad(patch, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
    return patch[:, :size, :size]


def fetch_satellite_observation(lat: float, lon: float, date_text: str, buffer_m: int = 160) -> dict | None:
    ready, _ = initialize_earth_engine()
    if not ready:
        return None

    point = ee.Geometry.Point([lon, lat])
    region = point.buffer(buffer_m).bounds()
    center_date = pd.Timestamp(date_text)
    start_date = (center_date - pd.Timedelta(days=12)).strftime("%Y-%m-%d")
    end_date = (center_date + pd.Timedelta(days=12)).strftime("%Y-%m-%d")

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 35))
    )

    try:
        image_count = int(collection.size().getInfo())
    except Exception:
        return None

    if image_count == 0:
        return None

    cloud_avg = None
    try:
        cloud_avg = float(collection.aggregate_mean("CLOUDY_PIXEL_PERCENTAGE").getInfo())
    except Exception:
        cloud_avg = None

    try:
        reference_image = ee.Image(collection.first())
        reference_projection = reference_image.select("B2").projection()
    except Exception:
        return None

    image = (
        collection.median()
        .select(["B2", "B3", "B4", "B5", "B6", "B7", "B8"])
        .divide(10000)
        .setDefaultProjection(reference_projection)
        .resample("bilinear")
        .unmask(0)
    )

    try:
        sample = image.sampleRectangle(region=region, defaultValue=0).getInfo()
        props = sample["properties"]
    except Exception:
        return None

    try:
        b2 = np.array(props["B2"], dtype=np.float32)
        b3 = np.array(props["B3"], dtype=np.float32)
        b4 = np.array(props["B4"], dtype=np.float32)
        b5 = np.array(props["B5"], dtype=np.float32)
        b6 = np.array(props["B6"], dtype=np.float32)
        b7 = np.array(props["B7"], dtype=np.float32)
        b8 = np.array(props["B8"], dtype=np.float32)
    except Exception:
        return None

    patch = _pad_and_crop_patch(np.stack([b2, b3, b4, b8], axis=0))
    patch = np.nan_to_num(patch, nan=0.0, posinf=1.0, neginf=0.0)
    patch = np.clip(patch, 0.0, 1.0)
    if patch_texture_score(patch) < 0.004:
        return None

    eps = 1e-6
    ndvi = float(np.clip(((b8 - b4) / (b8 + b4 + eps)).mean(), -1.0, 1.0))
    evi = float(
        np.clip((2.5 * ((b8 - b4) / (b8 + 6.0 * b4 - 7.5 * b2 + 1.0 + eps))).mean(), -1.0, 1.0)
    )
    savi = float(np.clip((1.5 * (b8 - b4) / (b8 + b4 + 0.5 + eps)).mean(), -1.0, 1.0))
    rep_matrix = 700.0 + 40.0 * ((((b4 + b7) / 2.0) - b5) / (b6 - b5 + eps))
    rep = float(np.clip(np.nanmean(rep_matrix), 680.0, 750.0))

    return {
        "patch": patch.astype(np.float32),
        "features": {"NDVI": ndvi, "EVI": evi, "SAVI": savi, "REP": rep},
        "meta": {
            "image_count": image_count,
            "cloud_avg": cloud_avg,
            "window_start": start_date,
            "window_end": end_date,
        },
    }


def patch_to_rgb_preview(patch: np.ndarray) -> np.ndarray:
    rgb = np.stack([patch[2], patch[1], patch[0]], axis=-1).astype(np.float32)
    p2, p98 = np.percentile(rgb, [2, 98])
    if p98 > p2:
        rgb = (rgb - p2) / (p98 - p2)
    else:
        rgb = np.clip(rgb, 0.0, 1.0)
    return np.clip(rgb, 0.0, 1.0)


@st.cache_data
def load_real_reference_dataset() -> pd.DataFrame | None:
    for candidate in [REAL_CSV_PATH, PATCH_REAL_CSV_PATH]:
        if not candidate.exists():
            continue
        try:
            df = pd.read_csv(candidate)
        except Exception:
            continue
        required = set(ALL_FEATURES + ["sample_id"])
        if required.issubset(df.columns):
            return df
    return None


def patch_texture_score(patch: np.ndarray) -> float:
    if patch is None or patch.ndim != 3:
        return 0.0
    band_std = np.std(patch, axis=(1, 2))
    return float(np.mean(band_std))


def load_patch_from_reference_row(row: pd.Series) -> np.ndarray | None:
    patch_path = None

    if "patch_path" in row and isinstance(row["patch_path"], str) and row["patch_path"].strip():
        patch_path = Path(row["patch_path"])
        if not patch_path.is_absolute():
            patch_path = BASE_DIR / patch_path
    elif "sample_id" in row and isinstance(row["sample_id"], str):
        sample_id = row["sample_id"]
        candidate_paths = [
            REAL_IMAGES_DIR / f"{sample_id}.npy",
            LEGACY_IMAGES_DIR / f"{sample_id}.npy",
        ]
        for candidate in candidate_paths:
            if candidate.exists():
                patch_path = candidate
                break

    if patch_path is None or not patch_path.exists():
        return None

    try:
        patch = np.load(patch_path)
    except Exception:
        return None

    if patch.ndim != 3:
        return None
    if patch.shape[0] != 4 and patch.shape[-1] == 4:
        patch = np.transpose(patch, (2, 0, 1))
    if patch.shape[0] < 4:
        return None

    patch = _pad_and_crop_patch(patch[:4], size=32)
    patch = np.nan_to_num(patch, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
    if np.nanmax(patch) > 1.5:
        patch = patch / 10000.0
    return np.clip(patch, 0.0, 1.0)


def find_closest_real_patch_for_manual(
    ndvi: float,
    evi: float,
    savi: float,
    rep: float,
    temp_max: float,
    temp_min: float,
    humidity: float,
    rainfall: float,
) -> dict | None:
    reference_df = load_real_reference_dataset()
    if reference_df is None or reference_df.empty:
        return None

    target = np.array([ndvi, evi, savi, rep, temp_max, temp_min, humidity, rainfall], dtype=np.float64)
    matrix = reference_df[ALL_FEATURES].astype(np.float64)

    means = matrix.mean(axis=0).values
    stds = matrix.std(axis=0).replace(0, 1.0).values
    norm_target = (target - means) / stds
    norm_matrix = (matrix.values - means) / stds

    distances = np.sqrt(np.sum((norm_matrix - norm_target) ** 2, axis=1))
    sorted_idx = np.argsort(distances)

    best_any = None
    for idx in sorted_idx[: min(30, len(sorted_idx))]:
        row = reference_df.iloc[int(idx)]
        patch = load_patch_from_reference_row(row)
        if patch is None:
            continue
        texture = patch_texture_score(patch)
        candidate = {
            "patch": patch,
            "row": row,
            "distance": float(distances[int(idx)]),
            "texture_score": texture,
        }
        if best_any is None:
            best_any = candidate
        if texture >= 0.004:
            return candidate

    return best_any


def predict_with_model(
    model,
    scaler,
    label_encoder,
    ndvi,
    evi,
    savi,
    rep,
    temp_max,
    temp_min,
    humidity,
    rainfall,
    spatial_patch_np=None,
):
    raw_df = pd.DataFrame(
        [[ndvi, evi, savi, rep, temp_max, temp_min, humidity, rainfall]],
        columns=ALL_FEATURES,
    )
    scaled_features = scaler.transform(raw_df)[0]

    spectral_tensor = torch.tensor(scaled_features[:4], dtype=torch.float32).unsqueeze(0)
    weather_tensor = torch.tensor(scaled_features[4:], dtype=torch.float32).unsqueeze(0)
    if spatial_patch_np is None:
        spatial_tensor = create_spatial_representation(spectral_tensor.squeeze(0)).unsqueeze(0)
    else:
        spatial_tensor = torch.tensor(spatial_patch_np, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        logits = model(spatial_tensor, spectral_tensor, weather_tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)

    predicted_idx = int(torch.argmax(probabilities).item())
    confidence = float(probabilities[predicted_idx].item())
    predicted_class = str(label_encoder.inverse_transform([predicted_idx])[0]).lower()

    classes = [str(label).lower() for label in label_encoder.classes_]
    prob_dict = {classes[i]: float(probabilities[i].item()) for i in range(len(classes))}

    disease_prob = prob_dict.get("diseased", 0.0)
    stress_prob = prob_dict.get("stressed", 0.0)
    disease_risk_score = min(1.0, disease_prob + (stress_prob * 0.5))

    if predicted_class == "diseased":
        risk_level = "High"
        action = "Immediate treatment required"
    elif predicted_class == "stressed":
        risk_level = "Medium"
        action = "Monitor closely and prepare treatment"
    else:
        risk_level = "Low"
        action = "Continue normal practices"

    stress_factors = []
    if ndvi < 0.3:
        stress_factors.append("Suboptimal NDVI")
    if humidity > 80 and temp_max > 25:
        stress_factors.append("Conditions favor fungal disease")
    if rainfall > 15:
        stress_factors.append("Excessive moisture stress")
    if temp_max - temp_min > 12:
        stress_factors.append("Large day-night temperature variation")

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "risk_level": risk_level,
        "action": action,
        "disease_risk_score": disease_risk_score,
        "stress_factors": stress_factors,
        "vegetation_health_score": float((ndvi + evi + savi) / 3.0),
        "probabilities": prob_dict,
    }


def classify_influence(score: float) -> str:
    if score >= 0.7:
        return "High"
    if score >= 0.4:
        return "Medium"
    return "Low"


def build_explainable_factors(
    ndvi: float,
    evi: float,
    savi: float,
    temp_max: float,
    temp_min: float,
    humidity: float,
    rainfall: float,
) -> list[dict]:
    factors: list[dict] = []

    ndvi_drop = float(np.clip((0.45 - ndvi) / 0.45, 0.0, 1.0))
    if ndvi_drop > 0:
        factors.append(
            {
                "factor": "NDVI drop",
                "score": ndvi_drop,
                "detail": f"NDVI at {ndvi:.2f} indicates reduced canopy vigor.",
            }
        )

    evi_drop = float(np.clip((0.30 - evi) / 0.30, 0.0, 1.0))
    if evi_drop > 0:
        factors.append(
            {
                "factor": "EVI decline",
                "score": evi_drop,
                "detail": f"EVI at {evi:.2f} suggests stressed photosynthetic activity.",
            }
        )

    savi_drop = float(np.clip((0.28 - savi) / 0.28, 0.0, 1.0))
    if savi_drop > 0:
        factors.append(
            {
                "factor": "SAVI weakness",
                "score": savi_drop,
                "detail": f"SAVI at {savi:.2f} indicates soil-vegetation stress interaction.",
            }
        )

    humidity_spike = float(np.clip((humidity - 75.0) / 25.0, 0.0, 1.0))
    if humidity_spike > 0:
        factors.append(
            {
                "factor": "Humidity spike",
                "score": humidity_spike,
                "detail": f"Humidity at {humidity:.1f}% favors fungal pressure.",
            }
        )

    rainfall_pressure = float(np.clip(rainfall / 20.0, 0.0, 1.0))
    if rainfall_pressure > 0:
        factors.append(
            {
                "factor": "Rainfall pressure",
                "score": rainfall_pressure,
                "detail": f"Rainfall at {rainfall:.1f} mm can increase leaf wetness duration.",
            }
        )

    thermal_stress = float(
        np.clip(max((temp_max - 35.0) / 10.0, (12.0 - temp_min) / 10.0), 0.0, 1.0)
    )
    if thermal_stress > 0:
        factors.append(
            {
                "factor": "Thermal stress",
                "score": thermal_stress,
                "detail": f"Temperature range {temp_min:.1f}C to {temp_max:.1f}C adds physiological stress.",
            }
        )

    diurnal_gap = float(np.clip(((temp_max - temp_min) - 12.0) / 12.0, 0.0, 1.0))
    if diurnal_gap > 0:
        factors.append(
            {
                "factor": "Day-night gap",
                "score": diurnal_gap,
                "detail": "Large day-night temperature swings can reduce plant resilience.",
            }
        )

    if not factors:
        factors.append(
            {
                "factor": "Stable crop conditions",
                "score": 0.15,
                "detail": "Vegetation and weather indicators are currently within safer operating ranges.",
            }
        )

    factors = sorted(factors, key=lambda item: item["score"], reverse=True)
    for item in factors:
        item["influence"] = classify_influence(item["score"])

    return factors


def compute_uncertainty(prob_dict: dict) -> dict:
    probs = np.array(list(prob_dict.values()), dtype=np.float64)
    sorted_probs = np.sort(probs)[::-1]
    top_prob = float(sorted_probs[0])
    second_prob = float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
    margin = top_prob - second_prob

    entropy = float(-np.sum(probs * np.log(probs + 1e-12)) / np.log(len(probs)))

    if top_prob >= 0.75 and margin >= 0.35 and entropy <= 0.45:
        level = "Low"
        guidance = "Prediction is stable. Manual verification is optional."
    elif top_prob >= 0.55 and margin >= 0.18 and entropy <= 0.75:
        level = "Medium"
        guidance = "Prediction is usable, but cross-check with field scouting."
    else:
        level = "High"
        guidance = "Needs manual verification before treatment decisions."

    return {
        "level": level,
        "entropy": entropy,
        "margin": margin,
        "needs_manual_verification": level == "High",
        "guidance": guidance,
    }


def project_weather_forecast(
    temp_max: float,
    temp_min: float,
    humidity: float,
    rainfall: float,
    days: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    day_index = np.arange(days)
    seasonal_wave = np.sin((2 * np.pi * day_index) / max(days, 7))

    forecast_temp_max = np.clip(
        temp_max + (seasonal_wave * 1.8) + rng.normal(0.0, 0.7, days), 12.0, 48.0
    )
    forecast_temp_min = np.clip(
        temp_min + (seasonal_wave * 1.2) + rng.normal(0.0, 0.6, days), 5.0, 35.0
    )
    forecast_humidity = np.clip(
        humidity + (seasonal_wave * 5.0) + rng.normal(0.0, 3.0, days), 20.0, 100.0
    )
    forecast_rainfall = np.clip(
        (rainfall * (0.7 + 0.25 * (seasonal_wave + 1.0))) + rng.normal(0.0, 1.2, days),
        0.0,
        60.0,
    )

    dates = pd.date_range(pd.Timestamp.today().normalize(), periods=days, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "temp_max": forecast_temp_max,
            "temp_min": forecast_temp_min,
            "humidity": forecast_humidity,
            "rainfall": forecast_rainfall,
        }
    )


def build_disease_pressure_forecast(
    base_risk: float,
    vegetation_health_score: float,
    temp_max: float,
    temp_min: float,
    humidity: float,
    rainfall: float,
    days: int,
) -> pd.DataFrame:
    forecast_df = project_weather_forecast(temp_max, temp_min, humidity, rainfall, days)
    vegetation_stress = float(np.clip((0.50 - vegetation_health_score) / 0.50, 0.0, 1.0))

    pressure_values = []
    risk_values = []
    rolling_risk = base_risk

    for _, row in forecast_df.iterrows():
        humidity_pressure = float(np.clip((row["humidity"] - 72.0) / 24.0, 0.0, 1.0))
        rainfall_pressure = float(np.clip(row["rainfall"] / 18.0, 0.0, 1.0))
        temp_pressure = float(
            np.clip(
                max((row["temp_max"] - 34.0) / 10.0, (12.0 - row["temp_min"]) / 8.0),
                0.0,
                1.0,
            )
        )

        daily_pressure = float(
            np.clip(
                (0.38 * humidity_pressure)
                + (0.26 * rainfall_pressure)
                + (0.21 * temp_pressure)
                + (0.15 * vegetation_stress),
                0.0,
                1.0,
            )
        )

        rolling_risk = float(np.clip((0.70 * rolling_risk) + (0.30 * daily_pressure), 0.0, 1.0))
        pressure_values.append(daily_pressure)
        risk_values.append(rolling_risk)

    forecast_df["disease_pressure"] = pressure_values
    forecast_df["risk_score"] = risk_values
    return forecast_df


def main() -> None:
    st.set_page_config(
        page_title="AgriGuard ML Prediction",
        page_icon="??",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_clean_theme()

    model, scaler, label_encoder = load_model_and_scaler()

    st.markdown(
        """
        <div class="hero-wrap">
            <p class="hero-title">🌾 AgriGuard: AI Crop Disease Early-Warning System</p>
            <p class="hero-subtitle">
                Multi-modal deep learning prediction from Sentinel-2 satellite imagery &amp; NASA weather data — Trained on 1000 real patches from Amaravati, Andhra Pradesh.
            </p>
            <div class="stats-banner">
                <span class="stats-badge">📡 1000 Satellite Patches</span>
                <span class="stats-badge">🌍 Amaravati, AP</span>
                <span class="stats-badge">🧠 Multi-Modal CNN (399K params)</span>
                <span class="stats-badge">🛰️ Sentinel-2 + NASA POWER</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.header("Field Analysis Parameters")
    prediction_mode = st.sidebar.radio(
        "Prediction Source",
        ["Manual Signals", "Specific Area (Lat/Lon + Date)"],
        index=1,
    )

    area_context = None
    area_patch = None
    manual_real_context = None

    if prediction_mode == "Specific Area (Lat/Lon + Date)":
        st.sidebar.subheader("Target Area")
        lat = st.sidebar.number_input("Latitude", min_value=-90.0, max_value=90.0, value=16.50, step=0.0001)
        lon = st.sidebar.number_input("Longitude", min_value=-180.0, max_value=180.0, value=80.50, step=0.0001)
        selected_date = st.sidebar.date_input(
            "Observation Date",
            value=(pd.Timestamp.today() - pd.Timedelta(days=7)).date(),
        )
        buffer_m = st.sidebar.slider("Area Radius (meters)", min_value=100, max_value=500, value=160, step=20)

        if st.sidebar.button("Fetch Real Area Data"):
            with st.spinner("Fetching Sentinel-2 and weather data for selected area..."):
                sat_data = fetch_satellite_observation(
                    lat=float(lat),
                    lon=float(lon),
                    date_text=str(selected_date),
                    buffer_m=int(buffer_m),
                )
                weather_data = fetch_weather_features(lat=float(lat), lon=float(lon), date_text=str(selected_date))

            if sat_data is None:
                st.sidebar.error("No valid Sentinel-2 data found for this point/date. Try nearby date or location.")
            elif weather_data is None:
                st.sidebar.error("Weather fetch failed for this point/date. Try again.")
            else:
                st.session_state["area_data"] = {
                    "lat": float(lat),
                    "lon": float(lon),
                    "date": str(selected_date),
                    "satellite": sat_data,
                    "weather": weather_data,
                }
                st.sidebar.success("Real area data loaded.")

        area_context = st.session_state.get("area_data")
        if area_context is None:
            st.info("Use sidebar to fetch a specific area. Until then, showing default manual signals.")
            prediction_mode = "Manual Signals"
        else:
            sat_features = area_context["satellite"]["features"]
            weather_features = area_context["weather"]
            ndvi = float(sat_features["NDVI"])
            evi = float(sat_features["EVI"])
            savi = float(sat_features["SAVI"])
            rep = float(sat_features["REP"])
            temp_max = float(weather_features["temp_max"])
            temp_min = float(weather_features["temp_min"])
            humidity = float(weather_features["humidity"])
            rainfall = float(weather_features["rainfall"])
            area_patch = area_context["satellite"]["patch"]

    if prediction_mode == "Manual Signals":
        st.sidebar.subheader("Satellite Vegetation Indices")
        ndvi = st.sidebar.slider("NDVI (Normalized Difference Vegetation Index)", -1.0, 1.0, 0.5, 0.01)
        evi = st.sidebar.slider("EVI (Enhanced Vegetation Index)", -1.0, 1.0, 0.3, 0.01)
        savi = st.sidebar.slider("SAVI (Soil Adjusted Vegetation Index)", -1.0, 1.0, 0.25, 0.01)
        rep = st.sidebar.slider("REP (Red Edge Position)", 680.0, 750.0, 715.0, 0.5)

        st.sidebar.subheader("Weather Conditions")
        temp_max = st.sidebar.slider("Maximum Temperature (C)", 15.0, 45.0, 30.0, 0.5)
        temp_min = st.sidebar.slider("Minimum Temperature (C)", 5.0, 35.0, 20.0, 0.5)
        humidity = st.sidebar.slider("Relative Humidity (%)", 20.0, 100.0, 70.0, 1.0)
        rainfall = st.sidebar.slider("Rainfall (mm)", 0.0, 50.0, 5.0, 0.5)

        st.sidebar.subheader("Manual Mode Patch Source")
        use_real_patch_for_manual = st.sidebar.checkbox(
            "Use nearest real satellite patch",
            value=True,
            help="Matches your manual signals to the closest real sample and uses that real patch for prediction.",
        )
        if use_real_patch_for_manual:
            manual_real_context = find_closest_real_patch_for_manual(
                ndvi=ndvi,
                evi=evi,
                savi=savi,
                rep=rep,
                temp_max=temp_max,
                temp_min=temp_min,
                humidity=humidity,
                rainfall=rainfall,
            )
            if manual_real_context is not None:
                area_patch = manual_real_context["patch"]
                st.sidebar.success(
                    f"Real patch linked: {manual_real_context['row'].get('sample_id', 'unknown')} "
                    f"(distance {manual_real_context['distance']:.2f})"
                )
                if manual_real_context.get("texture_score", 0.0) < 0.004:
                    st.sidebar.warning(
                        "Matched patch has low spatial texture. Prediction is valid, but image preview may look flat."
                    )
            else:
                st.sidebar.warning("No real patch match found, using synthetic spatial patch.")

    result = predict_with_model(
        model,
        scaler,
        label_encoder,
        ndvi,
        evi,
        savi,
        rep,
        temp_max,
        temp_min,
        humidity,
        rainfall,
        spatial_patch_np=area_patch,
    )
    uncertainty = compute_uncertainty(result["probabilities"])
    explainable_factors = build_explainable_factors(
        ndvi, evi, savi, temp_max, temp_min, humidity, rainfall
    )

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.header("Deep Learning Risk Analysis")
        st.markdown(
            '<p class="tiny-muted">Prediction, confidence, explainability, and action guidance.</p>',
            unsafe_allow_html=True,
        )
        if st.button("Analyze Field Conditions"):
            st.success("Neural network inference complete.")

        col_a, col_b, col_c, col_d = st.columns(4)

        with col_a:
            status_emoji = "🟢" if result["predicted_class"] == "healthy" else "🟡" if result["predicted_class"] == "stressed" else "🔴"
            st.metric(
                "Health Status",
                f"{status_emoji} {result['predicted_class'].title()}",
                f"{result['confidence']:.1%} confidence",
            )

        with col_b:
            st.metric("Risk Level", result["risk_level"])

        with col_c:
            st.metric("ML Disease Score", f"{result['disease_risk_score']:.2f}")

        with col_d:
            st.metric("Vegetation Health", f"{result['vegetation_health_score']:.2f}")

        st.subheader("Prediction Uncertainty")
        st.metric("Uncertainty Level", uncertainty["level"], f"decision margin: {uncertainty['margin']:.2f}")
        if uncertainty["needs_manual_verification"]:
            st.warning(uncertainty["guidance"])
        else:
            st.info(uncertainty["guidance"])

        st.subheader("Recommended Actions")
        if result["predicted_class"] == "diseased":
            st.error(f"**{result['action']}**")
            recommendations = [
                "Apply targeted fungicide treatment within 24-48 hours.",
                "Increase field monitoring to daily inspections.",
                "Improve drainage if excessive moisture is present.",
                "Consider soil treatment for root-borne diseases.",
            ]
        elif result["predicted_class"] == "stressed":
            st.warning(f"**{result['action']}**")
            recommendations = [
                "Monitor field conditions twice weekly.",
                "Adjust irrigation based on the weather pattern.",
                "Prepare preventive treatment options.",
                "Check manually for early disease symptoms.",
            ]
        else:
            st.success(f"**{result['action']}**")
            recommendations = [
                "Continue the regular monitoring schedule.",
                "Maintain current agricultural practices.",
                "Keep preventive measures in place.",
            ]

        for index, recommendation in enumerate(recommendations, start=1):
            st.write(f"**{index}.** {recommendation}")

        if result["stress_factors"]:
            st.subheader("Identified Stress Factors")
            for factor in result["stress_factors"]:
                st.write(f"- {factor}")

        st.subheader("Explainable AI: Why this prediction?")
        explain_df = pd.DataFrame(explainable_factors[:5])[["factor", "influence", "score", "detail"]]
        explain_df.columns = ["Factor", "Influence", "Score", "Reason"]
        explain_df["Score"] = explain_df["Score"].map(lambda value: f"{value:.2f}")
        st.dataframe(explain_df, use_container_width=True, hide_index=True)

        top_factor_plot = pd.DataFrame(explainable_factors[:3])
        fig_explain = go.Figure()
        fig_explain.add_trace(
            go.Bar(
                x=top_factor_plot["score"],
                y=top_factor_plot["factor"],
                orientation="h",
                marker_color=["#d62728", "#ff7f0e", "#bcbd22"],
                text=[f"{value:.2f}" for value in top_factor_plot["score"]],
                textposition="auto",
            )
        )
        fig_explain.update_layout(
            title="Top Contributors To Current Risk",
            xaxis_title="Influence Score",
            yaxis_title="Factor",
            xaxis=dict(range=[0, 1]),
            height=280,
        )
        st.plotly_chart(fig_explain, use_container_width=True)

        st.subheader("Detailed Analysis")
        indices = ["NDVI", "EVI", "SAVI"]
        values = [ndvi, evi, savi]
        colors = ["green" if value > 0.4 else "orange" if value > 0.2 else "red" for value in values]

        fig_veg = go.Figure()
        fig_veg.add_trace(
            go.Bar(
                x=indices,
                y=values,
                marker_color=colors,
                text=[f"{value:.3f}" for value in values],
                textposition="auto",
            )
        )
        fig_veg.update_layout(
            title="Vegetation Health Indicators",
            yaxis_title="Index Value",
            showlegend=False,
        )
        st.plotly_chart(fig_veg, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.header("Current Indicators")
        st.markdown(
            '<p class="tiny-muted">Spatial context, weather summary, and near-term pressure trend.</p>',
            unsafe_allow_html=True,
        )

        if area_context is not None and area_patch is not None:
            st.subheader("Selected Area")
            st.write(f"**Lat/Lon:** {area_context['lat']:.5f}, {area_context['lon']:.5f}")
            st.write(f"**Date:** {area_context['date']}")
            cloud_avg = area_context["satellite"]["meta"]["cloud_avg"]
            if cloud_avg is not None:
                st.write(f"**Avg cloud cover:** {cloud_avg:.1f}%")
            st.map(pd.DataFrame({"lat": [area_context["lat"]], "lon": [area_context["lon"]]}))
            st.image(
                patch_to_rgb_preview(area_patch),
                caption="Sentinel-2 patch (RGB composite)",
                use_column_width=True,
            )
        elif manual_real_context is not None and area_patch is not None:
            matched = manual_real_context["row"]
            st.subheader("Matched Real Patch (Manual Mode)")
            st.write(f"**Sample ID:** {matched.get('sample_id', 'N/A')}")
            if "region" in matched:
                st.write(f"**Region:** {matched.get('region', 'N/A')}")
            if "date" in matched:
                st.write(f"**Reference Date:** {matched.get('date', 'N/A')}")
            st.write(f"**Match Distance:** {manual_real_context['distance']:.2f}")
            st.write(f"**Texture Score:** {manual_real_context.get('texture_score', 0.0):.4f}")
            if manual_real_context.get("texture_score", 0.0) < 0.004:
                st.warning(
                    "This patch is almost uniform (low texture), usually due cloud/mask/weak scene variation."
                )
            st.image(
                patch_to_rgb_preview(area_patch),
                caption="Nearest real Sentinel-2 patch used for this manual prediction",
                use_column_width=True,
            )

        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=max(0.0, ndvi),
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "NDVI"},
                gauge={
                    "axis": {"range": [0, 1]},
                    "bar": {"color": "darkgreen"},
                    "steps": [
                        {"range": [0, 0.2], "color": "lightcoral"},
                        {"range": [0.2, 0.4], "color": "gold"},
                        {"range": [0.4, 1], "color": "lightgreen"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 0.2,
                    },
                },
            )
        )
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.subheader("Weather Summary")
        st.write(f"**Temperature:** {temp_min:.1f}C - {temp_max:.1f}C")
        st.write(f"**Humidity:** {humidity:.1f}%")
        st.write(f"**Rainfall:** {rainfall:.1f} mm")

        st.subheader("Class Probabilities")
        prob_table = pd.DataFrame(
            {
                "Class": [label.title() for label in result["probabilities"].keys()],
                "Probability": [f"{value:.1%}" for value in result["probabilities"].values()],
            }
        )
        st.dataframe(prob_table, use_container_width=True, hide_index=True)

        st.subheader("Disease Pressure Forecast")
        forecast_days = st.radio(
            "Forecast Horizon",
            options=[7, 14],
            horizontal=True,
            index=1,
            key="forecast_horizon_days",
        )
        forecast_df = build_disease_pressure_forecast(
            base_risk=result["disease_risk_score"],
            vegetation_health_score=result["vegetation_health_score"],
            temp_max=temp_max,
            temp_min=temp_min,
            humidity=humidity,
            rainfall=rainfall,
            days=forecast_days,
        )

        fig_timeline = go.Figure()
        fig_timeline.add_trace(
            go.Scatter(
                x=forecast_df["date"],
                y=forecast_df["risk_score"],
                mode="lines+markers",
                name="Forecast Risk",
                line=dict(color="red", width=3),
            )
        )
        fig_timeline.add_trace(
            go.Scatter(
                x=forecast_df["date"],
                y=forecast_df["disease_pressure"],
                mode="lines",
                name="Weather Pressure",
                line=dict(color="orange", width=2, dash="dash"),
            )
        )
        fig_timeline.update_layout(
            title=f"{forecast_days}-Day Disease Pressure Projection",
            xaxis_title="Date",
            yaxis_title="Risk Score",
            yaxis=dict(range=[0, 1]),
            height=280,
        )
        st.plotly_chart(fig_timeline, use_container_width=True)

        high_pressure_days = int((forecast_df["disease_pressure"] > 0.65).sum())
        st.write(f"**High pressure days ({forecast_days}d):** {high_pressure_days}")
        st.write(f"**Avg forecast humidity:** {forecast_df['humidity'].mean():.1f}%")
        st.write(f"**Avg forecast rainfall:** {forecast_df['rainfall'].mean():.1f} mm")
        st.markdown("</div>", unsafe_allow_html=True)

    # ── About Section ──
    st.markdown("---")
    with st.expander("ℹ️ About AgriGuard — System Architecture & Methodology", expanded=False):
        st.markdown("""
### System Overview
**AgriGuard** is a multi-modal deep learning system for early detection of crop diseases 
using satellite remote sensing and meteorological data. It is designed to provide actionable 
intelligence to farmers in the **Amaravati region of Andhra Pradesh**.

### Data Sources
| Source | Description | Resolution |
|--------|-------------|------------|
| **Sentinel-2 SR** | ESA satellite imagery (B2, B3, B4, B8 bands) | 10m spatial |
| **NASA POWER** | Daily weather: temperature, humidity, rainfall | Point-level |
| **Google Earth Engine** | Cloud-based geospatial processing | On-demand |

### Model Architecture
- **Spatial Encoder**: 3-layer CNN processing 32×32 satellite patches (4 bands)
- **Spectral Encoder**: MLP on vegetation indices (NDVI, EVI, SAVI, REP)
- **Weather Encoder**: MLP on meteorological features
- **Fusion Layer**: Concatenated multi-modal features → classification head
- **Total Parameters**: 399K | **Training Samples**: 1000 real patches

### Vegetation Indices Used
- **NDVI** — Normalized Difference Vegetation Index (canopy greenness)
- **EVI** — Enhanced Vegetation Index (improved sensitivity in dense canopy)
- **SAVI** — Soil Adjusted Vegetation Index (corrects for soil brightness)
- **REP** — Red Edge Position (early stress indicator, 680–750 nm)

### Technology Stack
`Python` · `PyTorch` · `PyTorch Lightning` · `Google Earth Engine` · `Streamlit` · `Plotly` · `scikit-learn`
        """)

    st.markdown(
        """
        <div style="text-align:center; padding:1.5rem 0 0.5rem 0; opacity:0.5; font-size:0.8rem;">
            AgriGuard v1.0 · Final Year Project · Powered by Sentinel-2, NASA POWER & PyTorch
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
