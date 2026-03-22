import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os

# ------------------------------
# Fix deterministic issue
# ------------------------------
torch.backends.cudnn.deterministic = False
torch.use_deterministic_algorithms(False)

# ------------------------------
# Project Root & Path Configuration
# ------------------------------
# Hardcoded paths as requested (using raw strings for Windows paths)
BASE_DIR = r"C:\Users\artha\OneDrive\Desktop\SSDDPP\AgriGuard"
CSV_PATH = r"C:\Users\artha\OneDrive\Desktop\SSDDPP\AgriGuard\data\processed\agriguard_training_dataset.csv"

print("Project Root:", BASE_DIR)
print("Dataset Path:", CSV_PATH)

# ------------------------------
# Dataset
# ------------------------------
class AgriDataset(Dataset):
    def __init__(self, data, features_scaler=None, label_encoder=None, target_col='health_status', mode='train'):
        self.data = data.copy()
        self.mode = mode
        self.target_col = target_col

        self.spectral_features = ['NDVI', 'EVI', 'SAVI', 'REP']
        self.weather_features = ['temp_max', 'temp_min', 'humidity', 'rainfall']

        self.all_features = self.spectral_features + self.weather_features

        # Scale features safely
        if features_scaler is None:
            self.features_scaler = StandardScaler()
            self.data[self.all_features] = self.features_scaler.fit_transform(self.data[self.all_features])
        else:
            self.features_scaler = features_scaler
            self.data[self.all_features] = self.features_scaler.transform(self.data[self.all_features])

        # Handle label encoding cleanly within the class
        if mode == 'train':
            self.label_encoder = LabelEncoder()
            self.data['encoded_label'] = self.label_encoder.fit_transform(self.data[target_col])
        else:
            self.label_encoder = label_encoder
            if self.label_encoder is not None:
                self.data['encoded_label'] = self.label_encoder.transform(self.data[target_col])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        spectral = torch.tensor([row[feat] for feat in self.spectral_features], dtype=torch.float32)
        weather = torch.tensor([row[feat] for feat in self.weather_features], dtype=torch.float32)

        spatial_patch = self.create_spatial_representation(spectral)

        if 'encoded_label' in self.data.columns:
            label = torch.tensor(row['encoded_label'], dtype=torch.long)
        else:
            label = torch.tensor(-1, dtype=torch.long)

        return {
            "spatial": spatial_patch,
            "spectral": spectral,
            "weather": weather,
            "label": label
        }

    # ------------------------------
    # Synthetic Spatial Patch
    # ------------------------------
    def create_spatial_representation(self, spectral_values):
        patch_size = 32
        n_bands = 4

        base_reflectance = torch.tensor([0.06, 0.10, 0.04, 0.50])

        ndvi_val = spectral_values[0].item()
        health_factor = max(0.3, (ndvi_val + 1) / 2)

        patch = torch.zeros(n_bands, patch_size, patch_size)

        for band_idx, base_val in enumerate(base_reflectance):
            spatial_noise = torch.randn(patch_size, patch_size) * 0.02
            health_variation = base_val * health_factor

            patch[band_idx] = health_variation + spatial_noise
            patch[band_idx] = torch.clamp(patch[band_idx], 0.01, 0.95)

        return patch


# ------------------------------
# Multi Modal CNN
# ------------------------------
class MultiModalCNN(pl.LightningModule):
    def __init__(self, num_classes=3, learning_rate=1e-3):
        super().__init__()
        self.learning_rate = learning_rate
        self.num_classes = num_classes

        # Spatial CNN
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
            nn.Dropout(0.3)
        )

        # Spectral encoder
        self.spectral_encoder = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

        # Weather encoder
        self.weather_encoder = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(128 + 64 + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, num_classes)
        )

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, spatial, spectral, weather):
        spatial_features = self.spatial_encoder(spatial)
        spectral_features = self.spectral_encoder(spectral)
        weather_features = self.weather_encoder(weather)

        combined = torch.cat([spatial_features, spectral_features, weather_features], dim=1)
        output = self.fusion(combined)

        return output

    def training_step(self, batch, batch_idx):
        outputs = self(
            batch["spatial"],
            batch["spectral"],
            batch["weather"]
        )

        loss = self.criterion(outputs, batch["label"])
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == batch["label"]).float().mean()

        self.log("train_loss", loss)
        self.log("train_acc", acc)

        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(
            batch["spatial"],
            batch["spectral"],
            batch["weather"]
        )

        loss = self.criterion(outputs, batch["label"])
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == batch["label"]).float().mean()

        self.log("val_loss", loss)
        self.log("val_acc", acc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


# ------------------------------
# Data Loaders
# ------------------------------
def create_dataloaders(train_df, val_df, batch_size=8):
    train_dataset = AgriDataset(train_df, mode="train")

    # Cleanly pass the saved scaler and encoder to the validation set
    val_dataset = AgriDataset(
        val_df,
        features_scaler=train_dataset.features_scaler,
        label_encoder=train_dataset.label_encoder,
        mode="val"
    )

    # Note: On Windows, keeping num_workers=0 (the default) in DataLoader prevents multi-processing crashing
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, train_dataset.label_encoder


# ------------------------------
# Training
# ------------------------------
def train_working_model():
    print("\nStarting AgriGuard Training")
    print("=" * 50)

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Dataset not found at {CSV_PATH}. Please check the path.")

    df = pd.read_csv(CSV_PATH)
    print("Dataset Loaded:", len(df))

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["health_status"],
        random_state=42
    )

    print("Train:", len(train_df), "Val:", len(val_df))

    train_loader, val_loader, label_encoder = create_dataloaders(train_df, val_df)

    batch = next(iter(train_loader))
    print("Batch spatial:", batch["spatial"].shape)

    model = MultiModalCNN(num_classes=len(label_encoder.classes_))

    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="auto", # Automatically detects GPU/CPU 
        devices=1,
        logger=False,
        enable_checkpointing=False
    )

    # Use explicit keyword arguments for the dataloaders
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Create the directory securely
    os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
    model_path = os.path.join(BASE_DIR, "models", "agriguard_model.pth")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "label_encoder": label_encoder
        },
        model_path
    )

    print("\nModel Saved:", model_path)
    return model, label_encoder


# ------------------------------
# Run
# ------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    model, label_encoder = train_working_model()
    print("\nTraining Completed Successfully")