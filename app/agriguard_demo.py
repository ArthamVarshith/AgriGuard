import streamlit as st
import torch
import torch.nn as nn
import pytorch_lightning as pl
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import os

# ------------------------------
# Hardcoded Paths
# ------------------------------
BASE_DIR = r"C:\Users\artha\OneDrive\Desktop\SSDDPP\AgriGuard"
CSV_PATH = os.path.join(BASE_DIR, "data", "processed", "agriguard_training_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "agriguard_model.pth")

# ------------------------------
# Model Architecture Definition
# ------------------------------
class MultiModalCNN(pl.LightningModule):
    def __init__(self, num_classes=3):
        super().__init__()
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(128 * 4 * 4, 128), nn.ReLU(), nn.Dropout(0.3)
        )
        self.spectral_encoder = nn.Sequential(
            nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 64)
        )
        self.weather_encoder = nn.Sequential(
            nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 32)
        )
        self.fusion = nn.Sequential(
            nn.Linear(128 + 64 + 32, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, num_classes)
        )

    def forward(self, spatial, spectral, weather):
        spatial_features = self.spatial_encoder(spatial)
        spectral_features = self.spectral_encoder(spectral)
        weather_features = self.weather_encoder(weather)
        combined = torch.cat([spatial_features, spectral_features, weather_features], dim=1)
        return self.fusion(combined)

# ------------------------------
# Setup & Caching
# ------------------------------
@st.cache_resource
def load_model_and_scaler():
    """Loads the model, label encoder, and recreates the scaler from the dataset."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at {MODEL_PATH}. Please train the model first.")
        st.stop()
        
    # Recreate the scaler using the original dataset
    df = pd.read_csv(CSV_PATH)
    spectral_features = ['NDVI', 'EVI', 'SAVI', 'REP']
    weather_features = ['temp_max', 'temp_min', 'humidity', 'rainfall']
    
    scaler = StandardScaler()
    scaler.fit(df[spectral_features + weather_features])
    
    # Load model and encoder
    checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)
    label_encoder = checkpoint['label_encoder']
    
    model = MultiModalCNN(num_classes=len(label_encoder.classes_))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval() # Set to evaluation mode
    
    return model, scaler, label_encoder

def create_spatial_representation(spectral_values):
    """Recreates the synthetic spatial patch needed for the CNN."""
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
# ML Prediction Logic
# ------------------------------
def predict_with_model(model, scaler, label_encoder, ndvi, evi, savi, rep, temp_max, temp_min, humidity, rainfall):
    # 1. Prepare raw inputs
    raw_features = np.array([[ndvi, evi, savi, rep, temp_max, temp_min, humidity, rainfall]])
    
    # 2. Scale features
    scaled_features = scaler.transform(raw_features)[0]
    
    # 3. Create Tensors (Add batch dimension using unsqueeze)
    spectral_tensor = torch.tensor(scaled_features[:4], dtype=torch.float32)
    weather_tensor = torch.tensor(scaled_features[4:], dtype=torch.float32)
    
    spatial_tensor = create_spatial_representation(spectral_tensor).unsqueeze(0)
    spectral_tensor = spectral_tensor.unsqueeze(0)
    weather_tensor = weather_tensor.unsqueeze(0)
    
    # 4. Inference
    with torch.no_grad():
        logits = model(spatial_tensor, spectral_tensor, weather_tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze()
        
    # 5. Decode predictions
    predicted_idx = torch.argmax(probabilities).item()
    confidence = probabilities[predicted_idx].item()
    predicted_class = label_encoder.inverse_transform([predicted_idx])[0].lower()
    
    # Create risk metrics based on probabilities mapping
    classes = list(label_encoder.classes_)
    prob_dict = {classes[i].lower(): probabilities[i].item() for i in range(len(classes))}
    
    # Calculate a composite disease risk score (0 to 1)
    disease_prob = prob_dict.get('diseased', 0)
    stress_prob = prob_dict.get('stressed', 0)
    disease_risk_score = disease_prob + (stress_prob * 0.5)
    
    # Formatting outputs for the UI
    if 'disease' in predicted_class:
        risk_level = 'High'
        action = 'Immediate treatment required'
    elif 'stress' in predicted_class:
        risk_level = 'Medium'
        action = 'Monitor closely and prepare treatment'
    else:
        risk_level = 'Low'
        action = 'Continue normal practices'

    # Determine stress factors for UI (hybrid logic based on raw inputs)
    stress_factors = []
    if ndvi < 0.3: stress_factors.append("Suboptimal NDVI")
    if humidity > 80 and temp_max > 25: stress_factors.append("Optimal fungal disease conditions")
    if rainfall > 15: stress_factors.append("Excessive moisture stress")
    
    return {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'risk_level': risk_level,
        'action': action,
        'disease_risk_score': disease_risk_score,
        'stress_factors': stress_factors,
        'vegetation_health_score': (ndvi + evi + savi) / 3 
    }

# ------------------------------
# Streamlit UI Definition
# ------------------------------
def main():
    st.set_page_config(
        page_title="AgriGuard ML Prediction",
        page_icon="🌾",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load model and scaler implicitly
    model, scaler, label_encoder = load_model_and_scaler()
    
    st.title("🌾 AgriGuard: AI-Powered Crop Disease Detection")
    st.markdown("**Multi-modal Deep Learning (PyTorch) using satellite imagery and weather data**")
    st.markdown("---")
    
    # Sidebar inputs
    st.sidebar.header("📊 Field Analysis Parameters")
    
    st.sidebar.subheader("🛰️ Satellite Vegetation Indices")
    ndvi = st.sidebar.slider("NDVI (Normalized Difference Vegetation Index)", -1.0, 1.0, 0.5, 0.01)
    evi = st.sidebar.slider("EVI (Enhanced Vegetation Index)", -1.0, 1.0, 0.3, 0.01)
    savi = st.sidebar.slider("SAVI (Soil Adjusted Vegetation Index)", -1.0, 1.0, 0.25, 0.01)
    rep = st.sidebar.slider("REP (Red Edge Position)", 680.0, 750.0, 715.0, 0.5)
    
    st.sidebar.subheader("🌤️ Weather Conditions")
    temp_max = st.sidebar.slider("Maximum Temperature (°C)", 15.0, 45.0, 30.0, 0.5)
    temp_min = st.sidebar.slider("Minimum Temperature (°C)", 5.0, 35.0, 20.0, 0.5)
    humidity = st.sidebar.slider("Relative Humidity (%)", 20.0, 100.0, 70.0, 1.0)
    rainfall = st.sidebar.slider("Rainfall (mm)", 0.0, 50.0, 5.0, 0.5)
    
    # Perform ML prediction
    result = predict_with_model(
        model, scaler, label_encoder, 
        ndvi, evi, savi, rep, temp_max, temp_min, humidity, rainfall
    )
    
    # Main analysis section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🎯 Deep Learning Risk Analysis")
        
        if st.button("🔍 Analyze Field Conditions", type="primary", use_container_width=True):
            st.success("✅ Neural Network Inference Complete!")
        
        col_a, col_b, col_c, col_d = st.columns(4)
        
        with col_a:
            status_emoji = "🟢" if 'health' in result['predicted_class'] else "🟡" if 'stress' in result['predicted_class'] else "🔴"
            st.metric(
                "Health Status",
                f"{status_emoji} {result['predicted_class'].title()}",
                f"{result['confidence']:.1%} confidence"
            )
        
        with col_b:
            st.metric("Risk Level", result['risk_level'])
        
        with col_c:
            st.metric("ML Disease Score", f"{result['disease_risk_score']:.2f}")
        
        with col_d:
            st.metric("Vegetation Health", f"{result['vegetation_health_score']:.2f}")
        
        st.subheader("📋 Recommended Actions")
        if 'disease' in result['predicted_class']:
            st.error(f"🚨 **{result['action']}**")
            recommendations = [
                "Apply targeted fungicide treatment within 24-48 hours",
                "Increase field monitoring to daily inspections",
                "Improve drainage if excessive moisture present",
                "Consider soil treatment for root-borne diseases"
            ]
        elif 'stress' in result['predicted_class']:
            st.warning(f"⚠️ **{result['action']}**")
            recommendations = [
                "Monitor field conditions twice weekly",
                "Adjust irrigation based on weather forecast",
                "Prepare preventive treatment options",
                "Check for early disease symptoms manually"
            ]
        else:
            st.success(f"✅ **{result['action']}**")
            recommendations = [
                "Continue regular monitoring schedule",
                "Maintain current agricultural practices",
                "Keep preventive measures in place"
            ]
        
        for i, rec in enumerate(recommendations, 1):
            st.write(f"**{i}.** {rec}")
        
        if result['stress_factors']:
            st.subheader("⚠️ Identified Stress Factors")
            for factor in result['stress_factors']:
                st.write(f"• {factor}")
        
        st.subheader("📊 Detailed Analysis")
        
        fig_veg = go.Figure()
        indices = ['NDVI', 'EVI', 'SAVI']
        values = [ndvi, evi, savi]
        colors = ['green' if v > 0.4 else 'orange' if v > 0.2 else 'red' for v in values]
        
        fig_veg.add_trace(go.Bar(
            x=indices, y=values, marker_color=colors,
            text=[f'{v:.3f}' for v in values], textposition='auto'
        ))
        
        fig_veg.update_layout(title="Vegetation Health Indicators", yaxis_title="Index Value", showlegend=False)
        st.plotly_chart(fig_veg, use_container_width=True)
        
    with col2:
        st.header("📈 Current Indicators")
        
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=ndvi,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "NDVI"},
            gauge={
                'axis': {'range': [0, 1]},
                'bar': {'color': "darkgreen"},
                'steps': [
                    {'range': [0, 0.2], 'color': "lightcoral"},
                    {'range': [0.2, 0.4], 'color': "gold"},
                    {'range': [0.4, 1], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.2
                }
            }
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        st.subheader("🌡️ Weather Summary")
        st.write(f"**Temperature:** {temp_min}°C - {temp_max}°C")
        st.write(f"**Humidity:** {humidity}%")
        st.write(f"**Rainfall:** {rainfall}mm")
        
        st.subheader("📅 Risk Forecast")
        dates = pd.date_range(pd.Timestamp.today().strftime('%Y-%m-%d'), periods=7, freq='D')
        base_risk = result['disease_risk_score']
        risk_trend = [base_risk * (1 + 0.1 * np.random.randn()) for _ in range(7)]
        risk_trend = [max(0, min(1, r)) for r in risk_trend]
        
        fig_timeline = go.Figure()
        fig_timeline.add_trace(go.Scatter(
            x=dates, y=risk_trend, mode='lines+markers', name='Disease Risk',
            line=dict(color='red', width=3)
        ))
        fig_timeline.update_layout(
            title="7-Day Risk Projection", xaxis_title="Date", yaxis_title="Risk Score", height=250
        )
        st.plotly_chart(fig_timeline, use_container_width=True)

if __name__ == "__main__":
    main()