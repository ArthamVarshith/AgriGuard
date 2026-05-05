import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import ee
import pandas as pd
import numpy as np
from datetime import datetime

def initialize_earth_engine():
    """Initialize Google Earth Engine"""
    try:
        ee.Initialize()
        print("✅ Google Earth Engine initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Error initializing Earth Engine: {e}")
        print("💡 Run: ee.Authenticate() in Python console first")
        return False

def collect_satellite_data():
    """Collect satellite data from Sentinel-2"""
    print("\n🛰️  COLLECTING SATELLITE DATA")
    print("-" * 40)
    
    # Define Amaravati region (major agricultural area in Andhra Pradesh)
    amaravati_region = ee.Geometry.Rectangle([80.20, 16.30, 80.80, 16.80])
    print("📍 Target region: Amaravati, Andhra Pradesh")
    
    # Get Sentinel-2 collection (using updated collection)
    collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                 .filterBounds(amaravati_region)
                 .filterDate('2024-06-01', '2024-11-30')
                 .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                 .sort('system:time_start'))
    
    image_count = collection.size().getInfo()
    print(f"📡 Found {image_count} usable satellite images")
    
    if image_count == 0:
        print("⚠️  No images found. Expanding date range...")
        collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                     .filterBounds(amaravati_region)
                     .filterDate('2024-01-01', '2024-12-31')
                     .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
                     .sort('system:time_start'))
        image_count = collection.size().getInfo()
        print(f"📡 Extended search found {image_count} images")
    
    # Get sample image for analysis
    sample_image = collection.first()
    
    # Calculate vegetation indices
    def calculate_indices(image):
        # NDVI - Normalized Difference Vegetation Index
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        
        # EVI - Enhanced Vegetation Index  
        evi = image.expression(
            '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
            {
                'NIR': image.select('B8').divide(10000),
                'RED': image.select('B4').divide(10000),
                'BLUE': image.select('B2').divide(10000)
            }
        ).rename('EVI')
        
        # SAVI - Soil Adjusted Vegetation Index
        savi = image.expression(
            '((NIR - RED) / (NIR + RED + 0.5)) * 1.5',
            {
                'NIR': image.select('B8').divide(10000),
                'RED': image.select('B4').divide(10000)
            }
        ).rename('SAVI')
        
        return image.addBands([ndvi, evi, savi])
    
    # Process collection
    processed_collection = collection.map(calculate_indices)
    
    # Get statistics from sample image
    sample_with_indices = calculate_indices(sample_image)
    
    stats = sample_with_indices.select(['NDVI', 'EVI', 'SAVI']).reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=amaravati_region,
        scale=100,
        maxPixels=1e9
    )
    
    stats_info = stats.getInfo()
    print("\n📊 Sample Vegetation Index Statistics:")
    for key, value in stats_info.items():
        if value is not None:
            print(f"  🌱 {key}: {value:.3f}")
    
    return processed_collection, amaravati_region

def start_satellite_export(collection, region):
    """Start export of satellite data to Google Drive"""
    print("\n📤 STARTING SATELLITE DATA EXPORT")
    print("-" * 40)
    
    # Create monthly composite
    composite = collection.median()
    
    # Select bands for export
    export_bands = ['B4', 'B3', 'B2', 'B8', 'NDVI', 'EVI', 'SAVI']
    
    # Create export task
    task = ee.batch.Export.image.toDrive(
        image=composite.select(export_bands),
        description='agriguard_amaravati_satellite_data',
        folder='AgriGuard_Data',
        region=region,
        scale=10,  # 10 meter resolution
        crs='EPSG:4326',
        maxPixels=1e9,
        fileFormat='GeoTIFF'
    )
    
    # Start the task
    task.start()
    
    print(f"🚀 Export task started!")
    print(f"📋 Task ID: {task.id}")
    print(f"📁 Folder: AgriGuard_Data")
    print(f"📏 Resolution: 10 meters")
    print(f"🎯 Bands: {', '.join(export_bands)}")
    print("⏳ Check your Google Drive in 10-15 minutes!")
    
    return task

def generate_weather_data():
    """Generate synthetic weather data with disease risk factors"""
    print("\n🌤️  GENERATING WEATHER DATA")
    print("-" * 40)
    
    # Create date range
    dates = pd.date_range('2024-06-01', '2024-11-30', freq='D')
    print(f"📅 Generating {len(dates)} days of weather data")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    weather_data = []
    
    for i, date in enumerate(dates):
        month = date.month
        day_of_year = date.dayofyear
        
        # Seasonal temperature pattern for Andhra Pradesh
        base_temp = 25 + 5 * np.sin(2 * np.pi * (month - 4) / 12)
        temp_max = base_temp + np.random.normal(5, 2)
        temp_min = base_temp - np.random.normal(5, 2)
        temp_max = max(temp_max, temp_min + 1)  # Ensure max > min
        
        # Humidity patterns (higher during monsoon)
        if month in [6, 7, 8, 9]:  # Monsoon months
            humidity = np.random.normal(80, 10)
        else:
            humidity = np.random.normal(65, 15)
        humidity = max(30, min(100, humidity))  # Clamp between 30-100%
        
        # Rainfall patterns
        if month in [6, 7, 8, 9]:  # Monsoon
            rainfall = np.random.exponential(3)
        else:
            rainfall = np.random.exponential(0.3)
        
        # Disease risk factors
        temp_stress = 1 if (temp_max > 35 or temp_min < 12) else 0
        humidity_risk = max(0, (humidity - 70) / 30)  # Risk increases above 70%
        wet_risk = min(1, rainfall / 10)  # Normalize rainfall risk
        
        # Combined disease favorability score
        disease_risk = (humidity_risk * 0.4 + wet_risk * 0.3 + temp_stress * 0.3)
        disease_risk = max(0, min(1, disease_risk))  # Clamp 0-1
        
        weather_data.append({
            'date': date,
            'temp_max': round(temp_max, 1),
            'temp_min': round(temp_min, 1),
            'humidity': round(humidity, 1),
            'rainfall': round(rainfall, 2),
            'wind_speed': round(np.random.normal(12, 3), 1),
            'temp_stress': temp_stress,
            'humidity_risk': round(humidity_risk, 3),
            'disease_risk': round(disease_risk, 3)
        })
    
    # Create DataFrame
    weather_df = pd.DataFrame(weather_data)
    
    # Save to CSV
    os.makedirs('data/raw', exist_ok=True)
    output_file = 'data/raw/amaravati_weather_2024.csv'
    weather_df.to_csv(output_file, index=False)
    
    print(f"💾 Weather data saved to: {output_file}")
    
    # Calculate statistics
    high_risk_days = (weather_df['disease_risk'] > 0.7).sum()
    medium_risk_days = ((weather_df['disease_risk'] > 0.4) & (weather_df['disease_risk'] <= 0.7)).sum()
    low_risk_days = (weather_df['disease_risk'] <= 0.4).sum()
    
    print(f"\n📊 Weather Summary:")
    print(f"  🌡️  Temperature: {weather_df['temp_max'].mean():.1f}°C ± {weather_df['temp_max'].std():.1f}")
    print(f"  💧 Humidity: {weather_df['humidity'].mean():.1f}% ± {weather_df['humidity'].std():.1f}")
    print(f"  🌧️  Total Rainfall: {weather_df['rainfall'].sum():.1f}mm")
    print(f"  ⚠️  Disease Risk Days:")
    print(f"     🔴 High (>0.7): {high_risk_days} days")
    print(f"     🟡 Medium (0.4-0.7): {medium_risk_days} days") 
    print(f"     🟢 Low (<0.4): {low_risk_days} days")
    
    return weather_df

def main():
    """Main execution function"""
    print("=" * 60)
    print("🌾 AGRIGUARD DATA COLLECTION PIPELINE")
    print("=" * 60)
    print("🎯 Collecting multi-modal data for crop disease detection")
    print("📍 Focus: Amaravati, Andhra Pradesh")
    print()
    
    # Initialize Earth Engine
    if not initialize_earth_engine():
        return
    
    try:
        # Collect satellite data
        collection, region = collect_satellite_data()
        
        # Start satellite export
        task = start_satellite_export(collection, region)
        
        # Generate weather data
        weather_df = generate_weather_data()
        
        # Success summary
        print("\n" + "=" * 60)
        print("✅ DATA COLLECTION PIPELINE COMPLETED!")
        print("=" * 60)
        
        print(f"\n🚀 Satellite export task: {task.id}")
        print("📱 Check Google Drive folder: AgriGuard_Data")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        print("💡 Troubleshooting:")
        print("  - Verify Google Earth Engine authentication")
        print("  - Check if Google Drive access")

if __name__ == "__main__":
    main()
