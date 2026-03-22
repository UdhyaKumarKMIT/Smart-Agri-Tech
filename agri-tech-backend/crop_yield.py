# crop_yield.py
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

class CropYieldPredictor:
    def __init__(self, models_path='crop-yield-models', data_path='crop-yield-data'):
        """
        Initialize the Crop Yield Predictor with trained models and data
        
        Args:
            models_path: Path to folder containing trained models
            data_path: Path to folder containing dataset
        """
        self.models_path = models_path
        self.data_path = data_path
        self.models = {}
        self.scaler = None
        self.district_encoder = None
        self.crop_encoder = None
        self.feature_columns = None
        self.ml_df = None
        self.stacking_model = None
        
        # Load all required artifacts
        self.load_artifacts()
        
    def load_artifacts(self):
        """Load all trained models, encoders, and data"""
        try:
            # Load stacking model (this is your main model)
            stacking_path = os.path.join(self.models_path, 'stacking_model.pkl')
            if os.path.exists(stacking_path):
                self.stacking_model = joblib.load(stacking_path)
                self.models['stacking'] = self.stacking_model
                print("✓ Stacking model loaded")
            
            # Load scaler (CRITICAL for correct predictions)
            scaler_path = os.path.join(self.models_path, 'scaler.pkl')
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                print("✓ Scaler loaded")
            
            # Load encoders
            district_encoder_path = os.path.join(self.models_path, 'district_encoder.pkl')
            if os.path.exists(district_encoder_path):
                self.district_encoder = joblib.load(district_encoder_path)
                print("✓ District encoder loaded")
            
            crop_encoder_path = os.path.join(self.models_path, 'crop_encoder.pkl')
            if os.path.exists(crop_encoder_path):
                self.crop_encoder = joblib.load(crop_encoder_path)
                print("✓ Crop encoder loaded")
            
            # Load feature columns (the EXACT columns used in training)
            feature_columns_path = os.path.join(self.models_path, 'feature_columns.pkl')
            if os.path.exists(feature_columns_path):
                self.feature_columns = joblib.load(feature_columns_path)
                print(f"✓ Feature columns loaded: {len(self.feature_columns)} features")
            
            # Load ml_df dataset
            ml_df_path = os.path.join(self.data_path, 'ml_df.csv')
            if os.path.exists(ml_df_path):
                self.ml_df = pd.read_csv(ml_df_path)
                print(f"✓ Dataset loaded: {self.ml_df.shape}")
            
            print("\n✅ All artifacts loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading artifacts: {str(e)}")
            raise
    
    def get_district_list(self):
        """Get list of all districts in the dataset"""
        if self.ml_df is not None:
            return sorted(self.ml_df['District'].unique())
        return []
    
    def get_crop_list(self):
        """Get list of all crops in the dataset"""
        if self.ml_df is not None:
            return sorted(self.ml_df['Crop'].unique())
        return []
    
    def get_crops_for_district(self, district):
        """Get crops for a specific district"""
        if self.ml_df is not None:
            district_data = self.ml_df[self.ml_df['District'] == district]
            return sorted(district_data['Crop'].unique())
        return []
    
    def get_district_insights(self, district):
        """Get insights for a specific district"""
        if self.ml_df is None:
            return None
        
        district_data = self.ml_df[self.ml_df['District'] == district]
        
        if len(district_data) == 0:
            return None
        
        # Calculate insights
        total_area = district_data['Area'].sum()
        total_production = district_data['Production'].sum()
        
        # Crop with highest production
        top_crop_idx = district_data['Production'].idxmax()
        top_crop = district_data.loc[top_crop_idx, 'Crop']
        top_production = district_data.loc[top_crop_idx, 'Production']
        
        # Crop with lowest production
        bottom_crop_idx = district_data['Production'].idxmin()
        bottom_crop = district_data.loc[bottom_crop_idx, 'Crop']
        bottom_production = district_data.loc[bottom_crop_idx, 'Production']
        
        # Crop with highest yield
        top_yield_idx = district_data['Yield'].idxmax()
        top_yield_crop = district_data.loc[top_yield_idx, 'Crop']
        top_yield_value = district_data.loc[top_yield_idx, 'Yield']
        
        # Crop with lowest yield
        bottom_yield_idx = district_data['Yield'].idxmin()
        bottom_yield_crop = district_data.loc[bottom_yield_idx, 'Crop']
        bottom_yield_value = district_data.loc[bottom_yield_idx, 'Yield']
        
        # Average yield across all crops
        avg_yield = district_data['Yield'].mean()
        
        # Crop statistics
        crop_stats = []
        for crop in district_data['Crop'].unique():
            crop_data = district_data[district_data['Crop'] == crop]
            crop_stats.append({
                'crop': crop,
                'area': float(crop_data['Area'].sum()),
                'production': float(crop_data['Production'].sum()),
                'yield': float(crop_data['Yield'].mean())
            })
        
        # Sort by production
        crop_stats.sort(key=lambda x: x['production'], reverse=True)
        
        return {
            'district': district,
            'total_area': float(total_area),
            'total_production': float(total_production),
            'avg_yield': float(avg_yield),
            'top_crop': {
                'name': top_crop,
                'production': float(top_production)
            },
            'bottom_crop': {
                'name': bottom_crop,
                'production': float(bottom_production)
            },
            'best_yield_crop': {
                'name': top_yield_crop,
                'yield': float(top_yield_value)
            },
            'worst_yield_crop': {
                'name': bottom_yield_crop,
                'yield': float(bottom_yield_value)
            },
            'crop_stats': crop_stats,
            'total_crops': len(district_data['Crop'].unique())
        }
    
    def get_district_crop_data(self, district, crop):
        """
        Fetch record for given district and crop from dataset
        
        Args:
            district: District name
            crop: Crop name
            
        Returns:
            Dictionary containing the record data or None if not found
        """
        if self.ml_df is None:
            return None
        
        # Filter dataset for the given district and crop
        record = self.ml_df[(self.ml_df['District'] == district) & 
                            (self.ml_df['Crop'] == crop)]
        
        if len(record) == 0:
            return None
        
        # Get the first matching record
        record = record.iloc[0].to_dict()
        return record
    
    def prepare_features(self, record):
        """
        Prepare features from record for prediction
        
        IMPORTANT: This uses the features DIRECTLY from the dataset
        No need to compute engineered features as they're already in the dataset
        
        Args:
            record: Dictionary containing district-crop data
            
        Returns:
            DataFrame with all required features in the EXACT order used in training
        """
        
        # All features are already in the record from ml_df.csv
        # We just need to extract them in the correct order
        
        feature_dict = {}
        
        # Use the feature columns from training if available
        if self.feature_columns is not None:
            for col in self.feature_columns:
                if col in record:
                    feature_dict[col] = record[col]
                else:
                    print(f"Warning: Feature {col} not found in record")
                    feature_dict[col] = 0
        else:
            # If feature_columns not available, use all expected features
            expected_features = [
                'NDVI', 'NDWI', 'EVI', 'SAVI', 'SMAI', 'Precipitation', 'CQI',
                'NDVI_NDWI_Ratio', 'EVI_SAVI_Ratio', 'Vegetation_Index',
                'Water_Veg_Interaction', 'Precip_NDVI', 'Quality_NDVI', 'Quality_Precip',
                'District_Encoded', 'Crop_Encoded', 'Area', 'Log_Area'
            ]
            
            for col in expected_features:
                if col in record:
                    feature_dict[col] = record[col]
                else:
                    print(f"Warning: Feature {col} not found in record")
                    feature_dict[col] = 0
        
        # Create DataFrame with single row
        features_df = pd.DataFrame([feature_dict])
        
        # Ensure columns are in the same order as training
        if self.feature_columns is not None:
            features_df = features_df[self.feature_columns]
        
        return features_df
    
    def predict(self, district, crop):
        """
        Predict yield for given district and crop using the trained Stacking model
        
        Args:
            district: District name
            crop: Crop name
            
        Returns:
            Dictionary with prediction results
        """
        # Fetch record from dataset
        record = self.get_district_crop_data(district, crop)
        
        if record is None:
            return {
                'success': False,
                'error': f"No data found for {district} - {crop}"
            }
        
        try:
            # Check if we have all required components
            if self.stacking_model is None:
                return {
                    'success': False,
                    'error': "Stacking model not loaded"
                }
            
            if self.scaler is None:
                return {
                    'success': False,
                    'error': "Scaler not loaded"
                }
            
            if self.feature_columns is None:
                return {
                    'success': False,
                    'error': "Feature columns not loaded"
                }
            
            # Prepare features (using the EXACT features from the dataset)
            features_df = self.prepare_features(record)
            
            # Scale features using the same scaler from training
            features_scaled = self.scaler.transform(features_df)
            
            # Make prediction using the Stacking model
            predicted_yield = self.stacking_model.predict(features_scaled)[0]
            
            # Ensure non-negative prediction
            predicted_yield = max(0, predicted_yield)
            
            # Calculate predicted production
            predicted_production = predicted_yield * record['Area']
            
            # Prepare response
           # Prepare response
            result = {
                'success': True,
                'district': district,
                'crop': crop,
                'area': round(float(record['Area']), 4),
                'predicted_yield': round(float(predicted_yield), 4),
                'predicted_production': round(float(predicted_production), 4),
                'model_used': 'stacking_ensemble',
                'features': {
                    'NDVI': round(float(record['NDVI']), 4),
                    'NDWI': round(float(record['NDWI']), 4),
                    'EVI': round(float(record['EVI']), 4),
                    'SAVI': round(float(record['SAVI']), 4),
                    'SMAI': round(float(record['SMAI']), 4),
                    'Precipitation': round(float(record['Precipitation']), 4),
                    'CQI': round(float(record['CQI']), 4)
                }
            }

            return result
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': f"Prediction error: {str(e)}"
            }


# Utility function to create the necessary folder structure
def create_folder_structure():
    """Create the required folder structure if it doesn't exist"""
    folders = ['crop-yield-models', 'crop-yield-data']
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Created folder: {folder}")
    
    print("✅ Folder structure ready")


# If run directly, test the predictor
if __name__ == "__main__":
    # Create folder structure
    create_folder_structure()
    
    # Initialize predictor
    predictor = CropYieldPredictor()
    
    # Test with first district and crop
    districts = predictor.get_district_list()
    
    if districts:
        test_district = districts[0]
        crops = predictor.get_crops_for_district(test_district)
        
        if crops:
            test_crop = crops[0]
            
            print(f"\n🔍 Testing prediction for {test_district} - {test_crop}")
            result = predictor.predict(test_district, test_crop)
            
            if result['success']:
                print(f"\n✅ Prediction successful!")
                print(f"   District: {result['district']}")
                print(f"   Crop: {result['crop']}")
                print(f"   Area: {result['area']:.2f} hectares")
                print(f"   Predicted Yield: {result['predicted_yield']:.2f} kg/ha")
                print(f"   Predicted Production: {result['predicted_production']:.2f} kg")
                
                # Also get actual yield from dataset for comparison
                record = predictor.get_district_crop_data(test_district, test_crop)
                if record and 'Yield' in record:
                    print(f"   Actual Yield (from dataset): {record['Yield']:.2f} kg/ha")
            else:
                print(f"\n❌ Prediction failed: {result['error']}")