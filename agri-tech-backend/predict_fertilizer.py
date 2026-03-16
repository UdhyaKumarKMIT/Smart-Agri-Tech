# predict_fertilizer.py
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import datetime
import threading
from fertilizer_report_generator import report_generator  # Import the report generator

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "fertilizer-recommendation")

class FertilizerPredictor:
    def __init__(self):
        self.model = None
        self.encoders = None
        self.prediction_history = []  # Store prediction history
        self.max_history = 100  # Maximum predictions to keep in memory
        self.report_generation_interval = 5  # Generate report every N predictions
        self.prediction_count = 0
        self.soil_types = ['Sandy Loam', 'Loamy', 'Sand', 'Clay Loam', 'Clay', 'Sandy', 'Loamy Sand',
                           'Loam', 'Red Clay Loam', 'Red Loam', 'Silty Loam', 'Alluvial', 'Black Soil']
        self.crop_types = ['Arhar/Tur', 'Bajra', 'Barley', 'Coriander', 'Cotton (Lint)', 'Cowpea (Lobia)',
                          'Dry Chillies', 'Garlic', 'Ginger', 'Gram (Chickpea)', 'Groundnut', 'Jowar',
                          'Linseed (Flax)', 'Maize (Grain)', 'Maize (Fodder)', 'Masoor (Red Lentil)',
                          'Moong (Green Gram)', 'Onion', 'Peas & Beans (Pulses)', 'Potato',
                          'Ragi (Finger Millet)', 'Rapeseed & Mustard', 'Rice', 'Safflower',
                          'Sugarcane', 'Sunflower', 'Turmeric', 'Urad (Black Gram)', 'Urad Bean', 'Wheat']
        self.fertilizers = ['MOP', '20-40-20', 'DAP', 'Urea', '40-20-20', '20-20-20', '15-15-15', '30-15-15',
                           'Compound NPK (0-20-20)', '0-20-20', '60-40-60', '40-30-40', '20-10-20',
                           'Organic compost', '120-60-60', '0-40-40', '20-30-20', '30-10-20', '25-10-15',
                           'SSP', '20-15-15', '30-15-20', '30-20-20', '20-10-10', '100-50-50']
        self.load_models()
    
    def load_models(self):
        """Load the trained model and encoders"""
        try:
            # Load model
            model_path = os.path.join(MODELS_DIR, 'stacked_ensemble.pkl')
            with open(model_path, 'rb') as file:
                self.model = pickle.load(file)
            
            # Load encoders
            encoder_path = os.path.join(MODELS_DIR, 'label_encoders.pkl')
            with open(encoder_path, 'rb') as file:
                self.encoders = pickle.load(file)
            
            print("✅ Fertilizer model loaded successfully!")
            return True
        except FileNotFoundError as e:
            print(f"❌ Error loading models: {e}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return False
    
    def generate_report_async(self):
        """Generate report in background thread"""
        try:
            print("📊 Auto-generating fertilizer report...")
            result = report_generator.generate_html_report()
            print(f"✅ Report generated: {result['latest_report']}")
        except Exception as e:
            print(f"❌ Error generating auto-report: {e}")
    
    def save_prediction_to_history(self, input_data, prediction_result):
        """Save prediction to history for future report generation"""
        try:
            # Create a record of the prediction
            record = {
                'timestamp': datetime.datetime.now().isoformat(),
                'input': input_data.copy(),
                'prediction': prediction_result.get('recommendation'),
                'alternatives': prediction_result.get('alternatives', [])
            }
            
            # Add to history
            self.prediction_history.append(record)
            
            # Trim history if too long
            if len(self.prediction_history) > self.max_history:
                self.prediction_history = self.prediction_history[-self.max_history:]
            
            # Save to CSV for persistence
            self.save_history_to_csv()
            
        except Exception as e:
            print(f"❌ Error saving prediction to history: {e}")
    
    def save_history_to_csv(self):
        """Save prediction history to CSV file"""
        try:
            history_dir = os.path.join(MODELS_DIR, "prediction_history")
            os.makedirs(history_dir, exist_ok=True)
            
            # Convert to DataFrame and save
            df = pd.DataFrame(self.prediction_history)
            if not df.empty:
                # Expand alternatives column
                df['alternatives_str'] = df['alternatives'].apply(
                    lambda x: '; '.join([f"{a['fertilizer']}({a['probability']}%)" for a in x]) if x else ''
                )
                
                # Expand input dictionary
                input_df = pd.json_normalize(df['input'])
                result_df = pd.concat([
                    df[['timestamp', 'prediction', 'alternatives_str']].reset_index(drop=True),
                    input_df
                ], axis=1)
                
                # Save to CSV
                csv_path = os.path.join(history_dir, 'prediction_history.csv')
                result_df.to_csv(csv_path, index=False)
                
        except Exception as e:
            print(f"❌ Error saving history to CSV: {e}")
    
    def predict(self, soil_type, crop_type, nitrogen, phosphorus, potassium,
                temperature, humidity, soil_moisture, auto_generate_report=True):
        """Make fertilizer prediction with optional auto-report generation"""
        if self.model is None or self.encoders is None:
            return {"error": "Model not loaded properly"}, 500
        
        try:
            # Convert inputs to float
            inputs = {
                'soil_type': soil_type,
                'crop_type': crop_type,
                'nitrogen': float(nitrogen),
                'phosphorus': float(phosphorus),
                'potassium': float(potassium),
                'temperature': float(temperature),
                'humidity': float(humidity),
                'soil_moisture': float(soil_moisture)
            }
            
            # Encode categorical variables
            soil_encoded = self.encoders['soil_encoder'].transform([soil_type])[0]
            crop_encoded = self.encoders['crop_encoder'].transform([crop_type])[0]
            
            # Create feature array
            features = np.array([[
                soil_encoded, crop_encoded,
                inputs['nitrogen'], inputs['phosphorus'], inputs['potassium'],
                inputs['temperature'], inputs['humidity'], inputs['soil_moisture']
            ]])
            
            # Make prediction
            prediction_encoded = self.model.predict(features)[0]
            fertilizer = self.encoders['fert_encoder'].inverse_transform([prediction_encoded])[0]
            
            # Get probabilities
            probabilities = self.model.predict_proba(features)[0]
            
            # Get top 3 predictions
            top_3_idx = np.argsort(probabilities)[-3:][::-1]
            top_3_fertilizers = self.encoders['fert_encoder'].inverse_transform(top_3_idx)
            top_3_probs = probabilities[top_3_idx] * 100
            
            # Prepare alternatives
            alternatives = []
            for fert, prob in zip(top_3_fertilizers, top_3_probs):
                alternatives.append({
                    'fertilizer': fert,
                    'probability': round(prob, 2)
                })
            
            # Prepare result
            result = {
                'success': True,
                'recommendation': fertilizer,
                'alternatives': alternatives,
                'input_summary': {
                    'soil_type': soil_type,
                    'crop_type': crop_type,
                    'nitrogen': inputs['nitrogen'],
                    'phosphorus': inputs['phosphorus'],
                    'potassium': inputs['potassium'],
                    'temperature': inputs['temperature'],
                    'humidity': inputs['humidity'],
                    'soil_moisture': inputs['soil_moisture']
                }
            }
            
            # Save to history
            self.save_prediction_to_history(inputs, result)
            
            # Auto-generate report based on conditions
            if auto_generate_report:
                self.prediction_count += 1
                
                # Generate report:
                # 1. On every 10th prediction
                # 2. If history has enough data (every 5 predictions)
                # 3. Or if manually triggered
                should_generate = (
                    self.prediction_count % self.report_generation_interval == 0 or
                    len(self.prediction_history) >= self.report_generation_interval
                )
                
                if should_generate:
                    # Generate report in background thread to not block response
                    thread = threading.Thread(target=self.generate_report_async)
                    thread.daemon = True
                    thread.start()
                    
                    # Add report info to result
                    result['report_generated'] = True
                    result['report_url'] = '/fertilizer-report/latest'
            
            return result
            
        except ValueError as e:
            return {"error": f"Invalid number format: {str(e)}"}, 400
        except Exception as e:
            return {"error": str(e)}, 500
    
    def get_metadata(self):
        """Return metadata about the model"""
        return {
            'soil_types': self.soil_types,
            'crop_types': self.crop_types,
            'fertilizers': self.fertilizers,
            'model_loaded': self.model is not None,
            'total_predictions': len(self.prediction_history),
            'prediction_count': self.prediction_count,
            'report_generation_interval': self.report_generation_interval
        }
    
    def force_generate_report(self):
        """Force generate a report immediately"""
        try:
            result = report_generator.generate_html_report()
            return {
                'success': True,
                'message': 'Report generated successfully',
                'report_path': result['latest_report']
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

# Create global instance
fertilizer_predictor = FertilizerPredictor()