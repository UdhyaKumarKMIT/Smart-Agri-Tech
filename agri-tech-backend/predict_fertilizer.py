# predict_fertilizer.py
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import datetime
import threading
from fertilizer_report_generator import report_generator

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "fertilizer-recommendation")


class FertilizerPredictor:
    def __init__(self):
        self.model            = None
        self.encoders         = None
        self.prediction_history             = []
        self.max_history                    = 100
        self.report_generation_interval     = 5
        self.prediction_count               = 0
        self.soil_types = [
            'Sandy Loam', 'Loamy', 'Sand', 'Clay Loam', 'Clay', 'Sandy',
            'Loamy Sand', 'Loam', 'Red Clay Loam', 'Red Loam', 'Silty Loam',
            'Alluvial', 'Black Soil'
        ]
        self.crop_types = [
            'Arhar/Tur', 'Bajra', 'Barley', 'Coriander', 'Cotton (Lint)',
            'Cowpea (Lobia)', 'Dry Chillies', 'Garlic', 'Ginger',
            'Gram (Chickpea)', 'Groundnut', 'Jowar', 'Linseed (Flax)',
            'Maize (Grain)', 'Maize (Fodder)', 'Masoor (Red Lentil)',
            'Moong (Green Gram)', 'Onion', 'Peas & Beans (Pulses)', 'Potato',
            'Ragi (Finger Millet)', 'Rapeseed & Mustard', 'Rice', 'Safflower',
            'Sugarcane', 'Sunflower', 'Turmeric', 'Urad (Black Gram)',
            'Urad Bean', 'Wheat'
        ]
        self.fertilizers = [
            'MOP', '20-40-20', 'DAP', 'Urea', '40-20-20', '20-20-20',
            '15-15-15', '30-15-15', 'Compound NPK (0-20-20)', '0-20-20',
            '60-40-60', '40-30-40', '20-10-20', 'Organic compost',
            '120-60-60', '0-40-40', '20-30-20', '30-10-20', '25-10-15',
            'SSP', '20-15-15', '30-15-20', '30-20-20', '20-10-10', '100-50-50'
        ]
        self.load_models()

    # ----------------------------------------------------------
    # MODEL LOADING
    # ----------------------------------------------------------

    def load_models(self):
        try:
            model_path = os.path.join(MODELS_DIR, 'stacked_ensemble.pkl')
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)

            encoder_path = os.path.join(MODELS_DIR, 'label_encoders.pkl')
            with open(encoder_path, 'rb') as f:
                self.encoders = pickle.load(f)

            print("✅ Fertilizer model loaded successfully!")
            return True
        except FileNotFoundError as e:
            print(f"❌ Error loading models: {e}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return False

    # ----------------------------------------------------------
    # FIX 1 + FIX 2 — pass input_data and prediction_result
    # ----------------------------------------------------------

    def generate_report_async(self, input_data, prediction_result):
        """
        Generate report in background thread.
        Receives the actual input and result so local XAI plots
        (SHAP local, LIME, breakdown, counterfactual) reflect
        the real prediction — not a fallback training row.
        """
        try:
            print("📊 Generating fertilizer report in background...")
            result = report_generator.generate_html_report(
                input_data        = input_data,        # ← FIX 1: pass input
                prediction_result = prediction_result  # ← FIX 2: pass result
            )
            print(f"✅ Report generated: {result.get('latest_report', '')}")
        except Exception as e:
            print(f"❌ Error generating report: {e}")
            import traceback; traceback.print_exc()

    # ----------------------------------------------------------
    # HISTORY
    # ----------------------------------------------------------

    def save_prediction_to_history(self, input_data, prediction_result):
        try:
            record = {
                'timestamp'   : datetime.datetime.now().isoformat(),
                'input'       : input_data.copy(),
                'prediction'  : prediction_result.get('recommendation'),
                'alternatives': prediction_result.get('alternatives', [])
            }
            self.prediction_history.append(record)
            if len(self.prediction_history) > self.max_history:
                self.prediction_history = self.prediction_history[-self.max_history:]
            self.save_history_to_csv()
        except Exception as e:
            print(f"❌ Error saving prediction to history: {e}")

    def save_history_to_csv(self):
        try:
            history_dir = os.path.join(MODELS_DIR, "prediction_history")
            os.makedirs(history_dir, exist_ok=True)

            df = pd.DataFrame(self.prediction_history)
            if not df.empty:
                df['alternatives_str'] = df['alternatives'].apply(
                    lambda x: '; '.join(
                        [f"{a['fertilizer']}({a['probability']}%)" for a in x]
                    ) if x else ''
                )
                input_df   = pd.json_normalize(df['input'])
                result_df  = pd.concat([
                    df[['timestamp', 'prediction',
                        'alternatives_str']].reset_index(drop=True),
                    input_df
                ], axis=1)
                csv_path = os.path.join(history_dir, 'prediction_history.csv')
                result_df.to_csv(csv_path, index=False)
        except Exception as e:
            print(f"❌ Error saving history to CSV: {e}")

    # ----------------------------------------------------------
    # PREDICT
    # ----------------------------------------------------------

    def predict(self, soil_type, crop_type, nitrogen, phosphorus, potassium,
                temperature, humidity, soil_moisture, auto_generate_report=True):

        if self.model is None or self.encoders is None:
            return {"error": "Model not loaded properly"}, 500

        try:
            inputs = {
                'soil_type'    : soil_type,
                'crop_type'    : crop_type,
                'nitrogen'     : float(nitrogen),
                'phosphorus'   : float(phosphorus),
                'potassium'    : float(potassium),
                'temperature'  : float(temperature),
                'humidity'     : float(humidity),
                'soil_moisture': float(soil_moisture)
            }

            soil_encoded = self.encoders['soil_encoder'].transform([soil_type])[0]
            crop_encoded = self.encoders['crop_encoder'].transform([crop_type])[0]

            features = np.array([[
                soil_encoded, crop_encoded,
                inputs['nitrogen'], inputs['phosphorus'], inputs['potassium'],
                inputs['temperature'], inputs['humidity'], inputs['soil_moisture']
            ]])

            prediction_encoded = self.model.predict(features)[0]
            fertilizer         = self.encoders['fert_encoder'].inverse_transform(
                                     [prediction_encoded])[0]

            probabilities  = self.model.predict_proba(features)[0]
            top_3_idx      = np.argsort(probabilities)[-3:][::-1]
            top_3_ferts    = self.encoders['fert_encoder'].inverse_transform(top_3_idx)
            top_3_probs    = probabilities[top_3_idx] * 100

            alternatives = [
                {'fertilizer': f, 'probability': round(p, 2)}
                for f, p in zip(top_3_ferts, top_3_probs)
            ]

            result = {
                'success'       : True,
                'recommendation': fertilizer,

                # ── FIX 3: add 'fertilizer' key so report generator finds it ──
                'fertilizer'    : fertilizer,

                'alternatives'  : alternatives,
                'input_summary' : {
                    'soil_type'    : soil_type,
                    'crop_type'    : crop_type,
                    'nitrogen'     : inputs['nitrogen'],
                    'phosphorus'   : inputs['phosphorus'],
                    'potassium'    : inputs['potassium'],
                    'temperature'  : inputs['temperature'],
                    'humidity'     : inputs['humidity'],
                    'soil_moisture': inputs['soil_moisture']
                }
            }

            self.save_prediction_to_history(inputs, result)

            if auto_generate_report:
                self.prediction_count += 1

                should_generate = (
                    self.prediction_count % self.report_generation_interval == 0 or
                    len(self.prediction_history) >= self.report_generation_interval
                )

                if should_generate:
                    # ── FIX 1+2: pass actual input and result to the thread ──
                    thread = threading.Thread(
                        target = self.generate_report_async,
                        args   = (inputs, result),   # ← pass both here
                        daemon = True
                    )
                    thread.start()

                    result['report_generated'] = True
                    result['report_url']       = '/fertilizer-report/latest'

            return result

        except ValueError as e:
            return {"error": f"Invalid number format: {str(e)}"}, 400
        except Exception as e:
            return {"error": str(e)}, 500

    # ----------------------------------------------------------
    # METADATA & FORCE REPORT
    # ----------------------------------------------------------

    def get_metadata(self):
        return {
            'soil_types'                : self.soil_types,
            'crop_types'                : self.crop_types,
            'fertilizers'               : self.fertilizers,
            'model_loaded'              : self.model is not None,
            'total_predictions'         : len(self.prediction_history),
            'prediction_count'          : self.prediction_count,
            'report_generation_interval': self.report_generation_interval
        }

    def force_generate_report(self, input_data=None, prediction_result=None):
        """Force generate a report immediately with optional real data."""
        try:
            result = report_generator.generate_html_report(
                input_data        = input_data,
                prediction_result = prediction_result
            )
            return {
                'success'    : True,
                'message'    : 'Report generated successfully',
                'report_path': result.get('latest_report', '')
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}


# Create global instance
fertilizer_predictor = FertilizerPredictor()