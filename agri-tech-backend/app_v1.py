from flask import Flask, jsonify, request
from flask_cors import CORS
from crop_yield import CropYieldPredictor
import pandas as pd

app = Flask(__name__)
CORS(app)

# Initialize the predictor
predictor = CropYieldPredictor(
    models_path='crop-yield-models',
    data_path='crop-yield-data'
)

@app.route('/api/districts', methods=['GET'])
def get_districts():
    """Get list of all districts"""
    try:
        districts = predictor.get_district_list()
        return jsonify(districts)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/crops/<district>', methods=['GET'])
def get_crops(district):
    """Get crops for a specific district"""
    try:
        crops = predictor.get_crops_for_district(district)
        return jsonify(crops)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/district-insights/<district>', methods=['GET'])
def get_district_insights(district):
    """Get insights for a specific district"""
    try:
        insights = predictor.get_district_insights(district)
        if insights:
            return jsonify(insights)
        else:
            return jsonify({'error': 'District not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Predict yield for district and crop using the trained Stacking model"""
    try:
        data = request.json
        district = data.get('district')
        crop = data.get('crop')
        
        if not district or not crop:
            return jsonify({
                'success': False,
                'error': 'District and crop are required'
            }), 400
        
        # Use the predictor's predict method which uses the actual Stacking model
        result = predictor.predict(district, crop)
        
        if result['success']:
            return jsonify(result)
        else:
            return jsonify(result), 404
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)