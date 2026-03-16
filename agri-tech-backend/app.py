# app.py (updated with fertilizer APIs)
from flask import Flask, request, jsonify, send_file, abort
from flask_cors import CORS
from predict_crop import predict_crop
import os
from flask import Flask, send_from_directory, abort, redirect, url_for

# Import fertilizer modules
from predict_fertilizer import fertilizer_predictor
from fertilizer_report_generator import report_generator

app = Flask(__name__)
CORS(app)

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Crop directories
CROP_REPORTS_DIR = os.path.join(BASE_DIR, "report_crop")
CROP_MODELS_DIR = os.path.join(BASE_DIR, "crop-recommendation", "crop-recommendation-models")
CROP_DATA_DIR = os.path.join(BASE_DIR, "crop-recommendation", "crop-recommendation-processed_data")

# Fertilizer directories
FERTILIZER_REPORTS_DIR = os.path.join(BASE_DIR, "report_fertilizer")

# ============================================
# CROP RECOMMENDATION APIs (Your existing code)
# ============================================

@app.route("/predict-crop", methods=["POST"])
def crop_prediction():
    """
    Endpoint for crop prediction
    Expects JSON with: nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall
    """
    try:
        data = request.json
        result = predict_crop(data)
        
        if "error" in result:
            return jsonify(result), 400
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict-crop-no-report", methods=["POST"])
def crop_prediction_no_report():
    """
    Endpoint for crop prediction without generating report
    """
    try:
        data = request.json
        result = predict_crop(data, generate_report=False)
        
        if "error" in result:
            return jsonify(result), 400
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/report/<report_id>")
def view_report(report_id):
    """
    View a specific crop report by ID
    """
    try:
        # Sanitize report_id to prevent directory traversal
        if '..' in report_id or '/' in report_id or '\\' in report_id:
            abort(400, "Invalid report ID")
        
        report_path = os.path.join(CROP_REPORTS_DIR, f"{report_id}.html")
        
        if not os.path.exists(report_path):
            # Try without .html extension
            report_path = os.path.join(CROP_REPORTS_DIR, report_id)
            if not os.path.exists(report_path):
                abort(404, "Report not found")
        
        return send_file(report_path)
    
    except Exception as e:
        abort(500, str(e))


# Serve static files from crop reports directory
@app.route('/crop-reports/<path:filename>')
def serve_crop_report_file(filename):
    """Serve any file from crop reports directory"""
    return send_from_directory(CROP_REPORTS_DIR, filename)

@app.route("/report/latest")
def view_latest_report():
    """Redirect to the latest crop report HTML"""
    latest_path = os.path.join(CROP_REPORTS_DIR, "latest_report.html")
    
    if not os.path.exists(latest_path):
        abort(404, "No crop reports found")
    
    # Redirect to the static file endpoint
    return redirect(url_for('serve_crop_report_file', filename='latest_report.html'))


@app.route("/reports/list")
def list_reports():
    """
    List all available crop reports
    """
    try:
        reports = []
        for filename in os.listdir(CROP_REPORTS_DIR):
            if filename.endswith('.html') and filename != 'latest_report.html':
                # Get report metadata
                report_path = os.path.join(CROP_REPORTS_DIR, filename)
                modified_time = os.path.getmtime(report_path)
                
                reports.append({
                    "id": filename.replace('.html', ''),
                    "filename": filename,
                    "created": modified_time,
                    "url": f"/report/{filename.replace('.html', '')}"
                })
        
        # Sort by creation time, newest first
        reports.sort(key=lambda x: x['created'], reverse=True)
        
        return jsonify({
            "total": len(reports),
            "reports": reports
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/report/image/<path:image_path>")
def serve_report_image(image_path):
    """
    Serve images for crop reports
    """
    try:
        # Security check
        if '..' in image_path:
            abort(400, "Invalid image path")
        
        image_full_path = os.path.join(CROP_REPORTS_DIR, "images", image_path)
        
        if not os.path.exists(image_full_path):
            abort(404, "Image not found")
        
        return send_file(image_full_path)
    
    except Exception as e:
        abort(500, str(e))

# ============================================
# FERTILIZER RECOMMENDATION APIs (New)
# ============================================

@app.route('/api/fertilizer/predict', methods=['POST'])
def predict_fertilizer():
    """
    Predict fertilizer based on soil and crop parameters
    Expected JSON:
    {
        "soil_type": "Loamy",
        "crop_type": "Wheat",
        "nitrogen": 50,
        "phosphorus": 30,
        "potassium": 40,
        "temperature": 25,
        "humidity": 65,
        "soil_moisture": 55
    }
    """
    try:
        data = request.json
        
        # Validate required fields
        required_fields = ['soil_type', 'crop_type', 'nitrogen', 'phosphorus', 
                          'potassium', 'temperature', 'humidity', 'soil_moisture']
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                "error": f"Missing required fields: {', '.join(missing_fields)}"
            }), 400
        
        # Make prediction
        result = fertilizer_predictor.predict(
            soil_type=data['soil_type'],
            crop_type=data['crop_type'],
            nitrogen=data['nitrogen'],
            phosphorus=data['phosphorus'],
            potassium=data['potassium'],
            temperature=data['temperature'],
            humidity=data['humidity'],
            soil_moisture=data['soil_moisture']
        )
        
        # Check if result is a tuple with error status
        if isinstance(result, tuple) and len(result) == 2 and result[1] >= 400:
            return jsonify(result[0]), result[1]
        
        # Generate report synchronously (waits for completion)
        try:
            print("📊 Generating fertilizer report after prediction...")
            report_result = report_generator.generate_html_report()
            print(f"✅ Report generated: {report_result['latest_report']}")
            
            # Add report info to result
            if isinstance(result, dict):
                result['report_generated'] = True
                result['report_url'] = '/fertilizer-report/latest'
                result['report_path'] = report_result['latest_report']
                
        except Exception as e:
            print(f"❌ Error generating report: {e}")
            if isinstance(result, dict):
                result['report_generated'] = False
                result['report_error'] = str(e)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/fertilizer/metadata', methods=['GET'])
def fertilizer_metadata():
    """Get metadata about the fertilizer model"""
    return jsonify(fertilizer_predictor.get_metadata())


@app.route('/api/fertilizer/soil-types', methods=['GET'])
def get_soil_types():
    """Get list of available soil types"""
    return jsonify({
        "soil_types": fertilizer_predictor.soil_types
    })


@app.route('/api/fertilizer/crop-types', methods=['GET'])
def get_crop_types():
    """Get list of available crop types"""
    return jsonify({
        "crop_types": fertilizer_predictor.crop_types
    })


@app.route('/api/fertilizer/fertilizers', methods=['GET'])
def get_fertilizers():
    """Get list of available fertilizers"""
    return jsonify({
        "fertilizers": fertilizer_predictor.fertilizers
    })


@app.route('/api/fertilizer/generate-report', methods=['POST'])
def generate_fertilizer_report():
    """Generate a comprehensive report with all fertilizer model plots"""
    try:
        result = report_generator.generate_html_report()
        return jsonify({
            "success": True,
            "message": "Fertilizer report generated successfully",
            "report_path": result['latest_report'],
            "timestamped_report": result['timestamped_report'],
            "images_generated": len(result['images'])
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/fertilizer/get-report', methods=['GET'])
def get_fertilizer_report():
    """Get the latest generated fertilizer report"""
    try:
        report_path = os.path.join(FERTILIZER_REPORTS_DIR, "latest_report.html")
        if os.path.exists(report_path):
            return send_file(report_path, as_attachment=False)
        else:
            # Generate report if it doesn't exist
            result = report_generator.generate_html_report()
            report_path = os.path.join(FERTILIZER_REPORTS_DIR, "latest_report.html")
            return send_file(report_path, as_attachment=False)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/fertilizer/report-images', methods=['GET'])
def list_fertilizer_report_images():
    """List all generated fertilizer report images"""
    try:
        images_dir = os.path.join(FERTILIZER_REPORTS_DIR, "images")
        if os.path.exists(images_dir):
            images = os.listdir(images_dir)
            return jsonify({
                "images": images,
                "count": len(images)
            })
        else:
            return jsonify({"images": [], "count": 0})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================
# FERTILIZER REPORT VIEWING ENDPOINTS
# ============================================

@app.route("/fertilizer-report/<report_id>")
def view_fertilizer_report(report_id):
    """
    View a specific fertilizer report by ID
    """
    try:
        # Sanitize report_id to prevent directory traversal
        if '..' in report_id or '/' in report_id or '\\' in report_id:
            abort(400, "Invalid report ID")
        
        report_path = os.path.join(FERTILIZER_REPORTS_DIR, f"{report_id}.html")
        
        if not os.path.exists(report_path):
            # Try without .html extension
            report_path = os.path.join(FERTILIZER_REPORTS_DIR, report_id)
            if not os.path.exists(report_path):
                abort(404, "Fertilizer report not found")
        
        return send_file(report_path)
    
    except Exception as e:
        abort(500, str(e))


# Serve static files from fertilizer reports directory
@app.route('/fertilizer-reports/<path:filename>')
def serve_fertilizer_report_file(filename):
    """Serve any file from fertilizer reports directory"""
    return send_from_directory(FERTILIZER_REPORTS_DIR, filename)


@app.route("/fertilizer-report/latest")
def view_latest_fertilizer_report():
    """Redirect to the latest fertilizer report HTML"""
    latest_path = os.path.join(FERTILIZER_REPORTS_DIR, "latest_report.html")
    
    if not os.path.exists(latest_path):
        abort(404, "No fertilizer reports found")
    
    # Redirect to the static file endpoint
    return redirect(url_for('serve_fertilizer_report_file', filename='latest_report.html'))


@app.route("/fertilizer-reports/list")
def list_fertilizer_reports():
    """
    List all available fertilizer reports
    """
    try:
        reports = []
        for filename in os.listdir(FERTILIZER_REPORTS_DIR):
            if filename.endswith('.html') and filename != 'latest_report.html':
                # Get report metadata
                report_path = os.path.join(FERTILIZER_REPORTS_DIR, filename)
                modified_time = os.path.getmtime(report_path)
                
                reports.append({
                    "id": filename.replace('.html', ''),
                    "filename": filename,
                    "created": modified_time,
                    "url": f"/fertilizer-report/{filename.replace('.html', '')}"
                })
        
        # Sort by creation time, newest first
        reports.sort(key=lambda x: x['created'], reverse=True)
        
        return jsonify({
            "total": len(reports),
            "reports": reports
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/fertilizer-report/image/<path:image_path>")
def serve_fertilizer_report_image(image_path):
    """
    Serve images for fertilizer reports
    """
    try:
        # Security check
        if '..' in image_path:
            abort(400, "Invalid image path")
        
        image_full_path = os.path.join(FERTILIZER_REPORTS_DIR, "images", image_path)
        
        if not os.path.exists(image_full_path):
            abort(404, "Image not found")
        
        return send_file(image_full_path)
    
    except Exception as e:
        abort(500, str(e))


# ============================================
# COMBINED APIs
# ============================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint for all services"""
    return jsonify({
        "status": "healthy",
        "timestamp": __import__('datetime').datetime.now().isoformat(),
        "services": {
            "crop_recommendation": {
                "status": "available",
                "endpoints": [
                    "/predict-crop",
                    "/predict-crop-no-report",
                    "/report/latest",
                    "/reports/list"
                ]
            },
            "fertilizer_recommendation": {
                "status": "available" if fertilizer_predictor.model else "model_not_loaded",
                "endpoints": [
                    "/api/fertilizer/predict",
                    "/api/fertilizer/metadata",
                    "/api/fertilizer/generate-report",
                    "/fertilizer-report/latest"
                ],
                "soil_types": len(fertilizer_predictor.soil_types),
                "crop_types": len(fertilizer_predictor.crop_types),
                "fertilizers": len(fertilizer_predictor.fertilizers)
            }
        }
    })


@app.route('/api/recommend-all', methods=['POST'])
def recommend_all():
    """
    Get both crop and fertilizer recommendations
    Expected JSON includes both crop and fertilizer parameters
    """
    try:
        data = request.json
        
        # Get fertilizer recommendation
        fertilizer_result = fertilizer_predictor.predict(
            soil_type=data.get('soil_type', 'Loamy'),
            crop_type=data.get('crop_type', 'Wheat'),
            nitrogen=data.get('nitrogen', 50),
            phosphorus=data.get('phosphorus', 30),
            potassium=data.get('potassium', 40),
            temperature=data.get('temperature', 25),
            humidity=data.get('humidity', 65),
            soil_moisture=data.get('soil_moisture', 55)
        )
        
        # Get crop recommendation
        crop_result = predict_crop(data)
        
        # Handle error cases
        if isinstance(fertilizer_result, tuple) and len(fertilizer_result) == 2:
            fertilizer_data = fertilizer_result[0]
        else:
            fertilizer_data = fertilizer_result
        
        return jsonify({
            "fertilizer_recommendation": fertilizer_data,
            "crop_recommendation": crop_result
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================
# HTML DEMO PAGES
# ============================================

@app.route('/fertilizer-demo', methods=['GET'])
def fertilizer_demo():
    """Simple HTML form for testing fertilizer recommendation"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Fertilizer Recommendation Demo</title>
        <style>
            body { font-family: Arial; max-width: 600px; margin: 50px auto; padding: 20px; background: #f5f5f5; }
            .container { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #2c3e50; text-align: center; }
            label { font-weight: bold; color: #34495e; margin-top: 10px; display: block; }
            input, select { width: 100%; padding: 8px; margin: 5px 0 15px; border: 1px solid #ddd; border-radius: 4px; box-sizing: border-box; }
            button { background: #27ae60; color: white; padding: 12px 20px; border: none; border-radius: 4px; cursor: pointer; width: 100%; font-size: 16px; }
            button:hover { background: #219a52; }
            .result { margin-top: 20px; padding: 20px; background: #f8f9fa; border-radius: 5px; border-left: 4px solid #27ae60; }
            .error { border-left-color: #e74c3c; }
            .nav-links { margin-top: 20px; text-align: center; }
            .nav-links a { color: #3498db; margin: 0 10px; text-decoration: none; }
            .nav-links a:hover { text-decoration: underline; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🌱 Fertilizer Recommendation</h1>
            <form id="fertilizerForm">
                <label>Soil Type:</label>
                <select name="soil_type" id="soil_type" required>
                    <option value="Loamy">Loamy</option>
                    <option value="Clay">Clay</option>
                    <option value="Sandy">Sandy</option>
                    <option value="Sandy Loam">Sandy Loam</option>
                    <option value="Clay Loam">Clay Loam</option>
                </select>
                
                <label>Crop Type:</label>
                <select name="crop_type" id="crop_type" required>
                    <option value="Wheat">Wheat</option>
                    <option value="Rice">Rice</option>
                    <option value="Maize">Maize</option>
                    <option value="Gram (Chickpea)">Gram (Chickpea)</option>
                    <option value="Sugarcane">Sugarcane</option>
                </select>
                
                <label>Nitrogen (N):</label>
                <input type="number" id="nitrogen" value="50" min="0" max="200" step="0.1" required>
                
                <label>Phosphorus (P):</label>
                <input type="number" id="phosphorus" value="30" min="0" max="200" step="0.1" required>
                
                <label>Potassium (K):</label>
                <input type="number" id="potassium" value="40" min="0" max="200" step="0.1" required>
                
                <label>Temperature (°C):</label>
                <input type="number" id="temperature" value="25" min="0" max="50" step="0.1" required>
                
                <label>Humidity (%):</label>
                <input type="number" id="humidity" value="65" min="0" max="100" step="0.1" required>
                
                <label>Soil Moisture (%):</label>
                <input type="number" id="soil_moisture" value="55" min="0" max="100" step="0.1" required>
                
                <button type="submit">Get Recommendation</button>
            </form>
            
            <div class="result" id="result" style="display: none;"></div>
            
            <div class="nav-links">
                <a href="/fertilizer-report/latest">📊 View Latest Report</a>
                <a href="/api/fertilizer/generate-report" target="_blank">📈 Generate New Report</a>
            </div>
        </div>
        
        <script>
            document.getElementById('fertilizerForm').onsubmit = async (e) => {
                e.preventDefault();
                const resultDiv = document.getElementById('result');
                resultDiv.style.display = 'block';
                resultDiv.innerHTML = '<p>Loading...</p>';
                resultDiv.className = 'result';
                
                const data = {
                    soil_type: document.getElementById('soil_type').value,
                    crop_type: document.getElementById('crop_type').value,
                    nitrogen: parseFloat(document.getElementById('nitrogen').value),
                    phosphorus: parseFloat(document.getElementById('phosphorus').value),
                    potassium: parseFloat(document.getElementById('potassium').value),
                    temperature: parseFloat(document.getElementById('temperature').value),
                    humidity: parseFloat(document.getElementById('humidity').value),
                    soil_moisture: parseFloat(document.getElementById('soil_moisture').value)
                };
                
                try {
                    const response = await fetch('/api/fertilizer/predict', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify(data)
                    });
                    
                    const result = await response.json();
                    
                    if (result.error) {
                        resultDiv.innerHTML = '<h3>❌ Error</h3><p>' + result.error + '</p>';
                        resultDiv.className = 'result error';
                    } else {
                        let html = '<h3>✅ Recommendation:</h3>';
                        html += '<p><strong>Fertilizer:</strong> ' + result.recommendation + '</p>';
                        html += '<h4>📊 Top Alternatives:</h4><ul>';
                        result.alternatives.forEach(alt => {
                            html += '<li><strong>' + alt.fertilizer + '</strong>: ' + alt.probability + '% confidence</li>';
                        });
                        html += '</ul>';
                        html += '<h4>📝 Input Summary:</h4>';
                        html += '<p>NPK: ' + result.input_summary.nitrogen + '-' + 
                                result.input_summary.phosphorus + '-' + result.input_summary.potassium + '<br>';
                        html += 'Temperature: ' + result.input_summary.temperature + '°C<br>';
                        html += 'Humidity: ' + result.input_summary.humidity + '%<br>';
                        html += 'Soil Moisture: ' + result.input_summary.soil_moisture + '%</p>';
                        resultDiv.innerHTML = html;
                    }
                } catch (error) {
                    resultDiv.innerHTML = '<h3>❌ Error</h3><p>' + error.message + '</p>';
                    resultDiv.className = 'result error';
                }
            };
        </script>
    </body>
    </html>
    """


@app.route('/dashboard', methods=['GET'])
def dashboard():
    """Combined dashboard for both services"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Agriculture Recommendation System</title>
        <style>
            body { font-family: Arial; margin: 0; padding: 20px; background: #f5f5f5; }
            .header { text-align: center; margin-bottom: 30px; }
            h1 { color: #2c3e50; }
            .container { max-width: 1200px; margin: 0 auto; }
            .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }
            .card { background: white; border-radius: 10px; padding: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .card h2 { color: #27ae60; margin-top: 0; border-bottom: 2px solid #27ae60; padding-bottom: 10px; }
            .card.crop h2 { color: #3498db; border-bottom-color: #3498db; }
            .btn { display: inline-block; padding: 10px 20px; margin: 10px 5px; text-decoration: none; border-radius: 5px; }
            .btn-primary { background: #27ae60; color: white; }
            .btn-secondary { background: #3498db; color: white; }
            .btn-info { background: #f39c12; color: white; }
            .stats { margin: 20px 0; padding: 15px; background: #f8f9fa; border-radius: 5px; }
            .endpoint-list { list-style: none; padding: 0; }
            .endpoint-list li { padding: 5px 0; border-bottom: 1px solid #eee; }
            .endpoint-list code { background: #f0f0f0; padding: 2px 5px; border-radius: 3px; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🌾 Agriculture Recommendation System</h1>
                <p>Integrated platform for crop and fertilizer recommendations</p>
            </div>
            
            <div class="grid">
                <!-- Crop Card -->
                <div class="card crop">
                    <h2>🌽 Crop Recommendation</h2>
                    <p>Get the best crop to grow based on soil nutrients and environmental conditions</p>
                    
                    <div class="stats">
                        <strong>API Endpoints:</strong>
                        <ul class="endpoint-list">
                            <li><code>POST /predict-crop</code> - Get crop recommendation with report</li>
                            <li><code>POST /predict-crop-no-report</code> - Quick prediction without report</li>
                            <li><code>GET /report/latest</code> - View latest crop report</li>
                            <li><code>GET /reports/list</code> - List all crop reports</li>
                        </ul>
                    </div>
                    
                    <a href="#" class="btn btn-secondary">Try Crop Demo</a>
                </div>
                
                <!-- Fertilizer Card -->
                <div class="card">
                    <h2>🧪 Fertilizer Recommendation</h2>
                    <p>Get the best fertilizer for your crop based on soil type and nutrient requirements</p>
                    
                    <div class="stats">
                        <strong>API Endpoints:</strong>
                        <ul class="endpoint-list">
                            <li><code>POST /api/fertilizer/predict</code> - Get fertilizer recommendation</li>
                            <li><code>GET /api/fertilizer/metadata</code> - Get model metadata</li>
                            <li><code>GET /fertilizer-report/latest</code> - View latest report</li>
                            <li><code>GET /api/fertilizer/generate-report</code> - Generate analysis report</li>
                        </ul>
                    </div>
                    
                    <a href="/fertilizer-demo" class="btn btn-primary">Try Fertilizer Demo</a>
                    <a href="/fertilizer-report/latest" class="btn btn-info">View Latest Report</a>
                </div>
            </div>
            
            <!-- Combined API -->
            <div class="card" style="margin-top: 20px;">
                <h2>🔄 Combined Recommendations</h2>
                <p>Get both crop and fertilizer recommendations in one API call</p>
                
                <div class="stats">
                    <strong>POST /api/recommend-all</strong><br>
                    <small>Send parameters for both services and get integrated results</small>
                </div>
                
                <pre style="background: #f0f0f0; padding: 10px; border-radius: 5px;">
{
  "soil_type": "Loamy",
  "crop_type": "Wheat", 
  "nitrogen": 50,
  "phosphorus": 30,
  "potassium": 40,
  "temperature": 25,
  "humidity": 65,
  "soil_moisture": 55,
  "ph": 6.5,
  "rainfall": 200
}
                </pre>
            </div>
            
            <!-- System Health -->
            <div class="card" style="margin-top: 20px;">
                <h2>📊 System Health</h2>
                <p>Check the status of all services</p>
                <a href="/api/health" target="_blank" class="btn btn-info">View Health Status</a>
            </div>
        </div>
    </body>
    </html>
    """


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    # Create crop reports directory if it doesn't exist
    os.makedirs(CROP_REPORTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(CROP_REPORTS_DIR, "images"), exist_ok=True)
    os.makedirs(os.path.join(CROP_REPORTS_DIR, "data"), exist_ok=True)
    
    # Create fertilizer reports directory if it doesn't exist
    os.makedirs(FERTILIZER_REPORTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(FERTILIZER_REPORTS_DIR, "images"), exist_ok=True)
    os.makedirs(os.path.join(FERTILIZER_REPORTS_DIR, "data"), exist_ok=True)
    
    print("=" * 60)
    print("🚀 Agriculture Recommendation System Starting...")
    print("=" * 60)
    
    print("\n📁 Directories:")
    print(f"   • Crop Reports: {CROP_REPORTS_DIR}")
    print(f"   • Fertilizer Reports: {FERTILIZER_REPORTS_DIR}")
    
    print("\n🌽 Crop Recommendation APIs:")
    print("   • POST   /predict-crop")
    print("   • POST   /predict-crop-no-report")
    print("   • GET    /report/latest")
    print("   • GET    /reports/list")
    
    print("\n🧪 Fertilizer Recommendation APIs:")
    print("   • POST   /api/fertilizer/predict")
    print("   • GET    /api/fertilizer/metadata")
    print("   • GET    /api/fertilizer/soil-types")
    print("   • GET    /api/fertilizer/crop-types")
    print("   • POST   /api/fertilizer/generate-report")
    print("   • GET    /fertilizer-report/latest")
    print("   • GET    /fertilizer-reports/list")
    
    print("\n🔄 Combined APIs:")
    print("   • POST   /api/recommend-all")
    print("   • GET    /api/health")
    
    print("\n🌐 Demo Pages:")
    print("   • GET    /dashboard")
    print("   • GET    /fertilizer-demo")
    
    print("\n" + "=" * 60)
    print(f"🌍 Server running at: http://localhost:5000")
    print("=" * 60)
    
    app.run(port=5000, debug=True)



