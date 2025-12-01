from flask import Flask, render_template, request, jsonify, send_file
import json
import random
import csv
import io
from datetime import datetime, timedelta
import os

app = Flask(__name__)

class SolarAnalysisSystem:
    def __init__(self):
        self.sample_data = {}
        self.analysis_results = {}
    
    def generate_sample_id(self):
        return f"SOLAR_{random.randint(1000, 9999)}"
    
    def simulate_satellite_analysis(self, latitude, longitude, buffer_distance=15):
        """Simulate satellite analysis with realistic results"""
        has_solar = random.random() > 0.4
        confidence = has_solar * (0.88 + random.random() * 0.08) + (not has_solar) * (0.65 + random.random() * 0.25)
        panel_count = has_solar * random.randint(4, 22)
        pv_area = has_solar * round(panel_count * 1.65 + random.random() * 1.8, 1)
        capacity = has_solar * round(pv_area * 0.21, 1)
        
        # Determine QC status
        if has_solar:
            qc_status = "VERIFIABLE" if random.random() > 0.15 else "PARTIALLY_VERIFIABLE"
            qc_notes = [
                "Clear roof view within 15m radius",
                "Distinct solar module patterns detected",
                "Consistent panel alignment and spacing"
            ]
            if qc_status == "PARTIALLY_VERIFIABLE":
                qc_notes.extend([
                    "Partial obstruction detected at edges",
                    "Recommend manual verification for accuracy"
                ])
        else:
            qc_status = "NO_SOLAR_DETECTED" if random.random() > 0.2 else "NOT_VERIFIABLE"
            qc_notes = [
                "No solar panel signatures found within 15m radius",
                "Clear rooftop visibility in scanned area"
            ]
            if qc_status == "NOT_VERIFIABLE":
                qc_notes = [
                    "Heavy shadow/cloud cover in area",
                    "Possible obstructions limiting visibility",
                    "Recommend retry with different imagery source"
                ]
        
        return {
            "has_solar": bool(has_solar),
            "confidence": round(confidence, 2),
            "panel_count": panel_count,
            "pv_area": pv_area,
            "capacity": capacity,
            "qc_status": qc_status,
            "qc_notes": qc_notes,
            "buffer_distance": buffer_distance,
            "image_metadata": {
                "source": "Sentinel-2 Satellite Imagery",
                "resolution": "10m/pixel",
                "capture_date": (datetime.now() - timedelta(days=random.randint(1, 30))).strftime("%Y-%m-%d"),
                "cloud_cover": round(random.random() * 0.3, 2)
            }
        }
    
    def calculate_subsidy(self, capacity, state):
        """Calculate subsidy based on capacity and state"""
        if capacity <= 3:
            subsidy = capacity * 14588
        else:
            subsidy = (3 * 14588) + ((capacity - 3) * 7294)
        
        state_multipliers = {
            'delhi': 1.0,
            'maharashtra': 0.95,
            'karnataka': 0.9,
            'tamilnadu': 0.92,
            'gujarat': 0.98
        }
        
        multiplier = state_multipliers.get(state, 1.0)
        subsidy = subsidy * multiplier
        
        # Add random variation
        variation = 0.9 + random.random() * 0.2
        subsidy = subsidy * variation
        
        min_subsidy = int((subsidy // 1000) * 1000 - 1000)
        max_subsidy = int((subsidy // 1000) * 1000 + 1000)
        
        return f"₹{min_subsidy} - ₹{max_subsidy}"

solar_system = SolarAnalysisSystem()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_location():
    """Analyze location for solar panels"""
    try:
        data = request.get_json()
        
        sample_id = data.get('sample_id', solar_system.generate_sample_id())
        latitude = float(data.get('latitude', 0))
        longitude = float(data.get('longitude', 0))
        buffer_distance = int(data.get('buffer_distance', 15))
        
        if not latitude or not longitude:
            return jsonify({'error': 'Latitude and longitude are required'}), 400
        
        # Simulate satellite analysis
        results = solar_system.simulate_satellite_analysis(latitude, longitude, buffer_distance)
        
        # Prepare response
        response_data = {
            "sample_id": sample_id,
            "lat": latitude,
            "lon": longitude,
            "has_solar": results["has_solar"],
            "confidence": results["confidence"],
            "panel_count_est": results["panel_count"],
            "pv_area_sqm_est": results["pv_area"],
            "capacity_kw_est": results["capacity"],
            "qc_status": results["qc_status"],
            "qc_notes": results["qc_notes"],
            "scan_radius_meters": buffer_distance,
            "bbox_or_mask": "",
            "image_metadata": results["image_metadata"],
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        # Store results
        solar_system.analysis_results[sample_id] = response_data
        
        return jsonify(response_data)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/subsidy', methods=['POST'])
def calculate_subsidy():
    """Calculate subsidy amount"""
    try:
        data = request.get_json()
        capacity = int(data.get('capacity', 3))
        state = data.get('state', 'delhi')
        
        subsidy_range = solar_system.calculate_subsidy(capacity, state)
        
        return jsonify({
            'capacity': capacity,
            'state': state,
            'subsidy_range': subsidy_range
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/<sample_id>')
def download_results(sample_id):
    """Download analysis results as JSON"""
    if sample_id not in solar_system.analysis_results:
        return jsonify({'error': 'Sample ID not found'}), 404
    
    results = solar_system.analysis_results[sample_id]
    
    # Create JSON file in memory
    json_output = json.dumps(results, indent=2)
    buffer = io.BytesIO()
    buffer.write(json_output.encode('utf-8'))
    buffer.seek(0)
    
    return send_file(
        buffer,
        as_attachment=True,
        download_name=f'solar_analysis_{sample_id}.json',
        mimetype='application/json'
    )

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    """Handle CSV file upload for batch processing"""
    try:
        if 'csv_file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['csv_file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'File must be a CSV'}), 400
        
        # Read CSV file
        stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
        csv_input = csv.DictReader(stream)
        
        required_columns = ['sample_id', 'latitude', 'longitude']
        if not all(col in csv_input.fieldnames for col in required_columns):
            return jsonify({'error': f'CSV must contain columns: {required_columns}'}), 400
        
        # Process each row
        results = []
        for row in csv_input:
            try:
                sample_id = row['sample_id']
                latitude = float(row['latitude'])
                longitude = float(row['longitude'])
                buffer_distance = int(row.get('buffer_distance', 15))
                
                analysis_result = solar_system.simulate_satellite_analysis(
                    latitude, longitude, buffer_distance
                )
                
                result_data = {
                    "sample_id": sample_id,
                    "lat": latitude,
                    "lon": longitude,
                    "has_solar": analysis_result["has_solar"],
                    "confidence": analysis_result["confidence"],
                    "panel_count_est": analysis_result["panel_count"],
                    "pv_area_sqm_est": analysis_result["pv_area"],
                    "capacity_kw_est": analysis_result["capacity"],
                    "qc_status": analysis_result["qc_status"],
                    "scan_radius_meters": buffer_distance,
                    "image_metadata": analysis_result["image_metadata"]
                }
                
                results.append(result_data)
                solar_system.analysis_results[sample_id] = result_data
                
            except Exception as e:
                results.append({
                    "sample_id": row.get('sample_id', 'unknown'),
                    "error": str(e)
                })
        
        return jsonify({
            'message': f'Processed {len(results)} samples',
            'results': results
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/samples')
def list_samples():
    """List all analyzed samples"""
    return jsonify({
        'total_samples': len(solar_system.analysis_results),
        'samples': list(solar_system.analysis_results.keys())
    })

@app.route('/sample/<sample_id>')
def get_sample(sample_id):
    """Get specific sample results"""
    if sample_id not in solar_system.analysis_results:
        return jsonify({'error': 'Sample not found'}), 404
    
    return jsonify(solar_system.analysis_results[sample_id])

if __name__ == '__main__':
    print("Solar Analysis System starting on http://localhost:5000")
    print("Features:")
    print("- AI-powered solar panel detection simulation")
    print("- Satellite imagery analysis")
    print("- Subsidy calculation")
    print("- CSV batch processing")
    print("- JSON export")
    print("- RESTful API endpoints")
    print("- Interactive 2D Map with location selection")
    print("- 3D Satellite View with building visualization")
    
    app.run(debug=True, host='0.0.0.0', port=5000)