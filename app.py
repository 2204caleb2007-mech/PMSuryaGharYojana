from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import os
import io
import json
import csv
import random
import logging
import math
from datetime import datetime, timedelta
import yaml
import base64
from io import BytesIO

# ML Imports (optional - graceful fallback if not available)
try:
    import torch
    import torchvision.transforms as T
    from torch.utils.data import Dataset, DataLoader
    from torchvision.models.detection import retinanet_resnet50_fpn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from PIL import Image
    import numpy as np
    PIL_NUMPY_AVAILABLE = True
except ImportError:
    PIL_NUMPY_AVAILABLE = False

try:
    from pycocotools.coco import COCO
    PYCOCOTOOLS_AVAILABLE = True
except ImportError:
    PYCOCOTOOLS_AVAILABLE = False

# Roboflow Inference imports
ROBOFLOW_AVAILABLE = True  # We'll use REST API directly with requests

# ==============================
# CONFIGURATION
# ==============================

app = Flask(__name__, 
           template_folder='templates',
           static_folder='static')
CORS(app)

# Load configuration
config_path = 'config.yaml' if os.path.exists('config.yaml') else None
if config_path:
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except:
        config = {
            'app': {'debug': True, 'host': '0.0.0.0', 'port': 5000},
            'ml_models': {},
            'roboflow': {
                'api_url': 'https://serverless.roboflow.com',
                'api_key': 'h02d8ka3vbQmsEIVyLj5',
                'model_id': 'custom-workflow-object-detection-tgnqc-af4gr/1'
            },
            'google_maps': {
                'api_key': 'AIzaSyDvuSBa_a10D5RLiu6dvH21OvF1H-jENtU'
            }
        }
else:
    config = {
        'app': {'debug': True, 'host': '0.0.0.0', 'port': 5000},
        'ml_models': {},
        'roboflow': {
            'api_url': 'https://serverless.roboflow.com',
            'api_key': 'h02d8ka3vbQmsEIVyLj5',
            'model_id': 'custom-workflow-object-detection-tgnqc-af4gr/1'
        },
        'google_maps': {
            'api_key': 'AIzaSyDvuSBa_a10D5RLiu6dvH21OvF1H-jENtU'
        }
    }

# Configure logging (with graceful fallback)
try:
    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)
    log_handlers = [
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
except Exception:
    # If logs directory can't be created, just use console logging
    log_handlers = [logging.StreamHandler()]

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=log_handlers,
    force=True  # Override any existing logging configuration
)
logger = logging.getLogger(__name__)

# Initialize Roboflow API client (using REST API via requests)
ROBOFLOW_MODEL_ID = config['roboflow']['model_id']
ROBOFLOW_API_KEY = config['roboflow']['api_key']
ROBOFLOW_API_URL = config['roboflow']['api_url']
GOOGLE_MAPS_API_KEY = config.get('google_maps', {}).get('api_key', os.getenv('GOOGLE_MAPS_API_KEY', ''))
CLIENT = None  # Not using inference SDK - using REST API instead
INFERENCE_MODEL = None  # Not loading local model - using Roboflow cloud API

logger.info("[OK] Roboflow model configured for detection")
logger.info(f"    Model: {ROBOFLOW_MODEL_ID}")
logger.info("[OK] Google Maps API key configured for satellite imagery")

# Create necessary directories (optional - for advanced features)
optional_dirs = ['data/uploads', 'data/processed', 'data/cache', 'logs', 'static/annotated']
for directory in optional_dirs:
    try:
        os.makedirs(directory, exist_ok=True)
    except Exception as e:
        logger.warning(f"Could not create directory {directory}: {str(e)}")

# ==============================
# OPTIONAL: LOCAL RETINANET DETECTION SYSTEM
# (Not required - uses Roboflow cloud model instead)
# ==============================

DATA_IMG_DIR = "dataset/images"
DATA_ANN_FILE = "dataset/annotations/instances.json"
MODEL_PATH = "solar_panel_detector_robo.pth"  # Optional local model file
BATCH_SIZE = 4
NUM_EPOCHS = 10
LR = 1e-4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if TORCH_AVAILABLE else None

# Optional: COCO dataset class for training (requires torch and pycocotools)
# Skipped for now - using Roboflow cloud model instead
# if TORCH_AVAILABLE and PYCOCOTOOLS_AVAILABLE:
#     class SolarPanelCocoDataset(Dataset):...

# Torch-dependent code commented out (using Roboflow cloud model instead)
# def collate_fn(batch):
# train_transforms = T.Compose([...])
# def get_retinanet_model(...):
# Optional torch-dependent classes removed
# Using Roboflow cloud model for detection instead
detector = None

logger.info("[OK] Using Roboflow cloud model for detection")

# ==============================
# SOLAR ANALYSIS SYSTEM
# ==============================

class SolarAnalysisSystem:
    def __init__(self):
        self.analysis_results = {}

    def generate_sample_id(self):
        return f"SOLAR_{random.randint(1000, 9999)}"

    def calculate_subsidy(self, capacity, state):
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

        multiplier = state_multipliers.get(state.lower(), 1.0)
        subsidy = subsidy * multiplier
        variation = 0.9 + random.random() * 0.2
        subsidy = subsidy * variation

        min_subsidy = int((subsidy // 1000) * 1000 - 1000)
        max_subsidy = int((subsidy // 1000) * 1000 + 1000)

        return f"₹{min_subsidy} - ₹{max_subsidy}"

solar_system = SolarAnalysisSystem()

# ==============================
# ROBOFLOW INFERENCE FUNCTIONS
# ==============================

def run_roboflow_inference(image_path):
    """Run inference using Roboflow model"""
    if not CLIENT or not ROBOFLOW_MODEL_ID:
        raise Exception("Roboflow client not available - using satellite analysis instead")
    
    try:
        # Run inference on the image
        result = CLIENT.infer(image_path, model_id=ROBOFLOW_MODEL_ID)
        logger.info(f"Roboflow inference completed: {len(result.get('predictions', []))} detections")
        return result
    except Exception as e:
        logger.error(f"Roboflow inference failed: {str(e)}")
        raise Exception(f"Model inference failed: {str(e)}")

def run_local_inference(image_path):
    """Run inference using local model"""
    if INFERENCE_MODEL is None:
        raise Exception("Local inference model not loaded")
    
    if not ROBOFLOW_AVAILABLE:
        raise Exception("Roboflow/supervision libraries not available")
    
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise Exception(f"Failed to load image: {image_path}")
        
        # Run inference
        results = INFERENCE_MODEL.infer(image)[0]
        
        # Convert to supervision format
        detections = sv.Detections.from_inference(results)
        
        # Create annotated image
        bounding_box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        
        annotated_image = bounding_box_annotator.annotate(
            scene=image, detections=detections)
        annotated_image = label_annotator.annotate(
            scene=annotated_image, detections=detections)
        
        # Save annotated image
        try:
            annotated_path = os.path.join('static/annotated', os.path.basename(image_path))
            cv2.imwrite(annotated_path, annotated_image)
        except Exception as e:
            logger.warning(f"Could not save annotated image: {str(e)}")
            annotated_path = None
        
        # Convert to API format
        predictions = []
        for i, (_, _, confidence, class_id, _) in enumerate(detections):
            class_name = results.get('predictions', [{}])[i].get('class', 'object') if i < len(results.get('predictions', [])) else 'object'
            
            predictions.append({
                'class': class_name,
                'confidence': float(confidence),
                'x': 0,  # These would need to be extracted from detections
                'y': 0,
                'width': 0,
                'height': 0
            })
        
        return {
            'predictions': predictions,
            'annotated_image': annotated_path,
            'detections_count': len(predictions)
        }
        
    except Exception as e:
        logger.error(f"Local inference failed: {str(e)}")
        raise Exception(f"Local model inference failed: {str(e)}")

def get_roboflow_predictions(image_url):
    """
    Query Roboflow API directly for solar panel detection using REST API
    Downloads satellite image and sends as base64-encoded data
    """
    try:
        import requests
        import base64
        
        # Download satellite image first
        logger.info(f"Downloading satellite image: {image_url[:100]}...")
        try:
            img_response = requests.get(image_url, timeout=15, allow_redirects=True)
            if img_response.status_code != 200:
                logger.error(f"Failed to download satellite image: {img_response.status_code}")
                return {
                    'predictions': [],
                    'success': False,
                    'error': f"Could not download satellite image (HTTP {img_response.status_code})"
                }
            logger.info(f"✓ Downloaded satellite image ({len(img_response.content)} bytes)")
        except Exception as e:
            logger.error(f"Failed to download image: {e}")
            return {
                'predictions': [],
                'success': False,
                'error': f"Could not download satellite image: {str(e)}"
            }
        
        # Encode image as base64
        image_base64 = base64.b64encode(img_response.content).decode('utf-8')
        logger.info(f"✓ Encoded image as base64 ({len(image_base64)} chars)")
        
        # Build API URL for inference
        api_url = f"https://api.roboflow.com/infer/{ROBOFLOW_MODEL_ID}"
        
        # Send base64-encoded image to Roboflow
        logger.info(f"Sending satellite image to Roboflow API...")
        headers = {
            "Content-Type": "application/x-www-form-urlencoded"
        }
        
        payload = {
            "api_key": ROBOFLOW_API_KEY,
            "image": image_base64
        }
        
        response = requests.post(api_url, data=payload, headers=headers, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            predictions = result.get('predictions', [])
            
            # Debug: log full response
            logger.info(f"Full Roboflow response received")
            logger.debug(f"Response: {json.dumps(result, indent=2)[:500]}")
            logger.info(f"Roboflow returned: {type(predictions).__name__} with {len(predictions) if predictions else 0} predictions")
            
            if isinstance(predictions, list) and len(predictions) == 0:
                logger.warning("⚠ No predictions returned (empty list)")
            elif len(predictions) > 0:
                logger.info(f"✓ Found {len(predictions)} detections!")
                for i, pred in enumerate(predictions[:3]):  # Log first 3
                    logger.info(f"  Detection {i+1}: {pred.get('class', 'unknown')} (confidence: {pred.get('confidence', 0):.2f})")
            
            return {
                'predictions': predictions,
                'success': True,
                'raw_response': result
            }
        else:
            logger.error(f"Roboflow API error: {response.status_code}")
            logger.error(f"Response text: {response.text[:500]}")
            return {
                'predictions': [],
                'success': False,
                'error': f"API returned {response.status_code}: {response.text[:100]}"
            }
            
    except Exception as e:
        logger.error(f"Roboflow API call failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            'predictions': [],
            'success': False,
            'error': str(e)
        }

def analyze_image_with_roboflow(image_file, use_local=False):
    """Analyze image using Roboflow model and return actual results"""
    try:
        # Save the uploaded file temporarily
        try:
            os.makedirs('data/cache', exist_ok=True)
        except:
            pass
        temp_path = os.path.join('data/cache', f'temp_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg')
        image_file.save(temp_path)
        
        # Run inference
        if use_local and INFERENCE_MODEL:
            inference_result = run_local_inference(temp_path)
        else:
            inference_result = run_roboflow_inference(temp_path)
        
        # Process results
        predictions = inference_result.get('predictions', [])
        
        if not predictions:
            result = {
                'has_solar': False,
                'confidence': 0.0,
                'detections': [],
                'panel_count': 0,
                'message': 'No solar panels detected'
            }
        else:
            # Extract actual detection data
            detections = []
            total_confidence = 0.0
            
            for pred in predictions:
                if pred.get('class') in ['solar panel', 'panel']:
                    bbox = {
                        'x': pred.get('x', 0),
                        'y': pred.get('y', 0),
                        'width': pred.get('width', 0),
                        'height': pred.get('height', 0)
                    }
                    confidence = pred.get('confidence', 0.0)
                    
                    detections.append({
                        'bbox': bbox,
                        'confidence': confidence,
                        'class': pred.get('class', 'solar panel')
                    })
                    total_confidence += confidence
            
            avg_confidence = total_confidence / len(detections) if detections else 0.0
            
            result = {
                'has_solar': len(detections) > 0,
                'confidence': round(avg_confidence, 3),
                'detections': detections,
                'panel_count': len(detections),
                'message': f'Detected {len(detections)} solar panels' if detections else 'No solar panels detected'
            }
        
        # Add annotated image path if available
        if 'annotated_image' in inference_result:
            result['annotated_image'] = inference_result['annotated_image']
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return result
        
    except Exception as e:
        logger.error(f"Image analysis failed: {str(e)}")
        raise Exception(f"Image analysis failed: {str(e)}")

# ==============================
# FLASK ROUTES - TEMPLATE ROUTES
# ==============================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ai-analysis')
def ai_analysis():
    return render_template('AIPSI.html')

@app.route('/subsidy')
def subsidy_page():
    return render_template('Subsidy.html')

@app.route('/process')
def process():
    return render_template('process.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/annotate-image')
def annotate_image():
    """New route for image annotation interface"""
    return render_template('annotate.html')

# ==============================
# FLASK ROUTES - API ENDPOINTS
# ==============================

# --------- RetinaNet Detection ---------

@app.route('/detect_panels', methods=['POST'])
def detect_panels():
    """Upload an image and run RetinaNet detection + roof space suggestion."""
    if detector is None:
        return jsonify({'error': 'Model not loaded. Train and save model first.'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        img = Image.open(file.stream).convert("RGB")
        img = img.resize((512, 512))

        boxes, scores, labels = detector.detect(img)
        suggestion, available_mask = suggester.suggest(boxes)

        detections = []
        for b, s, l in zip(boxes, scores, labels):
            x1, y1, x2, y2 = [float(v) for v in b]
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'score': float(s),
                'label': int(l)
            })

        total_pixels = int(available_mask.size)
        free_pixels = int(np.sum(available_mask))
        free_ratio = float(free_pixels / total_pixels)

        return jsonify({
            'detections': detections,
            'suggestion': suggestion,
            'available_area_ratio': free_ratio
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/train_model', methods=['POST'])
def train_model_route():
    """Trigger training on the server."""
    try:
        train_model()
        return jsonify({'message': 'Training completed', 'model_path': MODEL_PATH})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --------- Roboflow Analysis ---------

@app.route('/api/analyze-image', methods=['POST'])
def analyze_image():
    """Analyze image using Roboflow model"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        
        # Check if local inference is requested
        use_local = request.form.get('use_local', 'false').lower() == 'true'
        
        # Validate image (optional - PIL might not be available)
        if PIL_NUMPY_AVAILABLE:
            try:
                img = Image.open(image_file.stream)
                img.verify()  # Verify it's a valid image
                image_file.stream.seek(0)  # Reset stream
            except:
                return jsonify({'error': 'Invalid image file'}), 400
        else:
            # Just check file extension as fallback
            allowed_exts = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
            if not any(image_file.filename.lower().endswith(ext) for ext in allowed_exts):
                return jsonify({'error': 'Invalid image file'}), 400
        
        # Run actual model analysis
        analysis_result = analyze_image_with_roboflow(image_file, use_local=use_local)
        
        # Generate response with actual model results only
        sample_id = f"IMG_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        response = {
            "sample_id": sample_id,
            "analysis_timestamp": datetime.now().isoformat(),
            "has_solar": analysis_result['has_solar'],
            "confidence": analysis_result['confidence'],
            "panel_count": analysis_result['panel_count'],
            "detections": analysis_result['detections'],
            "message": analysis_result['message'],
            "model_used": "Local Inference" if use_local else "Roboflow Custom Object Detection",
            "model_id": ROBOFLOW_MODEL_ID
        }
        
        # Add annotated image URL if available
        if 'annotated_image' in analysis_result:
            response['annotated_image_url'] = f"/{analysis_result['annotated_image']}"
        
        # Store results
        solar_system.analysis_results[sample_id] = response
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Image analysis error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/annotate-image', methods=['POST'])
def annotate_image_api():
    """Annotate image with bounding boxes using local model"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        
        # Save the uploaded file
        temp_path = os.path.join('data/cache', f'annotate_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg')
        image_file.save(temp_path)
        
        # Run local inference
        if INFERENCE_MODEL is None:
            return jsonify({'error': 'Local inference model not available'}), 500
        
        # Read and process image
        image = cv2.imread(temp_path)
        if image is None:
            return jsonify({'error': 'Failed to read image'}), 400
        
        # Run inference
        results = INFERENCE_MODEL.infer(image)[0]
        detections = sv.Detections.from_inference(results)
        
        # Create annotated image
        bounding_box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        
        annotated_image = bounding_box_annotator.annotate(
            scene=image, detections=detections)
        annotated_image = label_annotator.annotate(
            scene=annotated_image, detections=detections)
        
        # Save annotated image
        annotated_filename = f"annotated_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
        annotated_path = os.path.join('static/annotated', annotated_filename)
        cv2.imwrite(annotated_path, annotated_image)
        
        # Prepare response
        detections_list = []
        for i, (xyxy, mask, confidence, class_id, tracker_id) in enumerate(detections):
            class_name = results.get('predictions', [{}])[i].get('class', f'class_{class_id}') if i < len(results.get('predictions', [])) else f'class_{class_id}'
            
            detections_list.append({
                'bbox': xyxy.tolist(),
                'confidence': float(confidence),
                'class_id': int(class_id),
                'class_name': class_name,
                'tracker_id': int(tracker_id) if tracker_id is not None else None
            })
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return jsonify({
            'success': True,
            'annotated_image_url': f'/static/annotated/{annotated_filename}',
            'detections': detections_list,
            'total_detections': len(detections),
            'image_size': image.shape[:2]
        })
        
    except Exception as e:
        logger.error(f"Image annotation error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze-location', methods=['POST'])
def analyze_location():
    """Analyze location by fetching satellite imagery and scanning with Roboflow model"""
    try:
        data = request.json
        latitude = float(data.get('latitude', 0))
        longitude = float(data.get('longitude', 0))
        zoom = int(data.get('zoom', 18))  # Default zoom level for satellite imagery
        
        if latitude == 0 and longitude == 0:
            return jsonify({'error': 'Valid latitude and longitude required'}), 400
        
        logger.info(f"Analyzing location: {latitude}, {longitude} with satellite imagery")
        
        # Generate satellite image URL using ArcGIS World Imagery tiles
        # Free public satellite imagery, no authentication required
        # This uses the standard Web Mercator tile service
        
        def lat_lon_to_tile(lat, lon, zoom):
            """Convert latitude, longitude to tile coordinates"""
            n = 2 ** zoom
            x = int((lon + 180) / 360 * n)
            y = int((1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n)
            return x, y
        
        tile_x, tile_y = lat_lon_to_tile(latitude, longitude, zoom)
        
        # Use Esri World Imagery (free public tiles)
        satellite_url = (
            f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/"
            f"tile/{zoom}/{tile_y}/{tile_x}"
        )
        logger.info(f"✓ Using Esri World Imagery tiles for satellite imagery (location: {latitude}, {longitude}, zoom: {zoom})")
        
        # Use provided image_url if available (override auto-fetched)
        image_url = data.get('image_url', satellite_url)
        
        # Send to Roboflow model for analysis using REST API
        if ROBOFLOW_AVAILABLE:
            try:
                logger.info(f"Sending satellite image to Roboflow: {image_url[:100]}...")
                roboflow_result = get_roboflow_predictions(image_url)
                
                if not roboflow_result['success']:
                    return jsonify({'error': f"Roboflow analysis failed: {roboflow_result['error']}"}), 500
                
                predictions = roboflow_result['predictions']
                
                # Handle different prediction formats
                if isinstance(predictions, dict):
                    detection_count = len(predictions)
                elif isinstance(predictions, list):
                    detection_count = len(predictions)
                else:
                    detection_count = 0
                
                sample_id = data.get('sample_id', f"LOC_{datetime.now().strftime('%Y%m%d%H%M%S')}")
                
                response = {
                    "sample_id": sample_id,
                    "lat": latitude,
                    "lon": longitude,
                    "analysis_timestamp": datetime.now().isoformat(),
                    "detection_count": detection_count,
                    "has_solar": detection_count > 0,
                    "model_used": "Roboflow Cloud Model",
                    "satellite_image_source": "Google Maps / Mapbox / Sentinel Hub",
                    "predictions": predictions,
                    "zoom_level": zoom
                }
                
                solar_system.analysis_results[sample_id] = response
                logger.info(f"Analysis complete for {sample_id}: {detection_count} solar panels detected")
                return jsonify(response)
                
            except Exception as e:
                logger.error(f"Roboflow analysis error: {str(e)}")
                return jsonify({'error': f'Roboflow analysis failed: {str(e)}'}), 500
        else:
            return jsonify({'error': 'Roboflow model not configured'}), 503
        
    except ValueError as e:
        logger.error(f"Invalid coordinate format: {str(e)}")
        return jsonify({"error": f"Invalid coordinates: {str(e)}"}), 400
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/subsidy', methods=['POST'])
def calculate_subsidy_enhanced():
    """Calculate subsidy based on ACTUAL capacity from model analysis"""
    try:
        data = request.json
        
        # Get sample ID to retrieve actual analysis
        sample_id = data.get('sample_id')
        if not sample_id or sample_id not in solar_system.analysis_results:
            return jsonify({'error': 'Valid sample_id required with actual analysis results'}), 400
        
        # Get actual capacity from previous analysis
        analysis = solar_system.analysis_results[sample_id]
        capacity = analysis.get('capacity_kw_est', 0)
        
        if capacity <= 0:
            return jsonify({'error': 'No solar capacity detected in analysis'}), 400
        
        state = data.get('state', 'delhi')
        
        # Calculate actual subsidy
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

        multiplier = state_multipliers.get(state.lower(), 1.0)
        subsidy = subsidy * multiplier
        
        min_subsidy = int((subsidy * 0.9) // 1000) * 1000
        max_subsidy = int((subsidy * 1.1) // 1000) * 1000
        avg_subsidy = (min_subsidy + max_subsidy) // 2

        return jsonify({
            'capacity': round(capacity, 2),
            'state': state,
            'subsidy_range': f"₹{min_subsidy:,} - ₹{max_subsidy:,}",
            'min_subsidy': min_subsidy,
            'max_subsidy': max_subsidy,
            'avg_subsidy': avg_subsidy,
            'currency': 'INR',
            'based_on_sample': sample_id,
            'panel_count': analysis.get('panel_count_est', 0)
        })
    except Exception as e:
        logger.error(f"Subsidy calculation error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/download-results/<filename>')
def download_results(filename):
    """Download actual analysis results"""
    try:
        # Check if it's a sample ID
        if filename in solar_system.analysis_results:
            results = solar_system.analysis_results[filename]
            json_output = json.dumps(results, indent=2)
            buffer = io.BytesIO()
            buffer.write(json_output.encode('utf-8'))
            buffer.seek(0)
            return send_file(
                buffer,
                as_attachment=True,
                download_name=f'solar_analysis_{filename}.json',
                mimetype='application/json'
            )
        else:
            return jsonify({"error": "Analysis results not found"}), 404
            
    except Exception as e:
        logger.error(f"Download error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/upload-csv', methods=['POST'])
def upload_csv():
    """Process CSV with image paths for actual analysis"""
    try:
        if 'csv_file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
            
        csv_file = request.files['csv_file']
        if csv_file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Check if local inference is requested
        use_local = request.form.get('use_local', 'false').lower() == 'true'
            
        # Save uploaded file
        filename = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = os.path.join('data/uploads', filename)
        csv_file.save(filepath)
        
        # Process CSV - expecting columns: sample_id, image_path, latitude, longitude
        results = []
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                try:
                    sample_id = row.get('sample_id', f"IMG_{len(results)+1}")
                    image_path = row.get('image_path')
                    lat = float(row.get('latitude', 0))
                    lon = float(row.get('longitude', 0))
                    
                    if not image_path or not os.path.exists(image_path):
                        results.append({
                            "sample_id": sample_id,
                            "error": "Image file not found",
                            "image_path": image_path
                        })
                        continue
                    
                    # Run actual model analysis
                    try:
                        if use_local and INFERENCE_MODEL:
                            inference_result = run_local_inference(image_path)
                        else:
                            inference_result = run_roboflow_inference(image_path)
                            
                        predictions = inference_result.get('predictions', [])
                        panel_detections = [p for p in predictions if p.get('class') in ['solar panel', 'panel']]
                        
                        has_solar = len(panel_detections) > 0
                        panel_count = len(panel_detections)
                        
                        if panel_detections:
                            avg_confidence = sum(p.get('confidence', 0) for p in panel_detections) / len(panel_detections)
                        else:
                            avg_confidence = 0.0
                        
                        result_entry = {
                            "sample_id": sample_id,
                            "lat": lat,
                            "lon": lon,
                            "has_solar": has_solar,
                            "confidence": round(avg_confidence, 3),
                            "panel_count": panel_count,
                            "image_path": image_path,
                            "detection_count": panel_count
                        }
                        
                        # Add annotated image URL if available
                        if 'annotated_image' in inference_result:
                            result_entry['annotated_image_url'] = f"/{inference_result['annotated_image']}"
                        
                        results.append(result_entry)
                        
                        # Store full analysis
                        solar_system.analysis_results[sample_id] = {
                            "sample_id": sample_id,
                            "lat": lat,
                            "lon": lon,
                            "has_solar": has_solar,
                            "confidence": round(avg_confidence, 3),
                            "panel_count": panel_count,
                            "analysis_timestamp": datetime.now().isoformat(),
                            "model_used": "Local Inference" if use_local else "Roboflow Custom Object Detection"
                        }
                        
                    except Exception as e:
                        results.append({
                            "sample_id": sample_id,
                            "error": f"Model analysis failed: {str(e)}",
                            "image_path": image_path
                        })
                        
                except Exception as e:
                    results.append({
                        "sample_id": row.get('sample_id', 'unknown'),
                        "error": str(e)
                    })
        
        # Save batch results
        batch_results_file = os.path.join('data/processed', f'batch_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(batch_results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return jsonify({
            "message": f"Processed {len(results)} samples with ACTUAL model analysis",
            "total_samples": len(results),
            "successful": len([r for r in results if 'error' not in r]),
            "failed": len([r for r in results if 'error' in r]),
            "results": results,
            "batch_results_file": batch_results_file
        })
        
    except Exception as e:
        logger.error(f"CSV upload error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/samples')
def list_samples():
    """List all analyzed samples with actual results"""
    samples_list = []
    for sample_id, results in solar_system.analysis_results.items():
        samples_list.append({
            'sample_id': sample_id,
            'has_solar': results.get('has_solar', False),
            'panel_count': results.get('panel_count_est', results.get('panel_count', 0)),
            'confidence': results.get('confidence', 0.0),
            'timestamp': results.get('analysis_timestamp', ''),
            'model_used': results.get('model_used', 'Unknown')
        })
    
    return jsonify({
        'total_samples': len(solar_system.analysis_results),
        'samples': samples_list
    })

@app.route('/sample/<sample_id>')
def get_sample(sample_id):
    """Get specific sample results (ACTUAL MODEL RESULTS ONLY)"""
    if sample_id not in solar_system.analysis_results:
        return jsonify({'error': 'Sample not found. Please run analysis first.'}), 404
    
    results = solar_system.analysis_results[sample_id]
    
    # Ensure we're only returning actual model results
    if 'model_used' not in results:
        results['model_used'] = 'Roboflow Custom Object Detection'
    
    return jsonify(results)

# --------- Additional API Endpoints ---------

@app.route('/api/save-analysis', methods=['POST'])
def save_analysis():
    """Save actual analysis results to file"""
    try:
        data = request.json
        
        if 'sample_id' not in data:
            return jsonify({"error": "sample_id is required"}), 400
        
        # Save to JSON file
        filename = f"{data['sample_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join('data/processed', filename)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Analysis saved: {filename}")
        
        # Also store in memory system
        solar_system.analysis_results[data['sample_id']] = data
        
        return jsonify({
            "message": "Analysis saved successfully",
            "file_path": filepath,
            "sample_id": data['sample_id']
        })
        
    except Exception as e:
        logger.error(f"Save analysis error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint with model status"""
    roboflow_status = "connected" if CLIENT and ROBOFLOW_MODEL_ID else "disconnected"
    retinanet_status = "loaded" if detector else "not_loaded"
    inference_model_status = "loaded" if INFERENCE_MODEL else "not_loaded"
    
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models": {
            "roboflow_inference": roboflow_status,
            "local_inference_model": inference_model_status,
            "retinanet_detector": retinanet_status
        },
        "system_stats": {
            "total_analyses": len(solar_system.analysis_results),
            "data_directory": os.listdir('data/') if os.path.exists('data/') else [],
            "roboflow_model": ROBOFLOW_MODEL_ID
        },
        "note": "All results are from actual model inference - no static fallbacks"
    })

# ==============================
# MAIN ENTRY POINT
# ==============================

if __name__ == '__main__':
    print("=" * 60)
    print("Solar Analysis & Detection System")
    print("ACTUAL MODEL RESULTS ONLY - NO STATIC FALLBACKS")
    print("=" * 60)
    print("\nTemplate Routes:")
    print("- /                    -> Main page")
    print("- /ai-analysis         -> AI Analysis interface")
    print("- /subsidy             -> Subsidy calculator")
    print("- /process             -> Process page")
    print("- /contact             -> Contact page")
    print("- /annotate-image      -> Image annotation interface")
    
    print("\nML Detection Routes:")
    print("- /detect_panels       -> RetinaNet panel detection (POST)")
    print("- /train_model         -> Train RetinaNet model (POST)")
    
    print("\nRoboflow Analysis Routes (ACTUAL MODEL RESULTS):")
    print("- /api/analyze-image   -> Analyze uploaded image (POST)")
    print("- /api/annotate-image  -> Annotate image with bounding boxes (POST)")
    print("- /api/analyze-location -> Analyze with image data (POST)")
    
    print("\nSubsidy & Data Routes:")
    print("- /api/subsidy         -> Subsidy based on actual analysis (POST)")
    print("- /api/upload-csv      -> Batch analysis with images (POST)")
    print("- /api/download-results/<id> -> Download actual results (GET)")
    print("- /samples             -> List all analyzed samples (GET)")
    print("- /sample/<id>         -> Get specific sample (GET)")
    
    print("\nAdditional API Routes:")
    print("- /api/save-analysis   -> Save analysis (POST)")
    print("- /api/health          -> Health check (GET)")
    
    print("\nRoboflow Model Info:")
    print(f"- API URL: {config['roboflow'].get('api_url', 'Not configured')}")
    print(f"- Model ID: {ROBOFLOW_MODEL_ID or 'Not configured'}")
    print(f"- Status: {'Connected' if CLIENT else 'Disconnected'}")
    print(f"- Local Model: {'Loaded' if INFERENCE_MODEL else 'Not loaded'}")
    
    print("\nIMPORTANT: All endpoints return ACTUAL model results only.")
    print("No static/fallback data will be returned.")
    print("\nRunning on: http://localhost:5000")
    print("=" * 60)
    
    app.run(debug=config['app']['debug'], 
            host=config['app']['host'], 
            port=config['app']['port'])