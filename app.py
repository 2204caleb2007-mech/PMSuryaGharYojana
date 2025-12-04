import os
import io
import json
import csv
import random
from datetime import datetime, timedelta

from flask import Flask, render_template, request, jsonify, send_file

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import retinanet_resnet50_fpn
from PIL import Image
import numpy as np

from pycocotools.coco import COCO

# ==============================
# CONFIG
# ==============================

DATA_IMG_DIR = "dataset/images"
DATA_ANN_FILE = "dataset/annotations/instances.json"
MODEL_PATH = "solar_panel_detector_robo.pth"
BATCH_SIZE = 4
NUM_EPOCHS = 10
LR = 1e-4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==============================
# DATASET & DATALOADER
# ==============================

class SolarPanelCocoDataset(Dataset):
    def __init__(self, img_dir, ann_file, transforms=None):
        self.img_dir = img_dir
        self.transforms = transforms
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        # adjust category names as per your Roboflow export
        self.cat_ids = self.coco.getCatIds(catNms=['solar panel', 'panel'])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id,
                                      catIds=self.cat_ids,
                                      iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)

        path = self.coco.loadImgs(img_id)[0]['file_name']
        img_path = os.path.join(self.img_dir, path)
        img = Image.open(img_path).convert("RGB")

        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(1)  # 1 for solar panel
            areas.append(ann['area'])
            iscrowd.append(ann.get('iscrowd', 0))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([img_id]),
            'area': areas,
            'iscrowd': iscrowd
        }

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))


train_transforms = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

# ==============================
# MODEL: TRAINING + INFERENCE
# ==============================

def get_retinanet_model(num_classes=2):
    model = retinanet_resnet50_fpn(pretrained=True)
    in_channels = model.head.classification_head.conv[0].in_channels
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head.cls_logits = torch.nn.Conv2d(
        in_channels,
        num_anchors * num_classes,
        kernel_size=3,
        stride=1,
        padding=1
    )
    model.head.classification_head.num_classes = num_classes
    return model


def train_model():
    dataset = SolarPanelCocoDataset(DATA_IMG_DIR, DATA_ANN_FILE,
                                    transforms=train_transforms)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )

    model = get_retinanet_model(num_classes=2)
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {epoch_loss/len(dataloader):.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


class SolarPanelDetector:
    def __init__(self, model_path, device=None, score_threshold=0.5):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.score_threshold = score_threshold

        self.model = get_retinanet_model(num_classes=2)
        self.model.load_state_dict(torch.load(model_path,
                                              map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.transform = train_transforms

    def detect(self, image_pil: Image.Image):
        img_t = self.transform(image_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(img_t)[0]

        mask = outputs['scores'] > self.score_threshold
        boxes = outputs['boxes'][mask].cpu()
        scores = outputs['scores'][mask].cpu()
        labels = outputs['labels'][mask].cpu()
        return boxes, scores, labels


class SolarPanelLocationSuggester:
    def __init__(self, min_area_ratio=0.2, size=(512, 512)):
        self.min_area_ratio = min_area_ratio
        self.size = size

    def suggest(self, boxes: torch.Tensor):
        # Create full roof mask (dummy assumption: entire image is roof)
        H, W = self.size
        roof_mask = np.ones((H, W), dtype=bool)
        existing_panels_mask = np.zeros((H, W), dtype=bool)

        for box in boxes:
            x1, y1, x2, y2 = box.int()
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            existing_panels_mask[y1:y2, x1:x2] = True

        available_area = np.logical_and(roof_mask,
                                        np.logical_not(existing_panels_mask))
        avail_ratio = np.sum(available_area) / np.sum(roof_mask)

        if avail_ratio < self.min_area_ratio:
            suggestion = "Limited space available for additional solar panels."
        else:
            suggestion = "Sufficient space detected for new solar panel installation."

        return suggestion, available_area.astype(np.uint8)


# Initialize detector if model file exists
detector = None
suggester = SolarPanelLocationSuggester(min_area_ratio=0.25, size=(512, 512))
if os.path.exists(MODEL_PATH):
    detector = SolarPanelDetector(MODEL_PATH, device=device, score_threshold=0.5)
    print("Loaded trained RetinaNet model.")
else:
    print("WARNING: Model file not found, detection endpoint will be disabled.")

# ==============================
# SIMULATED SATELLITE & SUBSIDY
# ==============================

class SolarAnalysisSystem:
    def __init__(self):
        self.sample_data = {}
        self.analysis_results = {}

    def generate_sample_id(self):
        return f"SOLAR_{random.randint(1000, 9999)}"

    def simulate_satellite_analysis(self, latitude, longitude, buffer_distance=15):
        has_solar = random.random() > 0.4
        confidence = has_solar * (0.88 + random.random() * 0.08) + \
                     (not has_solar) * (0.65 + random.random() * 0.25)
        panel_count = has_solar * random.randint(4, 22)
        pv_area = has_solar * round(panel_count * 1.65 + random.random() * 1.8, 1)
        capacity = has_solar * round(pv_area * 0.21, 1)

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
            if qc_status == "NO_SOLAR_DETECTED":
                qc_notes = [
                    "No solar panel signatures found within 15m radius",
                    "Clear rooftop visibility in scanned area"
                ]
            else:
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
                "capture_date": (datetime.now() - timedelta(
                    days=random.randint(1, 30)
                )).strftime("%Y-%m-%d"),
                "cloud_cover": round(random.random() * 0.3, 2)
            }
        }

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
# FLASK APP & ROUTES
# ==============================

app = Flask(__name__)


@app.route('/')
def index():
    # You can create templates/index.html if you want a UI.
    return "Solar Analysis & Detection API is running."


# --------- AI Detection Endpoint ---------

@app.route('/detect_panels', methods=['POST'])
def detect_panels():
    """
    Upload an image and run RetinaNet detection + roof space suggestion.

    Use multipart/form-data with field 'image'.
    """
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


# Optional: route to trigger training from API (be careful, long-running)
@app.route('/train_model', methods=['POST'])
def train_model_route():
    """
    Trigger training on the server.
    In production, this is usually done offline, not via HTTP.
    """
    try:
        train_model()
        return jsonify({'message': 'Training completed', 'model_path': MODEL_PATH})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# --------- Simulated Satellite Analysis ---------

@app.route('/analyze', methods=['POST'])
def analyze_location():
    try:
        data = request.get_json()
        sample_id = data.get('sample_id', solar_system.generate_sample_id())
        latitude = float(data.get('latitude', 0))
        longitude = float(data.get('longitude', 0))
        buffer_distance = int(data.get('buffer_distance', 15))

        if not latitude or not longitude:
            return jsonify({'error': 'Latitude and longitude are required'}), 400

        results = solar_system.simulate_satellite_analysis(
            latitude, longitude, buffer_distance
        )

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

        solar_system.analysis_results[sample_id] = response_data
        return jsonify(response_data)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/subsidy', methods=['POST'])
def calculate_subsidy():
    try:
        data = request.get_json()
        capacity = float(data.get('capacity', 3))
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
    if sample_id not in solar_system.analysis_results:
        return jsonify({'error': 'Sample ID not found'}), 404

    results = solar_system.analysis_results[sample_id]
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
    try:
        if 'csv_file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['csv_file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'File must be a CSV'}), 400

        stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
        csv_input = csv.DictReader(stream)

        required_columns = ['sample_id', 'latitude', 'longitude']
        if not all(col in csv_input.fieldnames for col in required_columns):
            return jsonify({'error': f'CSV must contain columns: {required_columns}'}), 400

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
    return jsonify({
        'total_samples': len(solar_system.analysis_results),
        'samples': list(solar_system.analysis_results.keys())
    })


@app.route('/sample/<sample_id>')
def get_sample(sample_id):
    if sample_id not in solar_system.analysis_results:
        return jsonify({'error': 'Sample not found'}), 404
    return jsonify(solar_system.analysis_results[sample_id])


if __name__ == '__main__':
    print("Solar Analysis & Detection System running on http://localhost:5000")
    print("- /detect_panels (POST, multipart/form-data, field 'image')")
    print("- /train_model (POST)  [optional]")
    print("- /analyze (POST, JSON)")
    print("- /subsidy (POST, JSON)")
    print("- /upload_csv (POST, CSV)")
    print("- /download/<sample_id> (GET)")
    print("- /samples (GET)")
    print("- /sample/<sample_id> (GET)")

    app.run(debug=True, host='0.0.0.0', port=5000)