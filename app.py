import tempfile
from flask import Flask, jsonify, render_template, request
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import tifffile
import requests
from datetime import datetime, timedelta
from huggingface_hub import hf_hub_download

UPLOAD_FOLDER = "static/uploads"
VIS_FOLDER = "static/visualizations"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VIS_FOLDER, exist_ok=True)

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



model_tif_path = hf_hub_download(repo_id= "AlshimaaAhmed/landclassification-models" , filename = "TIF_classification.h5")
model_rgb_path = hf_hub_download(repo_id= "AlshimaaAhmed/landclassification-models" , filename = "land_restnet_RGB.h5")
pca_model_path = hf_hub_download(repo_id= "AlshimaaAhmed/landclassification-models" , filename = "pca_model.pkl")

model_tif = load_model(model_tif_path)
model_rgb = load_model(model_rgb_path)
pca_model = joblib.load(pca_model_path)

LABELS = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
    'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake'
]

CLASS_COLORS = {
    'AnnualCrop': (255, 205, 86),
    'Forest': (34, 139, 34),
    'HerbaceousVegetation': (144, 238, 144),
    'Highway': (169, 169, 169),
    'Industrial': (128, 0, 128),
    'Pasture': (255, 165, 0),
    'PermanentCrop': (60, 179, 113),
    'Residential': (220, 20, 60),
    'River': (30, 144, 255),
    'SeaLake': (25, 25, 112)
}

CLIENT_ID = os.environ.get("CLIENT_ID")
CLIENT_SECRET = os.environ.get("CLIENT_SECRET")

def RGB_preprocessing(file_path):
    img = tf.keras.preprocessing.image.load_img(file_path, target_size=(224, 224))
    arr = tf.keras.preprocessing.image.img_to_array(img).astype(np.float32)
    arr = arr / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def preprocess_tif_inference(file_path, target_size=(64,64)):
    img = tifffile.imread(file_path).astype(np.float32)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)

    img = tf.image.resize(img, target_size).numpy()
    img = img / 10000.0
    img = np.expand_dims(img, axis=0)

    return img


def save_pil(image: Image.Image, path: str, quality=90):
    image.save(path, format='PNG', optimize=True)

def make_overlay_image(base_img_path, class_name, alpha=0.45, out_path=None):
    base = Image.open(base_img_path).convert("RGBA").resize((512,512))
    color = CLASS_COLORS.get(class_name, (255, 0, 0))
    overlay = Image.new("RGBA", base.size, color + (int(alpha*255),))
    combined = Image.alpha_composite(base, overlay)
    if out_path:
        combined.save(out_path)
    return combined

def make_bar_chart(probs, labels, out_path):
    plt.figure(figsize=(6,3))
    plt.barh(labels, probs, color=[np.array(CLASS_COLORS[l])/255.0 for l in labels])
    plt.xlim(0,1)
    plt.xlabel("Probability")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

def build_urls(saved, overlay, chart):
    return {
        "image_url": f"/static/uploads/{os.path.basename(saved)}",
        "overlay_url": f"/static/visualizations/{os.path.basename(overlay)}",
        "chart_url": f"/static/visualizations/{os.path.basename(chart)}",
    }

def get_access_token():
    if not CLIENT_ID or not CLIENT_SECRET:
        return None
    token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    data = {
        "grant_type": "client_credentials",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET
    }
    r = requests.post(token_url, data=data, timeout=30)
    if r.status_code == 200:
        return r.json().get("access_token")
    return None

def download_sentinel_rgb(lat, lon, out_path):
    token = get_access_token()
    if token is None:
        return None, "Missing CLIENT_ID/CLIENT_SECRET in environment."
    evalscript = """
    //VERSION=3
    function setup() {
        return { input: [{ bands: ["B04","B03","B02","SCL"], units: "DN" }], output: { bands: 3, sampleType: "AUTO" } };
    }
    function evaluatePixel(sample) {
        if (sample.SCL == 3 || sample.SCL == 8 || sample.SCL == 9 || sample.SCL == 10) { return [0,0,0]; }
        let gain = 2.5;
        return [Math.max(0,Math.min(1,gain*sample.B04/10000)),
                Math.max(0,Math.min(1,gain*sample.B03/10000)),
                Math.max(0,Math.min(1,gain*sample.B02/10000))];
    }
    """
    buffer = 0.008
    bbox = [lon-buffer, lat-buffer, lon+buffer, lat+buffer]
    request_body = {
        "input": {"bounds":{"bbox":bbox,"properties":{"crs":"http://www.opengis.net/def/crs/EPSG/0/4326"}},
                  "data":[{"type":"sentinel-2-l2a",
                           "dataFilter":{"timeRange":{"from":(datetime.now()-timedelta(days=60)).strftime("%Y-%m-%dT00:00:00Z"),
                                                     "to":datetime.now().strftime("%Y-%m-%dT23:59:59Z")},
                                         "maxCloudCoverage":30,"mosaickingOrder":"leastCC"}}]},
        "output":{"width":512,"height":512,"responses":[{"identifier":"default","format":{"type":"image/png"}}]},
        "evalscript": evalscript
    }
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    try:
        r = requests.post("https://sh.dataspace.copernicus.eu/api/v1/process", json=request_body, headers=headers, timeout=90)
        if r.status_code == 200 and 'image' in r.headers.get('content-type',''):
            img = Image.open(BytesIO(r.content)).convert("RGB").resize((512,512))
            img.save(out_path, "PNG")
            return out_path, None
        return None, f"Process API error: {r.status_code}, {r.text[:200]}"
    except Exception as e:
        return None, str(e)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_method = request.form.get('input_method')
    timestamp = int(datetime.now().timestamp())

    if input_method == 'map':
        try:
            lat = float(request.form.get('latitude'))
            lon = float(request.form.get('longitude'))
        except Exception:
            return jsonify({"success": False, "message": "Invalid coordinates"}), 400

        out_image_path = os.path.join(UPLOAD_FOLDER, f"sent_{lat:.6f}_{lon:.6f}_{timestamp}.png")
        downloaded, err = download_sentinel_rgb(lat, lon, out_image_path)
        if not downloaded:
            return jsonify({"success": False, "message": f"Could not download image: {err}"}), 500

        arr = RGB_preprocessing(downloaded)
        pred = model_rgb.predict(arr)[0]
        idx = int(np.argmax(pred))
        label = LABELS[idx]
        confidence = float(pred[idx]) * 100.0

        overlay_path = os.path.join(VIS_FOLDER, f"overlay_{timestamp}.png")
        chart_path = os.path.join(VIS_FOLDER, f"chart_{timestamp}.png")

        make_overlay_image(downloaded, label, alpha=0.45, out_path=overlay_path)
        make_bar_chart(pred, LABELS, chart_path)

        return jsonify({
            "success": True,
            "prediction": label,
            "confidence": f"{confidence:.1f}%",
            "location": f"({lat:.6f}, {lon:.6f})",
            "image_url": f"/static/uploads/{os.path.basename(downloaded)}",
            "overlay_url": f"/static/visualizations/{os.path.basename(overlay_path)}",
            "chart_url": f"/static/visualizations/{os.path.basename(chart_path)}"
        })

    elif input_method == 'upload':
        if 'file' not in request.files:
         return jsonify({"success": False, "message": "No file uploaded"}), 400

        f = request.files['file']
        secure_name = f"{timestamp}_{f.filename.replace(' ', '_')}"
        saved_path = os.path.join(UPLOAD_FOLDER, secure_name)
        f.save(saved_path)

        file_type = request.form.get('file_type', 'rgb')
        if file_type == 'rgb':
            arr = RGB_preprocessing(saved_path)
            pred = model_rgb.predict(arr)[0]

            overlay_path = os.path.join(VIS_FOLDER, f"overlay_{timestamp}.png")
            make_overlay_image(saved_path, LABELS[int(np.argmax(pred))], alpha=0.45, out_path=overlay_path)

        else:
            arr = preprocess_tif_inference(saved_path)
            pred = model_tif.predict(arr)[0]
            overlay_path = None

        idx = int(np.argmax(pred))
        label = LABELS[idx]
        confidence = float(pred[idx]) * 100.0

        chart_path = os.path.join(VIS_FOLDER, f"chart_{timestamp}.png")
        make_bar_chart(pred, LABELS, chart_path)

        result = {
            "success": True,
            "prediction": label,
            "confidence": f"{confidence:.1f}%",
            "image_url": f"/static/uploads/{os.path.basename(saved_path)}",
            "chart_url": f"/static/visualizations/{os.path.basename(chart_path)}"
        }

        if overlay_path:
            result["overlay_url"] = f"/static/visualizations/{os.path.basename(overlay_path)}"

        return jsonify(result)

    return jsonify({"success": False, "message": "Invalid input method"}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
