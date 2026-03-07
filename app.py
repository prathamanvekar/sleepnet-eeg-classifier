"""
app.py
======
Simple Flask web interface for sleep stage classification.
"""

import os
import numpy as np
import torch
from flask import Flask, render_template, request, jsonify
import mne
from io import BytesIO
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import SleepNet

app = Flask(__name__)

# Configuration
MODEL_PATH = "sleepnet.pth"
SFREQ = 100
EPOCH_SEC = 30
SAMPLES_PER_EPOCH = SFREQ * EPOCH_SEC
STAGE_NAMES = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SleepNet(n_classes=5).to(device)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"Model loaded from {MODEL_PATH}")
else:
    print(f"WARNING: Model not found at {MODEL_PATH}")


def preprocess_signal(data, sfreq=100):
    """Preprocess EEG signal."""
    # Bandpass filter
    data = mne.filter.filter_data(data.astype(np.float64), sfreq, 0.5, 30.0, verbose=False)
    # Normalize
    data = (data - np.mean(data)) / np.std(data)
    return data


def predict_stages(eeg_data):
    """Predict sleep stages for EEG data."""
    # Split into epochs
    n_epochs = len(eeg_data) // SAMPLES_PER_EPOCH
    epochs = []

    for i in range(n_epochs):
        start = i * SAMPLES_PER_EPOCH
        end = start + SAMPLES_PER_EPOCH
        epoch = eeg_data[start:end]
        if len(epoch) == SAMPLES_PER_EPOCH:
            epochs.append(epoch)

    if len(epochs) == 0:
        return []

    # Convert to tensor
    X = np.array(epochs, dtype=np.float32)
    X = torch.tensor(X).unsqueeze(1).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(X)
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()

    return predictions.tolist()


def create_hypnogram(predictions):
    """Create hypnogram visualization."""
    fig, ax = plt.subplots(figsize=(12, 4))

    time_hours = np.arange(len(predictions)) * 30 / 3600
    ax.plot(time_hours, predictions, 'b-', linewidth=1)
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_yticklabels(["W", "N1", "N2", "N3", "REM"])
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Sleep Stage')
    ax.set_title('Predicted Hypnogram')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)

    # Save to base64
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    return img_base64


# HTML Template (inline for simplicity)
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Sleep Stage Classification</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 900px; margin: 50px auto; padding: 20px; }
        h1 { color: #2c3e50; }
        .upload-form { background: #ecf0f1; padding: 20px; border-radius: 10px; margin: 20px 0; }
        .btn { background: #3498db; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
        .btn:hover { background: #2980b9; }
        .results { margin-top: 20px; }
        .metrics { background: #fff; padding: 15px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        img { max-width: 100%; margin-top: 20px; }
    </style>
</head>
<body>
    <h1>🌙 Sleep Stage Classification</h1>
    <p>Upload a CSV file with EEG data (30-second epochs at 100 Hz)</p>

    <div class="upload-form">
        <form id="upload-form" enctype="multipart/form-data">
            <input type="file" name="file" accept=".csv,.edf" required>
            <button type="submit" class="btn">Analyze</button>
        </form>
    </div>

    <div id="results" class="results" style="display: none;">
        <div class="metrics">
            <h3>Results</h3>
            <p>Total Epochs: <span id="num-epochs"></span></p>
            <p>Stage Distribution: <span id="distribution"></span></p>
        </div>
        <h3>Hypnogram</h3>
        <img id="hypnogram" src="" alt="Hypnogram">
    </div>

    <script>
        document.getElementById('upload-form').onsubmit = async function(e) {
            e.preventDefault();
            const formData = new FormData(this);
            const response = await fetch('/predict', {method: 'POST', body: formData});
            const data = await response.json();

            if (data.error) {
                alert(data.error);
                return;
            }

            document.getElementById('num-epochs').textContent = data.num_epochs;
            document.getElementById('distribution').textContent = data.distribution;
            document.getElementById('hypnogram').src = 'data:image/png;base64,' + data.hypnogram;
            document.getElementById('results').style.display = 'block';
        };
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return HTML_TEMPLATE


@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']

        if file.filename.endswith('.csv'):
            # Load CSV
            data = np.loadtxt(file, delimiter=',')
            if data.ndim > 1:
                data = data[:, 0]  # Take first column
        else:
            return jsonify({'error': 'Unsupported file format. Use CSV.'})

        # Preprocess
        data = preprocess_signal(data)

        # Predict
        predictions = predict_stages(data)

        if len(predictions) == 0:
            return jsonify({'error': 'Not enough data for analysis'})

        # Create hypnogram
        hypnogram = create_hypnogram(predictions)

        # Calculate distribution
        unique, counts = np.unique(predictions, return_counts=True)
        dist = ', '.join([f"{STAGE_NAMES[u]}: {c}" for u, c in zip(unique, counts)])

        return jsonify({
            'num_epochs': len(predictions),
            'distribution': dist,
            'hypnogram': hypnogram
        })

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True, port=5001)
