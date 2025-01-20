from flask import Flask, request, jsonify
from flask_cors import CORS
import torch.nn as nn
import base64
import numpy as np
import cv2
import logging
import io
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load model (assuming it works as expected)
class FacialExpressionModel(nn.Module):
    def __init__(self):
        super(FacialExpressionModel, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 12 * 12, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 8),
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.fc_layer(x)
        return x

# Load the pre-trained model
try:
    model = FacialExpressionModel()
    model.load_state_dict(torch.load('facial_expression_model.pth', map_location=torch.device('cpu')))
    model.eval()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

# Define emotion labels
EMOTIONS = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral",
    7: "Contempt"
}

# Preprocess the image
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    normalized = resized / 255.0
    tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return tensor

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if 'image' is in the JSON payload
        if 'image' not in request.json:
            return jsonify({"error": "No image provided"}), 400

        # Decode the base64 image
        image_data = request.json['image']
        image_bytes = base64.b64decode(image_data.split(",")[1])
        image = Image.open(io.BytesIO(image_bytes))
        image_np = np.array(image)

        # Preprocess the image
        processed_image = preprocess_image(image_np)

        # Model prediction
        with torch.no_grad():
            outputs = model(processed_image)
            probs = torch.softmax(outputs, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_class].item()

        # Response
        response = {
            'prediction': EMOTIONS[pred_class],
            'confidence': float(confidence),
            'emotion_code': pred_class
        }
        logger.info(f"Prediction successful: {response}")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return jsonify({'error': 'Internal server error'}), 500

# Start the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
