from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import torch
from model import FruitQualityModel
from torchvision import transforms
from PIL import Image

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Define transformation (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load your trained model
num_fruits = 14  # Total fruit categories
num_qualities = 2  # Good or Bad
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FruitQualityModel(num_fruits, num_qualities)
model.load_state_dict(torch.load('efficientnet_b0_fruit_quality_best.pth', map_location=device))
model = model.to(device)
model.eval()

# Category and quality mapping
category_to_label = ["Apple", "Banana", "Carrot", "Cucumber", "Grape", "Guava", "Mango", "Papaya", "Pomegranate", "Potato", "Strawberry", "Tomato", "Watermelon"]
quality_to_label = ["Good", "Bad"]

# Prediction function
def predict_image_class(image_path):
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        fruit_output, quality_output = model(img_tensor)
        _, fruit_pred = torch.max(fruit_output, 1)
        _, quality_pred = torch.max(quality_output, 1)

    fruit_name = category_to_label[fruit_pred.item()]
    quality_name = quality_to_label[quality_pred.item()]

    return fruit_name, quality_name

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Predict the image
    fruit, quality = predict_image_class(file_path)

    return jsonify({'fruit': fruit, 'quality': quality, 'image_path': file_path})

if __name__ == '__main__':
    app.run(debug=True)
