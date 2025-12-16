from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
import base64
import tensorflow as tf
from tensorflow import keras
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Load your trained model
MODEL_PATH = '../best_efficientnetb0.h5'
model = None

# Class names (you'll need to update this based on your model's classes)
CLASS_NAMES = [
    'Aphid', 'Ladybug', 'Caterpillar', 'Beetle', 'Grasshopper',
    'Spider', 'Ant', 'Bee', 'Butterfly', 'Moth'
]  # Update this with your actual class names

def load_model():
    """Load the trained model"""
    global model
    try:
        if os.path.exists(MODEL_PATH):
            model = keras.models.load_model(MODEL_PATH)
            print(f"Model loaded successfully from {MODEL_PATH}")
            return True
        else:
            print(f"Model file not found at {MODEL_PATH}")
            return False
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for model input"""
    try:
        # Resize image
        image = image.resize(target_size)
        # Convert to array
        img_array = np.array(image)
        # Normalize to [0, 1]
        img_array = img_array.astype('float32') / 255.0
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        raise Exception(f"Error preprocessing image: {str(e)}")

def predict_insect(image):
    """Predict insect class using the model"""
    try:
        # Preprocess image
        processed_image = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(processed_image, verbose=0)
        
        # Get predicted class
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        # Get class name
        predicted_class = CLASS_NAMES[predicted_class_idx] if predicted_class_idx < len(CLASS_NAMES) else f"Class_{predicted_class_idx}"
        
        # Get top 3 predictions
        top_indices = np.argsort(predictions[0])[-3:][::-1]
        top_predictions = [
            {
                'class': CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"Class_{idx}",
                'confidence': float(predictions[0][idx])
            }
            for idx in top_indices
        ]
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'top_predictions': top_predictions,
            'all_predictions': {CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"Class_{i}": float(predictions[0][i]) for i in range(len(predictions[0]))}
        }
    except Exception as e:
        raise Exception(f"Error making prediction: {str(e)}")

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict insect from uploaded image"""
    try:
        if model is None:
            return jsonify({
                'error': 'Model not loaded',
                'message': 'Please ensure the model file exists and is loaded correctly'
            }), 500
        
        # Get image from request
        if 'image' not in request.files and 'image' not in request.json:
            return jsonify({'error': 'No image provided'}), 400
        
        # Handle base64 image
        if 'image' in request.json:
            image_data = request.json['image']
            # Remove data URL prefix if present
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
        # Handle file upload
        elif 'image' in request.files:
            file = request.files['image']
            image = Image.open(file.stream)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Make prediction
        result = predict_insect(image)
        
        # Format response similar to Gemini API format
        response = {
            'success': True,
            'prediction': {
                'name': result['predicted_class'],
                'confidence': result['confidence'],
                'top_predictions': result['top_predictions']
            },
            'model_type': 'custom_efficientnetb0'
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500

@app.route('/predict-detailed', methods=['POST'])
def predict_detailed():
    """Predict with detailed information (combines model prediction with basic info)"""
    try:
        if model is None:
            return jsonify({
                'error': 'Model not loaded'
            }), 500
        
        # Get image
        if 'image' not in request.files and 'image' not in request.json:
            return jsonify({'error': 'No image provided'}), 400
        
        if 'image' in request.json:
            image_data = request.json['image']
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
        elif 'image' in request.files:
            file = request.files['image']
            image = Image.open(file.stream)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Make prediction
        result = predict_insect(image)
        predicted_class = result['predicted_class']
        
        # Basic information mapping (you can expand this)
        insect_info = {
            'Aphid': {
                'scientific': 'Aphidoidea',
                'family': 'Aphididae',
                'habitat': 'Crops, gardens, and plants',
                'harmful': 'Yes - Harmful',
                'recommendation': 'Use insecticidal soap or neem oil',
                'description': 'Aphids are small sap-sucking insects that can cause significant damage to crops.'
            },
            'Ladybug': {
                'scientific': 'Coccinellidae',
                'family': 'Coccinellidae',
                'habitat': 'Gardens and agricultural fields',
                'harmful': 'No - Beneficial',
                'recommendation': 'Keep - Helps control aphids',
                'description': 'Ladybugs are beneficial insects that prey on aphids and other harmful pests.'
            },
            'Caterpillar': {
                'scientific': 'Lepidoptera larvae',
                'family': 'Various',
                'habitat': 'Plants and crops',
                'harmful': 'Yes - Harmful',
                'recommendation': 'Remove manually or use Bt (Bacillus thuringiensis)',
                'description': 'Caterpillars are the larval stage of butterflies and moths.'
            }
        }
        
        # Get info for predicted class or use defaults
        info = insect_info.get(predicted_class, {
            'scientific': 'N/A',
            'family': 'N/A',
            'habitat': 'N/A',
            'harmful': 'Unknown',
            'recommendation': 'Consult with agricultural expert',
            'description': f'Predicted as {predicted_class} with {result["confidence"]*100:.1f}% confidence.'
        })
        
        # Format response in Gemini-compatible format
        response = {
            'name': predicted_class,
            'scientific': info['scientific'],
            'family': info['family'],
            'habitat': info['habitat'],
            'harmful': info['harmful'],
            'recommendation': info['recommendation'],
            'description': info['description'],
            'confidence': result['confidence'],
            'model_type': 'custom_efficientnetb0'
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    print("Loading model...")
    if load_model():
        print("Starting Flask server...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load model. Please check the model path.")

