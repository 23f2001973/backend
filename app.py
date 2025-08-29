from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app)  # Allow requests from Expo Go

print("Loading model...")
model = tf.keras.models.load_model('./model/model.h5')
print("Model loaded successfully!")

TARGET_SIZE = (224, 224)
CLASS_LABELS = [
    "Aloevera", "Amla", "Amruthaballi", "Arali", "Astma_weed", "Badipala", "Balloon_Vine", "Bamboo", "Beans", "Betel",
    "Bhrami", "Bringaraja", "Caricature", "Castor", "Catharanthus", "Chakte", "Chilly", "Citron lime (herelikai)",
    "Coffee", "Common rue(naagdalli)", "Coriender", "Curry", "Doddpathre", "Drumstick", "Ekka", "Eucalyptus", "Ganigale",
    "Ganike", "Gasagase", "Ginger", "Globe Amarnath", "Guava", "Henna", "Hibiscus", "Honge", "Insulin", "Jackfruit",
    "Jasmine", "Kambajala", "Kasambruga", "Kohlrabi", "Lantana", "Lemon", "Lemongrass", "Malabar_Nut", "Malabar_Spinach",
    "Mango", "Marigold", "Mint", "Neem", "Nelavembu", "Nerale", "Nooni", "Onion", "Padri", "Palak(Spinach)", "Papaya",
    "Parijatha", "Pea", "Pepper", "Pomoegranate", "Pumpkin", "Raddish", "Rose", "Sampige", "Sapota", "Seethaashoka",
    "Seethapala", "Spinach1", "Tamarind", "Taro", "Tecoma", "Thumbe", "Tomato", "Tulsi", "Turmeric", "ashoka",
    "camphor", "kamakasturi", "kepala"
]

@app.route('/test', methods=['GET'])
def test():
    return jsonify({'status': 'ok'})

@app.route('/predict', methods=['POST'])
def predict():
    print("=== /predict called ===")
    try:
        if 'file' not in request.files:
            print("No file part in request")
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        print(f"Received file: {file.filename}")

        # Read image
        image = Image.open(file.stream).convert('RGB')
        image = image.resize(TARGET_SIZE)
        image_array = np.array(image) / 255.0
        img_tensor = np.expand_dims(image_array, axis=0)

        print("Image processed, running prediction...")
        preds = model.predict(img_tensor)
        pred_index = preds.argmax(axis=1)[0]
        pred_label = CLASS_LABELS[pred_index]

        print(f"Prediction complete: {pred_label}")
        return jsonify({'prediction': pred_label})

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
