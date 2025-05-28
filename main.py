from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from io import BytesIO

app = Flask(__name__)
CORS(app)  # Habilita CORS para todas as rotas

# Load the model
try:
    model = load_model('finalcropnoses.keras')
    print("Modelo carregado com sucesso!")
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")
    model = None

class_names = ['Bacterial', 'Fungal', 'Healthy']

@app.route('/')
def home():
    return jsonify({"message": "API está funcionando!"})

@app.route('/predict', methods=['POST'])
def predict():
    print("Recebendo requisição de predição...")
    
    if model is None:
        print("Erro: Modelo não está carregado")
        return jsonify({"error": "Modelo não está carregado"}), 500

    if 'image' not in request.files:
        print("Erro: Nenhuma imagem enviada")
        return jsonify({"error": "Nenhuma imagem enviada"}), 400

    img_file = request.files['image']
    print(f"Nome do arquivo recebido: {img_file.filename}")

    if img_file.filename == '':
        print("Erro: Nome do arquivo vazio")
        return jsonify({"error": "No selected file"}), 400

    try:
        print("Iniciando processamento da imagem...")
        # Load and preprocess the image
        img = image.load_img(BytesIO(img_file.read()), target_size=(224, 224))
        print("Imagem carregada com sucesso")
        
        img_array = image.img_to_array(img)
        print("Imagem convertida para array")
        
        img_array = np.expand_dims(img_array, axis=0)  
        img_array /= 255.0  
        print("Imagem pré-processada")
        
        # Predict
        print("Iniciando predição...")
        predictions = model.predict(img_array)
        print(f"Predições obtidas: {predictions}")
        
        predicted_index = np.argmax(predictions)
        predicted_class = class_names[predicted_index]
        confidence = float(predictions[0][predicted_index])
        print(f"Classe predita: {predicted_class}, Confiança: {confidence}")

        return jsonify({
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4)
        })
    except Exception as e:
        print(f"Erro detalhado durante a predição: {str(e)}")
        import traceback
        print(f"Stack trace: {traceback.format_exc()}")
        return jsonify({"error": "Erro ao processar a imagem"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
