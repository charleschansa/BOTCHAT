from flask import Flask, request, jsonify, render_template
from PIL import Image
import io
import tensorflow as tf
import requests
import os
import google.generativeai as genai
from dotenv import load_dotenv
from tensorflow.keras.layers import InputLayer

load_dotenv()

app = Flask(__name__)

# Load your trained model with custom objects
custom_objects = {'InputLayer': InputLayer}
model = tf.keras.models.load_model('Cassava_Disease_Model.h5', custom_objects=custom_objects)

# Define the maize disease labels
disease_labels = [
    "Healthy",  # 0
    "Maize Common Rust",  # 1
    "Maize Gray Leaf Spot",  # 2
    "Maize Northern Leaf Blight",  # 3
    "Maize Southern Leaf Blight"  # 4
]

# Configure Generative AI
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

generation_config = {
    "temperature": 0,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}
safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
]

gen_model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    safety_settings=safety_settings,
    generation_config=generation_config,
    system_instruction="Respond as a knowledgeable and friendly agricultural expert specifically for Zambian farmers, providing clear and concise explanations, suggesting specific solutions, and reassuring the user with encouragement. Avoid technical jargon by using simple language, tailor your response to the specific question, and offer hopeful solutions. Do not answer anything out of agriculture.",
)

chat_session = gen_model.start_chat(history=[])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        img = Image.open(io.BytesIO(file.read()))
        img = img.resize((200, 200))  # Resize image to match model input size
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = tf.expand_dims(img, 0)  # Add batch dimension
        prediction = model.predict(img)
        # Assuming your model returns a class index
        predicted_class = tf.argmax(prediction[0]).numpy()
        disease_name = disease_labels[predicted_class]
        return jsonify({'prediction': disease_name})

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    response = chat_session.send_message(user_message)
    model_response = response.text
    chat_session.history.append({"role": "user", "parts": [user_message]})
    chat_session.history.append({"role": "model", "parts": [model_response]})
    return jsonify({'response': model_response})

if __name__ == '__main__':
    app.run(debug=True)