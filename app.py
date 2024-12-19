from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2
import os

app = Flask(__name__)

# Load the trained model
model = load_model("models/fine_tuned_vgg16.h5")

# Define the directory to save Grad-CAM images
static_folder = "static"
os.makedirs(static_folder, exist_ok=True)

def preprocess_image(image_file):
    img = Image.open(image_file).convert('RGB')  # Ensure the image has 3 channels (RGB)
    img = img.resize((224, 224))  # Resize to match the model's input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize pixel values to be between 0 and 1
    return img_array, img

def generate_grad_cam(img_array, model, class_idx, layer_name="block5_conv3"):
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_outputs = conv_outputs[0].numpy()
    pooled_grads = pooled_grads.numpy()
    
    for i in range(len(pooled_grads)):
        conv_outputs[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)  # ReLU
    heatmap /= np.max(heatmap)  # Normalize to [0, 1]
    return heatmap

def overlay_heatmap(heatmap, original_img, alpha=0.4):
    heatmap = cv2.resize(heatmap, (original_img.size[0], original_img.size[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(np.array(original_img), 1 - alpha, heatmap, alpha, 0)
    return Image.fromarray(overlay)

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_result = None
    grad_cam_image = None
    probabilities = None
    if request.method == 'POST':
        try:
            file = request.files['file']  # Retrieve the uploaded file
            preprocessed_image, original_image = preprocess_image(file)  # Preprocess the image
            
            prediction = model.predict(preprocessed_image)
            predicted_class = np.argmax(prediction)
            
            class_names = ['Mild Demented', 'Moderate Demented', 'Very Mild Demented', 'Non Demented']
            predicted_label = class_names[predicted_class]
            
            # Extract probabilities for all classes
            probabilities = {class_names[i]: f"{prediction[0][i] * 100:.2f}%" for i in range(len(class_names))}
            
            prediction_result = f"Prediction: {predicted_label} ({probabilities[predicted_label]})"
            
            # Generate Grad-CAM
            heatmap = generate_grad_cam(preprocessed_image, model, predicted_class)
            grad_cam_image = overlay_heatmap(heatmap, original_image)

            # Save Grad-CAM image to static folder
            grad_cam_path = os.path.join(static_folder, "grad_cam_image.jpg")
            grad_cam_image.save(grad_cam_path)

        except Exception as e:
            prediction_result = f"Error: {str(e)}"

    return render_template(
        'home.html',
        prediction_result=prediction_result,
        grad_cam_image="grad_cam_image.jpg" if grad_cam_image else None,
        probabilities=probabilities
    )
