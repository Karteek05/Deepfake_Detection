import gradio as gr
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.applications.xception import preprocess_input # type: ignore
from PIL import Image
import traceback

# ✅ Load the FIXED model instead of the overfitting one
print("✅ Loading fixed model...")
try:
    model = load_model("deepfake_detector_fixed.h5")  # 🔥 UPDATED HERE
    print("✅ Model loaded!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise

# Find the last convolutional layer dynamically
def get_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None

last_conv_layer_name = get_last_conv_layer(model)
if not last_conv_layer_name:
    raise ValueError("No convolutional layer found in the model!")
print(f"✅ Using last conv layer: {last_conv_layer_name}")

# ✅ Temperature Scaling to Fix Overconfident Predictions
def temperature_scaling(logits, T=2.5):
    """Applies temperature scaling to smooth confidence scores."""
    logits = np.array(logits)
    scaled_logits = logits / T
    return 1 / (1 + np.exp(-scaled_logits))  # Sigmoid scaling

# Define preprocessing
def preprocess_image(image_pil):
    try:
        image_pil = image_pil.resize((256, 256))  # Ensure size matches model input
        image_array = np.array(image_pil)
        if image_array.shape[-1] == 4:  # Handle RGBA images
            image_array = image_array[:, :, :3]
        image_array = preprocess_input(image_array)  # Normalize
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
        return image_array
    except Exception as e:
        print(f"❌ Error in preprocessing: {e}")
        traceback.print_exc()
        raise

# Grad-CAM function
def get_gradcam_heatmap(model, image_array, target_class):
    try:
        grad_model = tf.keras.models.Model(
            [model.input], 
            [model.get_layer(last_conv_layer_name).output, model.output]
        )
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image_array, training=False)
            loss = predictions[:, target_class] if predictions.shape[-1] > 1 else predictions[:, 0]
        
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_outputs = conv_outputs[0]  # Remove batch dimension
        heatmap = np.zeros(dtype=np.float32, shape=conv_outputs.shape[:-1])

        for i in range(pooled_grads.shape[-1]):
            heatmap += pooled_grads[i] * conv_outputs[:, :, i]

        heatmap = np.maximum(heatmap, 0)  # ReLU operation

        if np.max(heatmap) > 0:
            heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
        else:
            heatmap = np.zeros_like(heatmap)  # Avoid division by zero

        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.resize(heatmap, (256, 256))  # Resize to match input dimensions
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  # Convert to RGB

        return heatmap
    except Exception as e:
        print(f"❌ Error generating Grad-CAM: {e}")
        traceback.print_exc()
        return np.zeros((256, 256, 3), dtype=np.uint8)  # Return blank image on error

# Prediction function
def predict(image_np):
    if image_np is None:
        return "No image provided", {"Error": 1.0}, np.zeros((256, 256, 3), dtype=np.uint8)
    
    print("🔄 Received image for prediction...")
    try:
        image_pil = Image.fromarray(image_np)
        input_tensor = preprocess_image(image_pil)
        
        print("🔄 Running model inference...")
        predictions = model.predict(input_tensor, verbose=0)

        # 🔍 Debugging: Print raw predictions BEFORE any processing
        print(f"🔍 Raw Model Output: {predictions}")

        # ✅ DISABLE Temperature Scaling for Debugging
        adjusted_predictions = predictions  # Remove scaling for now

        # Determine output format
        if adjusted_predictions.shape[-1] == 1:  # Binary classification
            print("Detected single-output model")
            prediction_value = float(adjusted_predictions[0][0])

            # 🔍 Debugging: Print decision process
            print(f"🔍 Model thinks: Real Confidence = {prediction_value}, Fake Confidence = {1 - prediction_value}")

            # ✅ Adjust classification threshold if necessary
            if prediction_value >= 0.6:  # Switched condition for testing
                predicted_label = "Fake"
            else:
                predicted_label = "Real"

            print(f"🔍 Final Decision: {predicted_label}")

            real_confidence = prediction_value
            fake_confidence = 1 - prediction_value
            confidence = np.array([real_confidence, fake_confidence])
            class_labels = ["Real", "Fake"]
            target_class = 0  # Focus on what the model sees as "real" features
        else:  # Two outputs (softmax)
            print("Detected two-output model")
            confidence = tf.nn.softmax(adjusted_predictions[0]).numpy()
            class_labels = ["Real", "Fake"]
            predicted_label = class_labels[np.argmax(confidence)]
            target_class = np.argmax(confidence)

        print(f"✅ Prediction complete! Result: {predicted_label} with confidence {confidence}")

        print("🔄 Generating Grad-CAM heatmap...")
        heatmap = get_gradcam_heatmap(model, input_tensor, target_class=target_class)
        print("✅ Grad-CAM generated!")

        confidence_dict = {class_labels[i]: float(confidence[i]) for i in range(len(class_labels))}

        # ✅ Override: If fake_confidence > real_confidence, force the label to Fake
        if real_confidence > fake_confidence:
            predicted_label = "Real"
        else:
            predicted_label = "Fake"
            print(f"🟢 Final Decision: {predicted_label} (Real: {real_confidence:.2f}, Fake: {fake_confidence:.2f})")

        return predicted_label, confidence_dict, heatmap
    except Exception as e:
        print(f"❌ Error in prediction: {e}")
        traceback.print_exc()
        return f"Error: {str(e)}", {"Error": 1.0}, np.zeros((256, 256, 3), dtype=np.uint8)

# Gradio UI
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy"),
    outputs=[
        gr.Label(num_top_classes=2, label="Prediction"),
        gr.JSON(label="Confidence Scores"),
        gr.Image(type="numpy", label="Grad-CAM Heatmap")
    ],
    title="AI-Based Deepfake Detector",
    description="Upload an image to check if it's real or fake. The model provides explainability using Grad-CAM heatmaps.",
)

if __name__ == "__main__":
    print("🚀 Starting Gradio app...")
    demo.launch(share=True, debug=True)