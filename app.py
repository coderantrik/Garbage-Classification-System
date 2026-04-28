import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import json
from tensorflow.keras.applications.xception import preprocess_input

# ------------------------------
# Load TFLite Model
# ------------------------------
model_path = "garbage2_model.tflite"

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ------------------------------
# Load Class Mapping
# ------------------------------
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

categories = {v: k for k, v in class_indices.items()}
print("Class mapping:", categories)
# ------------------------------
# Waste Suggestions
# ------------------------------
waste_suggestions = {
    "paper": "♻️ Recycle or reuse as notepads.",
    "cardboard": "📦 Reuse or recycle.",
    "plastic": "🚯 Reduce usage and recycle.",
    "metal": "🔧 Recycle via scrap dealers.",
    "trash": "🗑️ Dispose properly.",
    "battery": "⚡ Use e-waste centers.",
    "shoes": "👟 Donate or recycle.",
    "clothes": "👕 Donate or reuse.",
    "green-glass": "🍾 Recycle.",
    "brown-glass": "🍶 Reuse or recycle.",
    "white-glass": "🥛 Handle carefully.",
    "biological": "🌱 Compost."
}

# ------------------------------
# Preprocessing (MATCH TRAINING)
# ------------------------------
def preprocess_image(img: Image.Image):
    input_shape = input_details[0]['shape'][1:3]

    img = img.convert("RGB")
    img = img.resize(tuple(input_shape))

    img_array = np.array(img, dtype=np.float32)
    img_array = preprocess_input(img_array)

    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ------------------------------
# Prediction Function
# ------------------------------
def classify_garbage(img: Image.Image):
    try:
        img_array = preprocess_image(img)

        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()

        output_data = interpreter.get_tensor(output_details[0]['index'])

        output_data = output_data / np.sum(output_data)

        class_idx = int(np.argmax(output_data))
        confidence = float(np.max(output_data))

        class_name = categories[class_idx]
        suggestion = waste_suggestions.get(class_name, "No suggestion available")

        return (
            f"🔍 Prediction: {class_name.capitalize()} ({confidence*100:.2f}%)",
            f"💡 Suggestion: {suggestion}"
        )

    except Exception as e:
        return "Error processing image", str(e)

# ------------------------------
# Gradio UI
# ------------------------------
iface = gr.Interface(
    fn=classify_garbage,
    inputs=gr.Image(type="pil"),
    outputs=["text", "text"],
    title="♻️ Garbage Classification System",
    description="Upload waste image to classify and get disposal suggestion."
)

# ------------------------------
# Run App
# ------------------------------
if __name__ == "__main__":
    iface.launch()