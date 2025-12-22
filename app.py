import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# ------------------------------
# Loading TFLite Model
# ------------------------------
model_path = r"C:\Users\antri\OneDrive\Desktop\Garbage_classi\garbage_model.tflite"

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ------------------------------
#  Defining Categories and Suggestions
# ------------------------------
categories = {
    0: 'paper', 1: 'cardboard', 2: 'plastic', 3: 'metal',
    4: 'trash', 5: 'battery', 6: 'shoes', 7: 'clothes',
    8: 'green-glass', 9: 'brown-glass', 10: 'white-glass',
    11: 'biological'
}

waste_suggestions = {
    "paper": "♻️ Recycle or reuse as notepads. Avoid burning as it causes pollution.",
    "cardboard": "📦 Reuse for storage or recycle at paper recycling centers.",
    "plastic": "🚯 Reduce usage. Recycle bottles, containers, and use eco-bricks.",
    "metal": "🔧 Send to scrap dealers for recycling. Can be melted & reused.",
    "trash": "🗑️ Dispose properly in dry waste bins. Avoid mixing with recyclables.",
    "battery": "⚡ Take to e-waste collection centers. Never throw in household bins.",
    "shoes": "👟 Donate if wearable. Otherwise recycle materials where possible.",
    "clothes": "👕 Donate to NGOs or reuse as cleaning cloth. Recycle if damaged.",
    "green-glass": "🍾 Recycle at glass collection centers. Can be remolded.",
    "brown-glass": "🍶 Reuse bottles or recycle. Keep separate by color.",
    "white-glass": "🥛 Reuse jars or recycle. Avoid breakage when disposing.",
    "biological": "🌱 Compost food and organic waste. Avoid plastic mixing."
}

# ------------------------------
#  Preprocessing for TFLite
# ------------------------------
def preprocess_image(img: Image.Image):
    # Resize according to model input
    input_shape = input_details[0]['shape'][1:3]  # e.g., (224, 224)
    img = img.resize(tuple(input_shape))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ------------------------------
# Prediction Function
# ------------------------------
def classify_garbage(img: Image.Image):
    try:
        img_array = preprocess_image(img)

        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], img_array)

        # Run inference
        interpreter.invoke()

        # Get prediction
        output_data = interpreter.get_tensor(output_details[0]['index'])
        class_idx = np.argmax(output_data)
        class_name = categories[class_idx]
        suggestion = waste_suggestions[class_name]

        return f"🔍 Prediction: {class_name.capitalize()}", f"💡 Suggestion: {suggestion}"
    except Exception as e:
        return "Error processing image", str(e)

# ------------------------------
#  Gradio Interface
# ------------------------------
iface = gr.Interface(
    fn=classify_garbage,
    inputs=gr.Image(type="pil", label="Upload an image of waste"),
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Textbox(label="Suggestion")
    ],
    title="♻️ Garbage Classification System (TFLite Version)",
    description="Upload an image to identify waste type and get disposal suggestions.",
    allow_flagging="never"
)

# ------------------------------
#  Run App
# ------------------------------
if __name__ == "__main__":
    iface.launch()
