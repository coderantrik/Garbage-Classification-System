import gradio as gr
import numpy as np
from PIL import Image
import json
from tensorflow.keras.models import load_model

# ------------------------------
# Load SavedModel
# ------------------------------
model = load_model("garbage_model_tf")

# ------------------------------
# Load Class Mapping
# ------------------------------
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

categories = {v: k for k, v in class_indices.items()}
# print("Class mapping:", categories)

# ------------------------------
# Waste Suggestions
# ------------------------------
waste_suggestions = {
    "paper": "♻️ Recycle or reuse as notepads,  Avoid burning as it causes pollution.",
    "cardboard": "📦 Reuse for storage or recycle at paper recycling centers.",
    "plastic": "🚯 Reduce usage. Recycle bottles, containers, and use eco-bricks.",
    "metal": "🔧 Send to scrap dealers for recycling. Can be melted & reused..",
    "trash": "🗑️ Dispose properly in dry waste bins. Avoid mixing with recyclables.",
    "battery": "⚡ Take to e-waste collection centers. Never throw in household bins.",
    "shoes": "👟 Donate if wearable. Otherwise recycle materials where possible",
    "clothes": "👕 Donate to NGOs or reuse as cleaning cloth. Recycle if damaged.",
    "green-glass": "🍾 Recycle at glass collection centers. Can be remolded.",
    "brown-glass": "🍶 Reuse bottles or recycle. Keep separate by color.",
    "white-glass": "🥛 Reuse jars or recycle. Avoid breakage when disposing.",
    "biological": "🌱 CoCompost food and organic waste. Avoid plastic mixing."
}

# ------------------------------
# Preprocessing Function
# ------------------------------
def preprocess_image(img: Image.Image):
    img = img.convert("RGB")
    img = img.resize((320, 320))

    img_array = np.array(img, dtype=np.float32)

    
    img_array = np.expand_dims(img_array, axis=0)
    

    return img_array

# ------------------------------
# Prediction Function
# ------------------------------
def classify_garbage(img: Image.Image):
    try:
        img_array = preprocess_image(img)

        # Debug: check input range
        print("Input range:", img_array.min(), img_array.max())

        # Model prediction
        preds = model.predict(img_array)[0]

        # Top-3 predictions
        top_indices = preds.argsort()[-3:][::-1]

        top_results = [
            f"{categories[i]} ({preds[i]*100:.2f}%)"
            for i in top_indices
        ]

        # Best prediction
        class_idx = top_indices[0]
        confidence = float(preds[class_idx])
        class_name = categories[class_idx]

        # Low confidence handling
        if confidence < 0.4:
            return (
                " Low confidence prediction",
                "Try a clearer image or better lighting"
            )

        suggestion = waste_suggestions.get(class_name, "No suggestion available")

        # Debug output
        print("Prediction vector:", preds)
        print("Top-3:", top_results)

        return (
            f" Prediction: {class_name.capitalize()} ({confidence*100:.2f}%)\n\n"
            f"Top-3 Predictions:\n" + "\n".join(top_results),
            f" Suggestion: {suggestion}"
        )

    except Exception as e:
        return "Error processing image", str(e)

# ------------------------------
# Gradio UI
# ------------------------------
iface = gr.Interface(
    fn=classify_garbage,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Textbox(label="Suggestion")
    ],
    title="♻️ AI-Powered Garbage Classification System",
    description="Upload waste image to classify and get disposal suggestion."
)

# ------------------------------
# Run App
# ------------------------------
if __name__ == "__main__":
    iface.launch()