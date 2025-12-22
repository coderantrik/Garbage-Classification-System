# ♻️ AI-Powered Garbage Classification System

An intelligent waste classification system leveraging **deep learning and transfer learning** to automatically identify garbage categories from images and provide **human-readable disposal recommendations**. The system is designed with a strong focus on **environmental sustainability**, **practical deployment**, and **edge-friendly inference**.

---

## 📌 Overview

Rapid urbanization has led to a significant increase in municipal solid waste, making manual segregation inefficient and error-prone. This project proposes an **AI-driven garbage classification framework** based on the **Xception convolutional neural network**, trained on a publicly available Kaggle dataset. The trained model is optimized and deployed using **TensorFlow Lite**, enabling lightweight and efficient inference through a **Gradio-based interactive interface**.

---

## 🚀 Key Features

- Image-based waste classification into **12 garbage categories**
- **Xception transfer learning** for high accuracy with limited training data
- Lightweight **TensorFlow Lite** deployment for efficient inference
- **Gradio web interface** for real-time user interaction
- Context-aware **waste disposal and recycling suggestions**
- Modular and clean project structure (training + inference separation)

---

## 🧠 Garbage Categories

The system classifies waste into the following categories:

- Paper  
- Cardboard  
- Plastic  
- Metal  
- Trash  
- Battery  
- Shoes  
- Clothes  
- Green Glass  
- Brown Glass  
- White Glass  
- Biological Waste  

---

## 🏗️ System Architecture

1. **Image Input** – User uploads an image of waste  
2. **Preprocessing** – Image resizing and normalization  
3. **Model Inference** – Xception-based CNN (TensorFlow Lite)  
4. **Prediction Output** – Waste category identification  
5. **Recommendation Engine** – Sustainable disposal guidance  
6. **User Interface** – Gradio-based web application  

---

## 🧪 Model Training

- **Architecture**: Xception (pretrained on ImageNet)
- **Training Strategy**: Transfer learning with frozen base layers
- **Dataset**: Kaggle Garbage Classification Dataset
- **Image Size**: 224 × 224
- **Optimizer**: Adam
- **Loss Function**: Categorical Cross-Entropy
- **Evaluation Metrics**:
  - Accuracy
  - Precision
  - Recall
  - F1-Score (Macro & Weighted)

📁 Training scripts are available in the `training/` folder.  
📦 The dataset is not included due to size and licensing constraints.

---

## 📊 Performance Summary

- **Test Accuracy**: ~93%
- **Macro F1-Score**: ~0.90
- Consistent performance across both majority and minority waste categories

These results demonstrate the feasibility of deep learning-based waste classification systems for real-world deployment.

---

## 💻 Deployment

- **Inference Format**: TensorFlow Lite (`.tflite`)
- **Platform**: Local desktop deployment
- **Interface**: Gradio web UI
- **Hardware Requirement**: CPU-only (no GPU required)

---

## ▶️ How to Run the Application

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
