# ♻️ AI-POWERED GARBAGE CLASSIFICATION SYSTEM FOR SUSTAINABLE WASTE MANAGEMENT

An intelligent waste classification system leveraging **deep learning and transfer learning** to automatically identify garbage categories from images and provide **human-readable disposal recommendations**. The system is designed with a strong focus on **environmental sustainability**, **practical deployment**, and **edge-friendly inference**.

---

## 📌 Overview

Rapid urbanization has led to a significant increase in municipal solid waste, making manual segregation inefficient and error-prone. This project proposes an **AI-driven garbage classification framework** based on the **Xception convolutional neural network**, trained on a publicly available Kaggle dataset. The trained model is optimized and deployed using **TensorFlow Savedmodel**, enabling lightweight and efficient inference through a **Gradio-based interactive interface**.

---

## 🚀 Key Features

- Image-based waste classification into **12 garbage categories**
- **Xception transfer learning** for high accuracy with limited training data
- **TensorFlow Savemodel** deployment for efficient inference
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
3. **Model Inference** – Xception-based CNN (TensorFlow savedmodel)  
4. **Prediction Output** – Waste category identification  
5. **Recommendation Engine** – Sustainable disposal guidance  
6. **User Interface** – Gradio-based web application  

---

## 🧪 Model Training

- **Architecture**: Xception (pretrained on ImageNet)
- **Training Strategy**: Transfer learning with frozen base layers
- **Dataset**: Kaggle Garbage Classification Dataset
- **Image Size**: 230 × 230
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

- **Inference Format**: TensorFlow  (`_tf`)
- **Platform**: Local desktop deployment
- **Interface**: Gradio web UI
- **Hardware Requirement**: CPU-only (no GPU required)

---

## ▶️ How to Run the Application

### Install dependencies
```bash
pip install -r requirements.txt
python app.py

The Gradio interface will open automatically in your browser.
Upload an image to receive the predicted waste category and disposal suggestion.

**## Project Structure**

Garbage-Classification-System/
│
├── app.py                    # Gradio-based inference app
├── garbage_model_tf      # Trained TF model
├── README.md
├── requirements.txt
│
├── training/
│   ├── README.md             # Training explanation
│   ├── model.py              # Xception model architecture
│   └── train.py              # Training pipeline
│
└── .gitignore

**##🌱 Environmental Impact##**

By enabling automated and accurate waste segregation, this system supports:

Reduced landfill overflow

Improved recycling efficiency

Safer handling of hazardous waste

Smarter waste management in smart cities



