# ♻️ AI-Powered Garbage Classification System for Sustainable Waste Management

An intelligent deep learning-based waste classification system designed to automate garbage segregation using **computer vision**, **transfer learning**, and **real-time inference**. The project leverages the **Xception Convolutional Neural Network** architecture to classify waste images into multiple categories and provide **eco-friendly disposal recommendations** for sustainable waste management.

Developed as a B.Tech final-year major project under the Department of CSE-Data Science at BBDITM, Lucknow.

---

# 📌 Problem Statement

Rapid urbanization and increasing consumer-driven lifestyles have drastically increased municipal solid waste generation. Traditional waste segregation methods rely heavily on manual labor, making the process:

- Time-consuming
- Error-prone
- Unsafe for sanitation workers
- Inefficient for large-scale recycling

Improper waste segregation reduces recycling efficiency and contributes to landfill overflow, pollution, and environmental degradation. This project addresses these challenges through an AI-powered automated garbage classification system capable of identifying waste categories in real time.

---

# 🎯 Objectives

The primary goals of the project are:

- Automate waste classification using deep learning
- Improve segregation accuracy and recycling efficiency
- Reduce human intervention in hazardous waste handling
- Enable real-time waste identification through an interactive interface
- Promote sustainable and eco-friendly disposal practices
- Build a scalable architecture suitable for future IoT and smart city integration

---

# 🚀 Key Features

✅ Deep learning-based image classification  
✅ Transfer learning using **Xception pretrained on ImageNet**  
✅ Classification across **12 distinct waste categories**  
✅ Real-time prediction through **Gradio Web Interface**  
✅ Sustainable disposal recommendations  
✅ TensorFlow SavedModel deployment  
✅ Modular training + inference pipeline  
✅ Edge-deployment ready architecture  
✅ Confidence-aware prediction workflow  
✅ Scalable for future IoT integration  

---

# 🧠 Waste Categories

| Category | Description |
|---|---|
| Paper | Newspapers, sheets, books |
| Cardboard | Packaging and boxes |
| Plastic | Bottles, containers, wrappers |
| Metal | Aluminum cans, metallic waste |
| Trash | Non-recyclable waste |
| Battery | Hazardous electronic waste |
| Shoes | Footwear waste |
| Clothes | Textile waste |
| Green Glass | Green-colored glass materials |
| Brown Glass | Brown-colored glass materials |
| White Glass | Transparent/white glass |
| Biological Waste | Organic and biodegradable waste |

---

# 🏗️ System Architecture

The system follows a modular layered architecture consisting of:

1. **Input Layer**  
   User uploads waste image through Gradio UI

2. **Preprocessing Layer**  
   - RGB conversion  
   - Image resizing  
   - Normalization  
   - Tensor conversion  

3. **Processing Layer**  
   Xception CNN performs feature extraction and classification

4. **Decision Layer**  
   - Probability evaluation  
   - Confidence threshold checking  
   - Top predictions generation  

5. **Output Layer**  
   - Predicted category  
   - Confidence score  
   - Waste disposal recommendation  

The architecture is designed for future extension into **IoT-enabled smart bins and automated sorting systems**.

---

# 🧪 Dataset Information

## Dataset Sources

- Kaggle Garbage Classification Dataset
- TrashNet Dataset
- Additional real-world waste images

## Dataset Characteristics

- 12 waste categories
- Real-world image variability
- Multiple lighting conditions
- Different object orientations
- Background complexity handling

---

# ⚙️ Data Preprocessing Pipeline

## Steps Performed

- Convert image to RGB
- Resize image to **320 × 320**
- Convert image to NumPy array
- Normalize pixel values
- Expand dimensions for batch inference

## Data Augmentation Techniques

| Technique | Purpose |
|---|---|
| Rotation | Improve orientation robustness |
| Horizontal Flip | Increase sample diversity |
| Zooming | Improve scale invariance |
| Brightness Adjustment | Handle lighting variation |

---

# 🧠 Deep Learning Model

## Architecture Used: Xception

The project uses the **Xception Architecture**, a powerful CNN model based on **depthwise separable convolutions**.

## Why Xception?

- Better feature extraction
- Higher classification accuracy
- Efficient parameter utilization
- Strong performance on visually similar categories

## Transfer Learning Strategy

The model uses:

- Pretrained ImageNet weights
- Frozen base layers initially
- Custom dense classification layers
- Softmax output layer

---

# 📊 Model Training Configuration

| Parameter | Value |
|---|---|
| Model | Xception |
| Input Size | 320 × 320 |
| Optimizer | Adam |
| Loss Function | Categorical Crossentropy |
| Training Method | Transfer Learning |
| Framework | TensorFlow / Keras |
| Deployment Format | TensorFlow SavedModel |

---

# 📈 Performance Results

| Metric | Performance |
|---|---|
| Training Accuracy | 99.4% |
| Validation Accuracy | 95.5% |
| Test Accuracy | ~93% |
| Macro F1 Score | ~0.90 |

The model demonstrated:

- Strong generalization capability
- Stable convergence
- Low overfitting
- Robust performance on minority classes

---

# 📉 Evaluation Metrics

The system was evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Training & Validation Curves

---

# 💻 Deployment

## Deployment Stack

| Component | Technology |
|---|---|
| Model Framework | TensorFlow |
| Interface | Gradio |
| Deployment Format | SavedModel |
| Runtime | Python |
| Hardware Support | CPU Compatible |

## Why SavedModel Instead of TensorFlow Lite?

The project intentionally uses the **full TensorFlow SavedModel** instead of TensorFlow Lite to:

- Preserve numerical precision
- Avoid quantization loss
- Maintain inference consistency
- Achieve higher classification reliability

---

# 🖥️ Gradio Interface

The application includes a lightweight and interactive Gradio web interface that allows users to:

- Upload waste images
- Receive real-time predictions
- View confidence scores
- Get disposal suggestions

---

# 🌱 Environmental Impact

This project contributes toward sustainable development by supporting:

♻️ Improved recycling efficiency  
🌍 Reduced landfill overflow  
⚠️ Safer hazardous waste handling  
🏙️ Smart city waste management initiatives  
🤖 Reduced dependency on manual segregation  
📈 Data-driven waste management systems  

---

# 🔮 Future Scope

Future enhancements may include:

- IoT-enabled smart waste bins
- Automated actuator-based waste sorting
- Mobile application integration
- Edge deployment using TensorFlow Lite
- Multi-modal sensing (weight + image)
- Cloud analytics dashboard
- Robotic waste segregation systems

---

# 🛠️ Tech Stack

## AI / ML
- TensorFlow
- Keras
- NumPy
- Pandas

## Visualization
- Matplotlib
- Seaborn

## Deployment
- Gradio
- Python

## Model Architecture
- Xception CNN
- Transfer Learning

---

# 📂 Project Structure

```bash
AI-Garbage-Classification-System/
│
├── app.py
├── requirements.txt
├── README.md
├── garbage_model_tf/
│
├── training/
│   ├── train.py
│   ├── model.py
│   ├── preprocessing.py
│   └── README.md
│
├── dataset/
│   └── (not included)
│
└── assets/
    └── sample_outputs/
```

---

# ▶️ Installation & Usage

## Clone Repository

```bash
git clone https://github.com/your-username/AI-Garbage-Classification-System.git

cd AI-Garbage-Classification-System
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Application

```bash
python app.py
```

The Gradio interface will automatically launch in your browser.

Upload a garbage image to:

- Predict waste category
- View confidence score
- Receive disposal recommendation

---

# 👨‍💻 Authors

- Sneha
- Antriksh Bhadauriya
- Vaibhav Tiwari
- Ritesh Tiwari

Under the guidance of **Dr. Usha Sharma**  
Department of CSE – Data Science  
Babu Banarasi Das Institute of Technology & Management, Lucknow

---
---

# 📄 Research Paper

The research work associated with this project has been published in the *International Research Journal of Modernization in Engineering Technology and Science (IRJMETS)*.

🔗 [Read Research Paper](https://www.irjmets.com/upload_newfiles/irjmets80400152656/paper_file/irjmets80400152656.pdf)

### Paper Title
**AI-Powered Garbage Classification System for Sustainable Waste Management**

The paper discusses:
- Deep learning-based waste classification
- Xception transfer learning architecture
- TensorFlow deployment pipeline
- Real-time Gradio inference system
- Sustainable waste management applications

---

# 📄 License

This project is developed for academic and research purposes.

For commercial usage, please contact the authors.
