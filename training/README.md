# Model Training (Kaggle)

This folder contains the training pipeline used to build the garbage classification model.

## Dataset
The model was trained using the **Garbage Classification dataset from Kaggle**, which contains images across 12 waste categories such as paper, plastic, metal, glass, clothes, and biological waste.

Due to dataset size and licensing restrictions, the dataset is **not included** in this repository.

Dataset link:
https://www.kaggle.com/asdasdasasdas/garbage-classification
https://www.kaggle.com/agrigorev/clothing-dataset-full

## Training Overview
- Transfer learning using the **Xception** convolutional neural network
- Image preprocessing and augmentation
- Dataset split into training, validation, and test sets
- Evaluation using accuracy and F1-score
- Model exported to TensorFlow Lite for lightweight deployment

The actual training code is provided in this folder for transparency and reproducibility.
