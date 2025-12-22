import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

from model import build_xception_model


# -----------------------------------------
# CONFIGURATION
# -----------------------------------------
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)

BATCH_SIZE = 64
EPOCHS = 60

# Dataset is NOT included in this repository
# Download from Kaggle and set the path here
DATASET_PATH = "path_to_garbage_dataset"


# -----------------------------------------
# LOAD DATASET INTO DATAFRAME
# -----------------------------------------
def load_dataset(dataset_path):
    data = []

    for class_name in os.listdir(dataset_path):
        class_dir = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_dir):
            continue

        for img in os.listdir(class_dir):
            data.append([f"{class_name}/{img}", class_name])

    return pd.DataFrame(data, columns=["filename", "category"])


df = load_dataset(DATASET_PATH)

train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)


# -----------------------------------------
# DATA GENERATORS (XCEPTION PREPROCESSING)
# -----------------------------------------
train_datagen = ImageDataGenerator(
    preprocessing_function=lambda x: x
)

val_datagen = ImageDataGenerator(
    preprocessing_function=lambda x: x
)

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    DATASET_PATH,
    x_col="filename",
    y_col="category",
    target_size=IMAGE_SIZE,
    class_mode="categorical",
    batch_size=BATCH_SIZE
)

val_generator = val_datagen.flow_from_dataframe(
    val_df,
    DATASET_PATH,
    x_col="filename",
    y_col="category",
    target_size=IMAGE_SIZE,
    class_mode="categorical",
    batch_size=BATCH_SIZE
)


# -----------------------------------------
# BUILD MODEL
# -----------------------------------------
model = build_xception_model(
    input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3),
    num_classes=train_generator.num_classes
)


# -----------------------------------------
# CALLBACKS
# -----------------------------------------
early_stop = EarlyStopping(
    monitor="val_categorical_accuracy",
    patience=5,
    restore_best_weights=True,
    verbose=1
)


# -----------------------------------------
# TRAIN MODEL
# -----------------------------------------
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[early_stop]
)


# -----------------------------------------
# SAVE TRAINED MODEL
# -----------------------------------------
model.save("garbage_model_trained")
print("Model training complete and saved.")
