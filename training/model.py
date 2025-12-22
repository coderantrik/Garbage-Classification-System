import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model


def build_xception_model(
    input_shape=(224, 224, 3),
    num_classes=12,
    dropout_rate=0.5
):
    """
    Builds an Xception-based classification model using transfer learning.
    """

    base_model = Xception(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape
    )

    # Freeze pretrained layers
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=outputs)

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"]
    )

    return model
