"""Model architectures: single branches and four fusion variants."""
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.optimizers import Adam

from config import IMG_SIZE, HUBER_DELTA, LEARNING_RATE


def _cnn_branch(inputs):
    # Custom branch
    x = layers.Rescaling(1.0 / 255)(inputs)
    for f in (32, 64, 128):
        x = layers.Conv2D(f, 3, activation="relu")(x)
        x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    return layers.Dense(128, activation="relu")(x)


def _mobilenet_branch(inputs):
    # Pretrained branch
    x = layers.Lambda(mobilenet_preprocess)(inputs)
    base = MobileNetV2(weights="imagenet", include_top=False, input_tensor=x)
    base.trainable = False
    z = layers.GlobalAveragePooling2D()(base.output)
    return layers.Dense(128, activation="relu")(z), base


def _head(z):
    z = layers.Dense(128, activation="relu")(z)
    z = layers.Dense(64, activation="relu")(z)
    return layers.Dense(1, activation="linear")(z)


def build_model(fusion, return_base=False):
    """fusion in {cnn_only, mobilenet_only, concat, mul, concat_mul, attention}."""
    inputs = Input(shape=(*IMG_SIZE, 3))
    base = None
    if fusion == "cnn_only":
        z = _cnn_branch(inputs)
    elif fusion == "mobilenet_only":
        z, base = _mobilenet_branch(inputs)
    else:
        f1 = _cnn_branch(inputs)
        f2, base = _mobilenet_branch(inputs)
        if fusion == "concat":
            z = layers.Concatenate()([f1, f2])
        elif fusion == "mul":
            z = layers.Multiply()([f1, f2])
        elif fusion == "concat_mul":  # proposed
            z = layers.Concatenate()([f1, f2, layers.Multiply()([f1, f2])])
        elif fusion == "attention":  # gated fusion
            g = layers.Dense(128, activation="sigmoid")(layers.Concatenate()([f1, f2]))
            z = layers.Add()([layers.Multiply()([g, f1]),
                              layers.Multiply()([layers.Lambda(lambda t: 1.0 - t)(g), f2])])
        else:
            raise ValueError(fusion)
    model = models.Model(inputs, _head(z))
    model.compile(optimizer=Adam(LEARNING_RATE),
                  loss=tf.keras.losses.Huber(delta=HUBER_DELTA), metrics=["mae"])
    return (model, base) if return_base else model
