import pandas as pd
import tensorflow as tf

from keras import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten
from keras.losses import MeanSquaredError, MeanAbsoluteError
from keras.optimizers import Adam


def build_model(aconf: dict, input_shape: tuple) -> Sequential:
    """
    Build a model according to the configuration
    :param aconf: model architecture configuration
    :param input_shape: shape of the input data
    :return: the model
    """
    model = Sequential()

    # first layer treated seperately
    first_layer = aconf["layers"][0]
    model.add(Dense(units=first_layer["units"], activation=first_layer["activation"], input_shape=input_shape))

    # rest of the layers
    for layer in aconf["layers"][1:]:
        layer_type = layer["type"]
        if layer_type == "dense":
            model.add(Dense(units=layer["units"], activation=layer["activation"]))
        elif layer_type == "dropout":
            model.add(Dropout(rate=layer["rate"]))
        # Assuming that convolutional networks will play a part in the project
        elif layer_type == "conv2d":
            model.add(Conv2D(filters=layer["filters"], kernel_size=layer["kernel_size"], activation=layer["activation"],
                             input_shape=layer.get("input_shape", None)))
        elif layer_type == "flatten":
            model.add(Flatten())

    # output layer will have linear activation for regression purposes
    model.add(Dense(units=1, activation=aconf["output_activation"]))
    return model


def compile_and_train(X_train: pd.DataFrame, y_train: pd.DataFrame,
                      X_val: pd.DataFrame, y_val: pd.DataFrame,
                      model: Sequential, tconf: dict) -> (Sequential, dict):
    """
    Compile and train the model
    :param X_train, y_train: training data
    :param X_val, y_val: validation data
    :param model: Model architecture has been resolved in build_model
    :param tconf: training configuration
    :return: The model and the training history
    """

    optimizer = Adam(learning_rate=tconf["learning_rate"])
    if tconf["loss"] == "mse":
        loss = MeanSquaredError()
    elif tconf["loss"] == "mae":
        loss = MeanAbsoluteError()
    elif tconf["loss"] == "weighted":
        loss = weighted_MAE_wrapper(tconf["parameters"]["a"])
    else:
        raise ValueError(f"Unsupported loss function: {tconf['loss']}")

    model.compile(optimizer=optimizer, loss=loss, metrics=['mae'])
    history = model.fit(X_train, y_train, epochs=tconf['epochs'],
                        validation_data=(X_val, y_val), batch_size=tconf['batch_size'])

    return model, history


def weighted_mae(y_true, y_pred, a):
    """
    Weighted Mean Absolute Error Loss with parameter `a`.
    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        a: Weighting exponent.

    Returns:
        Weighted MAE loss value.
    """

    # Compute weights
    weights = ((y_true+0.0000000001) / 5.0) ** a
    weights = tf.maximum(weights, 1.0)

    tf.debugging.check_numerics(weights, "Weights contain NaNs or Infs")

    # Compute per-sample MAE
    mae = tf.abs(y_true - y_pred)

    # Weighted MAE
    res = weights * mae

    # Return mean loss
    return tf.reduce_mean(res)


def weighted_MAE_wrapper(a: float) -> callable:
    """
    Wrapper for weighted_MAE in order to automate a
    :param a: parameter
    :return: the function to be used as loss
    """
    def loss(y_true, y_pred):
        return weighted_mae(y_true, y_pred, a)
    return loss

if __name__ == '__main__':
    pass
