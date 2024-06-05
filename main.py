import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from keras.metrics import MeanAbsoluteError
from keras.layers import Dense, Dropout, Conv2D, Flatten
import json


def json_to_dict(path="config.json"):
    """
    Reads a JSON file from the specified path and converts it into a Python dictionary.

    Parameters:
    path (str): The file path to the JSON configuration file. Default is "config.json".

    Returns:
    dict: A dictionary representation of the JSON configuration file.
    """
    try:
        with open(path, 'r') as file:
            config_dict = json.load(file)
            return config_dict
    except FileNotFoundError:
        print(f"Error: The file at {path} was not found.")
    except json.JSONDecodeError:
        print(f"Error: The file at {path} is not a valid JSON file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def read_data(dconf: dict[str, any]) -> (pd.DataFrame, pd.Series):
    df = pd.read_feather(dconf["path"])
    missing_features = [feature for feature in dconf["features"] if feature in df.columns]
    if dconf["target"] in df.columns and missing_features:
        X = df[dconf["features"]]
        y = df[dconf["target"]]
        return X, y
    else:
        return "Error. Target or features not in dataset."


def apply_scaling(X: pd.DataFrame, sconf: dict) -> (pd.DataFrame, dict):
    """
    Apply scaling and transformations to the DataFrame X based on the provided configuration.

    Parameters:
    X (pd.DataFrame): The input DataFrame to be transformed.
    config (dict): Configuration dictionary containing keys 'standard', 'minmax', and 'circular_transform' with lists of column names.

    Returns:
    (pd.DataFrame, dict): A tuple with the transformed DataFrame and a dictionary of fitted scalers.
    """
    scalers = {}

    # Standard scaling
    if sconf.get('standard'):
        standard_scaler = StandardScaler()
        standard_cols = sconf['standard']
        X[standard_cols] = standard_scaler.fit_transform(X[standard_cols])
        scalers['standard'] = standard_scaler

    # Min-Max scaling
    if sconf.get('minmax'):
        minmax_scaler = MinMaxScaler()
        minmax_cols = sconf['minmax']
        X[minmax_cols] = minmax_scaler.fit_transform(X[minmax_cols])
        scalers['minmax'] = minmax_scaler

    # Circular transform
    if sconf.get('circular'):
        for col in sconf['circular']:
            transformed = circular_transform(X[col])
            X = X.drop(columns=[col])
            X = pd.concat([X, transformed], axis=1)

    return X, scalers


def split_data(X: pd.DataFrame, y: pd.Series, tconf: dict[str, any]):
    """
    Possibly we'll add something to stratify here...
    :param X, y: Features and target
    :param tconf: Training config
    :return: Training and testing datasets.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=tconf["test_split"], random_state=42)
    return X_train, X_test, y_train, y_test


def circular_transform(column: pd.Series, degrees: bool = True) -> pd.DataFrame:
    """
    Transforms a column of angular data into sine and cosine components.

    Parameters:
    column (pd.Series): The input column with angular data.
    degrees (bool): If True, the input data is in degrees. If False, the data is in radians.

    Returns:
    pd.DataFrame: A DataFrame with two columns, 'sin' and 'cos', containing the sine and cosine of the input data.
    """
    if degrees:
        radians = np.deg2rad(column)
    else:
        radians = column

    sin_values = np.sin(radians)
    cos_values = np.cos(radians)

    result = pd.DataFrame({
        f'{column.name}_sin': sin_values,
        f'{column.name}_cos': cos_values
    })

    return result


def build_model(aconf, X_train):
    """
    Builds a Keras model based on the given configuration and training data.

    Parameters:
    config (dict): Configuration dictionary containing the model architecture.
    X_train (pd.DataFrame or np.ndarray): Training data to determine the input shape.

    Returns:
    model (Sequential): The constructed Keras model.
    """
    model = Sequential()

    # Determine input shape from X_train
    input_shape = X_train.shape[1:]

    # Add the input layer
    first_layer = aconf["layers"][0]
    model.add(Dense(units=first_layer["units"], activation=first_layer["activation"], input_shape=input_shape))

    # Add the remaining layers
    for layer in aconf["layers"][1:]:
        layer_type = layer["type"]
        if layer_type == "dense":
            model.add(Dense(units=layer["units"], activation=layer["activation"]))
        elif layer_type == "dropout":
            model.add(Dropout(rate=layer["rate"]))
        elif layer_type == "conv2d":
            model.add(Conv2D(filters=layer["filters"], kernel_size=layer["kernel_size"], activation=layer["activation"],
                             input_shape=layer.get("input_shape", None)))
        elif layer_type == "flatten":
            model.add(Flatten())

    model.add(Dense(units=1, activation=aconf["output_activation"]))

    return model


def compile_and_train(X_train, y_train, model, tconf):
    if tconf["optimizer"] == "adam":
        optimizer = Adam(learning_rate=tconf["learning_rate"])
    else:
        raise ValueError(f"Unsupported optimizer: {tconf['optimizer']}")

    if tconf["loss"] == "mse":
        loss = MeanSquaredError()
    else:
        raise ValueError(f"Unsupported loss function: {tconf['loss']}")

    model.compile(
        optimizer=optimizer, loss=loss, metrics=['mse']
    )

    history = model.fit(
        X_train, y_train, epochs=tconf['epochs'], validation_split=tconf['validation_split'],
        batch_size=tconf['batch_size']
    )

    return model, history


def prepare_report(X_test, y_test, model, history, report_filename='report.pdf'):
    # Calculate absolute error
    y_pred = model.predict(X_test)
    print(f"Predicted values: {y_pred}")
    absolute_error = mean_absolute_error(y_test, y_pred)

    # Create loss history plot
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss History')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_history.png')
    plt.close()

    # Create the PDF
    pdf = FPDF()

    # Add a page
    pdf.add_page()

    # Set title and subtitle
    pdf.set_font("Arial", size=16)
    pdf.cell(200, 10, txt="Model Evaluation Report", ln=True, align='C')
    pdf.ln(10)

    # Print the absolute error
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Mean Absolute Error: {absolute_error:.2f}", ln=True, align='L')
    pdf.ln(10)

    # Add the loss history plot
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Loss History:", ln=True, align='L')
    pdf.image('loss_history.png', x=10, y=None, w=190)
    pdf.ln(85)  # Adjust this value based on the image size

    # Save the PDF
    pdf.output(report_filename)

    print(f"Report saved as {report_filename}")


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', '{:.2f}'.format)

    config = json_to_dict()
    X, y = read_data(config["data"])
    X, scalers = apply_scaling(X, config["scaling"])
    X_train, X_test, y_train, y_test = split_data(X, y, config["training"])
    model = build_model(config["architecture"], X_train)
    model, history = compile_and_train(X_train, y_train, model, config["training"])
    prepare_report(X_test, y_test, model, history)

