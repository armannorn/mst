import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
from sklearn.model_selection import train_test_split, KFold
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
from io import BytesIO
from PIL import Image
import os
import logging
import time
from datetime import datetime


def json_to_dict(path="config.json"):
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
    scalers = {}
    if sconf.get('standard'):
        standard_scaler = StandardScaler()
        standard_cols = sconf['standard']
        for col in standard_cols:
            if col in X.columns:
                X[col] = standard_scaler.fit_transform(X[col].to_numpy().reshape(-1, 1))

        scalers['standard'] = standard_scaler

    if sconf.get('minmax'):
        minmax_scaler = MinMaxScaler()
        minmax_cols = sconf['minmax']
        for col in minmax_cols:
            if col in X.columns:
                X[col] = minmax_scaler.fit_transform(X[col].to_numpy().reshape(-1,1))
        scalers['minmax'] = minmax_scaler
    if sconf.get('circular'):
        for col in sconf['circular']:
            if col in X.columns:
                transformed = circular_transform(X[col])
                X = X.drop(columns=[col])
                X = pd.concat([X, transformed], axis=1)
    return X, scalers


def circular_transform(column: pd.Series, degrees: bool = True) -> pd.DataFrame:
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


def build_model(aconf, input_shape):
    model = Sequential()
    first_layer = aconf["layers"][0]
    model.add(Dense(units=first_layer["units"], activation=first_layer["activation"], input_shape=input_shape))
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


def prepare_report(cv_results, report_filename='report.pdf', config=None):
    if config:
        # Get the number of splits and features from the config
        n_splits = config.get("cross_validation", {}).get("n_splits", "N/A")
        features = config.get("data", {}).get("features", "N/A")

    # Calculate mean and std of absolute error
    mean_absolute_errors = [result['mae'] for result in cv_results]
    mean_mae = np.mean(mean_absolute_errors)
    std_mae = np.std(mean_absolute_errors)

    # Create the PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=16)
    pdf.cell(200, 10, txt="Cross-Validation Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)

    # Add number of splits and features used to the document
    if config:
        pdf.cell(200, 10, txt=f"Number of Cross-Validation Splits: {n_splits}", ln=True, align='L')
        pdf.ln(5)
        pdf.cell(200, 10, txt=f"Features Used: {', '.join(features)}", ln=True, align='L')
        pdf.ln(10)

    pdf.cell(200, 10, txt=f"Mean Absolute Error: {mean_mae:.2f} ± {std_mae:.2f}", ln=True, align='L')
    pdf.ln(10)

    for fold, result in enumerate(cv_results):
        history = result['history']
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'Training and Validation Loss History - Fold {fold + 1}')
        plt.legend()
        plt.grid(True)

        # Save the plot to a BytesIO object
        plot_buffer = BytesIO()
        plt.savefig(plot_buffer, format='png')
        plt.close()
        plot_buffer.seek(0)

        # Use PIL to open the image and save it in a compatible format for fpdf
        image = Image.open(plot_buffer)
        plot_filename = f'loss_history_fold_{fold + 1}.png'
        image.save(plot_filename)

        # Add the plot to the PDF
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt=f"Loss History for Fold {fold + 1}:", ln=True, align='L')
        pdf.image(plot_filename, x=10, y=None, w=190)
        pdf.ln(85)  # Adjust this value based on the image size

        # Remove the temporary plot file
        os.remove(plot_filename)

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

    kf = KFold(n_splits=config["cross_validation"]["n_splits"], shuffle=True, random_state=42)
    cv_results = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model = build_model(config["architecture"], X_train.shape[1:])
        model, history = compile_and_train(X_train, y_train, model, config["training"])

        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)

        cv_results.append({
            'model': model,
            'history': history,
            'mae': mae
        })

    formatted_time = datetime.now().strftime("%m-%dT%H:%M")
    prepare_report(cv_results, f'{formatted_time}_report.pdf', config)
