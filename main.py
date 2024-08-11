import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from keras.losses import MeanAbsoluteError
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
    elif tconf["loss"] == "mae":
        loss = MeanAbsoluteError()
    else:
        raise ValueError(f"Unsupported loss function: {tconf['loss']}")

    model.compile(
        optimizer=optimizer, loss=loss, metrics=['mae', 'mse']
    )

    history = model.fit(
        X_train, y_train, epochs=tconf['epochs'], validation_split=tconf['validation_split'],
        batch_size=tconf['batch_size']
    )
    return model, history


def prepare_report(cv_results, report_filename='report.pdf', config=None, start=datetime.now()):
    training_time = datetime.now() - start

    if config:
        # Get the number of splits and features from the config
        n_splits = config.get("cross_validation", {}).get("n_splits", "N/A")
        features = config.get("data", {}).get("features", "N/A")
        loss = config.get("training", {}).get("loss", "N/A")
        batch_size = config.get("training", {}).get("batch_size", "N/A")
        learning_rate = config.get("training", {}).get("learning_rate", "N/A")

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
        pdf.ln(5)
        pdf.cell(200, 10, txt=f"Loss function: {loss}", ln=True, align='L')
        pdf.ln(5)
        pdf.cell(200, 10, txt=f"Batch size: {batch_size}", ln=True, align='L')
        pdf.ln(5)
        pdf.cell(200, 10, txt=f"Training time: {training_time}", ln=True, align='L')
        pdf.ln(5)
        pdf.cell(200, 10, txt=f"Learning rate: {learning_rate}", ln=True, align='L')
        pdf.ln(10)

    pdf.cell(200, 10, txt=f"Mean Absolute Error: {mean_mae:.2f} ± {std_mae:.2f}", ln=True, align='L')
    pdf.ln(10)
    pdf.ln(10)

    # Add layered metrics for each fold
    for fold, result in enumerate(cv_results):
        layered_metrics = result['layered_metrics']

        pdf.add_page()
        pdf.set_font("Arial", size=14)
        pdf.cell(200, 10, txt=f"Layered Metrics - Fold {fold + 1}", ln=True, align='L')
        pdf.ln(10)
        pdf.set_font("Arial", size=12)

        # Table header
        pdf.cell(50, 10, txt="Range", border=1)
        pdf.cell(40, 10, txt="MAE", border=1)
        pdf.cell(40, 10, txt="MSE", border=1)
        pdf.cell(40, 10, txt="Samples", border=1)
        pdf.ln(10)

        # Table content
        for range_key, metrics in layered_metrics.items():
            pdf.cell(50, 10, txt=range_key, border=1)
            pdf.cell(40, 10, txt=f"{metrics['mae']:.2f}" if metrics['mae'] is not None else "N/A", border=1)
            pdf.cell(40, 10, txt=f"{metrics['mse']:.2f}" if metrics['mse'] is not None else "N/A", border=1)
            pdf.cell(40, 10, txt=str(metrics['n_samples']), border=1)
            pdf.ln(10)

    for fold, result in enumerate(cv_results):
        history = result['history']

        # Create the loss plot
        plt.figure(figsize=(5, 4))
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'Training and Validation Loss History - Fold {fold + 1}')
        plt.legend()
        plt.grid(True)

        # Save the plot to a BytesIO object
        loss_plot_buffer = BytesIO()
        plt.savefig(loss_plot_buffer, format='png')
        plt.close()
        loss_plot_buffer.seek(0)

        # Create the scatter plot
        y_test = result['y_test']
        y_pred = result['y_pred']

        plt.figure(figsize=(5, 4))
        plt.scatter(y_test, y_pred, alpha=0.5, s=10)  # Smaller dots with s=10
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Add red line y=x
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Predicted vs. Actual Values - Fold {fold + 1}')
        plt.grid(True)

        # Save the plot to a BytesIO object
        scatter_plot_buffer = BytesIO()
        plt.savefig(scatter_plot_buffer, format='png')
        plt.close()
        scatter_plot_buffer.seek(0)

        # Use PIL to open the image and save it in a compatible format for fpdf
        loss_image = Image.open(loss_plot_buffer)
        scatter_image = Image.open(scatter_plot_buffer)

        loss_plot_filename = f'loss_history_fold_{fold + 1}.png'
        scatter_plot_filename = f'pred_vs_actual_fold_{fold + 1}.png'

        loss_image.save(loss_plot_filename)
        scatter_image.save(scatter_plot_filename)

        # Add the plots to the PDF side by side
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt=f"Loss and Predicted vs. Actual Values - Fold {fold + 1}:", ln=True, align='L')
        pdf.image(loss_plot_filename, x=10, y=30, w=90)
        pdf.image(scatter_plot_filename, x=110, y=30, w=90)
        pdf.ln(5)  # Adjust this value based on the image size

        # Remove the temporary plot files
        os.remove(loss_plot_filename)
        os.remove(scatter_plot_filename)

    # Save the PDF
    pdf.output(report_filename)
    print(f"Report saved as {report_filename}")

# TODO: Results of only 'f15' linear regression
# TODO: Add results according to 'f' intervals, i.e. 10-20 m/s, 20-30 m/s, ...

if __name__ == '__main__':
    start = datetime.now()

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

        layered_metrics = {}
        ranges = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, float('inf'))]
        for r in ranges:
            mask = (y_test >= r[0]) & (y_test < r[1])
            y_test_range = y_test[mask]
            y_pred_range = y_pred[mask]

            if len(y_test_range) > 0:
                range_mae = mean_absolute_error(y_test_range, y_pred_range)
                range_mse = mean_squared_error(y_test_range, y_pred_range)
            else:
                range_mae = None
                range_mse = None

            layered_metrics[f'{r[0]}-{r[1]}'] = {
                'mae': range_mae,
                'mse': range_mse,
                'n_samples': len(y_test_range)
            }

        cv_results.append({
            'model': model,
            'history': history,
            'mae': mae,
            'y_test': y_test,
            'y_pred': y_pred,
            'layered_metrics': layered_metrics
        })

    formatted_time = datetime.now().strftime("%m-%dT%H:%M")

    prepare_report(cv_results, f'{formatted_time}_report.pdf', config, start)
