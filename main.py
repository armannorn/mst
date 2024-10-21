import colorama.win32
import colorcet.plotting
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

from keras.callbacks import EarlyStopping

import json
from io import BytesIO
from PIL import Image
import os
import logging
import time
from datetime import datetime
import sys


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


def resolve_columns(df: pd.DataFrame, fconf: dict[str, any]) -> list[str]:
    r: list[str] = []
    var_cols = fconf.get("various")
    pred_cols = fconf.get("predictions")
    max_e = fconf.get("elevation")
    stations_in = fconf.get("stations")

    if fconf.get("location"):
        loc_cols = ["lat", "lon", "h_meas"]
        r += [col for col in loc_cols if col in df.columns]

    if pred_cols:
        r += [col for col in pred_cols if col in df.columns]

    if max_e:
        e_cols = [col for col in df.columns if col.startswith("e") and len(col) >= 2 and col[1].isdigit()]
        for col in e_cols:
            if int(col[1:]) < max_e:
                r += [col]

    if var_cols:
        r += [col for col in var_cols if col in df.columns]

    if stations_in:
        r += [col for col in df.columns if col.startswith("station") and col != "station"]

    return r


def read_data(dconf: dict[str, any]) -> (pd.DataFrame, pd.Series):
    data_utility = dconf["utility"]
    df = pd.read_feather(dconf["path"])

    if data_utility < 1:
        df = df.sample(frac=data_utility, random_state=42)

    # Fix elevation
    e_columns = [col for col in df.columns if col.startswith('e') and col[1].isdigit()]
    overall_min = df[e_columns].min().min()
    df[e_columns] = df[e_columns].apply(lambda col: col.fillna(overall_min), axis=0)

    if "TRI" in df.columns:
        df["TRI"] = df["TRI"].fillna(0.0)

    if dconf["features"].get("stations"):
        df = ohe_stations(df)

    # Which columns are in X
    cols = resolve_columns(df, dconf["features"])

    if dconf["target"] in df.columns:
        X = df[cols]
        y = df[dconf["target"]]
        return X, y
    else:
        return "Error. Target or features not in dataset."


def ohe_stations(df):
    if "station" in df.columns:
        df["station"] = df["station"].apply(lambda x: str(int(x)))
        one_hot_df = pd.get_dummies(df['station'], prefix='station').astype(float)
        df = pd.concat([df, one_hot_df], axis=1)

    return df


def apply_scaling(X: pd.DataFrame, sconf: dict) -> (pd.DataFrame, dict):

    scalers = {}
    if sconf.get('standard'):
        standard_scaler = StandardScaler()
        standard_cols = sconf["standard"]
        if "predictions" in standard_cols:
            standard_cols.remove("predictions")
            standard_cols += ["f15", "p15", "t15", "theta15", "N2"]

        for col in standard_cols:
            if col in X.columns:
                X[col] = standard_scaler.fit_transform(X[col].to_numpy().reshape(-1, 1))

        scalers['standard'] = standard_scaler

    if sconf.get('minmax'):
        minmax_scaler = MinMaxScaler()
        minmax_cols = sconf['minmax']

        if "location" in minmax_cols:
            minmax_cols.remove("location")
            minmax_cols += ["lat", "lon", "height_ASL"]

        if "elevation" in minmax_cols:
            minmax_cols.remove("elevation")
            minmax_cols += [col for col in X.columns if col.startswith("e")]

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
    # Define early stopping
    if tconf["early_stopping"]["use"]:
        esconf = tconf["early_stopping"]
        early_stopping = EarlyStopping(
            monitor='val_loss',  # Metric to monitor (e.g., validation loss)
            patience=esconf["patience"],  # Number of epochs to wait after last improvement
            verbose=esconf["verbose"],  # Verbose output when stopping
            restore_best_weights=esconf["restore"]  # Restore model weights from the epoch with the best validation loss
        )
    else:
        early_stopping = None

    if tconf["optimizer"] == "adam":
        optimizer = Adam(learning_rate=tconf["learning_rate"])
    else:
        raise ValueError(f"Unsupported optimizer: {tconf['optimizer']}")

    if tconf["loss"] == "mse":
        loss = MeanSquaredError()
    elif tconf["loss"] == "mae" or tconf["loss"] == "weighted":
        loss = MeanAbsoluteError()
    else:
        raise ValueError(f"Unsupported loss function: {tconf['loss']}")

    model.compile(
        optimizer=optimizer, loss=loss, metrics=['mae'], weighted_metrics=[]
    )
    if tconf["loss"] == "weighted":
        sample_weights = resolve_sample_weights(y_train, tconf["parameters"])
        history = model.fit(
            X_train, y_train, epochs=tconf['epochs'], validation_split=tconf['validation_split'],
            batch_size=tconf['batch_size'], sample_weight=sample_weights
        )

    else:
        history = model.fit(
            X_train, y_train, epochs=tconf['epochs'], validation_split=tconf['validation_split'],
            batch_size=tconf['batch_size']
        )

    return model, history


def resolve_sample_weights(y, param):
    sample_weights = y.apply(lambda x: return_weight(x, param))
    return sample_weights.to_numpy()


def return_weight(sample, param):
    # Returns flat np.array with weights for X
    a = float(param["a"]) if "a" in param.keys() else 0

    return max((sample/5.0)**a, (30.0/5.0)**a)


def prepare_features(features):
    s = ""
    if features.get("location"):
        s += "lat, lon, height_ASL"

    if features.get("predictions"):
        s += ", " if len(s) > 0 else ""
        s += ', '.join(features["predictions"])

    if features.get("elevation"):
        s += ", elevation data"

    if features.get("various"):
        s += ", " if len(s) > 0 else ""
        s += ', '.join(features["various"])

    return s


def log_results(cv_results, config=None, start=datetime.now(), note=""):
    training_time = datetime.now() - start

    # Get features from config
    n_splits = config.get("cross_validation", {}).get("n_splits", "N/A")
    features = config.get("data", {}).get("features", "N/A")
    loss = config.get("training", {}).get("loss", "N/A")
    batch_size = config.get("training", {}).get("batch_size", "N/A")
    learning_rate = config.get("training", {}).get("learning_rate", "N/A")
    n_epochs = config.get("training", {}).get("epochs", "N/A")
    dataset = config.get("data", {}).get("path", "N/A").split('.')[0]
    note = config.get("note", "N/A")
    timestamp = start.strftime("%d-%m-%YT%H:%M:%S")

    # Calculate mean and std of absolute error
    mean_absolute_errors = [result['mae'] for result in cv_results]
    min_val_maes = [result['min_val_mae'] for result in cv_results]
    epochs = [result['min_val_mae_epoch'] for result in cv_results]
    mean_mae = np.mean(mean_absolute_errors)
    std_mae = np.std(mean_absolute_errors)
    min_epoch = np.mean(epochs)

    if loss == "weighted":
        a = config.get("training", {}).get("parameters", {}).get("a")
        loss = f"a={a} - weighted (f/5)^a"

    dictionary = {
        "timestamp": timestamp,
        "avg-min-epoch": min_epoch,
        "mae-all": np.round(np.mean(min_val_maes), 3),
        "mae-std": np.round(np.std(min_val_maes), 3),
        "test-mae-all": np.round(mean_mae, 3),
        "test-std-all": np.round(std_mae, 3),
    }

    ranges = [(0, 10), (10, 20), (20, 30), (30, float('inf'))]
    keys = [f'{r[0]}-{r[1]}' for r in ranges]
    combined_layered_metrics = [split["layered_metrics"] for split in cv_results]

    dictionary.update({
        f"mae-{key}": np.round(
            np.mean([clm[key]['mae'] for clm in combined_layered_metrics]), 3
        ) for key in keys
    })

    feature_string = prepare_features(features)

    dictionary.update({
        'n-splits': n_splits, 'epochs': n_epochs, 'training-time': training_time, 'loss': loss,
        'batch-size': batch_size, 'init-lr': learning_rate, 'features': feature_string,
        'dataset': dataset, 'note': note
    })

    try:
        previous_logs = pd.read_csv('logs.csv')
        new_logs = pd.concat([previous_logs, pd.DataFrame([dictionary])], axis=0, ignore_index=True)

    except Exception as e:
        new_logs = pd.DataFrame([dictionary])

    new_logs.to_csv('logs.csv', index=False)

    # Assuming 'cv_results' is already defined earlier in your code
    num_folds = len(cv_results)
    num_rows = int(np.ceil(num_folds / 2)) * 2 # Calculate the number of rows needed

    # Create a figure for subplots
    fig, axes = plt.subplots(num_rows, 2, figsize=(15, num_rows * 5))

    # If there's only one row, `axes` will not be a 2D array, so we need to make sure it's always treated as 2D.
    if num_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    for fold, result in enumerate(cv_results):
        history = result['history']

        # Get validation loss history
        val_loss = history.history['val_loss']

        # Find the epoch with the minimum validation loss
        min_val_mae = min(val_loss)
        min_val_mae_epoch = val_loss.index(min_val_mae)  # Epoch numbers are 1-indexed

        # Select the appropriate subplot for the loss plot
        ax_loss = axes[2 * fold]
        ax_loss.plot(history.history['loss'], label='Train Loss')
        ax_loss.plot(history.history['val_loss'], label='Validation Loss')
        ax_loss.set_xlabel('Epochs')
        ax_loss.set_ylabel('Loss')
        ax_loss.set_title(f'Training and Validation Loss History - Fold {fold + 1}')
        ax_loss.legend()
        ax_loss.grid(True)

        # Plot the point for the minimum validation loss (black point)
        ax_loss.scatter(min_val_mae_epoch, min_val_mae, color='black', s=100, zorder=5, label='Min Val Loss')

        # Annotate the point with the value of the minimum validation loss
        ax_loss.text(min_val_mae_epoch, min_val_mae, f'{min_val_mae:.4f}', color='black',
                     ha='center', va='bottom', fontsize=10)

        # Select the appropriate subplot for the scatter plot
        ax_scatter = axes[2 * fold + 1]
        y_test = result['y_test']
        y_pred = result['y_pred']
        ax_scatter.scatter(y_test, y_pred, alpha=0.5, s=0.5)
        ax_scatter.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Add red line y=x
        ax_scatter.set_xlabel('Actual Values')
        ax_scatter.set_ylabel('Predicted Values')
        ax_scatter.set_title(f'Predicted vs. Actual Values - Fold {fold + 1}')
        ax_scatter.grid(True)

    # Remove any empty subplots if the number of folds is odd
    if num_folds % 2 != 0:
        fig.delaxes(axes[-1])

    # Adjust layout to avoid overlap
    plt.tight_layout()

    if config.get("png_name") and note:
        combined_plot_filename = f"{os.path.join('figures', note+timestamp)}.png"
    else:
        combined_plot_filename = f"{os.path.join('figures', timestamp)}.png"

    plt.savefig(combined_plot_filename, format='png')
    plt.close()


def main(config=None):
    start = datetime.now()

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', '{:.2f}'.format)

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
        ranges = [(0, 10), (10, 20), (20, 30), (30, float('inf'))]
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

        # Get validation loss history
        val_mae = history.history['val_mae']

        # Find the epoch with the minimum validation loss
        min_val_mae = min(val_mae)
        min_val_mae_epoch = val_mae.index(min_val_mae) + 1  # Epoch numbers are 1-indexed

        cv_results.append({
            'model': model,
            'history': history,
            'mae': mae,
            'y_test': y_test,
            'y_pred': y_pred,
            'min_val_mae': min_val_mae,
            'layered_metrics': layered_metrics,
            'n_epochs': len(history.history["val_loss"]),
            'min_val_mae_epoch': min_val_mae_epoch
        })

    log_results(cv_results, config, start)


if __name__ == '__main__':
    # f15 + TRI
    config = json_to_dict()

    feat_conf_1 = {
        "location": False,
        "predictions": ["f15"],
        "elevation": 0,
        "stations": False,
        "various": [],
        "note": "1 BL wted"
    }

    feat_conf_2 = {
        "location": True,
        "predictions": ["f15"],
        "elevation": 0,
        "stations": False,
        "various": [],
        "note": "2 BL + loc wted"
    }

    feat_conf_3 = {
        "location": False,
        "predictions": ["f15"],
        "elevation": 20000,
        "stations": False,
        "various": [],
        "note": "3 BL + elev wted"
    }

    feat_conf_4 = {
        "location": True,
        "predictions": ["f15"],
        "elevation": 20000,
        "stations": False,
        "various": [],
        "note": "4 BL + elev + loc wted"
    }

    feat_conf_5 = {
        "location": True,
        "predictions": ["f15", "N2"],
        "elevation": 20000,
        "stations": False,
        "various": ["min_dist_to_ocean", "ocean_wind_indicator", "TRI"],
        "note": "5 All wted"
    }

    for a in [0, 1, 2]:
        for feat_conf in [feat_conf_1, feat_conf_2, feat_conf_3, feat_conf_4, feat_conf_5]:
            config["training"]["parameters"]["a"] = a
            config["data"]["features"] = feat_conf
            config["note"] = feat_conf["note"]
            main(config)
