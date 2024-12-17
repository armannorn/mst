import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error

from datetime import datetime
from dataprep import apply_scaling

from dataprep import read_data
from helpers import load_json
from custom_logging import log_results
from model import build_model, compile_and_train

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', '{:.2f}'.format)

def contains_nan(df):
    return df.isna().sum().sum() > 0

def run(config=None):
    start = datetime.now()

    X, y = read_data(config["data"])

    if contains_nan(X):
        raise ValueError("Data contains NaN values")
    if contains_nan(y):
        raise ValueError("Target contains NaN values")

    X, scalers = apply_scaling(X, config["scaling"])

    X_train_0, X_test, y_train_0, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    kf = KFold(n_splits=config["cross_validation"]["n_splits"], shuffle=True, random_state=42)
    cv_results = []

    for train_index, val_index in kf.split(X_train_0):
        X_train, X_val = X_train_0.iloc[train_index], X_train_0.iloc[val_index]
        y_train, y_val = y_train_0.iloc[train_index], y_train_0.iloc[val_index]

        model = build_model(config["architecture"], X_train.shape[1:])
        model, history = compile_and_train(X_train, y_train, X_val, y_val, model, config["training"])

        y_pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)

        layered_metrics = {}
        ranges = [(0, 10), (10, 20), (20, float('inf'))]
        for r in ranges:
            mask = (y_val >= r[0]) & (y_val < r[1])
            y_val_range = y_val[mask]
            y_pred_range = y_pred[mask]

            range_mae = mean_absolute_error(y_val_range, y_pred_range)
            range_mse = mean_squared_error(y_val_range, y_pred_range)

            layered_metrics[f'{r[0]}-{r[1]}'] = {
                'mae': range_mae,
                'mse': range_mse,
                'n_samples': len(y_val_range)
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
            'y_val': y_val,
            'y_pred': y_pred,
            'min_val_mae': min_val_mae,
            'layered_metrics': layered_metrics,
            'n_epochs': len(history.history["val_loss"]),
            'min_val_mae_epoch': min_val_mae_epoch
        })

    # Select the best-performing model:
    performances = [d.get("mae") for d in cv_results]
    best_performance, best_model = min(performances), None
    for cv in cv_results:
        if cv.get("mae") == best_performance:
            best_model = cv.get("model")

    y_pred = best_model.predict(X_test)
    test_mae = mean_absolute_error(y_test, y_pred)

    log_results(cv_results, config, start, test_mae)


if __name__ == '__main__':
    # f15 + TRI
    config = load_json()

    config["training"] = {
        "learning_rate": 0.0005,
        "loss": "weighted",
        "parameters": {"a": 0},
        "epochs": 150,
        "test_split": 0.2,
        "batch_size": 512
    }
    config["note"] = "a = 0"
    run(config)

    config["training"] = {
        "learning_rate": 0.0005,
        "loss": "weighted",
        "parameters": {"a": 1},
        "epochs": 150,
        "test_split": 0.2,
        "batch_size": 512
    }
    config["note"] = "a = 1"
    run(config)

    config["training"] = {
        "learning_rate": 0.0005,
        "loss": "weighted",
        "parameters": {"a": 1},
        "epochs": 150,
        "test_split": 0.2,
        "batch_size": 512
    }
    config["note"] = "a = 2"
    run(config)
