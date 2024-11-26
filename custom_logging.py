import os
from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def prepare_features(fconf):
    """
    Prepare a string of features to be logged
    :param fconf: feature configuration
    :return: string to log
    """
    s = ""
    if fconf.get("location"):
        s += "lat, lon, height_ASL"

    if fconf.get("predictions"):
        s += ", " if len(s) > 0 else ""
        s += ', '.join(fconf["predictions"])

    if fconf.get("elevation"):
        s += ", elevation data"

    if fconf.get("various"):
        s += ", " if len(s) > 0 else ""
        s += ', '.join(fconf["various"])

    return s


def log_results(cv_results, config=None, start=datetime.now(), test_mae=0.0, note=""):
    # Log training time
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
    min_val_maes = [result['min_val_mae'] for result in cv_results]
    epochs = [result['min_val_mae_epoch'] for result in cv_results]
    min_epoch = np.mean(epochs)

    if loss == "weighted":
        a = config.get("training", {}).get("parameters", {}).get("a")
        loss = f"a={a} - weighted (f/5)^a"

    dictionary = {
        "timestamp": timestamp,
        "avg-min-epoch": min_epoch,
        "mae-all": np.round(np.mean(min_val_maes), 3),
        "mae-std": np.round(np.std(min_val_maes), 3)
    }

    ranges = [(0, 10), (10, 20), (20, float('inf'))]
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
        'dataset': dataset, 'test-mae': test_mae, 'note': note
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
        y_val = result['y_val']
        y_pred = result['y_pred']
        ax_scatter.scatter(y_val, y_pred, alpha=0.5, s=0.5)
        ax_scatter.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], color='red')  # Add red line y=x
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
