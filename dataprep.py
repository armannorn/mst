import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def resolve_columns(df: pd.DataFrame, fconf: dict[str, any]) -> list[str]:
    """
    Resolve, according to configuration, which columns are used in the model
    :param df: Training data
    :param fconf: feature configuration,
    :return:
    """
    res: list[str] = []
    loc = fconf.get("location")             # Location columns (lat, lon, h_meas)
    var_cols = fconf.get("various")         # Various columns (e.g. TRI, ocean columns)
    pred_cols = fconf.get("predictions")    # Prediction based columns (e.g. f15, N2)
    max_e = fconf.get("elevation")          # Elevation goes up to max_e meters

    res += [col for col in ["lat", "lon", "h_meas"] if col in df.columns] if loc else []
    res += [col for col in pred_cols if col in df.columns] if pred_cols else []
    res += [col for col in var_cols if col in df.columns] if var_cols else []

    if max_e:
        e_cols = [col for col in df.columns if col.startswith("e") and len(col) >= 2 and col[1].isdigit()]
        for col in e_cols:
            if int(col[1:]) < max_e:
                res += [col]

    return res


def is_elevation_column(col: str) -> bool:
    """
    Check if a column is an elevation column
    :param col: column name
    :return: True if it is an elevation column
    """
    return col.startswith("e") and len(col) >= 2 and col[1].isdigit()


def read_data(dconf: dict[str, any]) -> (pd.DataFrame, pd.Series):
    """
    Read data from file
    :param dconf: data configuration
    :return: Training data with features and target
    """
    df = pd.read_feather(dconf["path"])
    # Utility represents a ratio of the data that is used to train. It is 1.0 except for testing purposes.
    data_utility = dconf["utility"]
    if data_utility < 1:
        df = df.sample(frac=data_utility, random_state=42)

    # Correct some discrepancies and empty entries in elevation data.
    e_columns = [col for col in df.columns if is_elevation_column(col)]
    overall_min = df[e_columns].min().min()
    df[e_columns] = df[e_columns].apply(lambda col: col.fillna(overall_min), axis=0)

    if "TRI" in df.columns:
        df["TRI"] = df["TRI"].fillna(0.0)

    # Which columns are in X
    cols = resolve_columns(df, dconf["features"])

    try:
        X, y = df[cols], df[dconf["target"]]
        return X, y
    except Exception as e:
        return f"Error reading data: {e}"


def apply_standard_scaling(X: pd.DataFrame, sconf: dict) -> (pd.DataFrame, StandardScaler):
    """
    Apply standard scaling to the columns specified as standard -- usually continuous and unimodal
    :param X: training data
    :param sconf: scaling configuration
    :return: modified training data and scaler
    """
    standard_scaler = StandardScaler()
    standard_cols = sconf["standard"]
    if "predictions" in standard_cols:
        standard_cols.remove("predictions")
        # Keeeping p15, t15 if circling back to those features
        standard_cols += ["f15", "p15", "t15", "theta15", "N2"]

    for col in standard_cols:
        if col in X.columns:
            X[col] = standard_scaler.fit_transform(X[col].to_numpy().reshape(-1, 1))

    return X, standard_scaler


def apply_minmax_scaling(X: pd.DataFrame, sconf: dict) -> (pd.DataFrame, MinMaxScaler):
    """
    Apply minmax scaling to the columns specified as minmax -- usually strictly positive or bounded
    :param X: training data
    :param sconf: scaling configuration
    :return: modified training data and scaler
    """
    minmax_scaler = MinMaxScaler()
    minmax_cols = sconf['minmax']

    # Replacing location with the columns it represents
    if "location" in minmax_cols:
        minmax_cols.remove("location")
        minmax_cols += ["lat", "lon", "height_ASL"]

    # Replacing elevation with the columns it represents
    if "elevation" in minmax_cols:
        minmax_cols.remove("elevation")
        minmax_cols += [col for col in X.columns if is_elevation_column(col)]

    for col in minmax_cols:
        if col in X.columns:
            X[col] = minmax_scaler.fit_transform(X[col].to_numpy().reshape(-1, 1))

    return X, minmax_scaler


def apply_scaling(X: pd.DataFrame, sconf: dict) -> (pd.DataFrame, dict):
    """
    Apply scaling to the training data
    :param X: Features
    :param sconf: scaling configuration
    :return: scaled data and scalers
    """

    scalers = {}

    if not sconf["use"]:
        return X, scalers

    if sconf.get('standard'):
        X, scalers['standard'] = apply_standard_scaling(X, sconf)

    if sconf.get('minmax'):
        X, scalers['minmax'] = apply_minmax_scaling(X, sconf)

    if sconf.get('circular'):
        for col in sconf['circular']:
            if col in X.columns:
                transformed = circular_transform(X[col])
                X = X.drop(columns=[col])
                X = pd.concat([X, transformed], axis=1)
    return X, scalers


def circular_transform(column: pd.Series, degrees: bool = True) -> pd.DataFrame:
    """
    Equivalent to scaling a feature of circular nature. For example, months or angles.
    It is transformed into sin and cos of the feature in question.
    Does not move e.g. months into ratios, must be done beforehand and passed either as
    degree angles or radians.
    :param column: Feature to be transformed
    :param degrees: Is the feature in degrees?
    :return: transformed feature
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
