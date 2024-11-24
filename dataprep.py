import pandas as pd


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
