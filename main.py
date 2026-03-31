"""CHAP model: malaria-chap-project

A probabilistic multistep random forest model for disease prediction.
"""

import pickle

import pandas as pd
from cyclopts import App
from simple_multistep_model import DataFrameMultistepModel, ResidualBootstrapModel
from sklearn.ensemble import RandomForestRegressor

N_TARGET_LAGS = 6
N_SAMPLES = 200
TARGET_VARIABLE = "disease_cases"
FEATURE_COLUMNS = ["smc_number", "rainfall", "mean_temperature", "rel_humidity", "population", "area", "median_elevation"]
INDEX_COLS = ["time_period", "location"]

app = App()


@app.command()
def train(train_data: str, model: str):
    """Train the model on the provided data.

    Parameters
    ----------
    train_data
        Path to the training data CSV file.
    model
        Path where the trained model will be saved.
    """
    df = pd.read_csv(train_data)
    X = df[INDEX_COLS + FEATURE_COLUMNS].fillna(0)
    y = df[INDEX_COLS + [TARGET_VARIABLE]].fillna(0)

    regressor = RandomForestRegressor(
        max_depth=10,
        min_samples_leaf=5,
        max_features="sqrt",
    )
    one_step = ResidualBootstrapModel(regressor)
    forecaster = DataFrameMultistepModel(one_step, N_TARGET_LAGS, TARGET_VARIABLE)
    forecaster.fit(X, y)

    with open(model, "wb") as f:
        pickle.dump(forecaster, f)
    print(f"Model saved to {model}")


@app.command()
def predict(model: str, historic_data: str, future_data: str, out_file: str):
    """Generate predictions using the trained model.

    Parameters
    ----------
    model
        Path to the trained model file.
    historic_data
        Path to historic data CSV file.
    future_data
        Path to future climate data CSV file.
    out_file
        Path where predictions will be saved.
    """
    with open(model, "rb") as f:
        forecaster = pickle.load(f)

    historic_df = pd.read_csv(historic_data)
    future_df = pd.read_csv(future_data)

    y_historic = historic_df[INDEX_COLS + [TARGET_VARIABLE]].fillna(0)
    n_steps = future_df.groupby("location").size().iloc[0]

    X = pd.concat(
        [historic_df[INDEX_COLS + FEATURE_COLUMNS], future_df[INDEX_COLS + FEATURE_COLUMNS]],
        ignore_index=True,
    ).fillna(0).sort_values(by=INDEX_COLS)

    output_df = forecaster.predict(y_historic, X, n_steps, N_SAMPLES)
    output_df.to_csv(out_file, index=False)
    print(f"Predictions saved to {out_file}")


if __name__ == "__main__":
    app()
