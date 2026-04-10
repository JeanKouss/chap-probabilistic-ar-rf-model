# KNOWME

This document explains the code in the order it actually runs.

The goal is not just to name the classes, but to make it clear:

1. what happens first,
2. what data looks like at each step,
3. why each class exists,
4. how training works,
5. how prediction works,
6. where uncertainty comes from.

If you read this top to bottom, you should be able to mentally execute the program.

---

## 1. The Big Picture in One Sentence

This project trains a random forest to predict the next disease count from:

- current known features such as rainfall and temperature,
- the last 6 observed disease counts,

then turns that point-prediction model into a probabilistic multi-step forecaster by:

- storing the model's training residuals,
- resampling those residuals,
- feeding sampled predictions back into the next forecast step.

So yes: this is an autoregressive residual-bootstrap random forest.

---

## 2. The Files and Their Roles

There are only a few files that matter for the core logic.

### `main.py`

This is the command-line entry point. It defines:

- how training is launched,
- how prediction is launched,
- which columns are used,
- which model objects are assembled.

This file does not contain the forecasting logic itself. It wires the components together.

### `simple_multistep_model/one_step_model.py`

This file defines the logic for a single-step probabilistic model.

Its main job is:

- take a normal regressor such as `RandomForestRegressor`,
- fit it,
- compute residuals,
- later use those residuals to generate samples.

### `simple_multistep_model/multistep.py`

This file defines the multi-step forecasting machinery.

Its main job is:

- build lagged target features,
- organize the data across locations and times,
- train one pooled model,
- recursively predict several future steps,
- convert between pandas and xarray.

### `simple_multistep_model/__init__.py`

This file only re-exports the public classes. It contains no forecasting logic.

---

## 3. Start at `main.py`

The code in `main.py` is the natural starting point because that is where execution begins.

### Constants

```python
N_TARGET_LAGS = 6
N_SAMPLES = 200
TARGET_VARIABLE = "disease_cases"
FEATURE_COLUMNS = [
    "smc_number",
    "rainfall",
    "mean_temperature",
    "rel_humidity",
    "population",
    "area",
    "median_elevation",
]
INDEX_COLS = ["time_period", "location"]
```

These constants define the entire modeling setup.

### What `N_TARGET_LAGS = 6` means

The model will always use the previous 6 disease counts as input features.

If the model is trying to predict disease count at time $t$, it uses:

$$
y_{t-6}, y_{t-5}, y_{t-4}, y_{t-3}, y_{t-2}, y_{t-1}
$$

This is the autoregressive part.

### What `N_SAMPLES = 200` means

When forecasting, the model does not produce just one future. It produces 200 sampled futures.

Each sample is one possible trajectory.

### What `FEATURE_COLUMNS` means

These are exogenous variables. They are not the target itself. They are extra predictors.

In this project they are:

- intervention or program feature: `smc_number`
- climate features: `rainfall`, `mean_temperature`, `rel_humidity`
- structural features: `population`, `area`, `median_elevation`

### What `INDEX_COLS` means

Every row is identified by:

- a time period,
- a location.

So the data is a panel time series: multiple locations, each observed over time.

---

## 4. Training Starts with `train(...)`

The function is:

```python
@app.command()
def train(train_data: str, model: str):
```

This command receives:

- `train_data`: a CSV path,
- `model`: where to save the trained model.

Now walk through it in the exact order it runs.

### Step 1. Read the CSV

```python
df = pd.read_csv(train_data)
```

At this point `df` is a pandas DataFrame with rows like:

| time_period | location | disease_cases | smc_number | rainfall | ... |
|-------------|----------|---------------|------------|----------|-----|

One row means: one location at one time period.

### Step 2. Split the data into `X` and `y`

```python
X = df[INDEX_COLS + FEATURE_COLUMNS].fillna(0)
y = df[INDEX_COLS + [TARGET_VARIABLE]].fillna(0)
```

This creates two DataFrames.

#### `X`

Contains:

- `time_period`
- `location`
- all exogenous features

#### `y`

Contains:

- `time_period`
- `location`
- `disease_cases`

### Why are `time_period` and `location` still present?

Because the code has not yet transformed the data into model matrices.

At this stage, those columns are needed to:

- pivot data by time and location,
- align target and features,
- build lag structure correctly for each location.

### Why `.fillna(0)`?

This forces missing values to zero before the model pipeline starts.

That is a modeling decision. It avoids crashes from missing values, but it also means missingness is interpreted as zero. Whether that is statistically ideal is a separate question; this is what the code does.

---

## 5. The Model Objects Created During Training

After reading the data, `train(...)` creates three nested objects.

```python
regressor = RandomForestRegressor(
    max_depth=10,
    min_samples_leaf=5,
    max_features="sqrt",
)
one_step = ResidualBootstrapModel(regressor)
forecaster = DataFrameMultistepModel(one_step, N_TARGET_LAGS, TARGET_VARIABLE)
```

You should think of these as three layers.

### Layer 1. `RandomForestRegressor`

This is the actual machine-learning predictor.

Its job is simple:

- input: a row of numeric features,
- output: one predicted disease count.

It is a standard scikit-learn regression model.

### Why these hyperparameters?

#### `max_depth=10`

Trees cannot grow arbitrarily deep.

This limits complexity and reduces overfitting.

#### `min_samples_leaf=5`

Every leaf must contain at least 5 training samples.

This prevents very tiny terminal regions, which usually means smoother, less noisy predictions.

#### `max_features="sqrt"`

At each split, each tree considers only a subset of available features.

This increases randomness across trees and usually improves generalization.

### What “conservative regularization” means here

It means the forest is deliberately constrained so it does not fit the training data too aggressively.

The model is allowed to learn useful structure, but not to create overly specific rules that only work on the training set.

In plain language:

- shallower trees,
- larger leaves,
- fewer candidate features per split,

all push the model toward smoother, more stable behavior.

### Layer 2. `ResidualBootstrapModel`

This wraps the random forest.

The random forest alone is a point predictor. It gives one number.

`ResidualBootstrapModel` turns it into a probabilistic one-step model.

Its idea is:

1. train the regressor,
2. compute the errors it made on training data,
3. store those errors,
4. when predicting later, add randomly sampled past errors to the point prediction.

### Layer 3. `DataFrameMultistepModel`

This wraps the one-step probabilistic model.

Its job is:

- accept pandas DataFrames,
- transform them into the internal format,
- build multi-step forecasting behavior,
- return pandas DataFrames again.

This is the object that gets pickled and saved.

---

## 6. The Chronological Training Flow

After the objects are created, this line runs:

```python
forecaster.fit(X, y)
```

This begins the actual training pipeline.

To understand it, follow the call chain.

### Call chain during training

```python
main.py: train(...)
    -> DataFrameMultistepModel.fit(X, y)
        -> target_to_xarray(y)
        -> features_to_xarray(X)
        -> MultistepModel.fit_multi(y_xr, X_xr)
            -> _build_lag_matrix_xr(y, n_lags)
            -> one_step_model.fit(X_np, y_np)
                -> ResidualBootstrapModel.fit(X_np, y_np)
                    -> RandomForestRegressor.fit(X_np, y_np)
```

This is the most important sequence in the whole project.

---

## 7. `DataFrameMultistepModel.fit(X, y)`

The method is:

```python
def fit(self, X, y) -> None:
    y_xr = target_to_xarray(y, self._target_variable)
    X_xr = features_to_xarray(X) if X is not None else None
    self._model.fit_multi(y_xr, X_xr)
```

This class is a facade.

It does not train the forest directly. It prepares the data and delegates.

### What is a facade here?

A facade is a wrapper that gives a simpler user-facing interface.

Without this class, the user would have to manually:

- convert pandas to xarray,
- pivot by location and time,
- manage feature dimensions,
- call lower-level methods.

This class hides that complexity.

### What happens first inside `fit`?

#### `target_to_xarray(y, self._target_variable)`

This converts the long-form target DataFrame into an xarray DataArray.

Input form:

| time_period | location | disease_cases |
|-------------|----------|---------------|

Output form:

- dims: `(location, time)`
- values: target values arranged as a 2D grid

You can think of it as a matrix:

| location \ time | t1 | t2 | t3 | ... |
|------------------|----|----|----|-----|
| locA             | 10 | 14 | 20 | ... |
| locB             |  3 |  5 |  9 | ... |

### Why does the code pivot the target?

Because lagged time-series features are easier to build when time is an explicit axis.

### What exactly does `target_to_xarray` do?

It does this:

```python
df = y_df.copy()
df["time_period"] = pd.to_datetime(df["time_period"])
target_wide = df.pivot(index="time_period", columns="location", values=target_variable)
target_wide = target_wide.sort_index().ffill().bfill()
```

#### Important details

- converts `time_period` to datetime,
- pivots to wide format,
- sorts by time,
- fills missing values forward and backward.

### Why `ffill().bfill()`?

If a location has a missing value in the middle or at an edge, this code fills it using nearby observations.

This is a practical alignment step, not a probabilistic treatment of missingness.

---

## 8. `features_to_xarray(X)`

This does for the exogenous features what `target_to_xarray` did for the target.

But now there are many feature columns, so the output is 3D.

### Output shape

- dims: `(location, time, feature)`

So for each location and time, you have a vector of feature values.

### Conceptually

For one location and one time period, the model sees something like:

```text
[smc_number, rainfall, mean_temperature, rel_humidity, population, area, median_elevation]
```

This method stacks those vectors across all locations and all times.

---

## 9. `MultistepModel.fit_multi(y_xr, X_xr)`

Now the training reaches the real modeling core.

This class works at the xarray / numpy level.

The method begins with:

```python
lags = _build_lag_matrix_xr(y, self.n_target_lags)
y_target = y.isel(time=slice(self.n_target_lags, None))
```

This is where autoregression enters.

---

## 10. The Most Important Concept: Lag Features

The random forest does not directly understand time series.

So the code converts the time-series problem into a tabular regression problem.

It does that by creating lagged target columns.

### Example with 6 lags

Suppose for one location the disease counts are:

```text
t1=10, t2=12, t3=8, t4=20, t5=18, t6=15, t7=16
```

To predict `t7`, the model uses:

```text
[t1, t2, t3, t4, t5, t6] -> predict t7
```

In the real model, exogenous features for `t7` are also appended.

### What `_build_lag_matrix_xr` does

```python
shifted = [y.shift(time=k) for k in range(n_lags, 0, -1)]
lag_matrix = xr.concat(shifted, dim="lag")
return lag_matrix.isel(time=slice(n_lags, None))
```

Chronologically:

1. shift the target by 6,
2. shift the target by 5,
3. shift the target by 4,
4. shift the target by 3,
5. shift the target by 2,
6. shift the target by 1,
7. stack those shifted arrays as a new `lag` dimension,
8. discard the first 6 time positions because they do not have enough history.

### Why discard the first 6 time steps?

Because you cannot create a full 6-lag input for them.

If you need 6 past values, the first time point that can be predicted is time step 7.

---

## 11. Building the Actual Training Matrix

After building lags, the code combines exogenous features with lag features.

```python
lags_feat = lags.rename(lag="feature")
if X is not None:
    X_trimmed = X.isel(time=slice(self.n_target_lags, None))
    features = xr.concat(
        [X_trimmed.transpose("feature", "location", "time"), lags_feat],
        dim="feature",
    )
else:
    features = lags_feat
```

### What this means conceptually

For each `(location, time)` training example, the final feature vector becomes:

```text
[smc_number,
 rainfall,
 mean_temperature,
 rel_humidity,
 population,
 area,
 median_elevation,
 y_t-6,
 y_t-5,
 y_t-4,
 y_t-3,
 y_t-2,
 y_t-1]
```

And the target is:

```text
y_t
```

That is the full supervised learning problem.

### Why is this powerful?

Because the forest is no longer solving a special time-series problem. It is solving a standard regression problem with well-crafted features.

---

## 12. Pooling All Locations Together

After features are built, the code stacks all locations and times together.

```python
features_stacked = features.stack(sample=("location", "time"))
y_stacked = y_target.stack(sample=("location", "time"))

X_np = features_stacked.transpose("sample", "feature").values
y_np = y_stacked.values
```

This is an important design choice.

### What it means

Instead of fitting one model per location, the code fits one shared model using all rows from all locations.

So one row of `X_np` is one `(location, time)` observation.

### Why do this?

Because many locations may not have enough data individually.

Pooling gives the model more examples and lets it learn shared structure.

### What assumption does this make?

It assumes the relationship between:

- climate,
- lagged disease counts,
- and next disease count,

is similar enough across locations that one shared model is sensible.

---

## 13. NaN Filtering Before Training

Then the code removes invalid rows.

```python
mask = ~(np.isnan(X_np).any(axis=1) | np.isnan(y_np))
self.one_step_model.fit(X_np[mask], y_np[mask])
```

This means:

- if any feature is NaN, drop the row,
- if the target is NaN, drop the row.

This is a final safety filter.

---

## 14. `ResidualBootstrapModel.fit(X, y)`

Now training reaches the one-step probabilistic wrapper.

The code is:

```python
self._regressor.fit(X, y)
predictions = self._regressor.predict(X)
self._residuals = y - predictions
```

This method does three things, in strict order.

### Step 1. Train the random forest

```python
self._regressor.fit(X, y)
```

At this point the forest learns a mapping:

$$
f(\text{features at time } t) \approx y_t
$$

where features include both exogenous variables and lagged target values.

### Step 2. Predict on the training data itself

```python
predictions = self._regressor.predict(X)
```

This gives one fitted value for every training row.

### Step 3. Compute residuals

```python
self._residuals = y - predictions
```

A residual is:

$$
	ext{residual} = y_{\text{true}} - y_{\text{predicted}}
$$

### Why store residuals?

Because the forest only gives a mean-like point prediction.

The residuals capture typical errors the model made during training.

Later, when predicting, the code will say:

- the forest thinks the next value is 20,
- historically the model often misses by values like -3, +1, +5, -2,
- so sample one of those errors and add it.

That is the bootstrap uncertainty mechanism.

---

## 15. What Exactly Gets Saved

Back in `main.py`, after `forecaster.fit(X, y)`, this runs:

```python
with open(model, "wb") as f:
    pickle.dump(forecaster, f)
```

The saved object contains:

- the `DataFrameMultistepModel`,
- inside it, the `MultistepModel`,
- inside it, the `ResidualBootstrapModel`,
- inside it, the fitted `RandomForestRegressor`,
- plus the stored training residuals.

So the saved artifact already knows both:

- how to make point predictions,
- how to sample uncertainty.

---

## 16. Prediction Starts with `predict(...)`

Now switch to the second command in `main.py`.

```python
@app.command()
def predict(model: str, historic_data: str, future_data: str, out_file: str):
```

This function receives:

- a saved model file,
- historical observed data,
- future known features,
- an output CSV path.

Chronologically, this is what happens.

### Step 1. Load the saved model

```python
with open(model, "rb") as f:
    forecaster = pickle.load(f)
```

### Step 2. Read the historical and future data

```python
historic_df = pd.read_csv(historic_data)
future_df = pd.read_csv(future_data)
```

### Step 3. Extract historical target values

```python
y_historic = historic_df[INDEX_COLS + [TARGET_VARIABLE]].fillna(0)
```

This is used to initialize the lag window.

### Step 4. Infer the horizon length

```python
n_steps = future_df.groupby("location").size().iloc[0]
```

This assumes every location has the same number of future rows.

If each location has 6 future periods, then `n_steps = 6`.

### Step 5. Concatenate historic and future features

```python
X = pd.concat(
    [historic_df[INDEX_COLS + FEATURE_COLUMNS], future_df[INDEX_COLS + FEATURE_COLUMNS]],
    ignore_index=True,
).fillna(0).sort_values(by=INDEX_COLS)
```

This is subtle and important.

### Why combine historic and future features?

Because the prediction wrapper expects a single DataFrame with continuous time ordering.

Later it will keep only the last `n_steps` rows per location as the true future feature block, but this combined DataFrame keeps time alignment consistent.

---

## 17. `DataFrameMultistepModel.predict(...)`

The method is:

```python
def predict(self, y_historic, X, n_steps: int, n_samples: int):
    y_xr = target_to_xarray(y_historic, self._target_variable)
    previous_y = y_xr.isel(time=slice(-self._model.n_target_lags, None))

    X_xr = features_to_xarray(X)
    X_future_xr = X_xr.isel(time=slice(-n_steps, None)).rename({"time": "step"})
    future_df = X.groupby("location", sort=False).tail(n_steps)
    predictions = self._model.predict_multi(previous_y, n_steps, n_samples, X_future_xr)

    return _predictions_to_dataframe(predictions, future_df)
```

Chronologically this means:

1. convert historical target values to `(location, time)`,
2. keep only the last 6 target observations per location,
3. convert combined features to `(location, time, feature)`,
4. slice only the last `n_steps` feature rows as future features,
5. run multi-location forecasting,
6. convert output back to a DataFrame.

### Why only the last 6 observed target values?

Because recursive forecasting only needs the latest lag window to start.

If `n_target_lags = 6`, the initial state is exactly the last 6 observed counts.

---

## 18. `MultistepModel.predict_multi(...)`

This method loops over locations.

```python
for loc in locations:
    prev = previous_y.sel(location=loc).values
    X_loc = X.sel(location=loc).values if X is not None else None
    dist = self.predict_proba(prev, n_steps, X_loc)
    samples = dist.sample(n_samples)
    results.append(samples)
```

### What does this mean?

For each location separately:

1. take the last 6 observed counts,
2. take that location's future feature block,
3. create a `MultistepDistribution`,
4. sample 200 forecast trajectories,
5. store them.

So training is pooled across locations, but forecast rollout is done per location.

---

## 19. `MultistepDistribution` Is Where the Recursive Forecast Happens

This class is the heart of the multi-step logic.

It is called a “distribution” because it can generate samples, but it is really a recursive simulator.

### Constructor

```python
def __init__(self, model, previous_y, n_steps, n_target_lags, X):
```

It stores:

- the one-step probabilistic model,
- the most recent observed target values,
- the forecast horizon,
- the number of target lags,
- future known exogenous features.

No prediction happens yet.

### Why no prediction happens yet?

Because this object is lazy.

It only computes when `.sample(n)` is called.

---

## 20. `MultistepDistribution.sample(n)` Step by Step

This method is the single most important method to understand.

It turns one-step uncertainty into a multi-step forecast distribution.

The code starts with:

```python
lag_window = xr.DataArray(
    np.tile(self._previous_y, (n, 1)),
    dims=["trajectory", "lag"],
)
```

### What is `lag_window`?

If `n = 200`, then `lag_window` has shape:

```text
(200, 6)
```

Each row is one forecast trajectory.

Initially, all 200 rows are identical because all trajectories start from the same observed history.

### Example

If the last 6 observed disease counts are:

```text
[8, 11, 10, 13, 15, 14]
```

then before any forecast step, all trajectories start with:

```text
[8, 11, 10, 13, 15, 14]
```

---

## 21. Forecast Step 1

Inside the loop:

```python
for step in range(self._n_steps):
```

At step 1, the code builds the feature vector for each trajectory.

```python
exog = xr.DataArray(
    np.tile(self._X[step], (n, 1)),
    dims=["trajectory", "feature"],
)
features = xr.concat([exog, lag_window.rename(lag="feature")], dim="feature")
```

### What does this do?

For each trajectory, it creates:

```text
[future climate/features at step 1 | last 6 disease counts]
```

At this moment all trajectories still have identical lag histories, so the only randomness has not happened yet.

Then the code does:

```python
dist = self._model.predict_proba(features.values)
step_samples = xr.DataArray(dist.sample(1)[0], dims=["trajectory"])
```

### What happens here?

#### `self._model.predict_proba(features.values)`

This calls `ResidualBootstrapModel.predict_proba(...)`.

That method:

1. asks the fitted forest for point predictions,
2. wraps them together with stored residuals,
3. returns a `ResidualDistribution`.

#### `dist.sample(1)`

This draws one sampled next value per trajectory.

Inside `ResidualDistribution.sample(1)`:

1. one training residual is sampled for each trajectory,
2. that residual is added to the forest prediction,
3. the result is clipped to be non-negative.

So after step 1, the 200 trajectories are no longer identical.

They diverge.

---

## 22. Updating the Lag Window

After step 1 samples are generated, the lag window is updated.

```python
lag_window = lag_window.roll(lag=-1)
lag_window[{"lag": -1}] = step_samples
```

### What does this mean?

Suppose one trajectory currently has lag window:

```text
[8, 11, 10, 13, 15, 14]
```

and the sampled prediction for step 1 is:

```text
18
```

Then the new lag window becomes:

```text
[11, 10, 13, 15, 14, 18]
```

The oldest value is dropped. The new sampled value is appended.

This is the autoregressive feedback loop.

---

## 23. Forecast Step 2 and Beyond

Now the process repeats.

But there is a major difference.

At step 2, the lag windows differ across trajectories, because step 1 was sampled stochastically.

So each trajectory now has its own feature vector.

That means uncertainty propagates forward.

### This is the key idea

The model is not just adding independent noise to each horizon step.

It is doing something stronger:

- random sampled prediction at step 1 changes lag inputs for step 2,
- random sampled prediction at step 2 changes lag inputs for step 3,
- and so on.

So later horizons naturally become more uncertain.

This is exactly what you expect in recursive forecasting.

---

## 24. `ResidualDistribution` in Plain Language

This class is very small, but conceptually important.

Its role is:

- hold point predictions,
- hold stored residuals,
- generate sampled noisy predictions on demand.

The code is essentially:

```python
drawn = rng.choice(self._residuals, size=(n_samples, n_rows), replace=True)
samples = self._predictions[np.newaxis, :] + drawn
return np.maximum(samples, 0.0)
```

### Why sample residuals with replacement?

Because bootstrap resampling means reusing the empirical error distribution seen during training.

The model is saying:

“Historically my mistakes looked like this set of errors. I will treat those errors as representative of future uncertainty.”

### Why clamp with `np.maximum(samples, 0.0)`?

Because disease case counts cannot be negative.

Without this, a large negative residual could create impossible predictions.

---

## 25. Returning the Forecasts

After all forecast steps are sampled, `MultistepDistribution.sample(n)` returns an array of shape:

```text
(n_samples, n_steps)
```

For this project that usually means:

```text
(200, forecast_horizon)
```

Then `predict_multi(...)` stacks location outputs into:

```text
(location, trajectory, step)
```

Then `_predictions_to_dataframe(...)` converts that into a pandas DataFrame with columns:

- `time_period`
- `location`
- `sample_0`
- `sample_1`
- ...
- `sample_199`

That is what gets saved to CSV.

---

## 26. Every Important Class and Why It Exists

This section summarizes all key classes, but now after the chronological explanation.

### `App` from `cyclopts`

Role:

- provides the command-line interface,
- lets `train` and `predict` be exposed as commands.

It is not a modeling class.

### `RandomForestRegressor`

Role:

- learns the one-step mapping from feature vector to next disease count.

It is the core predictive engine.

By itself it is deterministic and one-step only.

### `ResidualBootstrapModel`

Role:

- wraps the forest,
- stores training residuals,
- exposes `predict_proba`.

It converts a deterministic regressor into a probabilistic one-step model.

### `ResidualDistribution`

Role:

- holds point predictions and residuals,
- can sample possible future values.

It is the source of one-step uncertainty.

### `DataFrameMultistepModel`

Role:

- user-facing wrapper around the lower-level model,
- converts pandas DataFrames to xarray,
- returns pandas outputs.

It makes the system easy to call from `main.py`.

### `MultistepModel`

Role:

- builds lagged target features,
- trains across multiple locations,
- creates per-location multi-step distributions.

It is the main orchestration layer for recursive modeling.

### `MultistepDistribution`

Role:

- recursively simulates multiple future trajectories,
- updates lag windows after each sampled step.

It is the source of multi-step uncertainty propagation.

### `Distribution` protocol

Role:

- defines the required interface for distribution-like objects.

Anything with `.sample(n_samples)` returning an array of the expected shape fits the contract.

### `OneStepModel` protocol

Role:

- defines the required interface for a one-step probabilistic model.

It lets `MultistepModel` work with any compatible one-step probabilistic predictor, not only the bootstrap wrapper.

### `DeterministicOneStepModel`

Role:

- defines the interface for deterministic one-step predictors.

Used by the deterministic version of the recursive forecaster.

### `DeterministicMultistepModel`

Role:

- recursive multi-step forecasting without probabilistic sampling.

This class is present in the code, but it is not used in the default `main.py` workflow.

### `SkproWrapper` and `SkproDistribution`

Role:

- adapt external probabilistic regressors from `skpro` to this project's protocol.

These are extension points, not the default path.

---

## 27. The Whole Training Process in One Compact Story

Training, chronologically:

1. load CSV into pandas,
2. separate features and target,
3. create a random forest,
4. wrap it in a residual bootstrap one-step model,
5. wrap that in a pandas-friendly multi-step forecaster,
6. pivot target to `(location, time)`,
7. pivot features to `(location, time, feature)`,
8. build 6 lagged target features,
9. combine exogenous features and lagged target features,
10. flatten all `(location, time)` pairs into one training table,
11. fit the random forest,
12. compute and store training residuals,
13. pickle the whole forecaster.

---

## 28. The Whole Prediction Process in One Compact Story

Prediction, chronologically:

1. load the saved forecaster,
2. load historical observations,
3. load future known features,
4. keep the last 6 observed target values per location,
5. keep future exogenous features per location,
6. for each location, create 200 identical lag windows,
7. at each forecast step:
8. build features from future exogenous values plus current lag window,
9. get random-forest point predictions,
10. add one sampled residual per trajectory,
11. clip negatives to zero,
12. append sampled values to the lag window,
13. repeat until horizon is complete,
14. return 200 trajectories per location,
15. save them as CSV.

---

## 29. What This Model Is and Is Not

### What it is

- a pooled multi-location regression model,
- a tabularized autoregressive forecaster,
- a residual-bootstrap probabilistic predictor,
- a recursive multi-step simulator.

### What it is not

- not a state-space model,
- not a Bayesian generative model,
- not a model that directly learns a full probability distribution from first principles,
- not a separate model per location,
- not a direct multi-horizon model.

It learns one-step behavior, then rolls forward recursively.

---

## 30. The Most Important Intuition to Keep

If you keep only one mental model from this codebase, keep this one:

The project converts a time-series forecasting problem into a regular regression problem by adding lagged disease counts as features. Then it converts a normal random-forest point predictor into a probabilistic forecaster by resampling training residuals. Finally, it converts that one-step probabilistic predictor into a multi-step simulator by recursively feeding sampled predictions back into the lag inputs.

That single sentence is the entire design.

---

## 31. Minimal Mental Diagram

```text
historical panel data
    -> build lagged target features
    -> combine with climate/context features
    -> train random forest for one-step prediction
    -> store training residuals
    -> for forecasting: predict one step
    -> sample residual noise
    -> append sampled value to lag window
    -> repeat for next step
    -> produce many trajectories
```

---

## 32. Final Short Definitions of the Hard Concepts

### Autoregressive

The future target depends on past target values.

### Lag

A previous value of the same target series.

### Residual

The difference between observed value and model prediction.

### Residual bootstrap

Generate uncertainty by resampling stored model errors.

### Recursive forecasting

Predict one step ahead, then use that predicted value to help predict the next step.

### Pooled model

One shared model is fit using data from all locations.

### Exogenous features

Predictors that are not the target itself, such as rainfall or temperature.

### Probabilistic forecast

A set of possible futures, not just one number.

---

If useful, the next improvement would be to add a small worked example with a toy dataset of one location and 8 time points, showing the exact numeric lag matrix, the exact training table row by row, and one recursive sampled rollout by hand.
- `future_df` is used only for recovering original time-period label strings in the output.

---

### `DeterministicMultistepModel`

The point-prediction-only counterpart of `MultistepModel`. Uses `.predict()` instead of `.predict_proba()`. Feeds each point prediction forward as the next step's lag. No sampling, no uncertainty quantification. Useful for ablation studies or when only a forecast mean is needed.

---

## Full Training Data Flow

```
CSV file
  │
  ▼
pd.read_csv → DataFrame with [time_period, location, disease_cases, smc_number, rainfall, ...]
  │
  ▼
DataFrameMultistepModel.fit(X, y)
  │
  ├─► target_to_xarray(y)
  │       pivot → DataArray (location, time) of disease_cases
  │
  ├─► features_to_xarray(X)
  │       pivot each feature → DataArray (location, time, feature)
  │
  └─► MultistepModel.fit_multi(y_xr, X_xr)
          │
          ├─► _build_lag_matrix_xr(y, 6)
          │       shift y by 1..6 steps → DataArray (lag=6, location, time)
          │       drop first 6 time steps (NaN lags)
          │
          ├─► concat [X_trimmed (location,time,feature), lags (feature,location,time)]
          │       → DataArray (feature, location, time)
          │
          ├─► .stack(sample=("location","time"))
          │       → flat (N_total_samples, n_features+6) array
          │       (pooling all locations together)
          │
          ├─► drop NaN rows
          │
          └─► ResidualBootstrapModel.fit(X_np, y_np)
                  │
                  ├─► RandomForestRegressor.fit(X_np, y_np)
                  │       trains the RF on all (location, time) pairs
                  │
                  └─► residuals = y_np - RF.predict(X_np)
                          stores in-sample residuals for noise sampling
```

---

## Full Prediction Data Flow

```
historic CSV + future CSV
  │
  ▼
DataFrameMultistepModel.predict(y_historic, X_combined, n_steps=6, n_samples=200)
  │
  ├─► target_to_xarray(y_historic)  → y_xr (location, T_hist)
  ├─► previous_y = y_xr[:, -6:]     → last 6 observed values per location
  │
  ├─► features_to_xarray(X_combined) → X_xr (location, T_hist+n_steps, n_features)
  ├─► X_future = X_xr[:, -n_steps:] → (location, n_steps, n_features)
  │
  └─► MultistepModel.predict_multi(previous_y, n_steps=6, n_samples=200, X_future)
          │
          └─► for each location:
                  MultistepDistribution.sample(200)
                      │
                      └─► AR loop (6 steps):
                              step 1: features = [climate_step1 | lag_1..lag_6]
                                      → RF point pred + sample 1 residual × 200 trajectories
                                      → 200 sampled "step 1" values
                              step 2: features = [climate_step2 | sample_step1 | lag_1..lag_5]
                                      → 200 sampled "step 2" values (diverging)
                              ...
                              step 6: 200 trajectories fully diverged
                      returns (200, 6) array
          stacked → DataArray (location, trajectory=200, step=6)
          │
          └─► _predictions_to_dataframe(...)
                  → wide DataFrame: [time_period, location, sample_0, ..., sample_199]
                  saved to out_file CSV
```

---

## Key Statistical Concepts

### Autoregression (AR)
The model uses `N_TARGET_LAGS=6` past disease counts as predictors for the next value. This lets it capture seasonality and epidemic dynamics without explicitly modelling them — the lag structure implicitly encodes time-series patterns.

### Residual Bootstrap
After training, the model stores `residuals = y_true - y_pred` for all training points. At prediction time, uncertainty is quantified by adding randomly drawn training residuals to the point forecast. This is a non-parametric bootstrap — no distributional assumption (Gaussian etc.) is imposed.

### Recursive / Rollout Forecasting
When forecasting `h` steps ahead, the model only predicts one step at a time, then feeds that prediction back in as a lag feature. This means:
- Step 1 uncertainty: just one residual draw.
- Step 2 uncertainty: the step-1 sample propagates into the lag → uncertainty compounds.
- Step-h uncertainty grows with the horizon, which is statistically appropriate.

### Pooled Multi-location Training
All `(location, time)` pairs are treated as independent training examples. The RF learns a single shared function across all locations. This increases effective sample size but assumes the relationship between features and disease counts is **similar across locations** (after accounting for `population`, `area`, etc.).

### Non-negativity Clamping
`np.maximum(samples, 0.0)` in `ResidualDistribution.sample` enforces that disease case counts are non-negative. This is necessary because bootstrapped residuals can be large negatives that would otherwise push predictions below zero.
