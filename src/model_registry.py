import mlflow
import pandas as pd


# Use the MLflow database
mlflow.set_tracking_uri(
    "sqlite:///mlflow.db"
)


# Experiment name
experiment_name = "IMDB Movie Rating Prediction"


# Get the experiment
experiment = mlflow.get_experiment_by_name(
    experiment_name
)


if experiment is None:

    raise ValueError(
        "MLflow experiment not found."
    )


# Get successful runs
runs = mlflow.search_runs(
    experiment_ids=[
        experiment.experiment_id
    ],
    filter_string="status = 'FINISHED'",
    order_by=[
        "metrics.RMSE ASC"
    ]
)


# Check whether runs exist
if runs.empty:

    raise ValueError(
        "No successful MLflow runs found."
    )


# Select the best run
best_run = runs.iloc[0]


run_id = best_run["run_id"]

best_model = best_run["params.model"]

rmse = best_run["metrics.RMSE"]


print("================================")
print("BEST MODEL")
print("================================")

print("Model:", best_model)
print("RMSE:", rmse)
print("Run ID:", run_id)


# Model location inside the MLflow run
model_uri = (
    f"runs:/{run_id}/model"
)


# Registered model name
registered_model_name = (
    "IMDB_Movie_Rating_Model"
)


# Register the model
model_version = mlflow.register_model(
    model_uri=model_uri,
    name=registered_model_name
)


print("\n================================")
print("MODEL REGISTRATION COMPLETED")
print("================================")

print(
    "Registered Model:",
    registered_model_name
)

print(
    "Version:",
    model_version.version
)

print(
    "Model:",
    best_model
)

print(
    "Run ID:",
    run_id
)