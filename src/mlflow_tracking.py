import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from preprocess import (
    load_data,
    prepare_data,
    create_preprocessor
)


# Create MLflow experiment
mlflow.set_experiment(
    "IMDB Movie Rating Prediction"
)


# Load dataset
df = load_data()

# Prepare features and target
X, y = prepare_data(df)


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# Define models
models = {

    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200,
        random_state=42
    )
}


# Create preprocessing object
preprocessor = create_preprocessor()


# Train and track each model
for name, model in models.items():

    with mlflow.start_run(run_name=name):

        # Create pipeline
        pipeline = Pipeline([
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ])

        # Train
        pipeline.fit(
            X_train,
            y_train
        )

        # Predictions
        predictions = pipeline.predict(
            X_test
        )

        # Metrics
        mae = mean_absolute_error(
            y_test,
            predictions
        )

        rmse = mean_squared_error(
            y_test,
            predictions
        ) ** 0.5

        r2 = r2_score(
            y_test,
            predictions
        )

        # Log parameters
        mlflow.log_param(
            "model",
            name
        )

        mlflow.log_param(
            "test_size",
            0.20
        )

        mlflow.log_param(
            "random_state",
            42
        )

        # Log metrics
        mlflow.log_metric(
            "MAE",
            mae
        )

        mlflow.log_metric(
            "RMSE",
            rmse
        )

        mlflow.log_metric(
            "R2",
            r2
        )

        # Log trained model
        mlflow.sklearn.log_model(
    pipeline,
    "model",
    skops_trusted_types=[
        "sklearn.tree._tree.Tree"
    ]
)

        print(
            f"{name} logged successfully."
        )


print("\nMLflow experiment tracking completed.")