import os
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def evaluate_models():

    # Load test data
    test_data = pd.read_csv(
        "outputs/test_data.csv"
    )

    # Separate features and target
    X_test = test_data.drop(
        columns=["Rating"]
    )

    y_test = test_data["Rating"]

    # List of trained models
    model_files = [
        "linear_regression.pkl",
        "random_forest.pkl",
        "gradient_boosting.pkl"
    ]

    results = []

    # Evaluate each model
    for file in model_files:

        model_path = os.path.join(
            "models",
            file
        )

        # Load trained model
        model = joblib.load(
            model_path
        )

        # Generate predictions
        predictions = model.predict(
            X_test
        )

        # Calculate MAE
        mae = mean_absolute_error(
            y_test,
            predictions
        )

        # Calculate RMSE
        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                predictions
            )
        )

        # Calculate R2
        r2 = r2_score(
            y_test,
            predictions
        )

        # Store results
        model_name = file.replace(
            ".pkl",
            ""
        )

        results.append({
            "Model": model_name,
            "MAE": mae,
            "RMSE": rmse,
            "R2 Score": r2
        })

    # Convert results to DataFrame
    results_df = pd.DataFrame(
        results
    )

    # Sort by RMSE
    results_df = results_df.sort_values(
        "RMSE"
    )

    # Save results
    os.makedirs(
        "outputs",
        exist_ok=True
    )

    results_df.to_csv(
        "outputs/regression_results.csv",
        index=False
    )

    print("\n========== MODEL EVALUATION ==========")

    print(
        results_df.round(3)
    )

    print(
        "\nEvaluation results saved to:"
    )

    print(
        "outputs/regression_results.csv"
    )


if __name__ == "__main__":

    evaluate_models()