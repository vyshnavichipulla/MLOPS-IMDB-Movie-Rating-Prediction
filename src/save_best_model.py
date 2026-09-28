import joblib
import pandas as pd
import os


def save_best_model():

    # Load model evaluation results
    results = pd.read_csv(
        "outputs/regression_results.csv"
    )

    # Results are already sorted by RMSE
    best_model = results.iloc[0]["Model"]

    print("Best Model:", best_model)

    # Path of the best model
    model_path = (
        "models/"
        + best_model
        + ".pkl"
    )

    # Load the best model
    model = joblib.load(
        model_path
    )

    # Save as best_model.pkl
    os.makedirs(
        "models",
        exist_ok=True
    )

    joblib.dump(
        model,
        "models/best_model.pkl"
    )

    print(
        "Best model saved successfully."
    )

    print(
        "Location: models/best_model.pkl"
    )


if __name__ == "__main__":
    save_best_model()