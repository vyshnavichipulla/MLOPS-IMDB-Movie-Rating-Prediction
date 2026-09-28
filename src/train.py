import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from preprocess import (
    load_data,
    prepare_data,
    create_preprocessor
)


def train_models():

    # Load the dataset
    df = load_data()

    # Prepare features and target
    X, y = prepare_data(df)

    # Split the dataset into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    # Define the models
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

    # Create models folder
    os.makedirs("models", exist_ok=True)

    preprocessor = create_preprocessor()

    # Train every model
    for name, model in models.items():

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipeline.fit(
            X_train,
            y_train
        )

        # Create filename
        filename = (
            name.lower()
            .replace(" ", "_")
            + ".pkl"
        )

        # Save trained model
        joblib.dump(
            pipeline,
            f"models/{filename}"
        )

        print(
            f"{name} trained and saved successfully."
        )

    # Save test data for evaluation
    os.makedirs("outputs", exist_ok=True)

    test_data = X_test.copy()

    test_data["Rating"] = y_test

    test_data.to_csv(
        "outputs/test_data.csv",
        index=False
    )

    print("\nTest data saved successfully.")
    print("Training completed.")


if __name__ == "__main__":
    train_models()