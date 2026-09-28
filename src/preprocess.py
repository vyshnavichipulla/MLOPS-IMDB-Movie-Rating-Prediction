import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def load_data():

    # Dataset path
    path = "data/raw/IMDB-Movie-Data-selected-columns.csv"

    # Read dataset
    df = pd.read_csv(path)

    print("Dataset Shape:", df.shape)

    return df


def prepare_data(df):

    # Create a copy
    data = df.copy()

    # Extract the first genre
    data["Main_Genre"] = (
        data["Genre"]
        .str.split(",")
        .str[0]
        .str.strip()
    )

    # Features used for prediction
    features = [
        "Main_Genre",
        "Director",
        "Year",
        "Runtime (Minutes)",
        "Votes"
    ]

    X = data[features]

    # Target variable
    y = data["Rating"]

    return X, y


def create_preprocessor():

    # Categorical columns
    categorical_features = [
        "Main_Genre",
        "Director"
    ]

    # Numerical columns
    numerical_features = [
        "Year",
        "Runtime (Minutes)",
        "Votes"
    ]

    # Create preprocessing transformer
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            ),
            (
                "numerical",
                "passthrough",
                numerical_features
            )
        ]
    )

    return preprocessor


if __name__ == "__main__":

    # Load dataset
    df = load_data()

    # Prepare data
    X, y = prepare_data(df)

    print("\nFeatures:")
    print(X.head())

    print("\nTarget:")
    print(y.head())

    print("\nPreprocessing setup completed.")