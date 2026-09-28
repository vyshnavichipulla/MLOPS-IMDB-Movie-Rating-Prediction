# IMDB Movie Rating Prediction - MLOps Project

## Project Overview

This project implements an end-to-end Machine Learning and MLOps workflow for predicting IMDb movie ratings.

The project covers data preprocessing, model training, model evaluation, experiment tracking, ML pipeline automation, and model registration using MLflow.

## Dataset

The project uses the IMDb Movie Dataset.

Target variable:

- Rating

Features used:

- Main_Genre
- Director
- Year
- Runtime (Minutes)
- Votes

## Machine Learning Models

Three regression models are implemented:

1. Linear Regression
2. Random Forest Regressor
3. Gradient Boosting Regressor

## Data Preprocessing

The preprocessing workflow includes:

- Loading the IMDb dataset
- Extracting the main genre from the Genre column
- Separating features and target
- One-hot encoding categorical features
- Passing numerical features through preprocessing
- Splitting the dataset into training and testing data

## Model Evaluation

The models are evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

The evaluation results are stored in:

```text
outputs/regression_results.csv