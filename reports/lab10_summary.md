# Lab 10: Failure Injection Testing

## Aim
To test the robustness and recovery capability of the IMDb Movie Rating Prediction ML pipeline.

## Procedure
1. Executed the failure injection script.
2. Simulated a missing dataset failure.
3. Verified that the pipeline detected the failure.
4. Restored the dataset.
5. Executed the recovery pipeline.
6. Verified model training and evaluation.

## Results
- Expected failure detected.
- Dataset restored successfully.
- Preprocessing completed.
- Linear Regression, Random Forest and Gradient Boosting models trained.
- Gradient Boosting selected as the best model.
- Best model saved at models/best_model.pkl.
- Recovery pipeline completed successfully.

## Conclusion
The failure injection test demonstrated that the ML pipeline can detect an intentional failure and recover after restoring the dataset.
