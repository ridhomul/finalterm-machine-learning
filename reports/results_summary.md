# Results Summary

This summary is based on the generated result files in `artifacts/regression_results.csv` and `artifacts/fraud_results.csv`. MLflow stores tracking metadata in `mlflow.db` and run artifacts under `mlruns/`.

## Regression: `midterm-regresi-dataset.csv`

Lower MAE/RMSE is better, while higher R2 is better.

| Model | MAE | RMSE | R2 | Best Valid Loss | Notes |
|---|---:|---:|---:|---:|---|
| Deep MLP | 5.9307 | 8.4448 | 0.3710 | 0.6242 | Best RMSE, R2, and validation loss. |
| Baseline MLP | 5.9422 | 8.4861 | 0.3648 | 0.6333 | Competitive baseline, slightly below Deep MLP. |
| Optuna Residual MLP | 5.9072 | 8.6156 | 0.3453 | 0.6306 | Best MAE, but weaker RMSE/R2 than both fixed models. |

The Deep MLP is the best regression model overall because it achieves the lowest RMSE and highest R2. The Optuna-tuned residual model reduces MAE slightly, but its RMSE and R2 indicate less stable overall prediction quality.

## Fraud Classification: `train_transaction.csv`

For fraud detection, PR-AUC, recall, and F1 are more important than accuracy because the target class is imbalanced.

| Model | ROC-AUC | PR-AUC | Accuracy | Precision | Recall | F1 | Threshold | Best Valid Loss | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Deep MLP | 0.8938 | 0.4710 | 0.9769 | 0.5640 | 0.4245 | 0.4844 | 0.8900 | 0.7875 | Best overall fraud model. |
| Optuna Residual MLP | 0.8877 | 0.4666 | 0.9753 | 0.5236 | 0.4036 | 0.4559 | 0.8875 | 0.7933 | Improves recall/F1 over baseline, but not over Deep MLP. |
| Baseline MLP | 0.8916 | 0.4634 | 0.9763 | 0.5551 | 0.3802 | 0.4513 | 0.9045 | 0.8079 | Strong baseline, but misses more fraud cases. |

The Deep MLP is also the best fraud model because it has the highest PR-AUC, recall, F1, accuracy, precision, ROC-AUC, and lowest validation loss in this run. The Optuna residual model improves recall and F1 compared with the baseline, but its precision and ROC-AUC are lower.

## Analysis

- The Deep MLP performed best overall on both tasks.
- Optuna improved some metrics but did not produce the top final model in this run.
- For regression, the best model should be selected by RMSE and R2 rather than MAE alone.
- For fraud detection, accuracy is less informative because most transactions are non-fraud. PR-AUC, recall, and F1 give a better view of fraud detection quality.
- Remaining limitations include large dataset size, training time, class imbalance in fraud detection, and the limited number of Optuna trials.
