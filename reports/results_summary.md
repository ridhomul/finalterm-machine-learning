# Results Summary

Fill this report after running the notebooks. MLflow will store tracking metadata in `mlflow.db` and run artifacts under `mlruns/`.

## Regression: `midterm-regresi-dataset.csv`

| Model | MAE | RMSE | R2 | Notes |
|---|---:|---:|---:|---|
| Baseline MLP |  |  |  |  |
| Deep MLP |  |  |  |  |
| Optuna Residual MLP |  |  |  |  |

## Fraud Classification: `train_transaction.csv`

| Model | ROC-AUC | PR-AUC | Accuracy | Precision | Recall | F1 | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| Baseline MLP |  |  |  |  |  |  |  |
| Deep MLP |  |  |  |  |  |  |  |
| Optuna Residual MLP |  |  |  |  |  |  |  |

## Analysis Points

- Which architecture performed best, and on which metric?
- Did Optuna improve the validation score compared with fixed hyperparameters?
- For fraud classification, does the best model balance recall and precision well?
- What limitations remain, such as training time, class imbalance, missing values, or dataset size?
