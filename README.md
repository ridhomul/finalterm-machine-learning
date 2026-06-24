# Final Term Machine Learning

Repository for the UAS Machine Learning individual task: **Hands-On End-to-End Models Machine Learning**. The project builds deep learning pipelines for the provided tabular datasets, including preprocessing, feature engineering, model comparison, Optuna hyperparameter tuning, evaluation metrics, and MLflow experiment tracking.

## Student Identity

- Name: `Ridho Anugrah Mulyadi`
- Class: `<TK-47-05>`
- NIM: `<101032300028>`

## Project Overview

This repository contains two end-to-end deep learning experiments:

1. **Regression task** using `midterm-regresi-dataset.csv`.
   - The first column is treated as the regression target.
   - The remaining 90 columns are numerical features.
   - Models: baseline MLP, deeper MLP, and Optuna-tuned Residual MLP.
   - Metrics: MAE, RMSE, and R2 score.

2. **Fraud classification task** using `train_transaction.csv`.
   - The target column is `isFraud`.
   - Numerical and categorical transaction features are preprocessed separately.
   - Models: baseline MLP, deeper MLP, and Optuna-tuned Residual MLP.
   - Metrics: ROC-AUC, PR-AUC, accuracy, precision, recall, F1 score, and confusion matrix.

Both notebooks log parameters, validation losses, final metrics, and trained PyTorch models to MLflow.

## Repository Structure

```text
.
|-- agents.md
|-- midterm-regresi-dataset.csv
|-- train_transaction.csv
|-- README.md
|-- requirements.txt
|-- .gitattributes
|-- .gitignore
|-- notebooks/
|   |-- 01_regression_deep_learning_optuna_mlflow.ipynb
|   `-- 02_fraud_detection_deep_learning_optuna_mlflow.ipynb
|-- src/
|   |-- __init__.py
|   `-- tabular_dl.py
|-- reports/
|   `-- results_summary.md
|-- tools/
|   `-- create_notebooks.py
|-- artifacts/
|   `-- .gitkeep
`-- models/
    `-- .gitkeep
```

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m ipykernel install --user --name finalterm-machine-learning --display-name "FinalTerm ML"
```

Open the notebooks:

```powershell
jupyter lab
```

Start the MLflow UI after running experiments:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open the local URL shown by MLflow, usually `http://127.0.0.1:5000`. Tracking metadata is stored in `mlflow.db`; run artifacts are stored under `mlruns/`.

## Dataset Notes

The CSV files are large. GitHub normally rejects files over 100 MB, so this repository includes `.gitattributes` for Git LFS tracking of CSV files. If Git LFS is not available, keep the datasets locally in the project root with these exact filenames:

- `midterm-regresi-dataset.csv`
- `train_transaction.csv`

The notebooks use `QUICK_RUN = True` by default so they can run faster on a laptop. Set `QUICK_RUN = False` for fuller training before submitting final results.

## Notebook Guide

- `notebooks/01_regression_deep_learning_optuna_mlflow.ipynb` covers preprocessing, numeric feature engineering, MLP and Residual MLP training, Optuna tuning, MLflow logging, and regression metric comparison.
- `notebooks/02_fraud_detection_deep_learning_optuna_mlflow.ipynb` covers fraud data preprocessing, categorical encoding, class imbalance handling, neural-network training, Optuna tuning, MLflow logging, and classification metric comparison.
- `src/tabular_dl.py` contains reusable PyTorch model and training helpers shared by both notebooks.
- `reports/results_summary.md` is a submission-friendly summary template for the final metric tables and conclusions.

## Results Summary

The latest local run produced the following model comparison. Full tables are available in `reports/results_summary.md`.

| Task | Best Model | Main Result | Interpretation |
|---|---|---|---|
| Regression | Deep MLP | RMSE `8.4448`, MAE `5.9307`, R2 `0.3710` | Best overall regression model by RMSE and R2. |
| Fraud Classification | Deep MLP | PR-AUC `0.4710`, ROC-AUC `0.8938`, F1 `0.4844` | Best fraud model, especially on PR-AUC, recall, and F1. |

For regression, the Deep MLP outperformed the baseline and Optuna residual model on RMSE and R2, although the Optuna model had the lowest MAE. For fraud detection, the Deep MLP achieved the strongest overall performance across PR-AUC, ROC-AUC, accuracy, precision, recall, F1, and validation loss.

## Conclusion

The deeper fixed MLP architecture performed best across both tasks in the current experiments. Optuna helped explore hyperparameters and improved some metrics compared with the baseline, but the tuned residual model did not outperform the Deep MLP overall. For fraud detection, PR-AUC, recall, and F1 are prioritized over accuracy because the dataset is highly imbalanced.
