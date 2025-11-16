# BMLP Final Project — Clustering & Classification

End-to-end machine learning notebooks for unsupervised clustering and supervised classification. The project includes data loading, preprocessing, PCA + KMeans clustering, Decision Tree and Random Forest classifiers, evaluation metrics/plots, hyperparameter tuning, and persisted artifacts.

## Features
- Data loading from CSV datasets (`data_clustering.csv`, `data_clustering_inverse.csv`).
- Preprocessing: `LabelEncoder` for categorical features and `StandardScaler` for numeric scaling.
- Clustering: PCA dimensionality reduction + KMeans; includes silhouette score and cluster analysis.
- Classification: `DecisionTreeClassifier` and `RandomForestClassifier` with train/test split.
- Evaluation: Accuracy, Precision, Recall, F1-score, and confusion matrix visualizations (seaborn/matplotlib).
- Hyperparameter tuning: `RandomizedSearchCV` for Random Forest; best estimator saved.
- Artifacts (saved via joblib using `.h5` filenames):
	- `decision_tree_model.h5`
	- `explore_random_forest_classification.h5`
	- `tuning_classification.h5`
	- `model_clustering.h5`
	- `PCA_model_clustering.h5`

## Tech Stack
- Python 3 (tested with 3.9–3.11)
- Jupyter Notebook / JupyterLab
- NumPy, pandas
- scikit-learn
- seaborn, matplotlib
- joblib

## Installation
1) Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

2) Install dependencies:

```bash
pip install --upgrade pip
pip install jupyter numpy pandas scikit-learn seaborn matplotlib joblib
```

## How to Run
Option A — VS Code
- Open this folder in VS Code.
- Open `[Klasifikasi]_Submission_Akhir_BMLP_Nelson_Ahli.ipynb` or `[Clustering]_Submission_Akhir_BMLP_Nelson_Ahli.ipynb`.
- Select a Python 3 kernel, then “Run All”.

Option B — Jupyter (CLI)

```bash
jupyter lab
# or
jupyter notebook
```

Then open:
- Classification: `[Klasifikasi]_Submission_Akhir_BMLP_Nelson_Ahli.ipynb`
- Clustering: `[Clustering]_Submission_Akhir_BMLP_Nelson_Ahli.ipynb`

Notes for running:
- Keep the working directory at the repository root so relative data paths resolve.
- The notebooks save models automatically to the repo root (files listed above).

## Project Structure (reference)
```
.
├─ [Klasifikasi]_Submission_Akhir_BMLP_Nelson_Ahli.ipynb
├─ [Clustering]_Submission_Akhir_BMLP_Nelson_Ahli.ipynb
├─ data_clustering.csv
├─ data_clustering_inverse.csv
├─ decision_tree_model.h5
├─ explore_random_forest_classification.h5
├─ tuning_classification.h5
├─ model_clustering.h5
├─ PCA_model_clustering.h5
└─ README.md
```

## Notes
- Although the artifacts use the `.h5` extension, they are saved with `joblib.dump(...)` inside the notebooks.
- Exact metrics and plots are produced inside the notebooks.
