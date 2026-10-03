# Customer Segmentation with K-Means

## Project goal

This project demonstrates customer segmentation using the Mall Customers dataset
and the scikit-learn K-Means clustering algorithm.

## Segmentation objective

Customers are grouped by **Annual Income (k$)** and **Spending Score (1-100)**.
The workflow cleans invalid feature rows, standardizes the clustering features,
calculates Within-Cluster Sum of Squares (WCSS) for candidate cluster counts,
and selects the elbow automatically. Customer IDs and demographic columns are
preserved in the output but are not used as clustering features.

## Project structure

```text
dataset/
    Mall_Customers.csv
src/
    cluster_model.py
requirements.txt
README.md
```

## Installation

Python 3.9 or newer is recommended.

```bash
python -m venv .venv
```

Activate the environment, then install dependencies:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

python -m pip install -r requirements.txt
```

## Run the clustering workflow

From the project root:

```bash
python src/cluster_model.py
```

The script writes `dataset/Customer_Segments.csv`, `elbow_curve.png`, and
`customer_segments.png`. The source dataset is a small mock dataset intended to
verify the pipeline locally; replace it with the full Mall Customers dataset
when available.
