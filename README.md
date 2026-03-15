# Bank Marketing Classification

## Project Overview

This project predicts whether a bank client will subscribe to a term deposit on the Bank Marketing dataset. It includes three models:

- Logistic Regression
- MLP
- TabTransformer

The target column is `y`, where `yes -> 1` and `no -> 0`.

## Setup Instructions

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Required Dependencies

Dependencies are listed in [`requirements.txt`](/Users/jiangxin/mycode/ML_HW/final_project/requirements.txt):

- `torch`
- `numpy<2`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `tqdm`
- `pyyaml`
- `kagglehub`

## How To Reproduce Results

After setting up dependencies, you can just go to [`demo.ipynb`](demo.ipynb) to reproduce all the workflow. (Recommend)

Then you can skip the following content.

### or 

Run the following in order:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/preprocess.py --config configs/config.yaml
python src/train.py --model all
python src/evaluate.py --checkpoint checkpoints/best_model.pt
```


## Where To Download The Dataset

The Bank Marketing dataset is from:

<https://www.kaggle.com/datasets/sushant097/bank-marketing-dataset-full>

We download it with `kagglehub`:

```python
import kagglehub

path = kagglehub.dataset_download("sushant097/bank-marketing-dataset-full")
```
Then put it in `data/bank-full.csv`

## How To Preprocess The Data

Run:

```bash
python src/preprocess.py --config configs/config.yaml
```

If your CSV is not at `data/bank-full.csv`, run:

```bash
python src/preprocess.py --config configs/config.yaml --data-path /path/to/bank-full.csv
```

To run the ablation without `duration`:

```bash
python src/preprocess.py --config configs/config.yaml --exclude-duration
```

## How To Train The Model

Train one model:

```bash
python src/train.py --model logistic
python src/train.py --model mlp
python src/train.py --model tabtransformer
```

Train all experiments:

```bash
python src/train.py --model all
```

## How To Evaluate The Model

Evaluate the saved best PyTorch checkpoint:

```bash
python src/evaluate.py --checkpoint checkpoints/best_model.pt
```

Evaluate Logistic Regression:

```bash
python src/evaluate.py --model logistic
```

## Expected Outputs

After preprocessing:

- `data/processed/embedding_data.npz`
- `data/processed/logistic_data.npz`
- `data/processed/metadata.json`

After training:

- `results/experiment_results.csv`
- `results/logistic_metrics.csv`
- `results/mlp_metrics.csv`
- `results/tabtransformer_metrics.csv`
- `results/mlp_history.csv`
- `results/tabtransformer_history.csv`
- `checkpoints/best_model.pt`

After evaluation:

- `results/roc_curve.png`
- `results/confusion_matrix.png`

