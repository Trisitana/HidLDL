# HidLDL

Code and datasets for **Towards Better IncomLDL: We Are Unaware of Hidden Labels in Advance**.

This repository contains the implementation, demo scripts, visualization utilities, and benchmark datasets used for hidden-label incomplete label distribution learning.

## Structure

```text
demo/             Example scripts and quick-start notebook
pyldl/            Core LDL algorithms and utilities
visualization/    Result calculation and plotting scripts
dataset/          Benchmark .mat datasets
```

## Installation

Install the required Python packages:

```bash
pip install -r requirements.txt
```

The main dependencies include NumPy, SciPy, scikit-learn, TensorFlow, qpsolvers, quadprog, pandas, matplotlib, and related scientific Python packages.

## Quick Start

Run the demo:

```bash
python3 demo/demo.py
```

You can change the dataset in `demo/demo.py`:

```python
X, y = load_dataset('SJAFFE')
```

You can also change the missing-label rate:

```python
y_missing, mask = random_missing_real(y, missing_rate=.5)
```

## Experiments

Run the recovery experiment:

```bash
python3 demo/test_recover.py
```

Run the prediction experiment:

```bash
python3 demo/test_predict.py
```

Additional comparison methods can be registered in `pyldl/algorithms/__init__.py`.
