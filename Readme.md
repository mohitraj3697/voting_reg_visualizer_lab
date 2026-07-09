# Voting Regressor Visualizer Lab

An interactive visualization tool built with **matplotlib** and **scikit-learn** that demonstrates how a `VotingRegressor` ensemble combines predictions from multiple base regressors on a synthetic non-linear dataset.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange?logo=scikit-learn&logoColor=white)
![matplotlib](https://img.shields.io/badge/matplotlib-3.5%2B-green)

---

## Overview

Toggle and configure three base machine learning regressors via an interactive sidebar and observe how their unique mathematical properties influence the overall ensemble prediction curve:

| Model | Characteristics |
|---|---|
| **Linear Regression** | High bias, low variance — fits a straight line regardless of data complexity |
| **SVR** | Tunable complexity via kernel type, regularization (C), and epsilon-tube width |
| **Decision Tree** | High variance, low bias — can overfit with jagged, step-like predictions |

When **two or more** models are activated, their outputs are averaged through scikit-learn's `VotingRegressor` to produce a smooth, unified **blue ensemble curve** that delivers a balanced model fit.

---

## Features

- **Real-time model toggling** — activate/deactivate each base estimator with checkboxes
- **Interactive parameter tuning** — adjust SVR kernel, C, ε, and Decision Tree max depth with sliders
- **Live ensemble curve** — see the VotingRegressor output update instantly
- **Performance metrics panel** — R² and RMSE displayed for every active model
- **Dark-themed UI** — GitHub-inspired dark color palette

---

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/voting_reg_visualizer_lab.git
cd voting_reg_visualizer_lab

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

A matplotlib window will open with:
- **Left panel** — scatter plot of synthetic non-linear data with model prediction curves
- **Right sidebar** — checkboxes, sliders, and radio buttons to configure models

### Controls

| Control | Effect |
|---|---|
| ☑ Linear Regression | Toggle the linear model (dashed orange line) |
| ☑ SVR | Toggle the support vector regressor (dashed green line) |
| ☑ Decision Tree | Toggle the decision tree regressor (dashed red-orange line) |
| Kernel radio | Switch SVR kernel between `rbf`, `linear`, `poly` |
| C slider | Adjust SVR regularization parameter (0.1 – 100) |
| ε slider | Adjust SVR epsilon-tube width (0.01 – 1.0) |
| Depth slider | Set Decision Tree max depth (1 – 20; 20 ≈ unconstrained) |

---

## How It Works

1. **Synthetic data** is generated from `sin(x) + 0.5·cos(1.5x) + noise`
2. Each activated model is fit independently on the training data
3. Individual prediction curves are drawn as **dashed colored lines**
4. When ≥ 2 models are active, `VotingRegressor` averages their predictions into a **solid blue ensemble curve** with a glow effect
5. The metrics panel updates live with R² and RMSE for every model

---

## Requirements

- Python 3.8+
- numpy
- matplotlib ≥ 3.5
- scikit-learn ≥ 1.0

---

## License

This project is released under the [MIT License](LICENSE).
