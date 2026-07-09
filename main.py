import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import r2_score

st.title("Voting Ensemble Visualizer (Regression)")

np.random.seed(42)
X = np.sort(5 * np.random.rand(100, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

st.sidebar.header("Select Base Models")
use_lr = st.sidebar.checkbox("Linear Regression", value=True)
use_svr = st.sidebar.checkbox("Support Vector Regressor (SVR)", value=False)
use_dt = st.sidebar.checkbox("Decision Tree Regressor", value=False)

estimators = []
if use_lr:
    estimators.append(("lr", LinearRegression()))
if use_svr:
    estimators.append(("svr", SVR(kernel="rbf", C=10, gamma=0.1)))
if use_dt:
    estimators.append(("dt", DecisionTreeRegressor(max_depth=5)))

if len(estimators) == 0:
    st.error("Please select at least one base model from the sidebar.")
else:
    X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]  # dense grid for smooth curves

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X, y, color="black", alpha=0.5, label="Data points")

    scores = {}

    for name, model in estimators:
        model.fit(X, y)
        y_pred = model.predict(X)
        scores[name] = r2_score(y, y_pred)

        y_test_pred = model.predict(X_test)
        ax.plot(X_test, y_test_pred, label=f"{model.__class__.__name__}", linestyle="--")

    if len(estimators) > 1:  # ensembling only makes sense with 2+ base models
        vr = VotingRegressor(estimators=estimators)
        vr.fit(X, y)
        y_vr_pred = vr.predict(X)
        scores["Voting Regressor"] = r2_score(y, y_vr_pred)

        y_test_vr_pred = vr.predict(X_test)
        ax.plot(X_test, y_test_vr_pred, color="blue", linewidth=2.5, label="Voting Regressor")

    ax.set_xlabel("X")
    ax.set_ylabel("y")
    ax.legend()
    st.pyplot(fig)

    st.subheader("R² Scores")
    for model_name, score in scores.items():
        st.write(f"**{model_name}**: {score:.4f}")