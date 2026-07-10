
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import r2_score, mean_squared_error


st.set_page_config(
    page_title="Voting Regressor Visualizer Lab",
    page_icon="VR",
    layout="wide",
)


st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
        border-right: 1px solid #30363d;
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #bc8cff !important;
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Cards / metric boxes */
    div[data-testid="stMetric"] {
        background: rgba(22, 27, 34, 0.8);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 16px;
        backdrop-filter: blur(10px);
    }
    div[data-testid="stMetric"] label {
        color: #7d8590 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.8rem !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #e6edf3 !important;
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Title styling */
    .main-title {
        text-align: center;
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #bc8cff, #58a6ff, #3fb950);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
        padding-top: 0.5rem;
    }
    .sub-title {
        text-align: center;
        font-family: 'Inter', sans-serif;
        color: #7d8590;
        font-size: 1rem;
        margin-top: 0;
        margin-bottom: 2rem;
    }

    /* Separator styling */
    .sidebar-section {
        font-family: 'JetBrains Mono', monospace;
        color: #58a6ff;
        font-size: 0.85rem;
        font-weight: 600;
        border-bottom: 1px solid #21262d;
        padding-bottom: 4px;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }

    /* Score cards */
    .score-card {
        background: rgba(22, 27, 34, 0.9);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.6rem;
        backdrop-filter: blur(10px);
        font-family: 'JetBrains Mono', monospace;
    }
    .score-card .model-name {
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
    .score-card .score-row {
        display: flex;
        justify-content: space-between;
        font-size: 0.82rem;
        color: #7d8590;
    }
    .score-card .score-val {
        font-weight: 600;
        color: #e6edf3;
    }
    .ensemble-card {
        border-color: #58a6ff !important;
        box-shadow: 0 0 20px rgba(88, 166, 255, 0.15);
    }
</style>
""", unsafe_allow_html=True)



st.markdown('<div class="main-title">Voting Regressor Visualizer Lab</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Observe how ensemble averaging tames individual model quirks</div>', unsafe_allow_html=True)



np.random.seed(42)
X = np.sort(5 * np.random.rand(100, 1), axis=0)
y = np.sin(X).ravel() + 0.5 * np.cos(1.5 * X).ravel() + np.random.normal(0, 0.2, X.shape[0])
X_test = np.linspace(X.min() - 0.3, X.max() + 0.3, 500)[:, np.newaxis]



with st.sidebar:
    st.markdown("## Model Config")

    st.markdown('<div class="sidebar-section">LINEAR REGRESSION</div>', unsafe_allow_html=True)
    use_lr = st.checkbox("Enable Linear Regression", value=True)

    st.markdown('<div class="sidebar-section">SUPPORT VECTOR REGRESSOR</div>', unsafe_allow_html=True)
    use_svr = st.checkbox("Enable SVR", value=False)
    svr_kernel = st.selectbox("Kernel", ["rbf", "linear", "poly"], index=0)
    svr_c = st.slider("C (Regularization)", 0.1, 100.0, 10.0, step=0.1)
    svr_epsilon = st.slider("ε (Epsilon tube)", 0.01, 1.0, 0.1, step=0.01)

    st.markdown('<div class="sidebar-section">DECISION TREE REGRESSOR</div>', unsafe_allow_html=True)
    use_dt = st.checkbox("Enable Decision Tree", value=False)
    dt_max_depth = st.slider("Max Depth  (20 = unconstrained)", 1, 20, 5, step=1)

    st.markdown("---")
    st.caption("Toggle models & tune parameters to see how the ensemble adapts.")



estimators = []
fitted_models = {}
model_colors = {}

COLORS = {
    "lr":       "#ffa657",
    "svr":      "#3fb950",
    "dt":       "#f78166",
    "ensemble": "#58a6ff",
    "data":     "#ff7b72",
}

if use_lr:
    lr = LinearRegression().fit(X, y)
    estimators.append(("lr", LinearRegression()))
    fitted_models["Linear Regression"] = lr
    model_colors["Linear Regression"] = COLORS["lr"]

if use_svr:
    svr = SVR(kernel=svr_kernel, C=svr_c, epsilon=svr_epsilon).fit(X, y)
    estimators.append(("svr", SVR(kernel=svr_kernel, C=svr_c, epsilon=svr_epsilon)))
    fitted_models[f"SVR ({svr_kernel})"] = svr
    model_colors[f"SVR ({svr_kernel})"] = COLORS["svr"]

if use_dt:
    max_d = None if dt_max_depth >= 20 else dt_max_depth
    dt = DecisionTreeRegressor(max_depth=max_d, random_state=42).fit(X, y)
    estimators.append(("dt", DecisionTreeRegressor(max_depth=max_d, random_state=42)))
    depth_str = "None" if max_d is None else str(dt_max_depth)
    fitted_models[f"Decision Tree (d={depth_str})"] = dt
    model_colors[f"Decision Tree (d={depth_str})"] = COLORS["dt"]



if len(estimators) == 0:
    st.warning("Please enable at least one base model from the sidebar.")
    st.stop()


fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor("#0d1117")
ax.set_facecolor("#0d1117")

ax.grid(True, color="#21262d", alpha=0.5, linestyle="--", linewidth=0.5)
for spine in ax.spines.values():
    spine.set_color("#30363d")
ax.tick_params(colors="#7d8590")

ax.scatter(X, y, c=COLORS["data"], s=35, alpha=0.7,
           edgecolors="#ffa198", linewidths=0.5, zorder=5, label="Training data")

scores = {}
for name, model in fitted_models.items():
    y_pred_train = model.predict(X)
    y_pred_test = model.predict(X_test)
    r2 = r2_score(y, y_pred_train)
    rmse = np.sqrt(mean_squared_error(y, y_pred_train))
    scores[name] = {"r2": r2, "rmse": rmse, "color": model_colors[name]}
    ax.plot(X_test, y_pred_test, color=model_colors[name], lw=2,
            ls="--", alpha=0.85, zorder=10, label=f"{name}  R2={r2:.3f}")

if len(estimators) >= 2:
    vr = VotingRegressor(estimators=estimators).fit(X, y)
    y_vr_train = vr.predict(X)
    y_vr_test = vr.predict(X_test)
    r2 = r2_score(y, y_vr_train)
    rmse = np.sqrt(mean_squared_error(y, y_vr_train))
    scores["VotingRegressor (Ensemble)"] = {"r2": r2, "rmse": rmse, "color": COLORS["ensemble"]}

    ax.plot(X_test, y_vr_test, color="#1f6feb", lw=8, alpha=0.15, zorder=14)
    ax.plot(X_test, y_vr_test, color=COLORS["ensemble"], lw=3,
            alpha=0.95, zorder=15, label=f"Ensemble  R2={r2:.3f}")

ax.set_xlabel("Feature (X)", fontsize=12, color="#e6edf3", labelpad=8)
ax.set_ylabel("Target (y)", fontsize=12, color="#e6edf3", labelpad=8)
ax.set_title("Model Predictions on Non-Linear Data",
             fontsize=14, color="#e6edf3", pad=12, fontfamily="monospace")
ax.legend(loc="best", fontsize=9, facecolor="#161b22",
          edgecolor="#30363d", labelcolor="#e6edf3", framealpha=0.92)

st.pyplot(fig, use_container_width=True)
plt.close(fig)



st.markdown("### Performance Metrics")

cols = st.columns(len(scores))
for col, (name, vals) in zip(cols, scores.items()):
    is_ensemble = name.startswith("VotingRegressor")
    card_class = "score-card ensemble-card" if is_ensemble else "score-card"
    col.markdown(f"""
    <div class="{card_class}">
        <div class="model-name" style="color: {vals['color']};">{name}</div>
        <div class="score-row">
            <span>R2</span>
            <span class="score-val">{vals['r2']:.4f}</span>
        </div>
        <div class="score-row">
            <span>RMSE</span>
            <span class="score-val">{vals['rmse']:.4f}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)



with st.expander("How does VotingRegressor work?"):
    st.markdown("""
    **VotingRegressor** averages the predictions of its base estimators:

    ```
    ŷ_ensemble = (ŷ_model1 + ŷ_model2 + ... + ŷ_modelN) / N
    ```

    This simple averaging reduces the variance of high-variance models
    (like Decision Trees) while partially compensating for the bias of
    rigid models (like Linear Regression), resulting in a more robust
    prediction curve.

    | Model | Bias | Variance | Behavior |
    |---|---|---|---|
    | **Linear Regression** | High | Low | Straight line — underfits non-linear data |
    | **SVR (RBF)** | Low–Med | Medium | Smooth curve — tunable via C and ε |
    | **Decision Tree** | Low | High | Jagged steps — overfits at high depth |
    | **Ensemble** | **Balanced** | **Reduced** | **Smooth, superior fit** |
    """)