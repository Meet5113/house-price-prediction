import io
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import joblib

# ---------------------------- Page Config ----------------------------
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# ---------------------------- Utilities ----------------------------
FEATURE_ORDER = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude"
]
TARGET_COL = "MedHouseVal"

def _synthetic_housing(n=20000, seed=42):
    """Generate synthetic housing-like data as a fallback (no internet)."""
    rng = np.random.default_rng(seed)
    med_inc   = rng.normal(4.0, 1.5, n).clip(0.5, 15)         
    house_age = rng.integers(1, 52, n)
    ave_rooms = rng.normal(5.5, 1.2, n).clip(1, 12)
    ave_beds  = (ave_rooms * rng.uniform(0.18, 0.25, n)).clip(0.5, 5)
    pop       = rng.integers(200, 4000, n)
    ave_occ   = rng.normal(3.0, 1.0, n).clip(1, 7)
    lat       = rng.uniform(32, 42, n)
    lon       = rng.uniform(-124.5, -114, n)

    # A simple non-linear formula to derive target with noise
    price = (
        med_inc * 0.8
        + (ave_rooms - ave_beds) * 0.2
        + (42 - house_age) * 0.01
        + (lat - 37) * 0.05
        - (abs(lon + 119)) * 0.03
        - (pop / 10000) * 0.5
        + rng.normal(0, 0.5, n)
    )
    price = (price - price.min()) / (price.max() - price.min())
    price = price * 5.0  # scale to ~[0..5] like California target

    df = pd.DataFrame({
        "MedInc": med_inc,
        "HouseAge": house_age,
        "AveRooms": ave_rooms,
        "AveBedrms": ave_beds,
        "Population": pop,
        "AveOccup": ave_occ,
        "Latitude": lat,
        "Longitude": lon,
        "MedHouseVal": price,
    })
    return df

@st.cache_data(show_spinner=False)
def load_data():
    """Try to load California Housing; fall back to synthetic."""
    try:
        cali = fetch_california_housing(as_frame=True)
        df = cali.frame.copy()
        # Ensure expected column order
        df = df[FEATURE_ORDER + [TARGET_COL]]
        source = "California Housing (sklearn)"
    except Exception:
        df = _synthetic_housing()
        source = "Synthetic (offline fallback)"
    return df, source

def train_model(X, y, model_name="Linear Regression", test_size=0.2, random_state=42, standardize=True, cv_folds=0):
    """Train the selected model, evaluate on hold-out test, optionally do CV."""
    # Choose model
    if model_name == "Linear Regression":
        base_model = LinearRegression()
    elif model_name == "Random Forest":
        base_model = RandomForestRegressor(
            n_estimators=300, max_depth=None, random_state=random_state, n_jobs=-1
        )
    elif model_name == "Gradient Boosting":
        base_model = GradientBoostingRegressor(random_state=random_state)
    else:
        base_model = LinearRegression()

    steps = []
    if standardize:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", base_model))
    pipe = Pipeline(steps)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=float(test_size), random_state=random_state
    )
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    metrics = {
        "R2": r2_score(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": mean_squared_error(y_test, y_pred, squared=False),
    }

    cv_summary = None
    if cv_folds and cv_folds >= 2:
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        cv_scores = cross_val_score(pipe, X, y, scoring="r2", cv=kf, n_jobs=-1)
        cv_summary = {
            "CV Mean R2": float(np.mean(cv_scores)),
            "CV Std R2": float(np.std(cv_scores)),
            "Folds": int(cv_folds),
        }

    return pipe, metrics, (y_test, y_pred), cv_summary

def fig_actual_vs_pred(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.scatter(y_true, y_pred, alpha=0.4)
    lims = [
        min(y_true.min(), y_pred.min()),
        max(y_true.max(), y_pred.max())
    ]
    ax.plot(lims, lims, "--")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted")
    return fig

def fig_corr_heatmap(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(df.corr(numeric_only=True), annot=False, cmap="viridis", ax=ax)
    ax.set_title("Correlation Heatmap")
    return fig

def bytes_download(obj, filename="model.joblib"):
    buffer = io.BytesIO()
    joblib.dump(obj, buffer)
    buffer.seek(0)
    return buffer

# ---------------------------- Sidebar ----------------------------
st.sidebar.header("⚙️ Controls")
with st.sidebar:
    st.markdown("**Dataset Options**")
    df, data_source = load_data()
    st.write(f"📦 Data Source: **{data_source}**")
    st.caption("Target: `MedHouseVal` (median house value in $100,000s)")

    st.markdown("---")
    st.markdown("**Training Config**")
    model_choice = st.selectbox("Model", ["Linear Regression", "Random Forest", "Gradient Boosting"])
    test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
    rand_state = st.number_input("Random State", value=42, step=1)
    standardize = st.checkbox("Standardize Features", value=(model_choice == "Linear Regression"))
    cv_folds = st.selectbox("Cross-Validation Folds (optional)", [0, 3, 5, 10], index=0)

    st.markdown("---")
    st.markdown("**Prediction Settings**")
    default_vals = df[FEATURE_ORDER].median().to_dict()

# ---------------------------- Tabs ----------------------------
st.title("🏠 House Price Prediction")
st.caption("End-to-end ML demo: EDA → Train → Evaluate → Predict")

tab_overview, tab_eda, tab_train, tab_predict, tab_batch, tab_about = st.tabs(
    ["Overview", "EDA", "Train & Evaluate", "Predict (Single)", "Batch Predict (CSV)", "About"]
)

# ---------------------------- Overview ----------------------------
with tab_overview:
    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("Problem")
        st.write(
            "Predict median house values using features like income, house age, rooms/bedrooms, "
            "population, occupancy, and geolocation."
        )
        st.subheader("Features")
        st.write(pd.DataFrame({"Feature": FEATURE_ORDER}))
        st.subheader("Target")
        st.code("MedHouseVal  # median house value (in $100,000s)")
    with right:
        st.subheader("Sample Rows")
        st.dataframe(df.head(10), use_container_width=True)

# ---------------------------- EDA ----------------------------
with tab_eda:
    st.subheader("Exploratory Data Analysis")
    show_describe = st.checkbox("Show .describe()", value=True)
    if show_describe:
        st.dataframe(df.describe().T, use_container_width=True)

    st.markdown("### Correlation Heatmap")
    st.pyplot(fig_corr_heatmap(df))

    st.markdown("### Distributions")
    cols = st.multiselect("Choose features to visualize", FEATURE_ORDER, default=["MedInc", "HouseAge", "AveRooms"])
    if cols:
        ncols = min(3, len(cols))
        rows = (len(cols) + ncols - 1) // ncols
        for r in range(rows):
            ccols = cols[r*ncols:(r+1)*ncols]
            cols_stream = st.columns(len(ccols))
            for ax_col, feat in zip(cols_stream, ccols):
                with ax_col:
                    fig, ax = plt.subplots(figsize=(4, 3))
                    sns.histplot(df[feat], bins=30, ax=ax, kde=True)
                    ax.set_title(f"{feat} distribution")
                    st.pyplot(fig)

# ---------------------------- Train & Evaluate ----------------------------
with tab_train:
    st.subheader("Model Training & Evaluation")
    X = df[FEATURE_ORDER].copy()
    y = df[TARGET_COL].copy()

    train_btn = st.button("🚀 Train Model", type="primary")
    if train_btn:
        model, metrics, (y_test, y_pred), cv_summary = train_model(
            X, y,
            model_name=model_choice,
            test_size=test_size,
            random_state=int(rand_state),
            standardize=standardize,
            cv_folds=int(cv_folds)
        )
        st.session_state["model"] = model
        st.success("Training complete!")

        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric("R²", f"{metrics['R2']:.4f}")
        mcol2.metric("MAE", f"{metrics['MAE']:.4f}")
        mcol3.metric("RMSE", f"{metrics['RMSE']:.4f}")

        if cv_summary:
            st.info(f"CV Mean R²: {cv_summary['CV Mean R2']:.4f} ± {cv_summary['CV Std R2']:.4f}  "
                    f"(folds={cv_summary['Folds']})")

        st.pyplot(fig_actual_vs_pred(y_test, y_pred))

        # Download trained model
        st.download_button(
            "⬇️ Download Trained Model (.joblib)",
            data=bytes_download(model, "house_price_model.joblib"),
            file_name="house_price_model.joblib",
            mime="application/octet-stream"
        )
    else:
        st.caption("Click **Train Model** to fit the selected algorithm and view metrics & plots.")

# ---------------------------- Predict (Single) ----------------------------
with tab_predict:
    st.subheader("Single Prediction")
    if "model" not in st.session_state:
        st.warning("Please train a model first (see 'Train & Evaluate' tab).")
    else:
        ui_cols = st.columns(4)
        inputs = {}
        ranges = {
            "MedInc": (0.5, 15.0, default_vals["MedInc"]),
            "HouseAge": (1.0, 52.0, default_vals["HouseAge"]),
            "AveRooms": (1.0, 12.0, default_vals["AveRooms"]),
            "AveBedrms": (0.5, 5.0, default_vals["AveBedrms"]),
            "Population": (100.0, 10000.0, default_vals["Population"]),
            "AveOccup": (1.0, 8.0, default_vals["AveOccup"]),
            "Latitude": (32.0, 42.0, default_vals["Latitude"]),
            "Longitude": (-124.5, -114.0, default_vals["Longitude"]),
        }
        for i, feat in enumerate(FEATURE_ORDER):
            with ui_cols[i % 4]:
                lo, hi, mid = ranges[feat]
                if feat in ["Population"]:
                    val = st.number_input(feat, value=float(mid), min_value=lo, max_value=hi, step=50.0, format="%.1f")
                else:
                    val = st.number_input(feat, value=float(mid), min_value=lo, max_value=hi, step=0.1, format="%.3f")
                inputs[feat] = val

        if st.button("🔮 Predict Price"):
            model = st.session_state["model"]
            df_in = pd.DataFrame([inputs])[FEATURE_ORDER]
            pred = model.predict(df_in)[0]
            st.success(f"Estimated Median House Value: **${pred*100_000:,.0f}** "
                       f"(target unit is $100,000s → predicted={pred:.3f})")

# ---------------------------- Batch Predict (CSV) ----------------------------
with tab_batch:
    st.subheader("Batch Predictions (Upload CSV)")
    st.caption("CSV must contain columns: " + ", ".join(FEATURE_ORDER))
    if "model" not in st.session_state:
        st.warning("Please train a model first (see 'Train & Evaluate' tab).")
    else:
        up = st.file_uploader("Upload CSV", type=["csv"])
        if up:
            try:
                data_u = pd.read_csv(up)
                missing = [c for c in FEATURE_ORDER if c not in data_u.columns]
                if missing:
                    st.error(f"Missing required columns: {missing}")
                else:
                    preds = st.session_state["model"].predict(data_u[FEATURE_ORDER])
                    out = data_u.copy()
                    out["Predicted_MedHouseVal"] = preds
                    out["Predicted_Value_USD"] = (preds * 100000).round(2)
                    st.success("Predictions generated!")
                    st.dataframe(out.head(20), use_container_width=True)

                    # Download results
                    csv_bytes = out.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️ Download Predictions CSV",
                        data=csv_bytes,
                        file_name="house_price_predictions.csv",
                        mime="text/csv"
                    )
            except Exception as e:
                st.error(f"Failed to read/predict on CSV: {e}")

# ---------------------------- About ----------------------------
with tab_about:
    st.subheader("About this App")
    st.markdown(
        """
        **House Price Prediction** demo app for CSE/DA/ML projects.

        **Flow**
        1) Load dataset (California Housing or synthetic fallback)  
        2) EDA (describe, correlations, distributions)  
        3) Train & Evaluate (Linear/RandomForest/GradientBoosting)  
        4) Predict a single instance or batch via CSV  

        **Notes**
        - Target `MedHouseVal` is in units of **$100,000**.
        - The app standardizes features for linear models by default.
        - Use the download button to save your trained pipeline (.joblib).

        **Author:** Meet Vaghamshi
        """
    )
