"""
⛽ MPG Oracle — Automobile Fuel Efficiency Predictor
====================================================
End-to-end ML app (data extraction -> cleaning -> imputation -> scaling ->
encoding -> model selection -> prediction) served with Streamlit.

Run locally:   streamlit run app.py
Data file:     auto-mpg.csv  (same folder as this file)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ----------------------------------------------------------------------------
# 0. PAGE CONFIG + STYLING
# ----------------------------------------------------------------------------
st.set_page_config(page_title="MPG Oracle", page_icon="⛽", layout="wide")

st.markdown(
    """
    <style>
    .hero {
        background: linear-gradient(120deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        padding: 1.8rem 2rem; border-radius: 18px; color: white; margin-bottom: 1rem;
    }
    .hero h1 { margin: 0; font-size: 2.2rem; }
    .hero p  { margin: .3rem 0 0 0; opacity: .85; }
    .result-card {
        border-radius: 18px; padding: 1.6rem; text-align: center; color: white;
        box-shadow: 0 8px 24px rgba(0,0,0,.18);
    }
    .result-card .big   { font-size: 3.4rem; font-weight: 800; line-height: 1; }
    .result-card .unit  { font-size: 1rem; opacity: .9; }
    .result-card .badge {
        display:inline-block; margin-top:.7rem; padding:.25rem .9rem;
        background: rgba(255,255,255,.22); border-radius: 999px; font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

DATA_PATH = Path(__file__).parent / "auto-mpg.csv"
RANDOM_STATE = 42

ORIGIN_MAP = {1: "USA 🇺🇸", 2: "Europe 🇪🇺", 3: "Japan 🇯🇵"}

# Typos / abbreviations found in the raw 'car name' column
BRAND_FIXES = {
    "chevroelt": "chevrolet", "chevy": "chevrolet",
    "vw": "volkswagen", "vokswagen": "volkswagen",
    "maxda": "mazda", "toyouta": "toyota",
    "mercedes": "mercedes-benz",
}

NUM_COLS = ["cylinders", "displacement", "horsepower", "weight",
            "acceleration", "model_year", "power_to_weight"]
CAT_COLS = ["origin", "brand"]


# ----------------------------------------------------------------------------
# 1. DATA EXTRACTION + CLEANING
# ----------------------------------------------------------------------------
def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Raw CSV -> tidy dataframe. Missing values stay as NaN (imputed later,
    inside the pipeline, so there is no train/test leakage)."""
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # 'horsepower' contains '?' for unknown values -> force numeric, '?' -> NaN
    for col in ["mpg", "cylinders", "displacement", "horsepower", "weight",
                "acceleration", "model_year", "origin"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Brand = first word of the car name, with typos repaired
    brand = df["car_name"].astype(str).str.strip().str.lower().str.split().str[0]
    df["brand"] = brand.replace(BRAND_FIXES)

    # Group rare brands (<5 cars) into "other" so one-hot stays compact
    counts = df["brand"].value_counts()
    rare = counts[counts < 5].index
    df["brand"] = df["brand"].where(~df["brand"].isin(rare), "other")

    df["origin"] = df["origin"].astype("Int64").astype(str)  # categorical label
    df["model_year_full"] = 1900 + df["model_year"]

    df = df.drop_duplicates()
    df = df.dropna(subset=["mpg"])  # can't learn from rows without a target
    return df.reset_index(drop=True)


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering shared by training AND prediction."""
    X = X.copy()
    X["power_to_weight"] = X["horsepower"] / X["weight"] * 1000  # hp per 1000 lb
    return X


@st.cache_data(show_spinner=False)
def load_data(path_or_bytes) -> pd.DataFrame:
    return clean_raw(pd.read_csv(path_or_bytes))


# ----------------------------------------------------------------------------
# 2. PREPROCESSING + MODEL PIPELINE
# ----------------------------------------------------------------------------
def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),   # missing value treatment
        ("scale", StandardScaler()),                    # scaling
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),  # encoding
    ])
    return ColumnTransformer([
        ("num", numeric, NUM_COLS),
        ("cat", categorical, CAT_COLS),
    ])


def candidate_models() -> dict:
    return {
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=3,
            subsample=0.9, random_state=RANDOM_STATE),
    }


@st.cache_resource(show_spinner="🔧 Training models… (one-time, takes a few seconds)")
def train(df: pd.DataFrame):
    X = add_features(df[["cylinders", "displacement", "horsepower", "weight",
                         "acceleration", "model_year", "origin", "brand"]])
    y = df["mpg"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE)

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows, fitted = [], {}
    for name, model in candidate_models().items():
        pipe = Pipeline([("prep", build_preprocessor()), ("model", model)])
        cv_r2 = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="r2")
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        rows.append({
            "Model": name,
            "CV R² (mean)": cv_r2.mean(),
            "Test R²": r2_score(y_te, pred),
            "Test MAE": mean_absolute_error(y_te, pred),
            "Test RMSE": np.sqrt(mean_squared_error(y_te, pred)),
        })
        fitted[name] = pipe

    report = pd.DataFrame(rows).sort_values("CV R² (mean)", ascending=False)
    best_name = report.iloc[0]["Model"]

    # Refit the winner on ALL data for the best possible final model
    final = Pipeline([("prep", build_preprocessor()),
                      ("model", candidate_models()[best_name])]).fit(X, y)

    # Which inputs matter most? (permutation importance on the held-out set)
    imp = permutation_importance(fitted[best_name], X_te, y_te, n_repeats=10,
                                 random_state=RANDOM_STATE, scoring="r2")
    importance = (pd.DataFrame({"Feature": X_te.columns,
                                "Importance": imp.importances_mean})
                  .sort_values("Importance", ascending=False))

    holdout = pd.DataFrame({"Actual MPG": y_te.values,
                            "Predicted MPG": fitted[best_name].predict(X_te)})
    rmse = float(report.iloc[0]["Test RMSE"])
    return final, best_name, report, importance, holdout, rmse


# ----------------------------------------------------------------------------
# 3. HELPERS
# ----------------------------------------------------------------------------
def rating(mpg: float):
    if mpg < 18:
        return "🔥 Thirsty", "linear-gradient(135deg,#cb2d3e,#ef473a)"
    if mpg < 25:
        return "🚗 Average", "linear-gradient(135deg,#f7971e,#ffd200)"
    if mpg < 32:
        return "🌿 Efficient", "linear-gradient(135deg,#56ab2f,#a8e063)"
    return "⚡ Champion", "linear-gradient(135deg,#00b4db,#0083b0)"


MPG_TO_KML = 0.425144       # 1 US mpg = 0.425144 km/l
MPG_TO_L100 = 235.215       # L/100km = 235.215 / mpg


# ----------------------------------------------------------------------------
# 4. LOAD DATA
# ----------------------------------------------------------------------------
st.markdown(
    '<div class="hero"><h1>⛽ MPG Oracle</h1>'
    "<p>Tell me about a car — I'll predict how far it can go on one gallon.</p></div>",
    unsafe_allow_html=True,
)

if DATA_PATH.exists():
    data = load_data(DATA_PATH)
else:
    st.warning("`auto-mpg.csv` not found next to app.py — upload it below.")
    up = st.file_uploader("Upload auto-mpg.csv", type="csv")
    if up is None:
        st.stop()
    data = load_data(up)

model, best_name, report, importance, holdout, rmse = train(data)

# ----------------------------------------------------------------------------
# 5. SIDEBAR — CAR SPECS
# ----------------------------------------------------------------------------
d = data
with st.sidebar:
    st.header("🛠️ Build your car")
    cylinders = st.select_slider("Cylinders", options=[3, 4, 5, 6, 8], value=4)
    displacement = st.slider("Engine displacement (cu. inches)",
                             int(d.displacement.min()), int(d.displacement.max()),
                             int(d.displacement.median()))
    unknown_hp = st.checkbox("I don't know the horsepower",
                             help="The pipeline's imputer will fill it in.")
    if unknown_hp:
        horsepower = np.nan
        st.caption("Horsepower will be imputed (median of training data).")
    else:
        horsepower = st.slider("Horsepower", int(d.horsepower.min()),
                               int(d.horsepower.max()), int(d.horsepower.median()))
    weight = st.slider("Weight (lbs)", int(d.weight.min()), int(d.weight.max()),
                       int(d.weight.median()), step=10)
    acceleration = st.slider("0–60 mph time (sec)", float(d.acceleration.min()),
                             float(d.acceleration.max()),
                             float(d.acceleration.median()), step=0.1)
    year = st.slider("Model year", 1970, 1982, 1976)
    origin_label = st.radio("Origin", list(ORIGIN_MAP.values()), horizontal=True)
    brands = sorted(d.brand.unique())
    brand = st.selectbox("Brand", brands, index=brands.index("toyota")
                         if "toyota" in brands else 0)

    st.divider()
    st.subheader("💰 Running cost")
    km_year = st.number_input("Distance per year (km)", 1000, 100000, 12000, step=1000)
    fuel_price = st.number_input("Fuel price per litre", 1.0, 500.0, 100.0, step=1.0)

origin_code = str({v: k for k, v in ORIGIN_MAP.items()}[origin_label])

# ----------------------------------------------------------------------------
# 6. TABS
# ----------------------------------------------------------------------------
tab_predict, tab_explore, tab_model = st.tabs(
    ["🔮 Predict", "📊 Explore data", "🧠 Model report"])

# ---- PREDICT ----------------------------------------------------------------
with tab_predict:
    row = pd.DataFrame([{
        "cylinders": cylinders, "displacement": displacement,
        "horsepower": horsepower, "weight": weight, "acceleration": acceleration,
        "model_year": year - 1900, "origin": origin_code, "brand": brand,
    }])
    pred = float(model.predict(add_features(row))[0])
    label, gradient = rating(pred)

    kml = pred * MPG_TO_KML
    l100 = MPG_TO_L100 / pred
    yearly_cost = km_year / kml * fuel_price
    avg_mpg = d.mpg.mean()

    left, right = st.columns([1, 1.4])
    with left:
        st.markdown(
            f'<div class="result-card" style="background:{gradient}">'
            f'<div class="unit">Predicted fuel efficiency</div>'
            f'<div class="big">{pred:.1f}</div><div class="unit">miles per gallon</div>'
            f'<div class="badge">{label}</div></div>',
            unsafe_allow_html=True)
        st.caption(f"Typical error of this model ≈ ±{rmse:.1f} MPG "
                   f"(range {max(pred - rmse, 0):.1f} – {pred + rmse:.1f}).")

    with right:
        c1, c2, c3 = st.columns(3)
        c1.metric("km per litre", f"{kml:.1f}")
        c2.metric("L / 100 km", f"{l100:.1f}")
        c3.metric("vs. dataset average", f"{pred - avg_mpg:+.1f} MPG")
        st.metric(f"Estimated yearly fuel cost ({km_year:,} km)", f"{yearly_cost:,.0f}")
        pct = float((d.mpg < pred).mean())
        st.progress(min(pct, 1.0),
                    text=f"More efficient than {pct * 100:.0f}% of cars in the dataset")

    st.subheader("🧪 What-if: how does weight change the answer?")
    weights = np.linspace(d.weight.min(), d.weight.max(), 25)
    sweep = pd.concat([row] * len(weights), ignore_index=True)
    sweep["weight"] = weights
    curve = pd.DataFrame({"Weight (lbs)": weights,
                          "Predicted MPG": model.predict(add_features(sweep))})
    st.line_chart(curve, x="Weight (lbs)", y="Predicted MPG")

# ---- EXPLORE ----------------------------------------------------------------
with tab_explore:
    st.subheader("Dataset snapshot")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Cars", len(d))
    m2.metric("Avg MPG", f"{d.mpg.mean():.1f}")
    m3.metric("Best MPG", f"{d.mpg.max():.1f}")
    m4.metric("Missing horsepower (fixed)", int(d.horsepower.isna().sum()))

    st.dataframe(d.drop(columns=["model_year_full"]).head(50), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Weight vs MPG**")
        st.scatter_chart(d, x="weight", y="mpg", color="origin")
    with c2:
        st.markdown("**Average MPG by model year** (the oil-crisis effect ⛽)")
        st.line_chart(d.groupby("model_year_full")["mpg"].mean())

    st.markdown("**Average MPG by cylinders**")
    st.bar_chart(d.groupby("cylinders")["mpg"].mean())

# ---- MODEL REPORT -----------------------------------------------------------
with tab_model:
    st.subheader(f"🏆 Winning model: {best_name}")
    st.caption("Three models were compared with 5-fold cross-validation; the best "
               "one was refit on all the data and is used for predictions.")
    st.dataframe(report.style.format({c: "{:.3f}" for c in report.columns[1:]}),
                 use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Actual vs predicted (held-out cars)**")
        st.scatter_chart(holdout, x="Actual MPG", y="Predicted MPG")
    with c2:
        st.markdown("**What drives MPG? (permutation importance)**")
        st.bar_chart(importance.set_index("Feature"))

    with st.expander("🔍 Pipeline architecture"):
        st.code(
            """
raw CSV
 └─ clean_raw():   '?' → NaN, fix brand typos, group rare brands
 └─ add_features(): power_to_weight
 └─ ColumnTransformer
      ├─ numeric     → SimpleImputer(median) → StandardScaler
      └─ categorical → SimpleImputer(most_frequent) → OneHotEncoder
 └─ Regressor (Ridge / Random Forest / Gradient Boosting)
            """, language="text")

st.caption("Data: UCI Auto-MPG (1970–1982 cars, US gallons). Predictions reflect "
           "that era — treat them as a learning demo, not a modern car's rating.")
