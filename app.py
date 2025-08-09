# app.py
import sys, json, time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Geo deps for unpickling (FeatureBuilder inside pipeline)
import geopandas as gpd  # noqa: F401
from sklearn.base import BaseEstimator, TransformerMixin  # noqa: F401

# ---------- Paths ----------
ART = Path("artifacts")
P_LGBM    = ART / "full_pipeline_lightgbm.pkl"
P_RF      = ART / "full_pipeline_rf.pkl"
P_HIST    = ART / "full_pipeline_hist.pkl"
P_DEFAULT = ART / "full_pipeline.pkl"
P_METRICS = ART / "metrics.json"


CRS = "EPSG:4326"
DIST_CRS = 32610  # meters

# ---------- Pickle shim (class signature must match training) ----------
class FeatureBuilder(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    def __init__(self, coast_shp_path: Path, cities_csv_path: Path):
        self.coast_shp_path = Path(coast_shp_path)
        self.cities_csv_path = Path(cities_csv_path)
        self._coast = None
        self._cities_points = None

    def _ensure_lon_lat(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if "lon" not in X.columns and "longitude" in X.columns:
            X["lon"] = X["longitude"]
        if "lat" not in X.columns and "latitude" in X.columns:
            X["lat"] = X["latitude"]
        drop_cols = [c for c in ["longitude", "latitude"] if c in X.columns]
        if drop_cols:
            X = X.drop(columns=drop_cols)
        return X

    def fit(self, X, y=None):
        import geopandas as gpd
        self._coast = gpd.read_file(self.coast_shp_path)[["geometry"]].to_crs(CRS)
        cities = pd.read_csv(self.cities_csv_path)
        self._cities_points = gpd.GeoDataFrame(
            cities,
            geometry=gpd.points_from_xy(cities["Longitude"], cities["Latitude"]),
            crs=CRS,
        )
        return self

    def transform(self, X):
        import geopandas as gpd
        X = self._ensure_lon_lat(pd.DataFrame(X)).copy()
        gdf = gpd.GeoDataFrame(X, geometry=gpd.points_from_xy(X["lon"], X["lat"]), crs=CRS)

        # distance to ocean
        cj = gpd.sjoin_nearest(
            gdf.to_crs(DIST_CRS), self._coast.to_crs(DIST_CRS),
            how="left", distance_col="distance_to_ocean"
        )
        X["distance_to_ocean"] = (
            cj.groupby(cj.index)["distance_to_ocean"].min()
              .reindex(gdf.index).astype(float).values
        )

        # distance to nearest city
        sj = gpd.sjoin_nearest(
            gdf.to_crs(DIST_CRS), self._cities_points.to_crs(DIST_CRS),
            how="left", distance_col="distance_nearest_city"
        )
        X["distance_nearest_city"] = (
            sj.groupby(sj.index)["distance_nearest_city"].min()
              .reindex(gdf.index).astype(float).values
        )

        # ratio features
        hh = X["households"].replace(0, np.nan)
        tr = X["total_rooms"].replace(0, np.nan)
        X["rooms_per_household"]      = X["total_rooms"] / hh
        X["population_per_household"] = X["population"] / hh
        X["bedrooms_per_room"]        = X["total_bedrooms"] / tr

        return X.loc[:, ~X.columns.duplicated()].copy()

# ---------- Module aliases for unpickling ----------
sys.modules["main"] = sys.modules[__name__]
sys.modules["__main__"] = sys.modules[__name__]
sys.modules["train_full_pipeline"] = sys.modules[__name__]

# ---------- UI ----------
st.set_page_config(page_title="CA Housing — Predictions", page_icon="🏠", layout="wide")
st.markdown("""
<style>
  .stApp { background-color: #f7f9fc; }
  .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
  .metric-card { border-radius: 14px; padding: 14px 16px; background: #fff; box-shadow: 0 1px 12px rgba(0,0,0,.06); }
</style>
""", unsafe_allow_html=True)

st.markdown("<h2 style='text-align:center;'>California Housing — Model Predictions & Ensemble</h2>", unsafe_allow_html=True)
st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)

# ---------- Load metrics (optional) ----------
metrics = {}
if P_METRICS.exists():
    try:
        metrics = json.loads(P_METRICS.read_text())
    except Exception:
        metrics = {}

# ---------- Safe loader with retry ----------
def safe_load(p: Path):
    try:
        return joblib.load(p)
    except ModuleNotFoundError as e:
        missing = str(e).split("'")[1] if "'" in str(e) else None
        if missing:
            sys.modules[missing] = sys.modules[__name__]
            return joblib.load(p)
        raise
    except AttributeError:
        return joblib.load(p)

@st.cache_resource(show_spinner=True)
def load_pipelines():
    models = {}
    for name, path in [("LightGBM", P_LGBM), ("RandomForest", P_RF), ("HistGBM", P_HIST)]:
        if path.exists():
            models[name] = safe_load(path)
    if not models and P_DEFAULT.exists():
        models["LightGBM"] = safe_load(P_DEFAULT)
    return models

pipes = load_pipelines()
if not pipes:
    st.error("No model files found in `artifacts/`. Train first, then refresh.")
    st.stop()

# ---------- Helpers ----------
REQ_BASE = [
    "housing_median_age","total_rooms","total_bedrooms",
    "population","households","median_income","ocean_proximity"
]

def normalize_geo_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # accept latitude/longitude or lat/lon; standardize to lat/lon for UI/map
    if "lat" not in df.columns and "latitude" in df.columns:
        df.rename(columns={"latitude":"lat"}, inplace=True)
    if "lon" not in df.columns and "longitude" in df.columns:
        df.rename(columns={"longitude":"lon"}, inplace=True)
    return df

# ---------- Inputs ----------
left, right = st.columns([1,1], gap="large")

with left:
    st.subheader("Input source")
    src = st.radio("Select source", ["Manual", "From dataset row"], horizontal=True, label_visibility="collapsed")

    raw = None
    dataset_df = None

    if src == "From dataset row":
        file = st.file_uploader("Upload CSV with columns (lat/lon OR latitude/longitude) + required features", type=["csv"])
        if file:
            dataset_df = pd.read_csv(file)
            dataset_df = normalize_geo_columns(dataset_df)

            missing = [c for c in ["lat","lon"] if c not in dataset_df.columns]
            if missing:
                st.warning(f"Dataset missing columns: {missing}. Switching to Manual.")
                src = "Manual"
            else:
                # ensure feature columns exist (ok if extras present)
                for c in REQ_BASE:
                    if c not in dataset_df.columns:
                        st.warning(f"Dataset missing required column: `{c}`. Switching to Manual.")
                        src = "Manual"
                        break

        if src == "From dataset row" and dataset_df is not None:
            idx = st.number_input("Row index (0-based)", min_value=0, max_value=len(dataset_df)-1, value=0, step=1)
            row = dataset_df.iloc[[idx]].copy()
            st.caption("Selected row")
            st.dataframe(row, use_container_width=True)

            raw = row.loc[:, ["lon","lat"] + REQ_BASE].copy()

    if src == "Manual" or raw is None:
        st.subheader("Manual inputs")
        lat = st.number_input("Latitude", value=37.880000, format="%.6f")
        lon = st.number_input("Longitude", value=-122.230000, format="%.6f")
        housing_median_age = st.number_input("housing_median_age", value=41, step=1)
        total_rooms        = st.number_input("total_rooms", value=880, step=1)
        total_bedrooms     = st.number_input("total_bedrooms", value=129.0, step=1.0)
        population         = st.number_input("population", value=322, step=1)
        households         = st.number_input("households", value=126, step=1)
        median_income      = st.number_input("median_income (in $10k units)", value=8.3252, format="%.4f")
        ocean_proximity    = st.selectbox(
            "ocean_proximity", ["NEAR BAY", "INLAND", "NEAR OCEAN", "ISLAND", "<1H OCEAN"], index=0
        )
        raw = pd.DataFrame([{
            "lon": lon, "lat": lat,
            "housing_median_age": housing_median_age,
            "total_rooms": total_rooms, "total_bedrooms": total_bedrooms,
            "population": population, "households": households,
            "median_income": median_income, "ocean_proximity": ocean_proximity
        }])

with right:
    st.subheader("Map")
    st.map(pd.DataFrame([{"lat": float(raw.iloc[0]['lat']), "lon": float(raw.iloc[0]['lon'])}]),
           zoom=10, use_container_width=True)

st.subheader("Your input row")
st.dataframe(raw, use_container_width=True)

# ---------- Predict ----------
if st.button("Predict from all models 🚀", type="primary"):
    rows, preds = [], []
    w_preds, weights = [], []

    for name, model in pipes.items():
        t0 = time.perf_counter()
        try:
            y = float(model.predict(raw)[0])
        except Exception as e:
            y = np.nan
        dt_ms = (time.perf_counter() - t0) * 1000.0

        rows.append({"Model": name, "Prediction ($)": y, "Time (ms)": dt_ms})
        if np.isfinite(y):
            preds.append(y)
            # Weighted ensemble (1/RMSE) if metrics available
            mkey = name.lower().replace(" ", "_")
            try:
                rmse = float(metrics[mkey]["actual"]["RMSE"])
                if rmse > 0:
                    w = 1.0 / rmse
                    w_preds.append(y * w); weights.append(w)
            except Exception:
                pass

    out = pd.DataFrame(rows)
    if "Prediction ($)" in out:
        out["Prediction ($)"] = out["Prediction ($)"].map(lambda v: f"${v:,.2f}" if pd.notnull(v) else "—")
    if "Time (ms)" in out:
        out["Time (ms)"] = out["Time (ms)"].map(lambda v: f"{v:.1f}")
    st.subheader("Predictions")
    st.dataframe(out, use_container_width=True)

    if preds:
        avg = float(np.mean(preds))
        st.markdown(
            f"<div class='metric-card'>🟢 <b>Average ensemble</b>: <b>${avg:,.2f}</b></div>",
            unsafe_allow_html=True
        )

    if weights:
        w_avg = float(np.sum(w_preds) / np.sum(weights))
        st.markdown(
            f"<div class='metric-card'>🔵 <b>Weighted ensemble (1/RMSE)</b>: <b>${w_avg:,.2f}</b></div>",
            unsafe_allow_html=True
        )
