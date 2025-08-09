# train_full_pipeline.py
import json
import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from feature_engine.selection import SmartCorrelatedSelection
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
import joblib

# -------- Paths (local) --------
ROOT       = Path(__file__).resolve().parent
ARTIFACTS  = ROOT / "artifacts"
DATA_CSV   = ARTIFACTS / "housing.csv"               # <-- your training CSV
COAST_SHP  = ARTIFACTS / "westcoast" / "US_Westcoast.shp"
CITIES_CSV = ARTIFACTS / "cal_cities_lat_long.csv"

OUT_LGBM   = ARTIFACTS / "full_pipeline_lightgbm.pkl"
OUT_RF     = ARTIFACTS / "full_pipeline_rf.pkl"
OUT_HIST   = ARTIFACTS / "full_pipeline_hist.pkl"
# keep a default path for backwards-compatibility (we’ll save LightGBM here too)
OUT_DEFAULT = ARTIFACTS / "full_pipeline.pkl"

CRS = "EPSG:4326"
DIST_CRS = 32610  # meters (UTM Zone 10N)

# ----- Custom FeatureBuilder -----
class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Adds distance_to_ocean, distance_nearest_city, and ratio features.
       Accepts lon/lat or longitude/latitude; normalizes to lon/lat."""
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
        X = X.drop(columns=[c for c in ["longitude","latitude"] if c in X.columns])
        return X

    def fit(self, X, y=None):
        self._coast = gpd.read_file(self.coast_shp_path)[["geometry"]].to_crs(CRS)
        cities = pd.read_csv(self.cities_csv_path)
        self._cities_points = gpd.GeoDataFrame(
            cities,
            geometry=gpd.points_from_xy(cities["Longitude"], cities["Latitude"]),
            crs=CRS
        )
        return self

    def transform(self, X):
        X = self._ensure_lon_lat(pd.DataFrame(X)).copy()

        gdf = gpd.GeoDataFrame(X, geometry=gpd.points_from_xy(X["lon"], X["lat"]), crs=CRS)

        # distance to ocean (handle ties)
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

        # ratios (imputer will fill NaNs from zero-div)
        hh = X["households"].replace(0, np.nan)
        tr = X["total_rooms"].replace(0, np.nan)
        X["rooms_per_household"]      = X["total_rooms"] / hh
        X["population_per_household"] = X["population"] / hh
        X["bedrooms_per_room"]        = X["total_bedrooms"] / tr

        X = X.loc[:, ~X.columns.duplicated()].copy()
        return X


def build_preprocessor():
    """ColumnTransformer that selects columns by dtype AFTER FeatureBuilder."""
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),  # median for numeric
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    pre = ColumnTransformer([
        ("num", num_pipe, make_column_selector(dtype_include=np.number)),
        ("cat", cat_pipe, make_column_selector(dtype_exclude=np.number)),
    ])
    return pre


def build_full_pipeline(base_model, use_log: bool):
    """Full pipeline: FeatureBuilder → preprocess → corr prune → model (with optional log target)."""
    pre = build_preprocessor()
    corr_selector = SmartCorrelatedSelection(
        variables=None, method="pearson", threshold=0.8, missing_values="ignore"
    )
    reg = base_model if not use_log else TransformedTargetRegressor(
        regressor=base_model, func=np.log1p, inverse_func=np.expm1
    )
    return Pipeline([
        ("features", FeatureBuilder(COAST_SHP, CITIES_CSV)),
        ("preprocess", pre),
        ("corr_prune", corr_selector),
        ("model", reg),
    ])


def train_eval_save(name: str, pipe: Pipeline, X_train, y_train, X_test, y_test, out_path: Path, use_log: bool):
    print(f"\n=== Training {name} ===")
    pipe.fit(X_train, y_train)

    # Evaluate on holdout (predictions already inverse-transformed if use_log=True)
    y_pred = pipe.predict(X_test)
    mae_actual = mean_absolute_error(y_test, y_pred)
    mse_actual = mean_squared_error(y_test, y_pred)
    rmse_actual = np.sqrt(mse_actual)

    print(f"[{name}] MAE  (actual $): {mae_actual:,.2f}")
    print(f"[{name}] MSE  (actual $): {mse_actual:,.2f}")
    print(f"[{name}] RMSE (actual $): {rmse_actual:,.2f}")

    metrics = {
        "actual": {"MAE": float(mae_actual), "MSE": float(mse_actual), "RMSE": float(rmse_actual)}
    }

    if use_log:
        # Log-space diagnostics (optional, just for your sanity)
        y_test_log = np.log1p(y_test)
        y_pred_log = np.log1p(np.maximum(y_pred, 0))
        mae_log = mean_absolute_error(y_test_log, y_pred_log)
        mse_log = mean_squared_error(y_test_log, y_pred_log)
        print(f"[{name}] MAE_log: {mae_log:.4f}")
        print(f"[{name}] MSE_log: {mse_log:.4f}")
        metrics["log"] = {"MAE": float(mae_log), "MSE": float(mse_log)}

    joblib.dump(pipe, out_path)
    print(f"[{name}] ✅ Saved → {out_path}")
    return metrics


def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    # ---- Load data ----
    df = pd.read_csv(DATA_CSV) if DATA_CSV.suffix.lower()==".csv" else pd.read_excel(DATA_CSV)
    y = df["median_house_value"]
    X = df.drop(columns=["median_house_value"])

    # ---- Split ----
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ---- Config ----
    use_log = True  # keep like your last run (preds returned in actual $ via inverse transform)

    # ---- Define base models ----
    model_lgbm = LGBMRegressor(
        n_estimators=800, learning_rate=0.05, max_depth=-1,
        subsample=0.8, colsample_bytree=0.8, random_state=42
    )
    model_rf   = RandomForestRegressor(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1)
    model_hist = HistGradientBoostingRegressor(max_iter=500, random_state=42)

    # ---- Build pipelines ----
    pipe_lgbm = build_full_pipeline(model_lgbm, use_log)
    pipe_rf   = build_full_pipeline(model_rf,   use_log)
    pipe_hist = build_full_pipeline(model_hist, use_log)

    # ---- Train, evaluate, save ----
    all_metrics = {}
    all_metrics["lightgbm"] = train_eval_save("LightGBM", pipe_lgbm, X_train, y_train, X_test, y_test, OUT_LGBM, use_log)
    all_metrics["random_forest"] = train_eval_save("RandomForest", pipe_rf, X_train, y_train, X_test, y_test, OUT_RF, use_log)
    all_metrics["hist_gbm"] = train_eval_save("HistGradientBoosting", pipe_hist, X_train, y_train, X_test, y_test, OUT_HIST, use_log)

    # also save LGBM to default filename to keep your current app working
    joblib.dump(pipe_lgbm, OUT_DEFAULT)

    # ---- Save metrics.json for the app to show ----
    (ARTIFACTS / "metrics.json").write_text(json.dumps(all_metrics, indent=2))
    print(f"\n📄 Metrics written → {ARTIFACTS / 'metrics.json'}")
    print("\nDone.")

if __name__ == "__main__":
    main()
