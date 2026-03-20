"""
Model wrappers for the CFL forecasting pipeline.

Each wrapper exposes the same interface:

    model.fit(X_train, y_train)
    model.predict(X_pred) -> np.ndarray

Statistical models (ARIMA, Holt-Winters) only use the target series, so
``X_train`` should be a 1-D array-like of chronologically ordered target
values and ``X_pred`` is ignored (they forecast one step ahead).

ML models (LightGBM, XGBoost, Random Forest) accept feature matrices.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.statespace.sarimax import SARIMAX

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

from sklearn.ensemble import RandomForestRegressor


# ─────────────────────────────── base class ──────────────────────────────────

class BaseModel:
    """Minimal interface shared by all model wrappers."""

    name: str = "base"

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "BaseModel":
        raise NotImplementedError

    def predict(self, X_pred: np.ndarray) -> np.ndarray:
        raise NotImplementedError


# ─────────────────────────── statistical models ──────────────────────────────

class HoltWintersModel(BaseModel):
    """Holt-Winters exponential smoothing (additive trend + additive season)."""

    name = "holt_winters"

    def __init__(self, seasonal_periods: int = 4):
        self.seasonal_periods = seasonal_periods
        self._model = None
        self._last_forecast: Optional[float] = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "HoltWintersModel":
        series = np.asarray(y_train, dtype=float)
        series = np.where(np.isfinite(series), series, np.nanmean(series))
        series = np.maximum(series, 0.0)

        n = len(series)
        if n < 2 * self.seasonal_periods:
            # Fall back to simple exponential smoothing
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    es = ExponentialSmoothing(
                        series, trend="add", seasonal=None
                    ).fit(optimized=True, disp=False)
                    self._last_forecast = float(es.forecast(1)[0])
            except Exception:
                self._last_forecast = float(np.mean(series[-4:]))
        else:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    hw = ExponentialSmoothing(
                        series,
                        trend="add",
                        seasonal="add",
                        seasonal_periods=self.seasonal_periods,
                    ).fit(optimized=True, disp=False)
                    self._last_forecast = float(hw.forecast(1)[0])
            except Exception:
                self._last_forecast = float(np.mean(series[-4:]))
        return self

    def predict(self, X_pred: np.ndarray) -> np.ndarray:
        return np.array([self._last_forecast])


class ARIMAModel(BaseModel):
    """SARIMA(p,d,q)(P,D,Q,m) wrapper with automatic order selection fallback."""

    name = "arima"

    def __init__(
        self,
        order: tuple[int, int, int] = (1, 1, 1),
        seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 4),
    ):
        self.order = order
        self.seasonal_order = seasonal_order
        self._last_forecast: Optional[float] = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "ARIMAModel":
        series = np.asarray(y_train, dtype=float)
        series = np.where(np.isfinite(series), series, np.nanmean(series))
        series = np.maximum(series, 0.0)

        orders_to_try = [
            (self.order, self.seasonal_order),
            ((1, 1, 0), (0, 0, 0, 0)),
            ((1, 0, 0), (0, 0, 0, 0)),
            ((0, 1, 1), (0, 0, 0, 0)),
        ]

        for order, seas_order in orders_to_try:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = SARIMAX(
                        series,
                        order=order,
                        seasonal_order=seas_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    ).fit(disp=False, maxiter=100)
                    fc = float(model.forecast(1)[0])
                    if np.isfinite(fc):
                        self._last_forecast = fc
                        return self
            except Exception:
                continue

        self._last_forecast = float(np.mean(series[-4:]))
        return self

    def predict(self, X_pred: np.ndarray) -> np.ndarray:
        return np.array([self._last_forecast])


class NaiveSeasonalModel(BaseModel):
    """Same-quarter-last-year naive baseline."""

    name = "naive_seasonal"

    def __init__(self, seasonal_periods: int = 4):
        self.seasonal_periods = seasonal_periods
        self._last_forecast: Optional[float] = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "NaiveSeasonalModel":
        series = np.asarray(y_train, dtype=float)
        valid = series[np.isfinite(series)]
        if len(valid) >= self.seasonal_periods:
            self._last_forecast = float(valid[-self.seasonal_periods])
        elif len(valid) > 0:
            self._last_forecast = float(valid[-1])
        else:
            self._last_forecast = 0.0
        return self

    def predict(self, X_pred: np.ndarray) -> np.ndarray:
        return np.array([self._last_forecast])


# ────────────────────────────── ML models ────────────────────────────────────

class LightGBMModel(BaseModel):
    """LightGBM regressor."""

    name = "lightgbm"

    def __init__(self, **params: object):
        default = {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 4,
            "num_leaves": 15,
            "min_child_samples": 2,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "random_state": 42,
            "verbose": -1,
        }
        default.update(params)
        self.params = default
        self._model: Optional[object] = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "LightGBMModel":
        if not _HAS_LGB:
            raise ImportError("lightgbm is not installed")
        X = np.nan_to_num(np.asarray(X_train, dtype=float), nan=0.0)
        y = np.asarray(y_train, dtype=float)
        self._model = lgb.LGBMRegressor(**self.params)
        self._model.fit(X, y)
        return self

    def predict(self, X_pred: np.ndarray) -> np.ndarray:
        X = np.nan_to_num(np.asarray(X_pred, dtype=float), nan=0.0)
        return self._model.predict(X)

    @property
    def feature_importances_(self) -> Optional[np.ndarray]:
        if self._model is not None:
            return self._model.feature_importances_
        return None


class XGBoostModel(BaseModel):
    """XGBoost regressor."""

    name = "xgboost"

    def __init__(self, **params: object):
        default = {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 4,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "random_state": 42,
            "verbosity": 0,
        }
        default.update(params)
        self.params = default
        self._model: Optional[object] = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "XGBoostModel":
        if not _HAS_XGB:
            raise ImportError("xgboost is not installed")
        X = np.nan_to_num(np.asarray(X_train, dtype=float), nan=0.0)
        y = np.asarray(y_train, dtype=float)
        self._model = xgb.XGBRegressor(**self.params)
        self._model.fit(X, y)
        return self

    def predict(self, X_pred: np.ndarray) -> np.ndarray:
        X = np.nan_to_num(np.asarray(X_pred, dtype=float), nan=0.0)
        return self._model.predict(X)

    @property
    def feature_importances_(self) -> Optional[np.ndarray]:
        if self._model is not None:
            return self._model.feature_importances_
        return None


class RandomForestModel(BaseModel):
    """Random Forest regressor."""

    name = "random_forest"

    def __init__(self, **params: object):
        default = {
            "n_estimators": 200,
            "max_depth": 6,
            "min_samples_leaf": 2,
            "random_state": 42,
            "n_jobs": -1,
        }
        default.update(params)
        self.params = default
        self._model: Optional[object] = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "RandomForestModel":
        X = np.nan_to_num(np.asarray(X_train, dtype=float), nan=0.0)
        y = np.asarray(y_train, dtype=float)
        self._model = RandomForestRegressor(**self.params)
        self._model.fit(X, y)
        return self

    def predict(self, X_pred: np.ndarray) -> np.ndarray:
        X = np.nan_to_num(np.asarray(X_pred, dtype=float), nan=0.0)
        return self._model.predict(X)

    @property
    def feature_importances_(self) -> Optional[np.ndarray]:
        if self._model is not None:
            return self._model.feature_importances_
        return None


# ─────────────────────────── model registry ──────────────────────────────────

def get_default_models() -> list[BaseModel]:
    """Return the default list of model instances used by the pipeline."""
    models: list[BaseModel] = [
        HoltWintersModel(),
        ARIMAModel(),
        NaiveSeasonalModel(),
        RandomForestModel(),
    ]
    if _HAS_LGB:
        models.append(LightGBMModel())
    if _HAS_XGB:
        models.append(XGBoostModel())
    return models
