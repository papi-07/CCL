"""
Unit tests for the CFL forecasting pipeline.

Run with:  pytest tests/ -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecasting_pipeline.backtesting import accuracy_score, rolling_backtest
from forecasting_pipeline.data_loader import _fy_to_cal, _FY_TO_CAL, load_all
from forecasting_pipeline.ensemble import ensemble_predict
from forecasting_pipeline.feature_engineering import (
    build_feature_matrix,
    classify_products,
    get_feature_columns,
)
from forecasting_pipeline.models import (
    ARIMAModel,
    HoltWintersModel,
    LightGBMModel,
    NaiveSeasonalModel,
    RandomForestModel,
    XGBoostModel,
)
from forecasting_pipeline.postprocessing import (
    cap_outliers,
    clip_negatives,
    postprocess,
    smooth_jump,
)


# ──────────────────────────── data loader tests ───────────────────────────────

class TestFyToCalMapping:
    def test_fy23q2_to_2023q2(self):
        assert _fy_to_cal("FY23Q2") == "2023Q2"

    def test_fy26q1_to_2026q1(self):
        assert _fy_to_cal("FY26Q1") == "2026Q1"

    def test_fy26q2_to_2026q2(self):
        assert _fy_to_cal("FY26Q2") == "2026Q2"

    def test_all_quarters_in_map(self):
        expected = [
            "FY23Q2", "FY23Q3", "FY23Q4",
            "FY24Q1", "FY24Q2", "FY24Q3", "FY24Q4",
            "FY25Q1", "FY25Q2", "FY25Q3", "FY25Q4",
            "FY26Q1", "FY26Q2",
        ]
        for q in expected:
            assert q in _FY_TO_CAL, f"{q} missing from _FY_TO_CAL"

    def test_no_duplicates_in_cal(self):
        cal_vals = list(_FY_TO_CAL.values())
        assert len(cal_vals) == len(set(cal_vals)), "Duplicate cal_quarter values"


class TestDataLoading:
    @pytest.fixture(scope="class")
    def tables(self):
        return load_all()

    def test_actuals_shape(self, tables):
        df = tables["actuals"]
        assert df.shape[0] == 360, f"Expected 360 rows, got {df.shape[0]}"

    def test_actuals_30_products(self, tables):
        assert tables["actuals"]["product"].nunique() == 30

    def test_actuals_12_quarters(self, tables):
        assert tables["actuals"]["quarter"].nunique() == 12

    def test_no_null_products(self, tables):
        assert tables["actuals"]["product"].isna().sum() == 0

    def test_scms_quarters(self, tables):
        qs = set(tables["scms"]["cal_quarter"].unique())
        assert "2026Q1" in qs
        assert "2023Q1" in qs

    def test_vms_total_mostly_non_negative(self, tables):
        # A few negatives from returns are acceptable (< 5 % of rows)
        neg = (tables["vms"]["vms_total"] < 0).sum()
        assert neg / len(tables["vms"]) < 0.05

    def test_big_deal_8_quarters(self, tables):
        assert tables["big_deal"]["cal_quarter"].nunique() == 8


# ────────────────────────── feature engineering tests ─────────────────────────

class TestProductClassification:
    def make_actuals(self, values, product="P1", life_cycle="Sustaining"):
        quarters = [f"FY23Q{i+2}" for i in range(len(values))]
        return pd.DataFrame({
            "product": product,
            "life_cycle": life_cycle,
            "quarter": quarters,
            "actual_units": values,
        })

    def test_new_product_few_obs(self):
        df = self.make_actuals([100.0, 200.0, np.nan])
        result = classify_products(df)
        assert result["ts_class"].iloc[0] == "new"

    def test_stable_product(self):
        vals = [1000.0, 1050.0, 980.0, 1020.0, 1010.0, 990.0, 1030.0, 1000.0]
        df = self.make_actuals(vals)
        result = classify_products(df)
        assert result["ts_class"].iloc[0] == "stable"

    def test_intermittent_product(self):
        # High zero fraction
        vals = [500.0, 0.0, 0.0, 600.0, 0.0, 0.0, 0.0, 400.0]
        df = self.make_actuals(vals)
        result = classify_products(df)
        assert result["ts_class"].iloc[0] == "intermittent"


class TestBuildFeatureMatrix:
    @pytest.fixture(scope="class")
    def feature_df(self):
        tables = load_all()
        return build_feature_matrix(
            actuals=tables["actuals"],
            big_deal=tables["big_deal"],
            scms=tables["scms"],
            vms=tables["vms"],
        )

    def test_target_row_exists(self, feature_df):
        target = feature_df[feature_df["quarter"] == "FY26Q2"]
        assert len(target) == 30, f"Expected 30 target rows, got {len(target)}"

    def test_target_row_no_actuals(self, feature_df):
        target = feature_df[feature_df["quarter"] == "FY26Q2"]
        assert target["actual_units"].isna().all()

    def test_lag1_is_fy26q1(self, feature_df):
        # For the target quarter row, lag-1 should equal the FY26Q1 actual
        target_row = feature_df[
            (feature_df["quarter"] == "FY26Q2") &
            (feature_df["product"] == "SWITCH Enterprise 48-Port UPOE")
        ].iloc[0]
        fy26q1_row = feature_df[
            (feature_df["quarter"] == "FY26Q1") &
            (feature_df["product"] == "SWITCH Enterprise 48-Port UPOE")
        ].iloc[0]
        assert target_row["actual_units_lag1"] == fy26q1_row["actual_units"]

    def test_no_future_leak_in_lag(self, feature_df):
        # For FY23Q2 (first row per product), lag1 must be NaN
        first_rows = feature_df[feature_df["quarter"] == "FY23Q2"]
        assert first_rows["actual_units_lag1"].isna().all()

    def test_feature_columns_all_numeric(self, feature_df):
        feat_cols = get_feature_columns(feature_df)
        for col in feat_cols:
            assert feature_df[col].dtype in [np.float64, np.int64, float, int], \
                f"Column {col} is not numeric"


# ───────────────────────────── model tests ────────────────────────────────────

class TestStatisticalModels:
    series = np.array([100, 110, 95, 105, 100, 115, 90, 110], dtype=float)

    def test_holt_winters_fit_predict(self):
        m = HoltWintersModel()
        m.fit(self.series, self.series)
        pred = m.predict(np.zeros((1, 5)))
        assert np.isfinite(pred[0])
        assert pred[0] >= 0

    def test_arima_fit_predict(self):
        m = ARIMAModel()
        m.fit(self.series, self.series)
        pred = m.predict(np.zeros((1, 5)))
        assert np.isfinite(pred[0])

    def test_naive_seasonal_exact(self):
        m = NaiveSeasonalModel(seasonal_periods=4)
        m.fit(self.series, self.series)
        pred = m.predict(np.zeros((1, 5)))
        # Should return series[-4] = 100
        assert pred[0] == 100.0

    def test_arima_short_series(self):
        short = np.array([100.0, 110.0, 95.0])
        m = ARIMAModel()
        m.fit(short, short)
        pred = m.predict(np.zeros((1, 5)))
        assert np.isfinite(pred[0])


class TestMLModels:
    @pytest.fixture(autouse=True)
    def setup(self):
        rng = np.random.default_rng(42)
        n_train = 20
        self.X_train = rng.standard_normal((n_train, 5))
        self.y_train = rng.uniform(100, 1000, n_train)
        self.X_pred = rng.standard_normal((1, 5))

    def test_random_forest(self):
        m = RandomForestModel(n_estimators=10)
        m.fit(self.X_train, self.y_train)
        pred = m.predict(self.X_pred)
        assert pred.shape == (1,)
        assert np.isfinite(pred[0])

    def test_lightgbm(self):
        m = LightGBMModel(n_estimators=10)
        m.fit(self.X_train, self.y_train)
        pred = m.predict(self.X_pred)
        assert np.isfinite(pred[0])
        assert m.feature_importances_ is not None

    def test_xgboost(self):
        m = XGBoostModel(n_estimators=10)
        m.fit(self.X_train, self.y_train)
        pred = m.predict(self.X_pred)
        assert np.isfinite(pred[0])

    def test_nan_in_features_handled(self):
        X_nan = self.X_train.copy()
        X_nan[0, 0] = np.nan
        m = RandomForestModel(n_estimators=10)
        m.fit(X_nan, self.y_train)  # should not raise
        pred = m.predict(self.X_pred)
        assert np.isfinite(pred[0])


# ──────────────────────────── backtesting tests ───────────────────────────────

class TestAccuracyScore:
    def test_perfect_forecast(self):
        assert accuracy_score(100.0, 100.0) == pytest.approx(1.0)

    def test_zero_actual_returns_nan(self):
        result = accuracy_score(100.0, 0.0)
        assert np.isnan(result)

    def test_off_by_50_pct(self):
        assert accuracy_score(150.0, 100.0) == pytest.approx(0.5)

    def test_clamped_at_zero(self):
        # 3× overforecast → raw = 1 - 2 = -1, clamped to 0
        assert accuracy_score(300.0, 100.0) == pytest.approx(0.0)

    def test_negative_actual_treated_as_zero_accuracy(self):
        result = accuracy_score(100.0, -50.0)
        # abs(actual) is 50 → raw = 1 - 3 = -2 → clamped to 0
        assert result == pytest.approx(0.0)


class TestRollingBacktest:
    @pytest.fixture(scope="class")
    def backtest_result(self):
        tables = load_all()
        feat_df = build_feature_matrix(
            actuals=tables["actuals"],
            big_deal=tables["big_deal"],
            scms=tables["scms"],
            vms=tables["vms"],
        )
        feat_cols = get_feature_columns(feat_df)
        product = "SWITCH Enterprise 48-Port UPOE"
        product_df = feat_df[feat_df["product"] == product].sort_values("quarter_idx")
        models = [NaiveSeasonalModel(), RandomForestModel(n_estimators=10)]
        return rolling_backtest(product_df, feat_cols, models=models, min_train_size=4)

    def test_returns_dataframe(self, backtest_result):
        assert isinstance(backtest_result, pd.DataFrame)

    def test_has_accuracy_columns(self, backtest_result):
        assert "accuracy_naive_seasonal" in backtest_result.columns
        assert "accuracy_random_forest" in backtest_result.columns

    def test_accuracy_in_range(self, backtest_result):
        for col in ["accuracy_naive_seasonal", "accuracy_random_forest"]:
            vals = backtest_result[col].dropna()
            assert (vals >= 0).all() and (vals <= 1).all()

    def test_at_least_one_eval_window(self, backtest_result):
        assert len(backtest_result) >= 1


# ──────────────────────────── ensemble tests ─────────────────────────────────

class TestEnsemblePredict:
    def test_equal_weights(self):
        preds = {"m1": 100.0, "m2": 200.0}
        weights = {"m1": 1.0, "m2": 1.0}
        result = ensemble_predict(preds, weights)
        assert result == pytest.approx(150.0)

    def test_skewed_weights(self):
        preds = {"m1": 100.0, "m2": 200.0}
        weights = {"m1": 3.0, "m2": 1.0}
        result = ensemble_predict(preds, weights)
        assert result == pytest.approx(125.0)

    def test_ignores_nan_pred(self):
        preds = {"m1": 100.0, "m2": float("nan")}
        weights = {"m1": 1.0, "m2": 1.0}
        result = ensemble_predict(preds, weights)
        assert result == pytest.approx(100.0)

    def test_missing_weight_defaults_to_one(self):
        preds = {"m1": 100.0, "m2": 200.0}
        weights = {"m1": 1.0}   # m2 not in weights
        result = ensemble_predict(preds, weights)
        assert result == pytest.approx(150.0)


# ─────────────────────────── postprocessing tests ────────────────────────────

class TestPostProcessing:
    def test_clip_negatives(self):
        assert clip_negatives(-100.0) == 0.0
        assert clip_negatives(200.0) == 200.0

    def test_cap_outliers_clips_high(self):
        hist = np.array([100.0, 110.0, 95.0, 105.0, 100.0, 115.0])
        result = cap_outliers(10000.0, hist, iqr_multiplier=2.0)
        assert result < 500.0

    def test_cap_outliers_preserves_normal(self):
        hist = np.array([100.0, 110.0, 95.0, 105.0, 100.0, 115.0])
        result = cap_outliers(108.0, hist, iqr_multiplier=2.0)
        assert result == pytest.approx(108.0)

    def test_smooth_jump_caps_large_increase(self):
        result = smooth_jump(10000.0, last_actual=100.0, max_change_ratio=3.0)
        assert result == pytest.approx(300.0)

    def test_smooth_jump_caps_large_decrease(self):
        result = smooth_jump(1.0, last_actual=100.0, max_change_ratio=3.0)
        assert result == pytest.approx(100.0 / 3.0)

    def test_postprocess_returns_integer(self):
        hist = np.array([100.0, 110.0, 95.0, 105.0, 100.0, 115.0])
        result = postprocess(107.3, hist, last_actual=115.0)
        assert result == float(int(result))

    def test_postprocess_non_negative(self):
        hist = np.array([100.0, 110.0, 95.0, 105.0])
        result = postprocess(-50.0, hist, last_actual=100.0)
        assert result >= 0.0


# ───────────────────────────── integration test ───────────────────────────────

class TestEndToEndPipeline:
    @pytest.fixture(scope="class")
    def forecast_df(self):
        from forecasting_pipeline.forecast import run_pipeline

        return run_pipeline(
            output_csv=None,
            models=[NaiveSeasonalModel(), RandomForestModel(n_estimators=20)],
            verbose=False,
        )

    def test_returns_30_products(self, forecast_df):
        assert len(forecast_df) == 30

    def test_all_forecasts_non_negative(self, forecast_df):
        assert (forecast_df["fy26q2_forecast"] >= 0).all()

    def test_no_null_forecasts(self, forecast_df):
        assert forecast_df["fy26q2_forecast"].isna().sum() == 0

    def test_has_required_columns(self, forecast_df):
        for col in ["cost_rank", "product", "fy26q2_forecast", "ts_class"]:
            assert col in forecast_df.columns, f"Missing column: {col}"

    def test_forecasts_are_integers(self, forecast_df):
        for val in forecast_df["fy26q2_forecast"]:
            assert val == float(int(val)), f"Non-integer forecast: {val}"
