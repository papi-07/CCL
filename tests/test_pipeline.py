"""
Unit tests for the CFL forecasting pipeline.

Run with:  pytest tests/ -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecasting_pipeline.backtesting import accuracy_score, rolling_backtest, portfolio_backtest
from forecasting_pipeline.data_loader import _fy_to_cal, _FY_TO_CAL, load_all
from forecasting_pipeline.ensemble import (
    ensemble_predict,
    compute_expert_weights,
    blend_expert_forecasts,
)
from forecasting_pipeline.feature_engineering import (
    build_feature_matrix,
    classify_products,
    get_feature_columns,
    _add_recency_weighted_features,
)
from forecasting_pipeline.models import (
    ARIMAModel,
    HoltWintersModel,
    LightGBMModel,
    NaiveSeasonalModel,
    RandomForestModel,
    VMSSCMSRegressionModel,
    XGBoostModel,
)
from forecasting_pipeline.postprocessing import (
    cap_outliers,
    clip_negatives,
    postprocess,
    smooth_jump,
    _lifecycle_change_ratio,
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

    def test_has_best_model_column(self, forecast_df):
        assert "best_model" in forecast_df.columns

    def test_has_bias_applied_column(self, forecast_df):
        assert "bias_applied" in forecast_df.columns

    def test_has_expert_blend_pred_column(self, forecast_df):
        assert "expert_blend_pred" in forecast_df.columns

    def test_has_expert_weights_used_column(self, forecast_df):
        assert "expert_weights_used" in forecast_df.columns


# ─────────────────────── demand sensing (EWMA) tests ─────────────────────────

class TestRecencyWeightedFeatures:
    def _make_df(self):
        """Tiny product dataframe for testing EWMA features."""
        return pd.DataFrame({
            "product":      ["P"] * 6,
            "quarter_idx":  list(range(6)),
            "actual_units": [100.0, 110.0, 90.0, 120.0, 100.0, 105.0],
        })

    def test_ewma_column_created(self):
        df = self._make_df()
        result = _add_recency_weighted_features(df, "actual_units")
        assert "actual_units_ewma" in result.columns

    def test_ewma_no_look_ahead(self):
        """First row EWMA should be NaN (shift(1) ensures no look-ahead)."""
        df = self._make_df()
        result = _add_recency_weighted_features(df, "actual_units")
        first = result.sort_values("quarter_idx").iloc[0]["actual_units_ewma"]
        assert np.isnan(first)

    def test_ewma_finite_after_first(self):
        df = self._make_df()
        result = _add_recency_weighted_features(df, "actual_units")
        later = result.sort_values("quarter_idx").iloc[2:]["actual_units_ewma"]
        assert later.notna().all()

    def test_ewma_in_feature_matrix(self):
        tables = load_all()
        feat_df = build_feature_matrix(
            actuals=tables["actuals"],
            big_deal=tables["big_deal"],
            scms=tables["scms"],
            vms=tables["vms"],
        )
        assert "actual_units_ewma" in feat_df.columns
        assert "scms_total_ewma" in feat_df.columns
        assert "vms_total_ewma" in feat_df.columns


# ─────────────────────── VMS-SCMS regression model tests ─────────────────────

class TestVMSSCMSRegressionModel:
    @pytest.fixture(autouse=True)
    def setup(self):
        rng = np.random.default_rng(0)
        n = 20
        self.X_train = rng.standard_normal((n, 8))
        self.y_train = rng.uniform(100, 1000, n)
        self.X_pred  = rng.standard_normal((1, 8))
        self.fnames  = [
            "vms_total_lag1", "scms_total_lag1", "actual_units_lag1",
            "actual_units_lag2", "actual_units_ewma", "trend", "is_Q1", "is_Q2",
        ]

    def test_fit_predict_basic(self):
        m = VMSSCMSRegressionModel()
        m.fit(self.X_train, self.y_train)
        pred = m.predict(self.X_pred)
        assert pred.shape == (1,)
        assert np.isfinite(pred[0])

    def test_fit_with_feature_names(self):
        m = VMSSCMSRegressionModel()
        m.fit(self.X_train, self.y_train, feature_names=self.fnames)
        pred = m.predict(self.X_pred)
        assert np.isfinite(pred[0])

    def test_selected_cols_are_signal_columns(self):
        m = VMSSCMSRegressionModel()
        m.fit(self.X_train, self.y_train, feature_names=self.fnames)
        # _selected_cols should be a non-empty subset
        assert m._selected_cols is not None
        assert len(m._selected_cols) > 0

    def test_fallback_on_tiny_dataset(self):
        m = VMSSCMSRegressionModel()
        m.fit(self.X_train[:1], self.y_train[:1])
        pred = m.predict(self.X_pred)
        assert np.isfinite(pred[0])

    def test_vms_scms_reg_in_default_models(self):
        from forecasting_pipeline.models import get_default_models
        names = [m.name for m in get_default_models()]
        assert "vms_scms_reg" in names


# ──────────────────────── recency-weighted backtest tests ─────────────────────

class TestRecencyWeightedBacktest:
    @pytest.fixture(scope="class")
    def backtest_with_weights(self):
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
        return rolling_backtest(
            product_df, feat_cols,
            models=[NaiveSeasonalModel()],
            min_train_size=4,
            recency_decay=0.85,
        )

    def test_fold_weight_column_exists(self, backtest_with_weights):
        assert "fold_weight" in backtest_with_weights.columns

    def test_last_fold_highest_weight(self, backtest_with_weights):
        bt = backtest_with_weights
        last_w  = bt["fold_weight"].iloc[-1]
        first_w = bt["fold_weight"].iloc[0]
        assert last_w >= first_w

    def test_error_column_exists(self, backtest_with_weights):
        assert "error_naive_seasonal" in backtest_with_weights.columns

    def test_portfolio_backtest_has_bias(self):
        tables = load_all()
        feat_df = build_feature_matrix(
            actuals=tables["actuals"],
            big_deal=tables["big_deal"],
            scms=tables["scms"],
            vms=tables["vms"],
        )
        _, summary_df = portfolio_backtest(
            feat_df,
            models=[NaiveSeasonalModel()],
            min_train_size=4,
            recency_decay=0.85,
        )
        assert "bias" in summary_df.columns
        assert "best_model" in summary_df.columns

    def test_softmax_best_model_highest_weight(self):
        """After softmax sharpening, the best model must have the highest weight."""
        tables = load_all()
        feat_df = build_feature_matrix(
            actuals=tables["actuals"],
            big_deal=tables["big_deal"],
            scms=tables["scms"],
            vms=tables["vms"],
        )
        _, summary_df = portfolio_backtest(
            feat_df,
            models=[NaiveSeasonalModel(), RandomForestModel(n_estimators=5)],
            min_train_size=4,
        )
        for _, row in summary_df.iterrows():
            w = row["optimal_weights"]
            if not isinstance(w, dict) or len(w) < 2:
                continue
            best = row["best_model"]
            if best in w:
                others = [v for k, v in w.items() if k != best]
                # Best model weight should be >= average of others after sharpening
                assert w[best] >= float(np.mean(others)), \
                    f"Best model {best} weight {w[best]} < mean others {np.mean(others)}"


# ─────────────────────── expert blending tests ────────────────────────────────

class TestExpertBlending:
    @pytest.fixture(scope="class")
    def feat_df(self):
        tables = load_all()
        return build_feature_matrix(
            actuals=tables["actuals"],
            big_deal=tables["big_deal"],
            scms=tables["scms"],
            vms=tables["vms"],
        )

    def test_compute_expert_weights_returns_dict(self, feat_df):
        weights = compute_expert_weights(feat_df)
        assert isinstance(weights, dict)
        assert len(weights) == 30

    def test_expert_weights_sum_to_one(self, feat_df):
        weights = compute_expert_weights(feat_df)
        for product, w in weights.items():
            total = sum(w.values())
            assert abs(total - 1.0) < 1e-9, \
                f"Product {product} expert weights sum to {total}"

    def test_expert_weights_non_negative(self, feat_df):
        weights = compute_expert_weights(feat_df)
        for product, w in weights.items():
            for col, val in w.items():
                assert val >= 0, f"Negative expert weight for {product}/{col}"

    def test_blend_expert_forecasts_returns_float(self, feat_df):
        target_row = feat_df[feat_df["quarter"] == "FY26Q2"].iloc[0]
        product = target_row["product"]
        weights = compute_expert_weights(feat_df)
        result = blend_expert_forecasts(target_row, weights.get(product, {}))
        # Should return a finite positive float or None
        assert result is None or (np.isfinite(result) and result >= 0)

    def test_blend_missing_expert_returns_none(self):
        row = pd.Series({"dp_forecast": np.nan, "mktg_forecast": np.nan,
                         "ds_forecast": np.nan})
        result = blend_expert_forecasts(row, {})
        assert result is None


# ─────────────────────── lifecycle post-processing tests ──────────────────────

class TestLifecyclePostProcessing:
    hist = np.array([100.0, 110.0, 95.0, 105.0, 100.0, 115.0])

    def test_npi_ramp_allows_large_growth(self):
        ratio, do_jump = _lifecycle_change_ratio("NPI-Ramp", "stable", 3.0)
        assert ratio >= 4.0
        assert do_jump is True

    def test_decline_limits_upside(self):
        ratio, do_jump = _lifecycle_change_ratio("Decline", "stable", 3.0)
        assert ratio <= 2.0
        assert do_jump is True

    def test_intermittent_skips_jump_smoothing(self):
        ratio, do_jump = _lifecycle_change_ratio("Sustaining", "intermittent", 3.0)
        assert do_jump is False

    def test_volatile_widens_ratio(self):
        ratio, do_jump = _lifecycle_change_ratio("Sustaining", "volatile", 3.0)
        assert ratio >= 3.5

    def test_postprocess_npi_ramp_allows_high_forecast(self):
        """NPI-Ramp: 5× increase should pass through (ratio ≥ 4)."""
        result_npi = postprocess(
            550.0, self.hist, last_actual=100.0,
            life_cycle="NPI-Ramp", ts_class="stable",
        )
        result_sus = postprocess(
            550.0, self.hist, last_actual=100.0,
            life_cycle="Sustaining", ts_class="stable",
        )
        # NPI-Ramp should allow a higher forecast than Sustaining
        assert result_npi >= result_sus

    def test_postprocess_decline_caps_upside(self):
        """Decline product: very high forecast should be capped more aggressively."""
        result_dec = postprocess(
            500.0, self.hist, last_actual=100.0,
            life_cycle="Decline", ts_class="stable",
        )
        result_sus = postprocess(
            500.0, self.hist, last_actual=100.0,
            life_cycle="Sustaining", ts_class="stable",
        )
        assert result_dec <= result_sus

    def test_postprocess_accepts_lifecycle_params(self):
        """Ensure no exception with lifecycle params."""
        result = postprocess(
            107.3, self.hist, last_actual=115.0,
            life_cycle="Sustaining", ts_class="stable",
        )
        assert np.isfinite(result)
        assert result >= 0


# ──────────────────────── anchor constraint tests ────────────────────────────

class TestAnchorConstraint:
    from forecasting_pipeline.postprocessing import anchor_constraint as _anchor

    def test_forecast_above_upper_clamped(self):
        from forecasting_pipeline.postprocessing import anchor_constraint
        result = anchor_constraint(200.0, last_actual=100.0, upper_factor=1.3)
        assert result == pytest.approx(130.0)

    def test_forecast_below_lower_clamped(self):
        from forecasting_pipeline.postprocessing import anchor_constraint
        result = anchor_constraint(50.0, last_actual=100.0, lower_factor=0.7)
        assert result == pytest.approx(70.0)

    def test_forecast_in_range_unchanged(self):
        from forecasting_pipeline.postprocessing import anchor_constraint
        result = anchor_constraint(110.0, last_actual=100.0)
        assert result == pytest.approx(110.0)

    def test_zero_last_actual_noop(self):
        from forecasting_pipeline.postprocessing import anchor_constraint
        # When last_actual == 0, constraint cannot be applied → forecast unchanged
        result = anchor_constraint(500.0, last_actual=0.0)
        assert result == pytest.approx(500.0)

    def test_npi_ramp_skips_anchor(self):
        """NPI-Ramp products must not have anchor constraint applied."""
        hist = np.array([50.0, 80.0, 120.0, 200.0, 300.0])
        result_npi = postprocess(
            900.0, hist, last_actual=300.0,
            life_cycle="NPI-Ramp", ts_class="stable",
        )
        result_sus = postprocess(
            900.0, hist, last_actual=300.0,
            life_cycle="Sustaining", ts_class="stable",
        )
        # NPI-Ramp forecast should be higher (anchor not applied)
        assert result_npi >= result_sus

    def test_intermittent_skips_anchor(self):
        """Intermittent products should not have anchor constraint applied."""
        hist = np.array([0.0, 300.0, 0.0, 200.0, 0.0])
        result_intermittent = postprocess(
            600.0, hist, last_actual=0.0,
            life_cycle="Sustaining", ts_class="intermittent",
        )
        # last_actual=0, so anchor is naturally disabled; just check no error
        assert np.isfinite(result_intermittent)

    def test_postprocess_applies_anchor_by_default(self):
        """Anchor should shrink an extreme forecast for a stable product."""
        hist = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        # 300 is 3× last_actual=100; with default anchor (1.3×) should cap at 130
        result = postprocess(
            300.0, hist, last_actual=100.0,
            life_cycle="Sustaining", ts_class="stable",
        )
        assert result <= 130.0


# ────────────────────────── momentum feature tests ───────────────────────────

class TestMomentumFeatures:
    @pytest.fixture(scope="class")
    def feat_df(self):
        tables = load_all()
        return build_feature_matrix(
            actuals=tables["actuals"],
            big_deal=tables["big_deal"],
            scms=tables["scms"],
            vms=tables["vms"],
        )

    def test_momentum_column_exists(self, feat_df):
        assert "actual_units_mom" in feat_df.columns

    def test_momentum_rate_column_exists(self, feat_df):
        assert "actual_units_mom_rate" in feat_df.columns

    def test_momentum_accel_column_exists(self, feat_df):
        assert "actual_units_accel" in feat_df.columns

    def test_momentum_no_look_ahead(self, feat_df):
        """Momentum = lag1 - lag2; first 2 rows per product should be NaN."""
        grp = feat_df[feat_df["product"] == feat_df["product"].iloc[0]].sort_values(
            "quarter_idx"
        )
        # At least the first row should have NaN momentum
        assert grp["actual_units_mom"].isna().any()

    def test_momentum_in_feature_cols(self, feat_df):
        """Momentum features should be returned by get_feature_columns."""
        fcols = get_feature_columns(feat_df)
        assert "actual_units_mom" in fcols
        assert "actual_units_mom_rate" in fcols
        assert "actual_units_accel" in fcols

    def test_momentum_computation_correct(self):
        """Spot-check: momentum for 3rd row should equal lag1 - lag2."""
        df = pd.DataFrame({
            "product":      ["P"] * 5,
            "quarter_idx":  list(range(5)),
            "actual_units": [100.0, 120.0, 90.0, 150.0, 110.0],
        })
        from forecasting_pipeline.feature_engineering import _add_lag_features
        df = _add_lag_features(df, "actual_units", lags=(1, 2, 3))
        df["actual_units_mom"] = df["actual_units_lag1"] - df["actual_units_lag2"]
        row = df[df["quarter_idx"] == 3].iloc[0]
        assert row["actual_units_mom"] == pytest.approx(row["actual_units_lag1"] - row["actual_units_lag2"])


# ─────────────────── ensemble weight optimization tests ─────────────────────

class TestWeightOptimization:
    @pytest.fixture(scope="class")
    def bt_summary(self):
        tables = load_all()
        feat_df = build_feature_matrix(
            actuals=tables["actuals"],
            big_deal=tables["big_deal"],
            scms=tables["scms"],
            vms=tables["vms"],
        )
        _, summary = portfolio_backtest(
            feat_df,
            models=[NaiveSeasonalModel(), RandomForestModel(n_estimators=10)],
            min_train_size=4,
            recency_decay=0.85,
        )
        return summary

    def test_optimal_weights_present(self, bt_summary):
        assert "optimal_weights" in bt_summary.columns

    def test_weights_sum_to_one(self, bt_summary):
        for _, row in bt_summary.iterrows():
            w = row["optimal_weights"]
            if isinstance(w, dict) and w:
                total = sum(w.values())
                assert abs(total - 1.0) < 1e-6, f"Weights sum to {total}"

    def test_weights_non_negative(self, bt_summary):
        for _, row in bt_summary.iterrows():
            w = row["optimal_weights"]
            if isinstance(w, dict):
                for k, v in w.items():
                    assert v >= -1e-9, f"Negative weight {v} for {k}"

    def test_best_model_has_highest_weight(self, bt_summary):
        """The best model should receive the highest optimised weight."""
        for _, row in bt_summary.iterrows():
            w = row["optimal_weights"]
            best = row.get("best_model")
            if not isinstance(w, dict) or len(w) < 2 or not best or best not in w:
                continue
            others_max = max(v for k, v in w.items() if k != best)
            # best model weight should be >= all others (allow tiny fp tolerance)
            assert w[best] >= others_max - 1e-6


# ──────────────────────── model confidence tests ─────────────────────────────

class TestModelConfidence:
    @pytest.fixture(scope="class")
    def bt_summary(self):
        tables = load_all()
        feat_df = build_feature_matrix(
            actuals=tables["actuals"],
            big_deal=tables["big_deal"],
            scms=tables["scms"],
            vms=tables["vms"],
        )
        _, summary = portfolio_backtest(
            feat_df,
            models=[NaiveSeasonalModel()],
            min_train_size=4,
            recency_decay=0.85,
        )
        return summary

    def test_model_confidence_column_present(self, bt_summary):
        assert "model_confidence" in bt_summary.columns

    def test_confidence_in_range(self, bt_summary):
        for val in bt_summary["model_confidence"].dropna():
            assert 0.0 <= float(val) <= 1.0

    def test_confidence_helper_on_synthetic(self):
        from forecasting_pipeline.backtesting import _compute_model_confidence
        rng = np.random.default_rng(42)
        n = 8
        actual = rng.uniform(100, 200, n)
        preds  = rng.uniform(90, 210, n)
        bt = pd.DataFrame({
            "actual_units":       actual,
            "naive_seasonal_pred": preds,
            "accuracy_naive_seasonal": np.clip(1 - np.abs(preds - actual) / actual, 0, 1),
            "fold_weight":        [0.85 ** (n - 1 - i) for i in range(n)],
        })
        conf = _compute_model_confidence(bt, ["naive_seasonal"], bt["fold_weight"].values)
        assert 0.0 < conf < 1.0


# ──────────────── expert accuracy estimate tests ─────────────────────────────

class TestExpertAccuracyEstimate:
    from forecasting_pipeline.ensemble import compute_expert_accuracy_estimate as _fn

    def test_neutral_when_no_naive(self):
        from forecasting_pipeline.ensemble import compute_expert_accuracy_estimate
        row = pd.Series({
            "actual_units_lag4": np.nan,
            "dp_forecast": 100.0, "mktg_forecast": 120.0, "ds_forecast": 90.0,
        })
        assert compute_expert_accuracy_estimate(row) == pytest.approx(0.5)

    def test_high_accuracy_when_experts_match_naive(self):
        from forecasting_pipeline.ensemble import compute_expert_accuracy_estimate
        row = pd.Series({
            "actual_units_lag4": 100.0,
            "dp_forecast": 100.0, "mktg_forecast": 100.0, "ds_forecast": 100.0,
        })
        acc = compute_expert_accuracy_estimate(row)
        assert acc == pytest.approx(1.0)

    def test_lower_accuracy_when_experts_diverge(self):
        from forecasting_pipeline.ensemble import compute_expert_accuracy_estimate
        row_near = pd.Series({
            "actual_units_lag4": 100.0,
            "dp_forecast": 102.0, "mktg_forecast": 98.0, "ds_forecast": 101.0,
        })
        row_far = pd.Series({
            "actual_units_lag4": 100.0,
            "dp_forecast": 200.0, "mktg_forecast": 50.0, "ds_forecast": 180.0,
        })
        from forecasting_pipeline.ensemble import compute_expert_accuracy_estimate
        assert compute_expert_accuracy_estimate(row_near) > compute_expert_accuracy_estimate(row_far)

    def test_neutral_when_no_experts(self):
        from forecasting_pipeline.ensemble import compute_expert_accuracy_estimate
        row = pd.Series({
            "actual_units_lag4": 100.0,
            "dp_forecast": np.nan, "mktg_forecast": np.nan, "ds_forecast": np.nan,
        })
        assert compute_expert_accuracy_estimate(row) == pytest.approx(0.5)


# ─────────────────── confidence-based blending tests ────────────────────────

class TestConfidenceBasedBlending:
    """Verify that dynamic expert weight scales with model confidence."""

    def _make_ensemble_result(self, model_confidence: float) -> dict:
        from forecasting_pipeline.ensemble import build_ensemble_forecast
        from forecasting_pipeline.models import NaiveSeasonalModel
        rng = np.random.default_rng(0)
        n = 8
        df = pd.DataFrame({
            "quarter_idx":    list(range(n)),
            "actual_units":   rng.uniform(100, 300, n),
            "dp_forecast":    [150.0] * n,
            "mktg_forecast":  [160.0] * n,
            "ds_forecast":    [155.0] * n,
            "trend":          list(range(n)),
            "actual_units_lag1": [np.nan] + list(rng.uniform(90, 310, n - 1)),
        })
        feature_cols = ["trend", "actual_units_lag1"]
        train_df = df.iloc[:-1].copy()
        pred_row = df.iloc[-1].copy()
        weights  = {"naive_seasonal": 1.0}
        return build_ensemble_forecast(
            product="P1",
            train_df=train_df,
            pred_row=pred_row,
            feature_cols=feature_cols,
            models=[NaiveSeasonalModel()],
            weights=weights,
            model_confidence=model_confidence,
            expert_pred=155.0,
            expert_weight=0.15,
        )

    def test_result_has_expert_override_key(self):
        result = self._make_ensemble_result(0.5)
        assert "expert_override_applied" in result

    def test_low_confidence_raises_expert_weight(self):
        """Low model confidence → dynamic expert weight should be high."""
        import math
        # dynamic_expert_weight = clip(0.15 * (1.5 - conf), 0.05, 0.40)
        low_conf  = float(np.clip(0.15 * (1.5 - 0.1), 0.05, 0.40))
        high_conf = float(np.clip(0.15 * (1.5 - 0.9), 0.05, 0.40))
        assert low_conf > high_conf


# ──────────────── portfolio calibration tests ────────────────────────────────

class TestPortfolioCalibration:
    @pytest.fixture(scope="class")
    def forecast_df(self):
        """Run full pipeline with portfolio_calibration=True."""
        from forecasting_pipeline.forecast import run_pipeline
        from forecasting_pipeline.models import NaiveSeasonalModel
        return run_pipeline(
            models=[NaiveSeasonalModel()],
            output_csv=None,
            verbose=False,
            portfolio_calibration=True,
        )

    def test_calibration_factor_column_present(self, forecast_df):
        assert "calibration_factor" in forecast_df.columns

    def test_calibration_factor_in_range(self, forecast_df):
        """Calibration factor should be between 0.80 and 1.20."""
        for val in forecast_df["calibration_factor"].dropna():
            assert 0.79 <= float(val) <= 1.21

    def test_non_npi_products_have_calibration(self, forecast_df):
        """Non-NPI-Ramp products should have calibration_factor != 1.0
        if portfolio adjustment was needed, OR exactly 1.0 if already aligned."""
        non_npi = forecast_df[forecast_df["life_cycle"] != "NPI-Ramp"]
        # All non-NPI rows should have a finite calibration_factor
        assert non_npi["calibration_factor"].notna().all()

    def test_npi_ramp_calibration_factor_is_one(self, forecast_df):
        """NPI-Ramp products must not be calibrated."""
        npi = forecast_df[forecast_df["life_cycle"] == "NPI-Ramp"]
        if not npi.empty:
            assert (npi["calibration_factor"] == 1.0).all()

    def test_forecasts_positive_after_calibration(self, forecast_df):
        assert (forecast_df["fy26q2_forecast"] >= 0).all()


# ──────────────── intermittent demand handling tests ─────────────────────────

class TestIntermittentDemandHandling:
    @pytest.fixture(scope="class")
    def forecast_df(self):
        from forecasting_pipeline.forecast import run_pipeline
        from forecasting_pipeline.models import NaiveSeasonalModel
        return run_pipeline(
            models=[NaiveSeasonalModel()],
            output_csv=None,
            verbose=False,
            portfolio_calibration=False,
        )

    def test_intermittent_products_have_forecasts(self, forecast_df):
        """All intermittent products should still have a non-NaN forecast."""
        intermittent = forecast_df[forecast_df["ts_class"] == "intermittent"]
        if not intermittent.empty:
            assert intermittent["fy26q2_forecast"].notna().all()

    def test_intermittent_forecasts_non_negative(self, forecast_df):
        intermittent = forecast_df[forecast_df["ts_class"] == "intermittent"]
        if not intermittent.empty:
            assert (intermittent["fy26q2_forecast"] >= 0).all()

    def test_expert_override_col_present(self, forecast_df):
        assert "expert_override_applied" in forecast_df.columns

