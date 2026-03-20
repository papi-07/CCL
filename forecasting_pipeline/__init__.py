"""
Cisco Forecast League Phase 1 - End-to-End Forecasting Pipeline.

Modules
-------
data_loader        : Load and clean all data sheets from the Excel data pack.
feature_engineering: Create lag features, rolling statistics, VMS/SCMS signals.
models             : Statistical (ARIMA, Holt-Winters) and ML (LightGBM, XGBoost,
                     Random Forest) model wrappers.
backtesting        : Rolling-origin backtesting with the competition accuracy metric.
ensemble           : Validation-optimised weighted ensemble.
postprocessing     : Non-negative clipping, outlier smoothing, rounding.
forecast           : Main orchestrator – trains models and produces FY26 Q2 forecasts.
"""
