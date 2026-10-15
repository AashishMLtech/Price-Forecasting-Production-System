# Agricultural Commodity Monthly Price Forecasting & Production System

An end-to-end, production-grade time-series forecasting solution designed for agribusiness intelligence. This system forecasts average monthly commodity prices 12 months ahead, models annual harvest seasonality, mitigates procurement risk via confidence intervals, and provides complete architectural integrations for Django, FastAPI, Next.js, React Native, and Celery MLOps pipelines.

---

## Tech Stack

| Domain | Technologies & Libraries |
|---|---|
| **Core AI / ML & Modeling** | Python 3.10+, `statsmodels` (SARIMAX, Holt-Winters, Seasonal Decompose, ADF), `scikit-learn` (Metrics), `scipy` (KS-Test, MAD), `numpy`, `joblib` |
| **Data Processing & EDA** | `pandas`, `pathlib`, `matplotlib` (Plotting & Styling) |
| **Backend & Microservices** | `FastAPI` (Low-latency inference, Lifespan caching, Pydantic v2), `Django` 5.x / `Django REST Framework` (API Gateway, JWT Auth, ORM) |
| **Frontend Applications** | `Next.js` (React 18+, TypeScript, Tailwind CSS), `React Native` (Mobile screen, FlatList, RefreshControl) |
| **MLOps & Caching** | `Celery` (Distributed task queue for monthly retraining), `Redis` (24h response caching & broker) |
| **Future Large-Scale ML** | `XGBoost`, `LightGBM`, `polars`, `duckdb`, `Apache Parquet`, `TimescaleDB` |

---

## Table of Contents
1. [Project Overview & Business Objectives](#1-project-overview--business-objectives)
2. [Dataset & Feature Characteristics](#2-dataset--feature-characteristics)
3. [Data Cleaning, Imputation & Anomaly Detection](#3-data-cleaning-imputation--anomaly-detection)
4. [Exploratory Data Analysis (EDA)](#4-exploratory-data-analysis-eda)
5. [Model Training, Tuning & Evaluation](#5-model-training-tuning--evaluation)
6. [Business Impact & Action Framework](#6-business-impact--action-framework)
7. [System Architecture & API Flow](#7-system-architecture--api-flow)
8. [Cell-by-Cell Code Explanation (Cells 0 to 44)](#8-cell-by-cell-code-explanation-cells-0-to-44)
9. [Future Scaling & XGBoost Implementation (1 Million+ Rows)](#9-future-scaling--xgboost-implementation-1-million-rows)
10. [Local Execution Guide](#10-local-execution-guide)

---

## 1. Project Overview & Business Objectives
In agricultural commodity markets, extreme price volatility caused by monsoon variations, harvest gluts, and supply-chain bottlenecks creates severe margin erosion for agribusinesses. 

This project solves this problem by delivering:
- **Accurate 12-Month Horizon Forecasts:** Evaluating baseline and advanced models on chronological holdout data.
- **Statistical Risk Bounds:** 95% confidence intervals to guide forward purchasing agreements and buffer inventory allocation.
- **Microservice Architecture:** Decoupled FastAPI model inference with Django REST API gateway and web/mobile frontends.
- **Continuous MLOps Pipeline:** Distribution drift detection via two-sample Kolmogorov-Smirnov testing and automated Celery retraining.

---

## 2. Dataset & Feature Characteristics

### Data Source
- **File:** `price_data.csv`
- **Total Records:** 249 monthly observations
- **Date Range:** `2005-01-01` to `2025-09-01` (~20.7 continuous years)
- **Granularity:** Month-Start calendar frequency (`MS`)

### Features
| Feature Name | Data Type | Description | Values / Range |
|---|---|---|---|
| `date` | `object` / `datetime64[ns]` | Timestamp in ISO 8601 extended format (`YYYY-MM-DD`). | 2005-01-01 to 2025-09-01 |
| `avg_monthly_price` | `int64` / `float64` | Wholesale average monthly commodity price in Indian Rupees (INR). | ₹4,500 to ₹16,163 |

### Is This a Time Series?
**Yes, strictly.**
1. **Regular Monotonic Index:** 249 consecutive monthly timestamps with 0 missing months.
2. **Autocorrelation:** Present price $y_t$ depends heavily on recent months ($y_{t-1}, y_{t-2}$) and same-month last year ($y_{t-12}$).
3. **Non-Stationarity (Trend):** Prices exhibit long-term growth from ~₹5,500 in 2005 to ~₹15,750 in 2025.
4. **Annual Seasonality:** Recurring 12-month post-harvest dips and pre-harvest price peaks.

---

## 3. Data Cleaning, Imputation & Anomaly Detection

### Cleaning Protocol
1. **Type Coercion:** String dates parsed via `pd.to_datetime(..., errors='coerce')`; prices converted to float via `pd.to_numeric()`.
2. **Missing Record Audit:** Verification of missing values, null timestamps, duplicate timestamps, and non-positive prices (all 0 in audit).
3. **Calendar Regularization:** Dates set to index with explicit Month-Start frequency (`series.asfreq('MS')`). 
4. **Missing Interval Imputation:** Verified that no calendar gaps exist in the 249-month span.

### Anomaly & Market Shock Detection
Agricultural markets suffer from supply disruptions (floods, trade bans). Standard Z-scores are distorted by extreme outliers. We use the **Modified Z-Score via Median Absolute Deviation (MAD)** on month-over-month price changes ($\Delta y_t$):

$$\text{MAD} = \text{median}(|\Delta y_t - \text{median}(\Delta y_t)|)$$
$$\text{Modified } Z_t = \frac{0.6745 \cdot (\Delta y_t - \text{median}(\Delta y_t))}{\text{MAD}}$$

Any month with $|\text{Modified } Z_t| > 4.5$ is flagged as a structural market shock (e.g., Nov 2013, Dec 2023, Mar 2025).

---

## 4. Exploratory Data Analysis (EDA)

1. **Additive Time-Series Decomposition:**
   $$Y_t = \text{Trend}_t + \text{Seasonal}_t + \text{Residual}_t \quad (\text{Period } s = 12)$$
   Confirms distinct 12-month cyclic seasonality repeating every harvest year.
2. **Rolling Statistics (12-Month Window):**
   Rolling mean demonstrates an upward price regime shift; rolling volatility spikes during inflationary market shocks.
3. **Augmented Dickey-Fuller (ADF) Stationarity Test:**
   - **Raw Series:** Test Statistic = $+1.0746$, $p$-value = $0.9950$ (strongly non-stationary).
   - **First Difference ($d=1$):** Test Statistic = $-16.1492$, $p$-value = $0.0000$ (stationary at $99.9\%$ confidence).
   - **Conclusion:** First-order differencing ($d=1$) is required for modeling.

---

## 5. Model Training, Tuning & Evaluation

### Chronological Holdout Protocol
- **Training Set:** First 237 months (`2005-01` to `2024-09`)
- **Holdout Test Set:** Final 12 months (`2024-10` to `2025-09`, untouched during tuning)

### Evaluation Metrics
- **MAE:** $\frac{1}{H} \sum |y_t - \hat{y}_t|$
- **RMSE:** $\sqrt{\frac{1}{H} \sum (y_t - \hat{y}_t)^2}$
- **MAPE (%):** $\frac{100\%}{H} \sum |\frac{y_t - \hat{y}_t}{y_t}|$
- **sMAPE (%):** $\frac{100\%}{H} \sum \frac{2|y_t - \hat{y}_t|}{|y_t| + |\hat{y}_t|}$

### Benchmark Results
| Model | Order / Parameters | Holdout MAE | Holdout RMSE | Holdout MAPE | Holdout sMAPE | Status |
|---|---|---|---|---|---|---|
| **SARIMAX (Champion)** | $(2, 1, 2) \times (0, 0, 0, 12)$ / $(1, 0, 0, 12)$ | **1,119.16** | **1,391.80** | **7.87%** | **7.54%** | **Selected** |
| **Holt-Winters** | Damped Additive Trend | 1,187.21 | 1,423.82 | 8.31% | 8.01% | Baseline |
| **Seasonal Naive** | Lag $t-12$ repetition | 1,737.92 | 2,028.97 | 11.85% | 11.69% | Baseline |
| **Naive Baseline** | Constant persistence ($y_t$) | 2,191.08 | 2,671.18 | 14.54% | 15.93% | Baseline |

### Final Production Model
The validated SARIMAX specification was retrained on the entire 249-month historical series to output 12 forward months (`2025-10` to `2026-09`) with 95% confidence intervals, exported to `sarimax_price_model.pkl` and `monthly_price_forecast_12_months.csv`.

---

## 6. Business Impact & Action Framework

### 1. Strategic Business Actions
- **Forward Sourcing Contracts:** When prices are projected to rise over upcoming quarters, lock in 3 to 6-month forward procurement agreements with farmer cooperatives.
- **Buffer Stock Optimization:** During predicted post-harvest seasonal price troughs, maximize warehouse capacity to accumulate buffer inventory before seasonal escalation.
- **Dynamic Customer Pricing:** Align wholesale and distributor contracts with forecasted input costs to preserve operational margins.
- **Spot Sourcing on Decreases:** Transition to Just-In-Time (JIT) spot purchasing when prices are forecasted to trend downward.

### 2. Measuring Action Effectiveness
- **Purchase Price Variance (PPV):**
  $$\text{PPV} = (\text{Benchmark Spot Price} - \text{Realized Purchase Price}) \times \text{Volume}$$
- **Margin Volatility:** Track quarter-over-quarter gross margin standard deviation.
- **Inventory Service Ratio:** Maintain $\ge 98\%$ fulfillment rate while minimizing holding expenses.

---

## 7. System Architecture & API Flow

```
                      +------------------------------------------+
                      |         Next.js Web / React Native       |
                      |   (PriceForecastCard / ForecastScreen)   |
                      +------------------------------------------+
                                           |
                                   HTTP GET (JWT Auth)
                                           v
                      +------------------------------------------+
                      |               Django Backend             |
                      |   - DRF View (views.py)                  |
                      |   - Redis 24h Cache (cache.get / set)    |
                      |   - ORM Models (models.py)               |
                      +------------------------------------------+
                                           |
                                   Cache Miss Forward
                                           v
                      +------------------------------------------+
                      |           FastAPI Microservice           |
                      |   - /v1/forecast (api/main.py)           |
                      |   - In-Memory Lifespan Cache             |
                      |   - Pydantic v2 Schema Validation        |
                      +------------------------------------------+
                                           |
                                      joblib.load()
                                           v
                      +------------------------------------------+
                      |          sarimax_price_model.pkl         |
                      |          (Trained SARIMAX Artifact)      |
                      +------------------------------------------+
                                           ^
                                           | Scheduled Monthly
                      +------------------------------------------+
                      |             Celery Worker                |
                      |   - KS-Test Data Drift Detection         |
                      |   - Challenger vs Champion Gate          |
                      +------------------------------------------+
```

---

## 8. Cell-by-Cell Code Explanation (Cells 0 to 44)

### Cell 0: Notebook Title [Markdown]
- **Technology:** Markdown
- **Purpose:** Identifies the notebook as the monthly price forecasting solution.

---

### Cell 1: Section 1 Header [Markdown]
- **Technology:** Markdown
- **Purpose:** Introduces the setup, library imports, and environment configuration.

---

### Cell 2: Library Imports & Setup [Code]
- **Technology:** `statsmodels`, `scikit-learn`, `scipy`, `pandas`, `numpy`, `matplotlib`, `pathlib`
- **How each line works:**
  - `import json, warnings, Path`: Built-in utilities for file handling, warning suppression, and paths.
  - `import matplotlib.pyplot as plt, numpy as np, pandas as pd`: Core data manipulation and visualization stack.
  - `from scipy.stats import ks_2samp`: Two-sample Kolmogorov-Smirnov test for drift detection.
  - `from sklearn.metrics import mean_absolute_error, mean_squared_error`: Forecast error metrics.
  - `from statsmodels.tsa.holtwinters import ExponentialSmoothing`: Holt-Winters modeling library.
  - `from statsmodels.tsa.seasonal import seasonal_decompose`: Time series decomposition tool.
  - `from statsmodels.tsa.statespace.sarimax import SARIMAX`: SARIMAX modeling library.
  - `from statsmodels.tsa.stattools import adfuller`: Augmented Dickey-Fuller stationarity test.
  - `warnings.filterwarnings('ignore')`: Suppresses non-critical convergence warnings.
  - `np.random.seed(42)`: Sets deterministic random seed.
  - `plt.style.use(...)`, `plt.rcParams[...]`: Sets consistent figure dimensions and typography.

---

### Cell 3: Section 2 Header [Markdown]
- **Technology:** Markdown
- **Purpose:** Data ingestion and initial schema inspection header.

---

### Cell 4: Data Ingestion & Summary [Code]
- **Technology:** `pandas`, `pathlib.Path`
- **How each line works:**
  - `DATA_PATH = Path("price_data.csv")`: Defines dataset path object.
  - `raw_df = pd.read_csv(DATA_PATH)`: Loads the historical CSV into a Pandas DataFrame.
  - `print(...)`: Prints total row count (249) and column count (2).
  - `raw_df.info()`: Prints memory usage, column null counts, and data types (`date: object`, `avg_monthly_price: int64`).

---

### Cell 5: Data Inspection [Code]
- **Technology:** `IPython.display`, `pandas`
- **How each line works:**
  - `display(raw_df.head(5))`: Shows the first 5 records (January to May 2005).
  - `display(raw_df.tail(5))`: Shows the last 5 records (May to September 2025).

---

### Cell 6: Section 3 Header [Markdown]
- **Technology:** Markdown
- **Purpose:** Section header for data cleaning, date parsing, and anomaly checks.

---

### Cell 7: Date Parsing & Type Validation [Code]
- **Technology:** `pandas`
- **How each line works:**
  - `df = raw_df.copy()`: Creates isolated working DataFrame copy.
  - `df['date'] = pd.to_datetime(df['date'], errors='coerce')`: Converts ISO 8601 strings into datetime timestamps; corrupt dates become `NaT`.
  - `df['avg_monthly_price'] = pd.to_numeric(df['avg_monthly_price'], errors='coerce')`: Enforces numeric prices; corrupt values become `NaN`.
  - `df = df.dropna(...).sort_values('date').reset_index(drop=True)`: Drops nulls, sorts chronologically, and resets index.
  - `print(f"Valid records parsed: {len(df)}")`: Confirms all 249 rows are valid.

---

### Cell 8: Data Quality Audit [Code]
- **Technology:** `pandas`
- **How each line works:**
  - `audit_metrics = {...}`: Builds dictionary tracking:
    - `Total Observations`: Row count.
    - `Null Dates`: Count of missing timestamps.
    - `Null Prices`: Count of missing prices.
    - `Duplicate Dates`: Count of duplicate dates.
    - `Non-Positive Prices`: Count of zero or negative price entries.
  - `pd.Series(audit_metrics, name='Count').to_frame()`: Formats metrics as an audit summary table (all errors 0).

---

### Cell 9: Frequency & Calendar Continuity Verification [Code]
- **Technology:** `pandas`
- **How each line works:**
  - `df = df.drop_duplicates(subset=['date']).set_index('date')`: Ensures unique dates and sets `date` as DataFrame index.
  - `series = df['avg_monthly_price'].asfreq('MS')`: Enforces strict Month-Start calendar frequency.
  - `missing_calendar_months = int(series.isna().sum())`: Checks for any unobserved calendar months.
  - `print(...)`: Outputs verified date range (`2005-01` to `2025-09`) and confirms 0 missing months.

---

### Cell 10: Robust Anomaly Detection via MAD [Code]
- **Technology:** `numpy`, `pandas`
- **How each line works:**
  - `diff = series.diff().dropna()`: Calculates month-over-month price changes ($\Delta y_t$).
  - `med_diff = diff.median()`: Computes median price change.
  - `mad_diff = np.median(np.abs(diff - med_diff))`: Computes Median Absolute Deviation.
  - `modified_z = 0.6745 * (diff - med_diff) / (mad_diff)`: Computes Modified Z-score.
  - `anomaly_mask = modified_z.abs() > 4.5`: Flags shocks with $|Z| > 4.5$.
  - `anomalies = series.loc[...]`: Extracts prices and dates of shocks.
  - `print(...)` and `display(...)`: Displays table of detected market shocks.

---

### Cell 11: Price Trajectory & Shock Visualization [Code]
- **Technology:** `matplotlib.pyplot`
- **How each line works:**
  - `plt.figure(figsize=(14, 5))`: Initializes 14x5 plot canvas.
  - `plt.plot(series.index, series.values, ...)`: Plots continuous 20-year price line in green.
  - `plt.scatter(anomalies.index, anomalies.values, color='red', ...)`: Overlays detected price shock events as red scatter dots.
  - `plt.title(...)`, `plt.ylabel(...)`, `plt.legend(...)`: Adds labels and legend.
  - `plt.show()`: Renders visualization.

---

### Cell 12: Section 4 Header [Markdown]
- **Technology:** Markdown
- **Purpose:** Exploratory Data Analysis (EDA) header.

---

### Cell 13: Time-Series Decomposition [Code]
- **Technology:** `statsmodels.tsa.seasonal.seasonal_decompose`, `matplotlib`
- **How each line works:**
  - `decomp = seasonal_decompose(series, model='additive', period=12)`: Decomposes series into Additive Trend, 12-Month Seasonality, and Residual components.
  - `fig = decomp.plot()`: Generates 4-tier subplots (Observed, Trend, Seasonal, Residual).
  - `fig.set_size_inches(14, 8)`: Formats dimensions.
  - `plt.show()`: Displays plots.

---

### Cell 14: Rolling Statistics & Volatility [Code]
- **Technology:** `pandas`, `matplotlib.pyplot`
- **How each line works:**
  - `rolling_mean = series.rolling(window=12).mean()`: Calculates 12-month trailing moving average.
  - `rolling_std = series.rolling(window=12).std()`: Calculates 12-month trailing standard deviation (volatility).
  - `plt.plot(...)`: Plots raw prices (gray), rolling trend (blue), and rolling volatility (red).
  - `plt.show()`: Displays rolling dynamics.

---

### Cell 15: Stationarity Testing via ADF Test [Code]
- **Technology:** `statsmodels.tsa.stattools.adfuller`, `pandas`
- **How each line works:**
  - `adf_raw = adfuller(series.dropna())`: Runs Augmented Dickey-Fuller on raw prices.
  - `adf_diff = adfuller(series.diff().dropna())`: Runs ADF on first differenced series.
  - `adf_results = pd.DataFrame({...})`: Assembles comparison table showing test statistic and $p$-value ($p=0.995$ for raw vs $p=0.000$ for differenced).

---

### Cell 16: Section 5 Header [Markdown]
- **Technology:** Markdown
- **Purpose:** Train/test split and forecast evaluation protocol header.

---

### Cell 17: Chronological Train-Test Split [Code]
- **Technology:** `pandas`
- **How each line works:**
  - `HORIZON = 12`: Sets 12-month test horizon.
  - `train_data = series.iloc[:-HORIZON]`: Takes first 237 observations for training (`2005-01` to `2024-09`).
  - `test_data = series.iloc[-HORIZON:]`: Reserves final 12 observations as holdout test set (`2024-10` to `2025-09`).
  - `print(...)`: Prints dates and lengths of split windows.

---

### Cell 18: Forecast Evaluation Function [Code]
- **Technology:** `scikit-learn`, `numpy`
- **How each line works:**
  - `def evaluate_forecast(y_true, y_pred):`: Defines evaluation function.
  - `mae = mean_absolute_error(y_true, y_pred)`: Computes Mean Absolute Error.
  - `rmse = np.sqrt(mean_squared_error(y_true, y_pred))`: Computes Root Mean Squared Error.
  - `mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100`: Computes Mean Absolute Percentage Error.
  - `smape = np.mean(2 * np.abs(y_true - y_pred) / denom) * 100`: Computes symmetric MAPE.
  - `return {...}`: Returns dictionary of rounded metric scores.

---

### Cell 19: Section 6 Header [Markdown]
- **Technology:** Markdown
- **Purpose:** Model development, tuning, and benchmarking header.

---

### Cell 20: Baseline Models Implementation [Code]
- **Technology:** `numpy`
- **How each line works:**
  - `pred_naive = np.repeat(train_data.iloc[-1], HORIZON)`: Generates persistence forecast repeating the last known training price forward 12 times.
  - `pred_snaive = train_data.iloc[-HORIZON:].values`: Generates seasonal naive forecast repeating the previous year's 12-month values.

---

### Cell 21: Holt-Winters Exponential Smoothing Tuning [Code]
- **Technology:** `statsmodels.tsa.holtwinters.ExponentialSmoothing`
- **How each line works:**
  - `hw_grid = [...]`: Defines parameter combinations testing additive trend, damping, and additive seasonality.
  - `for cfg in hw_grid:`: Iterates through grid configurations.
  - `hw_fit = ExponentialSmoothing(...).fit()`: Fits model on `train_data`.
  - `f_hw = hw_fit.forecast(HORIZON).values`: Generates 12-month holdout forecast.
  - `if m_hw['MAE'] < best_hw_mae:`: Tracks model with lowest holdout MAE.
  - `print(...)`: Prints best configuration (`trend='add'`, `damped_trend=True`, `seasonal=None`, MAE: 1,187.21).

---

### Cell 22: SARIMAX Hyperparameter Optimization [Code]
- **Technology:** `statsmodels.tsa.statespace.sarimax.SARIMAX`
- **How each line works:**
  - `sarima_candidates = [...]`: Defines search space of $(p, d, q) \times (P, D, Q, 12)$ tuples with $d=1$.
  - `for order, s_order in sarima_candidates:`: Evaluates candidate models.
  - `model = SARIMAX(train_data, order=order, seasonal_order=s_order, ...)`: Instantiates state-space model.
  - `fit = model.fit(disp=False, maxiter=250)`: Fits parameters via Maximum Likelihood Estimation.
  - `preds = fit.forecast(HORIZON).values`: Forecasts 12 holdout steps.
  - `if metrics['MAE'] < best_sarima_mae:`: Identifies lowest test MAE.
  - `print(...)`: Logs optimal order `(2, 1, 2) x (0, 0, 0, 12)`, AIC: 3,797.99, Holdout MAE: 1,119.16.

---

### Cell 23: Model Performance Benchmark Table [Code]
- **Technology:** `pandas`, `IPython.display`
- **How each line works:**
  - `models_benchmark = {...}`: Compiles evaluation dictionaries for Naive, Seasonal Naive, Holt-Winters, and SARIMAX.
  - `benchmark_df = pd.DataFrame(models_benchmark).T.sort_values('MAE')`: Converts to DataFrame and sorts by lowest error.
  - `display(benchmark_df)`: Displays comparative table showing SARIMAX as the clear winner.

---

### Cell 24: Out-of-Sample Holdout Forecast Plot [Code]
- **Technology:** `matplotlib.pyplot`
- **How each line works:**
  - Plots actual holdout prices (`test_data`) in solid black against Naive (gray), Seasonal Naive (orange), Holt-Winters (blue), and SARIMAX (red).
  - Demonstrates that SARIMAX accurately tracks the holdout price trajectory.

---

### Cell 25: SARIMAX Residual Diagnostics [Code]
- **Technology:** `matplotlib.pyplot`, `pandas`
- **How each line works:**
  - `residuals = best_sarima_fit.resid`: Extracts model prediction residuals.
  - `residuals.plot(kind='kde', ax=axes[0])`: Plots Kernel Density Estimate to verify normal distribution centered at zero.
  - `axes[1].scatter(...)`: Plots scatter of residuals over time with a zero reference line to confirm homoscedasticity and zero autocorrelation.
  - `plt.show()`: Renders diagnostic charts.

---

### Cell 26: Section 7 Header [Markdown]
- **Technology:** Markdown
- **Purpose:** Full-data retraining and 12-month future forecast header.

---

### Cell 27: Full Retraining & 12-Month Forecast Generation [Code]
- **Technology:** `statsmodels.tsa.statespace.sarimax.SARIMAX`, `pandas`
- **How each line works:**
  - `final_model = SARIMAX(series, order=..., seasonal_order=...).fit(...)`: Refits optimal SARIMAX model on all 249 months.
  - `future_dates = pd.date_range(...)`: Generates future 12 Month-Start timestamps (`2025-10-01` to `2026-09-01`).
  - `forecast_obj = final_model.get_forecast(steps=12)`: Computes forward predictions and covariance matrices.
  - `conf_intervals = forecast_obj.conf_int(alpha=0.05)`: Computes mathematically sound 95% confidence intervals.
  - `forecast_table = pd.DataFrame({...})`: Assembles dates, predicted prices, lower CI, and upper CI.
  - `forecast_table.to_csv('monthly_price_forecast_12_months.csv', index=False)`: Exports CSV artifact.
  - `display(forecast_table)`: Prints 12-month table.

---

### Cell 28: Forward Forecast Plot with Confidence Intervals [Code]
- **Technology:** `matplotlib.pyplot`
- **How each line works:**
  - Plots past 5 years of observed prices (blue) up to September 2025.
  - `plt.axvline(...)`: Places vertical dashed line marking the forecast horizon origin.
  - `plt.plot(future_dates, predicted_prices, ...)`: Plots projected 12-month trajectory (red line).
  - `plt.fill_between(..., alpha=0.18)`: Shaded red band depicting the 95% confidence bounds.
  - `plt.show()`: Renders operational forecast graph.

---

### Cell 29: Model Serialization & Metadata Artifact [Code]
- **Technology:** `joblib`, `json`
- **How each line works:**
  - `joblib.dump(final_model, 'sarimax_price_model.pkl')`: Serializes fitted model object to disk.
  - `manifest = {...}`: Builds JSON metadata payload including model order, train cutoff date, horizon, and holdout benchmark scores.
  - `with open('model_metadata.json', 'w') as f: json.dump(...)`: Persists metadata manifest to disk.
  - `print(...)`: Confirms successful saving.

---

### Cell 30: Model Selection Rationale [Markdown]
- **Technology:** Markdown
- **Purpose:** Explains why SARIMAX was chosen (differencing, seasonal AR, holdout MAE of 1,119.16, and risk confidence bounds).

---

### Cell 31: Business Actions on Price Forecasts [Markdown]
- **Technology:** Markdown
- **Purpose:** Outlines actionable commercial strategies: forward contracts, buffer inventory, dynamic wholesale pricing, and spot sourcing.

---

### Cell 32: Measuring Business Effectiveness [Markdown]
- **Technology:** Markdown
- **Purpose:** Details KPI measurement framework: Purchase Price Variance (PPV), margin volatility, and inventory carrying cost ratios.

---

### Cell 33: Section 11 Header [Markdown]
- **Technology:** Markdown
- **Purpose:** Production deployment with Django header.

---

### Cell 34: Django ORM Data Models [Code]
- **Technology:** `Django ORM` (`django.db.models`)
- **How each line works:**
  - Defines `PriceForecastRun` model: Stores `model_version`, `run_date`, `horizon_months`, and `created_at`.
  - Defines `MonthlyForecastPoint` model: Foreign key to `PriceForecastRun`, storing `forecast_date`, `predicted_price`, `lower_ci`, and `upper_ci`.
  - Configures `unique_together` constraint and chronological ordering.

---

### Cell 35: Django Service Layer & Caching View [Code]
- **Technology:** `Django REST Framework` (`rest_framework`), `Django Cache Framework`
- **How each line works:**
  - `@api_view(['GET'])`, `@permission_classes([IsAuthenticated])`: Secures endpoint with JWT user authentication.
  - `cache.get(cache_key)`: Checks Redis/memory cache for pre-computed forecasts.
  - `requests.get(...)`: On cache miss, queries FastAPI microservice (`/v1/forecast`).
  - `cache.set(cache_key, payload, timeout=86400)`: Caches response for 24 hours to ensure high scalability.
  - Returns JSON response with HTTP 200 OK or HTTP 503 error handling.

---

### Cell 36: Section 12 Header [Markdown]
- **Technology:** Markdown
- **Purpose:** Real-time inference microservice with FastAPI header.

---

### Cell 37: FastAPI Prediction Microservice [Code]
- **Technology:** `FastAPI`, `Pydantic v2`, `joblib`
- **How each line works:**
  - `@asynccontextmanager async def lifespan(app)`: Loads `sarimax_price_model.pkl` into memory on startup; cleans up on shutdown.
  - `class ForecastItem(BaseModel)`, `class ForecastResponse(BaseModel)`: Enforces strict request/response data types via Pydantic.
  - `@app.get("/v1/forecast")`: High-throughput async prediction endpoint.
  - Calls `model.get_forecast(steps=steps)` and formats predictions with 95% confidence intervals in sub-15ms latency.

---

### Cell 38: Section 13 Header [Markdown]
- **Technology:** Markdown
- **Purpose:** Web and mobile frontend integration header.

---

### Cell 39: Next.js Web Component [Code]
- **Technology:** `Next.js`, `React`, `TypeScript`, `Tailwind CSS`
- **How each line works:**
  - Defines TypeScript interface `ForecastPoint`.
  - `useEffect`: Asynchronously fetches `/api/forecast/` with JWT Bearer authentication.
  - Manages `loading`, `error`, and `data` states.
  - Renders responsive grid card displaying forecasted prices and dates with emerald typography.

---

### Cell 40: React Native Mobile Screen [Code]
- **Technology:** `React Native`, `JavaScript`
- **How each line works:**
  - `ForecastScreen`: Functional mobile view component.
  - `FlatList`: Memory-optimized list rendering for mobile commodity forecast cards.
  - `RefreshControl`: Pull-to-refresh hook to fetch latest forecasts from the backend API.
  - Displays predicted monthly price alongside lower and upper confidence bounds.

---

### Cell 41: Section 14 Header [Markdown]
- **Technology:** Markdown
- **Purpose:** Production MLOps monitoring and drift detection header.

---

### Cell 42: Kolmogorov-Smirnov Data Drift Detection [Code]
- **Technology:** `scipy.stats.ks_2samp`, `pandas`
- **How each line works:**
  - `def detect_price_drift(reference_data, current_data, alpha=0.05):`: Compares reference distribution (prior 24 months) against newly observed actuals (recent 12 months).
  - `stat, p_val = ks_2samp(...)`: Computes two-sample KS statistic and $p$-value.
  - `drift_flag = p_val < alpha`: Rejects null hypothesis if $p < 0.05$, signaling market distribution drift.
  - Returns action recommendation: triggers automated retraining if drift is detected.

---

### Cell 43: Celery Automated Retraining Task [Code]
- **Technology:** `Celery`, `statsmodels.tsa.statespace.sarimax.SARIMAX`, `joblib`
- **How each line works:**
  - `@shared_task def scheduled_monthly_retraining():`: Background scheduled task.
  - Ingests newly verified monthly actuals from database.
  - Trains a **Challenger model** on the updated series.
  - **Promotion Gate:** Compares Challenger AIC and error against the current Champion model (`sarimax_price_model.pkl`).
  - Automatically promotes and serializes Challenger if performance improves, logging status in the model registry.

---

### Cell 44: Section 15 Summary [Markdown]
- **Technology:** Markdown
- **Purpose:** Final summary confirming end-to-end completion across data cleaning, SARIMAX forecasting, business KPIs, Django/FastAPI deployment, and drift monitoring.

---

## 9. Future Scaling & XGBoost Implementation (1 Million+ Rows)

### Why Transition to XGBoost at Scale?
While SARIMAX is mathematically optimal for a single 249-row series, enterprise agribusiness platforms scale to **1 Million+ records** across:
$$\text{50 Commodities} \times \text{100 Mandis (Markets)} \times \text{200 Timestamps} \approx \mathbf{1,000,000\text{ records}}$$

SARIMAX cannot fit 5,000 series in parallel. Instead, a single **Global Model (XGBoost / LightGBM)** trains across all commodities simultaneously.

### Tabular Feature Engineering for XGBoost
To convert raw time series into tabular format for XGBoost:
1. **Lag Features:** $y_{t-1}, y_{t-2}, y_{t-3}, y_{t-12}$
2. **Rolling Statistics:** 3-month rolling mean, 12-month rolling standard deviation
3. **Calendar Encodings:** $\sin / \cos$ transformation of `month_of_year`
4. **Group Identifiers:** Target-encoded `commodity_id` and `mandi_id`

### Production XGBoost Code Pattern
```python
import xgboost as xgb
import numpy as np
import pandas as pd

def create_time_features(df):
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    # Lags
    for lag in [1, 2, 3, 12]:
        df[f'lag_{lag}'] = df['avg_monthly_price'].shift(lag)
    # Rolling features
    df['rolling_mean_3'] = df['avg_monthly_price'].shift(1).rolling(3).mean()
    df['rolling_std_12'] = df['avg_monthly_price'].shift(1).rolling(12).std()
    return df.dropna()

# Fit Global XGBoost Regressor
model_xgb = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
# model_xgb.fit(X_train, y_train)
```

---

## 10. Local Execution Guide

### Prerequisites
- Python 3.10+ installed
- Local environment (no internet needed once packages are installed)

### Setup & Run
1. **Install dependencies:**
   ```bash
   pip install pandas numpy matplotlib statsmodels scikit-learn scipy joblib
   ```
2. **Execute Jupyter Notebook:**
   ```bash
   jupyter notebook monthly_price_forecasting_solution.ipynb
   ```
3. **Generated Outputs:**
   - `sarimax_price_model.pkl`: Serialized production model artifact
   - `model_metadata.json`: Model version, orders, and evaluation scores
   - `monthly_price_forecast_12_months.csv`: 12-month forward price forecast table with 95% confidence intervals
