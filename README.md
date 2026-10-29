# Foreign-exchange-forecast

Daily forecasting of three currency pairs — **GBP/USD, USD/EUR and USD/JPY** — with machine learning. The project is an alternative-data reproduction of the CS229 paper *Foreign Currency Exchange Rate Forecast*: the paper used Bloomberg FX and CRSP macro data for 2001–2018, while this project uses a public daily FX dataset and FRED macro series for 2004–2026.

Three questions are studied:

1. **Classification** — will the pair rise or fall over the next observation?
2. **Regression** — what will the next-observation log return be?
3. **Trading** — would acting on the direction forecasts have produced a useful hypothetical result?

## Project structure

```
src/fx_utils.py        shared data preparation: pair construction, features, targets, split
data/
  daily_forex_rates.csv    raw input (one row per date and currency, quoted per EUR)
  processed/               derived datasets written by the notebooks
notebooks/             the pipeline, run in numerical order
results/               result tables written by the notebooks
```

| Notebook | Purpose |
|---|---|
| `01_data_exploration` | Inspect the raw data and build the three pairs by cross-rate |
| `02_feature_engineering` | Build features and next-observation targets; save model datasets |
| `03_classification` | Random Forest, SVM-RBF and MLP direction models; test predictions |
| `04_regression` | Random Forest and SVR return models against a mean-return baseline |
| `05_final_results` | Collect and plot the saved results (no retraining) |
| `06_macro_features` | Download and align FRED macro factors with publication lags |
| `07_trading_backtest` | Long-or-cash backtest of the test-period direction forecasts |
| `08_gru` | GRU recurrent network on 20-observation sequences, compared with the baselines and earlier models |

## Method

**Pairs.** The source quotes every currency per EUR, so the pairs are derived as `GBP_USD = USD / GBP`, `USD_EUR = 1 / USD` and `USD_JPY = JPY / USD`. Only weekdays are kept: the source adds weekend quotes from late 2023 onward, which have near-zero returns and do not exist earlier in the history. This leaves 5,749 dates from 2004-09-27 to 2026-10-09.

**Feature sets** (per pair, using only data up to day *t*):

- **7** — current and previous log return, and moving averages of returns over 5/10/20/60/120 observations.
- **15-paper** — the 7 plus the paper's price-level terms: log price, its square and cube, and log-price moving averages over 5/10/20/60/220 observations.
- **15-stationary** — the 7 plus relative versions of the level terms: distance of the log price from its 5/10/20/60/220 moving averages, 20- and 60-observation volatility, and a 220-observation z-score. These keep a stable range when prices move outside the training range.

**Targets.** The log return from *t* to *t+1*, and its direction (1 if positive).

**Evaluation.** A chronological 80/10/10 train/validation/test split with no shuffling. Models and feature sets are compared on validation only; the chosen configuration is refit on train + validation and scored once on test against a baseline (majority class for classification, training-mean return for regression).

**Macro factors.** Notebook 06 downloads US series from FRED (Federal Reserve Economic Data), shifts each by its publication lag, and forward-fills them onto the FX dates (never backward-filled). `fx_utils.create_macro_features` then makes five model-ready features:

| Feature | FRED series | Transformation |
|---|---|---|
| `NASDAQ_log_return_lag1` | `NASDAQCOM` (daily) | Daily log return, lagged one day because US markets close after the European FX fixing |
| `CPI_mom_pct_change` | `CPIAUCSL` (monthly) | Month-over-month % change, available 1 month + 15 days after the reference month |
| `FEDFUNDS_rate` | `FEDFUNDS` (monthly) | Effective rate level, available 1 month later |
| `NBER_recession` | `USREC` (monthly) | 1 during a US recession, otherwise 0; available 1 month later |
| `Federal_debt_qoq_log_growth` | `GFDEBTN` (quarterly) | Log growth since the previous release, available 4 months after the quarter starts; the raw level is not used because it trends upward and would act as a time stamp |

S&P 500 and Dow Jones are excluded because public FRED only provides their last 10 years; S&P 100 is no longer available. Notebooks 03, 04 and 08 compare each FX feature set with and without these five factors (`7+macro`, `15-paper+macro`, `15-stationary+macro`), and Notebook 05 summarises the difference.

**Backtest.** Hold the pair when the Random Forest predicts a rise, otherwise hold cash; 2 bps cost per position change; compared with buy-and-hold and cash over the test period.

## Results summary

From the current files in `results/`. Each model uses the feature set chosen on validation (Random Forest and SVM: 15-paper; SVR: 15-paper; GRU: 15-stationary).

| Test set | Majority baseline | Random Forest | SVM-RBF | GRU | SVR R² | GRU R² |
|---|---|---|---|---|---|---|
| GBP/USD | 50.5% | 52.4% | 51.2% | 45.2% | +0.0106 | +0.0008 |
| USD/EUR | 47.6% | 52.6% | 51.0% | 49.9% | -0.0224 | -0.0044 |
| USD/JPY | 55.7% | 51.2% | 53.5% | 52.6% | -0.0794 | -0.0073 |

Accuracy columns are next-day direction; R² columns are next-day return regression (0 = no better than predicting the mean return).

- **Classification:** validation selected the 15-paper features for both Random Forest and SVM. Test accuracy is 51–54% with balanced accuracy of about 0.48–0.53, beating the majority baseline for GBP/USD and USD/EUR but not USD/JPY. This is a weak directional signal at best.
- **Regression:** every configuration's R² is close to zero or negative; no model clearly beats predicting the mean return. The 15-paper features make Random Forest unstable when prices leave the training range, which the 15-stationary set avoids.
- **GRU:** validation selected the 15-stationary features for both tasks, and early stopping often chose 1–3 epochs, meaning the network began fitting noise almost immediately. Test accuracy is 45–53%, below the majority baseline for GBP/USD and USD/JPY; test R² is within ±0.01 of zero. The sequence model finds no more signal than the single-row models.
- **Macro factors:** adding them changed mean validation balanced accuracy by between −0.018 and +0.012 and made validation RMSE slightly worse for most regression models. Validation never selected a `+macro` feature set for any model, so the test results above are unchanged by them. At a daily horizon, slow-moving US-only macro data adds no detectable signal.
- **Backtest:** the strategy beat buy-and-hold for all three pairs over the ~2-year test period, but this is a single short window with weak underlying accuracy, so it is not evidence of a reliable edge.

## Running

Requires Python with `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `torch` and `jupyter`. Run the notebooks in order from the `notebooks/` folder: 01 → 02 → 06 → 03 → 04 → 08 → 05 → 07. Notebook 06 needs internet access to download the FRED series.

## Limitations

- Different data source and period from the paper, so comparisons with its figures are indicative only.
- Macro factors use calendar publication lags rather than true point-in-time vintages. NBER recession dates in particular are announced many months after the fact, so the recession flag uses information that was not available in real time (it is 0 throughout validation and test, so this affects training only).
- The macro factors are US-only; exchange rates respond to differences between two economies (e.g. US vs UK interest rates), which these features do not capture.
- The backtest is illustrative: one pair at a time, no shorting, leverage, spreads beyond the modelled cost, or market impact.
- GRU results come from a single random seed and fixed settings (20-observation window, 32 hidden units), so they carry extra run-to-run variance.
