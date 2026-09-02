# Exploratory Input Ablation and Paired Uncertainty Analysis

## Scope

This is a post-hoc exploratory analysis conducted after the main test results had been observed. It does not reselect the primary model and must not be described as preregistered confirmatory evidence. All dates use UTC.

The existing full-input ordinary LSTM was reused as the control. Two fixed three-seed ablations used the same optimizer, early-stopping protocol, data split, clipping, and seeds:

- **History-only:** historical power, UTC calendar features, and zone embedding; all future NWP removed.
- **NWP-only:** future NWP, UTC calendar features, and zone embedding; no historical encoder or history-derived hidden state.

## Main findings

| Model | Test MAE | Test RMSE |
|---|---:|---:|
| XGBoost | 0.031186 | 0.068945 |
| Full-input ordinary LSTM | 0.032660 | 0.068195 |
| Attention-LSTM | 0.033690 | 0.069177 |
| History-only LSTM | 0.043347 | 0.088319 |
| NWP-only LSTM | 0.034300 | 0.072061 |

Removing NWP increased ordinary-LSTM test MAE/RMSE by `32.72%/29.51%`. Removing historical power increased them by `5.02%/5.67%`. Under this fixed model and training protocol, both input sources provide complementary predictive information, with forecast-origin-available NWP contributing more. This is a predictive-information result, not a physical causal claim.

## Paired moving-block bootstrap

The bootstrap resampled consecutive UTC forecast-origin date blocks while retaining all three zones and all 24 horizons. Each replicate recomputed each seed's full-window metric before averaging across seeds. The primary analysis used 7-day blocks, 2,000 replicates, and seed `20260901`; 3-day and 14-day blocks were sensitivity checks.

Differences are candidate minus reference:

| Comparison | Metric difference | 95% interval |
|---|---:|---:|
| Ordinary LSTM − XGBoost | MAE +0.001474 | [0.000549, 0.002393] |
| Ordinary LSTM − XGBoost | RMSE −0.000749 | [−0.002865, 0.001653] |
| Attention-LSTM − ordinary LSTM | MAE +0.001030 | [0.000604, 0.001472] |
| Attention-LSTM − ordinary LSTM | RMSE +0.000982 | [0.000365, 0.001725] |
| History-only − ordinary LSTM | RMSE +0.020124 | [0.010303, 0.026353] |
| NWP-only − ordinary LSTM | RMSE +0.003866 | [0.001932, 0.006037] |

The evidence supports XGBoost's lower MAE. Although ordinary LSTM has the lowest RMSE point estimate, its paired interval against XGBoost crosses zero, so the available test window does not establish a stable RMSE advantage. Attention-LSTM is worse than ordinary LSTM on both test metrics. An interval crossing zero does not establish equivalence.

High-power and ramp-period results are descriptive subgroup diagnostics only. They do not show that a model can identify these conditions in advance or reduce dispatch costs.
