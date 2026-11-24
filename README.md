# Does What the Fed Says Move Tomorrow's Yield?
AI for Finance – Final Capstone (Type A: free application)

**Question.** Can machine-learning models applied to the text of Federal Reserve speeches forecast the
next-day change in the US 10-year Treasury yield well enough to trade profitably after costs?

**Answer (short).** No. On 1,314 out-of-sample speech days (2012–2023) the gross edge is statistically zero for
every text model, and all strategies lose money after a 0.25 bp/side cost. See the manuscript.

## Reproduce everything
```bash
pip install -r requirements.txt
python download_data.py                     # FRED DGS10 + CBS speeches (Fed only) -> data/
python speeches_yields_capstone.py --speeches data/fed_speeches.csv --yields data/DGS10.csv --out outputs
# ~12 min on one CPU core. Or simply:  ./run_all.sh
python speeches_yields_capstone.py --synthetic --out outputs_synthetic   # 30 s smoke test (fake data)
python speeches_yields_capstone.py --eval-only --out outputs             # rebuild tables from saved forecasts (seconds)
```
Seed 42. Tested with Python 3.13, numpy 2.5, pandas 3.0, scikit-learn 1.9, scipy 1.18, matplotlib 3.11.

## Data
| Source | What | Link |
|---|---|---|
| CBS dataset (Campiglio, Deyris, Romelli, Scalisi, 2025, *European Economic Review*) | 35,487 central-bank speeches; we keep the 6,607 with Country == "USA" (1986–2023) | https://cbspeeches.com |
| FRED DGS10 | Daily 10-year Treasury constant-maturity yield | https://fred.stlouisfed.org/series/DGS10 |

`data/fed_speeches.csv` (124 MB) is rebuilt by `download_data.py` and is git-ignored. The CBS data are free for
academic / personal non-commercial use.

## Where each result in the manuscript comes from
| Manuscript item | File in `outputs/` |
|---|---|
| Table 1 (main results) | `table_main.csv` |
| Table 2 (sub-periods) / Figure 2 | `table_subperiod_sharpe.csv` / `fig_subperiod_sharpe.png` |
| Table 3 (costs) | `table_cost_sensitivity_sharpe.csv` |
| Figure 1 (cumulative net return) | `fig_cumulative_pnl.png` |
| Bootstrap confidence intervals | `table_bootstrap_ci.csv` |
| Delayed-execution check | `table_timing_check.csv` |
| Speaker type / volatility regime / signal strength | `table_by_speaker_type_sharpe.csv`, `table_by_vol_regime_sharpe.csv`, `table_by_signal_strength.csv` |
| Top words (interpretability) | `table_top_words.csv` |
| Raw out-of-sample forecasts | `oos_predictions.csv` |
| Console log of the full run | `run_log.txt` |

## Method in one paragraph
One observation per speech day. Features use information up to the close of day *t*; target is the yield change
from *t* to *t+1* (bp). Annual expanding-window walk-forward from 2012 with a 7-day embargo; TF-IDF/SVD/winsorising
are fitted on training data only; hyper-parameters tuned by time-series CV inside each training set. Trading rule:
sign of forecast, one-day hold, cost 0.25 bp per side, duration 8.

## Changes made to the starter script
1. **Bug fix** – the speech filter looked for "united states"/"federal reserve" but the data codes the Fed as `USA`, so every speech was silently dropped.
2. **Bug fix** – zero-change days are excluded from hit rate and its binomial test.
3. New robustness / error-analysis tables, new sub-periods, gross-and-net bootstrap, `--eval-only`, faster tone scoring.

## Generative-AI use
Claude (Anthropic) was used for code review/debugging, extra analysis code, and drafting documents; see the manuscript's acknowledgement.
