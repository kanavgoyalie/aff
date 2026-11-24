"""
Central-bank speech tone -> next-day change in the US 10Y Treasury yield.
AI for Finance capstone (Type A: free application).

Pipeline
--------
1. Load Fed speeches (BIS speeches dataset, cbspeeches.com) and FRED DGS10.
2. Build event-day dataset: one row per speech day, features known at close of day t,
   target = change in 10Y yield (bp) from close t to close t+1.
3. Expanding-window walk-forward evaluation (retrained every calendar year, with embargo).
4. Models: naive baselines, lag-only ridge, dictionary tone, TF-IDF ridge, TF-IDF/SVD + GBM.
5. Trading rule: sign(forecast) -> short (+1) or long (-1) duration for one day,
   net of transaction costs. Report Sharpe, drawdown, turnover, hit rate, sub-periods.

Usage
-----
  python download_data.py                                             # fetch + prepare real data
  python speeches_yields_capstone.py --speeches data/fed_speeches.csv --yields data/DGS10.csv
  python speeches_yields_capstone.py --synthetic                      # smoke test, no data needed

Data
----
  fed_speeches.csv : Fed speeches from the CBS dataset (cbspeeches.com), built by download_data.py.
  DGS10.csv        : https://fred.stlouisfed.org/series/DGS10 -> Download -> CSV.

Changes relative to the starter version (see README for the reasoning)
----------------------------------------------------------------------
  * FIX  load_speeches: the CBS file codes the Fed as Country == "USA"; the old regex
         ("united states|federal reserve|...") matched nothing and silently dropped every speech.
  * FIX  hit rate: days with a 0bp yield change (common, yields are quoted in 1bp steps) can
         never be "hit"; they are now excluded from the hit-rate and its binomial test.
  * NEW  timing/placebo check (execute one day late), speaker-type split (Board vs Reserve Banks),
         volatility-regime split, signal-strength split, economically motivated sub-periods
         (zero-lower-bound, COVID, hiking cycle) and a Sharpe-by-period figure.
  * SPEED tone_score uses one compiled regex per word list (identical counts, much faster).
"""
import argparse
import os
import re
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)

# ----------------------------- configuration ---------------------------------
COST_BP_PER_SIDE = 0.25   # transaction cost in yield-bp per side (assumption; vary in robustness)
DURATION = 8.0            # approx. modified duration of 10Y note, converts yield-bp to price return
EMBARGO_DAYS = 7          # calendar days dropped before each test year (avoids target overlap)
MIN_TRAIN = 150           # minimum training observations
FIRST_TEST_YEAR = 2012
SUBPERIODS = {"2012-2015 (zero lower bound)": ("2012-01-01", "2015-12-31"),
              "2016-2019 (normalisation)": ("2016-01-01", "2019-12-31"),
              "2020-2021 (COVID)": ("2020-01-01", "2021-12-31"),
              "2022-2023 (hiking cycle)": ("2022-01-01", "2100-01-01")}
BOARD_NAME = "board of governors"   # speeches by the Board of Governors vs regional Reserve Banks

HAWKISH = ["inflation", "inflationary", "tighten", "tightening", "hike", "raise", "raising",
           "overheating", "restrictive", "vigilant", "price stability", "upside risks", "strong",
           "robust", "normalization", "normalize", "balance sheet reduction", "wage growth"]
DOVISH = ["accommodation", "accommodative", "easing", "ease", "cut", "lower", "downside",
          "weak", "weakness", "slack", "unemployment", "patient", "stimulus", "support",
          "uncertainty", "subdued", "disinflation", "recession", "asset purchases"]


# ----------------------------- data loading ----------------------------------
def load_speeches(path, country_codes=("usa", "us", "united states")):
    """Read the speech file; keep Fed speeches; flag Board-of-Governors speeches."""
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    date_col = next(c for c in df.columns if "date" in c)
    text_col = next(c for c in df.columns if c in ("text", "speech", "content", "body", "fulltext"))
    if "country" in df.columns:
        df = df[df["country"].astype(str).str.lower().str.strip().isin(country_codes)]
    if len(df) == 0:
        raise ValueError("No speeches left after the country filter - check the country column.")
    bank = df["centralbank"].astype(str).str.lower() if "centralbank" in df.columns else ""
    out = pd.DataFrame({"date": pd.to_datetime(df[date_col], errors="coerce"),
                        "text": df[text_col].astype(str),
                        "board": (bank.str.contains(BOARD_NAME) if "centralbank" in df.columns
                                  else False)}).dropna(subset=["date", "text"])
    return out


def load_yields(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    date_col = df.columns[0]
    val_col = "DGS10" if "DGS10" in df.columns else df.columns[1]
    y = pd.DataFrame({"date": pd.to_datetime(df[date_col]),
                      "y": pd.to_numeric(df[val_col], errors="coerce")}).dropna()
    return y.sort_values("date").reset_index(drop=True)


def make_synthetic(n_years=20):
    """Synthetic data ONLY for smoke-testing the code. Results are meaningless."""
    rng = np.random.default_rng(SEED)
    days = pd.bdate_range("2004-01-01", periods=int(252 * n_years))
    vol = 6 * np.exp(np.cumsum(rng.normal(0, 0.02, len(days))) * 0.3)
    dy = rng.normal(0, vol)
    y = pd.DataFrame({"date": days, "y": 3 + np.cumsum(dy) / 100})
    vocab = ["economy", "growth", "policy", "committee", "markets", "financial", "outlook", "labor",
             "credit", "global", "banks", "risk", "data", "rates", "consumer", "firms"]
    rows = []
    for i, d in enumerate(days[:-1]):
        if rng.random() < 0.25:
            tone = rng.normal(0, 1)
            words = list(rng.choice(vocab, 120))
            words += list(rng.choice(HAWKISH[:6], max(0, int(3 + 3 * tone)))) \
                   + list(rng.choice(DOVISH[:6], max(0, int(3 - 3 * tone))))
            rng.shuffle(words)
            rows.append((d, " ".join(words), bool(rng.random() < 0.4)))
            dy[i + 1] += 0.8 * tone  # weak planted signal
    y["y"] = 3 + np.cumsum(dy) / 100
    return pd.DataFrame(rows, columns=["date", "text", "board"]), y


# ----------------------------- feature building ------------------------------
def _compile(words):
    return re.compile(r"\b(?:" + "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True)) + r")\b")


_HAWK_RE, _DOVE_RE = _compile(HAWKISH), _compile(DOVISH)


def tone_score(text):
    """(hawkish - dovish) word count per 1,000 words. Positive = hawkish."""
    t = text.lower()
    n = max(len(t.split()), 1)
    return 1000.0 * (len(_HAWK_RE.findall(t)) - len(_DOVE_RE.findall(t))) / n


def build_dataset(speeches, yields):
    y = yields.copy()
    y["dy0"] = y["y"].diff() * 100                    # today's change (bp), known at close t
    y["y_next"] = y["dy0"].shift(-1)                   # target: change t -> t+1 (bp)
    y["y_next2"] = y["dy0"].shift(-2)                  # t+1 -> t+2 (used ONLY for the timing check)
    y["vol5"] = y["dy0"].rolling(5).std()              # uses data up to t only
    y["mom5"] = y["dy0"].rolling(5).sum()
    g = speeches.assign(day=speeches["date"].dt.normalize()).groupby("day")
    sp = pd.DataFrame({"text": g["text"].apply(" ".join), "board": g["board"].any()}).reset_index()
    sp = sp.rename(columns={"day": "date"})
    df = sp.merge(y, on="date", how="inner").sort_values("date").reset_index(drop=True)
    df["tone"] = df["text"].map(tone_score)
    # tone surprise: tone minus expanding mean of PREVIOUS speeches (shift(1) -> no look-ahead)
    df["tone_surprise"] = df["tone"] - df["tone"].expanding().mean().shift(1)
    df = df.dropna(subset=["y_next", "y_next2", "dy0", "vol5", "mom5", "tone_surprise"]).reset_index(drop=True)
    return df


# ----------------------------- models ----------------------------------------
LAGS = ["dy0", "vol5", "mom5"]


def winsor(y, p=0.01):
    lo, hi = np.quantile(y, [p, 1 - p])
    return np.clip(y, lo, hi)


def m_always_long(train, test):          # buy-and-hold duration (pos = -1 means long bonds)
    return -np.ones(len(test))


def m_momentum(train, test):             # trade in the direction of today's yield change
    return test["dy0"].values


def m_ar(train, test):                   # lag-only ridge (no text)
    mdl = RidgeCV(alphas=np.logspace(-1, 3, 9), cv=TimeSeriesSplit(5))
    mdl.fit(train[LAGS], winsor(train["y_next"].values))
    return mdl.predict(test[LAGS])


def m_dictionary(train, test):           # parameter-free: hawkish surprise -> yields up
    return test["tone_surprise"].values


def _vec():
    return TfidfVectorizer(min_df=5, max_df=0.8, ngram_range=(1, 2), stop_words="english",
                           max_features=20000, sublinear_tf=True)


def m_tfidf_ridge(train, test, return_model=False):
    vec = _vec()
    Xtr, Xte = vec.fit_transform(train["text"]), vec.transform(test["text"])   # fit on train only
    mdl = RidgeCV(alphas=np.logspace(-1, 3, 9), cv=TimeSeriesSplit(5))
    mdl.fit(Xtr, winsor(train["y_next"].values))
    pred = mdl.predict(Xte)
    return (pred, vec, mdl) if return_model else pred


def m_svd_gbm(train, test):
    vec = _vec()
    Xtr, Xte = vec.fit_transform(train["text"]), vec.transform(test["text"])
    k = int(min(50, Xtr.shape[1] - 1, Xtr.shape[0] - 1))
    svd = TruncatedSVD(k, random_state=SEED).fit(Xtr)
    cols = ["tone", "tone_surprise"] + LAGS
    Ztr = np.hstack([svd.transform(Xtr), train[cols].values])
    Zte = np.hstack([svd.transform(Xte), test[cols].values])
    y = winsor(train["y_next"].values)
    best, best_mse = None, np.inf
    for depth in (2, 3):                                 # small inner time-series grid search
        for iters in (100, 300):
            mses = []
            for tr, va in TimeSeriesSplit(4).split(Ztr):
                g = HistGradientBoostingRegressor(max_depth=depth, max_iter=iters, learning_rate=0.05,
                                                  l2_regularization=1.0, random_state=SEED)
                g.fit(Ztr[tr], y[tr])
                mses.append(np.mean((g.predict(Ztr[va]) - y[va]) ** 2))
            if np.mean(mses) < best_mse:
                best_mse, best = np.mean(mses), (depth, iters)
    g = HistGradientBoostingRegressor(max_depth=best[0], max_iter=best[1], learning_rate=0.05,
                                      l2_regularization=1.0, random_state=SEED).fit(Ztr, y)
    return g.predict(Zte)


MODELS = {"Always long duration": m_always_long, "Momentum (lag dy)": m_momentum,
          "Lag-only ridge": m_ar, "Dictionary tone": m_dictionary,
          "TF-IDF ridge": m_tfidf_ridge, "SVD+GBM": m_svd_gbm}


# ----------------------------- walk-forward ----------------------------------
def walk_forward(df, first_test_year=FIRST_TEST_YEAR):
    folds = []
    for Y in sorted(df["date"].dt.year.unique()):
        if Y < first_test_year:
            continue
        start, end = pd.Timestamp(f"{Y}-01-01"), pd.Timestamp(f"{Y + 1}-01-01")
        train = df[df["date"] < start - pd.Timedelta(days=EMBARGO_DAYS)]
        test = df[(df["date"] >= start) & (df["date"] < end)]
        if len(train) < MIN_TRAIN or test.empty:
            continue
        row = test[["date", "y_next", "y_next2", "vol5", "board"]].copy()
        for name, fn in MODELS.items():
            row[name] = fn(train, test)
        folds.append(row)
        print(f"  fold {Y}: train={len(train)} test={len(test)}")
    return pd.concat(folds).reset_index(drop=True)


# ----------------------------- evaluation ------------------------------------
def trade_pnl(pred, y_next, cost=COST_BP_PER_SIDE):
    pos = np.sign(pred)                                  # +1 short duration, -1 long duration
    gross_bp = pos * y_next                              # profit in yield-bp
    net_bp = gross_bp - 2 * cost * np.abs(pos)           # enter + exit each event day
    return pos, gross_bp, net_bp


def metrics(pred, y_next, dates, cost=COST_BP_PER_SIDE):
    pos, gross, net = trade_pnl(pred, y_next, cost)
    ret = net * DURATION / 100.0                         # % return per trade on notional
    years = max((dates.max() - dates.min()).days / 365.25, 1e-9)
    tpy = len(ret) / years
    cum = np.cumsum(ret)
    dd = (cum - np.maximum.accumulate(cum)).min()
    nz = y_next != 0                                     # a 0bp day can never be called right/wrong
    hits = int(np.sum(np.sign(pred[nz]) == np.sign(y_next[nz])))
    hit = hits / max(nz.sum(), 1)
    p_hit = stats.binomtest(hits, int(nz.sum()), 0.5).pvalue
    r2 = 1 - np.sum((y_next - pred) ** 2) / np.sum(y_next ** 2)
    return {"n_trades": len(ret), "hit_rate": hit, "hit_p": p_hit,
            "mean_net_bp": net.mean(), "mean_gross_bp": gross.mean(),
            "sharpe_net": ret.mean() / ret.std(ddof=1) * np.sqrt(tpy) if ret.std() > 0 else np.nan,
            "cum_ret_pct": cum[-1], "max_dd_pct": dd,
            "turnover_per_yr": 2 * np.mean(np.abs(pos)) * tpy,
            "oos_r2_vs_zero": r2}


def evaluate(oos, outdir):
    names = list(MODELS)
    main = pd.DataFrame({n: metrics(oos[n].values, oos["y_next"].values, oos["date"]) for n in names}).T
    # R2 is only meaningful for models that output a forecast in bp
    main.loc[["Always long duration", "Momentum (lag dy)", "Dictionary tone"], "oos_r2_vs_zero"] = np.nan
    main.to_csv(f"{outdir}/table_main.csv")

    sub = {}
    for label, (a, b) in SUBPERIODS.items():
        m = (oos["date"] >= a) & (oos["date"] <= b)
        if m.sum() > 20:
            for n in names:
                sub[(label, n)] = metrics(oos.loc[m, n].values, oos.loc[m, "y_next"].values,
                                          oos.loc[m, "date"])["sharpe_net"]
    pd.Series(sub).unstack().to_csv(f"{outdir}/table_subperiod_sharpe.csv")

    cost_tbl = {}
    for c in (0.0, 0.25, 0.5, 1.0):                      # robustness to transaction costs
        for n in names:
            cost_tbl[(c, n)] = metrics(oos[n].values, oos["y_next"].values, oos["date"], c)["sharpe_net"]
    pd.Series(cost_tbl).unstack().to_csv(f"{outdir}/table_cost_sensitivity_sharpe.csv")

    # bootstrap CI for mean P&L per trade (bp) of each text model, GROSS and NET of costs.
    # (i.i.d. bootstrap over event days; ignores any dependence between neighbouring events)
    rng = np.random.default_rng(SEED)
    ci = {}
    for n in ("Dictionary tone", "TF-IDF ridge", "SVD+GBM"):
        _, gross, net = trade_pnl(oos[n].values, oos["y_next"].values)
        idx = [rng.integers(0, len(net), len(net)) for _ in range(2000)]
        bg = [gross[i].mean() for i in idx]
        bn = [net[i].mean() for i in idx]
        ci[n] = {"mean_gross_bp": gross.mean(), "gross_ci_low": np.percentile(bg, 2.5),
                 "gross_ci_high": np.percentile(bg, 97.5), "mean_net_bp": net.mean(),
                 "net_ci_low": np.percentile(bn, 2.5), "net_ci_high": np.percentile(bn, 97.5)}
    pd.DataFrame(ci).T.to_csv(f"{outdir}/table_bootstrap_ci.csv")

    # ---- robustness / error analysis (all use the SAME out-of-sample forecasts) ----------
    # (a) timing / placebo: execute one day late. If forecasts only capture a fast reaction to the
    #     speech, the edge should vanish here. Also guards against speeches delivered after the close.
    timing = {n: {"sharpe_t+1": metrics(oos[n].values, oos["y_next"].values, oos["date"])["sharpe_net"],
                  "sharpe_delayed_t+2": metrics(oos[n].values, oos["y_next2"].values, oos["date"])["sharpe_net"]}
              for n in names}
    pd.DataFrame(timing).T.to_csv(f"{outdir}/table_timing_check.csv")

    # (b) who is speaking: Board of Governors vs regional Reserve Banks; (c) volatility regime
    def split_table(mask_a, label_a, mask_b, label_b):
        out = {}
        for lab, m in ((label_a, mask_a), (label_b, mask_b)):
            if m.sum() > 20:
                for n in names:
                    r = metrics(oos.loc[m, n].values, oos.loc[m, "y_next"].values, oos.loc[m, "date"])
                    out[(lab, n)] = r["sharpe_net"]
        return pd.Series(out).unstack()
    board = oos["board"].astype(bool)
    split_table(board, "Board speech day", ~board, "Reserve-Bank-only day").to_csv(
        f"{outdir}/table_by_speaker_type_sharpe.csv")
    hi = oos["vol5"] >= oos["vol5"].median()           # descriptive split of the OOS sample
    split_table(hi, "High-vol regime", ~hi, "Low-vol regime").to_csv(f"{outdir}/table_by_vol_regime_sharpe.csv")

    # (d) signal strength: are bigger forecasts more reliable? (descriptive, not a tradable rule)
    strength = {}
    for n in ("Dictionary tone", "TF-IDF ridge", "SVD+GBM"):
        a = np.abs(oos[n].values)
        q = pd.qcut(pd.Series(a).rank(method="first"), 3, labels=["weak", "medium", "strong"]).values
        for lab in ("weak", "medium", "strong"):
            m = q == lab
            pos, g, net = trade_pnl(oos[n].values[m], oos["y_next"].values[m])
            nz = oos["y_next"].values[m] != 0
            strength[(n, lab)] = {"n": int(m.sum()), "hit_rate": np.mean(np.sign(oos[n].values[m][nz]) ==
                                  np.sign(oos["y_next"].values[m][nz])), "mean_gross_bp": g.mean(),
                                  "mean_net_bp": net.mean()}
    pd.DataFrame(strength).T.to_csv(f"{outdir}/table_by_signal_strength.csv")

    # figure 2: Sharpe by sub-period
    sp = pd.read_csv(f"{outdir}/table_subperiod_sharpe.csv", index_col=0)
    ax2 = sp.T.plot(kind="bar", figsize=(10, 5), width=0.8)
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_ylabel("net annualised Sharpe ratio")
    ax2.set_title("Net Sharpe ratio by sub-period (out of sample)")
    ax2.legend(fontsize=7)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(f"{outdir}/fig_subperiod_sharpe.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(9, 5))
    for n in names:
        ret = trade_pnl(oos[n].values, oos["y_next"].values)[2] * DURATION / 100
        ax.plot(oos["date"], np.cumsum(ret), label=n)
    ax.set_title("Out-of-sample cumulative net return (event-day trades)")
    ax.set_ylabel("cumulative % of notional")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{outdir}/fig_cumulative_pnl.png", dpi=150)
    plt.close(fig)
    return main


def top_words(df, outdir, k=15):
    """Interpretability: words the ridge assigns to higher / lower next-day yields (last fold)."""
    last = df["date"].dt.year.max()
    train = df[df["date"] < pd.Timestamp(f"{last}-01-01") - pd.Timedelta(days=EMBARGO_DAYS)]
    _, vec, mdl = m_tfidf_ridge(train, df.tail(5), return_model=True)
    vocab = np.array(vec.get_feature_names_out())
    order = np.argsort(mdl.coef_)
    out = pd.DataFrame({"words_yields_down": vocab[order[:k]], "words_yields_up": vocab[order[::-1][:k]]})
    out.to_csv(f"{outdir}/table_top_words.csv", index=False)
    return out


# ----------------------------- main ------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speeches"), ap.add_argument("--yields")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--eval-only", action="store_true",
                    help="skip the 10-15 min walk-forward; rebuild tables/figures from <out>/oos_predictions.csv")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--first-test-year", type=int, default=FIRST_TEST_YEAR)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    if a.eval_only:
        oos = pd.read_csv(f"{a.out}/oos_predictions.csv", parse_dates=["date"])
        oos["board"] = oos["board"].astype(bool)
        print(evaluate(oos, a.out).round(3).to_string())
        return
    if a.synthetic:
        print("*** SYNTHETIC DATA: smoke test only, results are meaningless ***")
        speeches, yields = make_synthetic()
    else:
        speeches, yields = load_speeches(a.speeches), load_yields(a.yields)
    df = build_dataset(speeches, yields)
    print(f"Event-day sample: {len(df)} obs, {df.date.min().date()} to {df.date.max().date()}")
    oos = walk_forward(df, a.first_test_year)
    oos.to_csv(f"{a.out}/oos_predictions.csv", index=False)
    main_tbl = evaluate(oos, a.out)
    pd.set_option("display.width", 200)
    print(main_tbl.round(3).to_string())
    print(top_words(df, a.out))


if __name__ == "__main__":
    main()
