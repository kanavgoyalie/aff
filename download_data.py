"""
Download and prepare the two raw datasets used in the capstone.

  1. FRED DGS10  : daily US 10-year Treasury constant-maturity yield (percent).
  2. CBS dataset : Central Bank Speeches (Campiglio, Deyris, Romelli, Scalisi), cbspeeches.com.
                   The full file is ~690 MB (35k speeches from 131 central banks). We keep
                   only Federal Reserve speeches (Country == "USA") and write a small CSV.

Usage:  python download_data.py            (writes data/DGS10.csv and data/fed_speeches.csv)
        python download_data.py --keep-raw (also keeps the 690 MB raw file)

The CBS dataset is free for academic research / personal non-commercial use. Please cite:
Campiglio, Deyris, Romelli & Scalisi (2025), "Warning words in a warming world: central bank
communication and climate change", European Economic Review.
"""
import argparse
import os
import urllib.request

import pandas as pd

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10"
CBS_URL = ("https://www.dropbox.com/scl/fi/n2tsmswnk4ymm3zz0wteg/CBS_dataset_v1.0.csv"
           "?rlkey=sdxk6b9xpid0xe7w7s4nm6s87&dl=1")
DATA = "data"


def fetch(url, dest):
    print(f"Downloading {url[:70]}... -> {dest}")
    req = urllib.request.Request(url)  # default UA (FRED rejects browser-like UAs)
    with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-raw", action="store_true")
    a = ap.parse_args()
    os.makedirs(DATA, exist_ok=True)

    fetch(FRED_URL, f"{DATA}/DGS10.csv")

    raw = f"{DATA}/CBS_raw.csv"
    if not os.path.exists(raw):
        fetch(CBS_URL, raw)
    cols = ["Date", "Authorname", "Role", "CentralBank", "Country", "Title", "text"]
    df = pd.read_csv(raw, usecols=cols, low_memory=False)
    fed = df[df["Country"] == "USA"].copy()
    fed["Date"] = pd.to_datetime(fed["Date"], errors="coerce")
    fed = fed.dropna(subset=["Date", "text"]).sort_values("Date")
    fed.to_csv(f"{DATA}/fed_speeches.csv", index=False)
    print(f"Saved {len(fed)} Fed speeches ({fed.Date.min().date()} to {fed.Date.max().date()})")
    if not a.keep_raw:
        os.remove(raw)


if __name__ == "__main__":
    main()
