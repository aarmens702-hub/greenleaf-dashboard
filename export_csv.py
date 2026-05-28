"""Export the derived parquet tables to CSV for human viewing (e.g. in VS Code).

Run: python export_csv.py

Reads data/derived/*.parquet and writes matching CSVs to data/derived_csv/.
Parquet stays the source of truth — these CSVs are view-only copies. Dates are
written as ISO strings, so reload them with parse_dates=['date'] if you need
real datetimes back.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
SRC = ROOT / "data" / "derived"
OUT = ROOT / "data" / "derived_csv"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    parquets = sorted(SRC.glob("*.parquet"))
    if not parquets:
        print("[csv] no parquet files in data/derived/ — run build_derived.py first")
        return
    for p in parquets:
        df = pd.read_parquet(p)
        out = OUT / f"{p.stem}.csv"
        df.to_csv(out, index=False)
        print(f"[csv] wrote {out.name}: {len(df)} rows, {df.shape[1]} cols")
    print("[csv] done")


if __name__ == "__main__":
    main()
