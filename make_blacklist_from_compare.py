# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import pandas as pd

INP = "features_compare_summary.csv"
OUT = "feature_blacklist.txt"

def main():
    df = pd.read_csv(INP)
    if "feature" not in df.columns or "mismatch_cnt" not in df.columns:
        raise SystemExit("Invalid summary CSV: missing 'feature' or 'mismatch_cnt' columns.")
    bad = df.loc[df["mismatch_cnt"] > 0, "feature"].astype(str).tolist()
    if not bad:
        print("No mismatched features found.")
        return
    with open(OUT, "w", encoding="utf-8") as f:
        for name in bad:
            f.write(name.strip() + "\n")
    print(f"Wrote {len(bad)} names to {OUT}.")

if __name__ == "__main__":
    main()
