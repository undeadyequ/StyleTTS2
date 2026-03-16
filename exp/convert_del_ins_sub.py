"""Convert quality_aggregated.json to del_ins_sub.csv.

Usage:
    python exp/convert_del_ins_sub.py <quality_aggregated.json> [--no-latex] [--out <csv_path>]

The JSON is produced by exp2/pipeline/quality_stage.py and has the structure:
    {model_name: {deletions, insertions, substitutions, wer, utmos_mean, utmos_std}}
"""

import argparse
import json
from pathlib import Path

import pandas as pd


COLUMNS = ["deletions", "insertions", "substitutions"]


def convert(json_path: str, need_print_latex: bool = True, out_csv: str = "") -> pd.DataFrame:
    with open(json_path, "r") as f:
        data = json.load(f)

    rows = []
    for model, metrics in data.items():
        rows.append({
            "model": model,
            "deletions":     round(metrics.get("deletions", 0), 2),
            "insertions":    round(metrics.get("insertions", 0), 2),
            "substitutions": round(metrics.get("substitutions", 0), 2),
        })

    df = pd.DataFrame(rows).set_index("model")[COLUMNS]

    # Save CSV
    if not out_csv:
        out_csv = str(Path(json_path).parent / "del_ins_sub.csv")
    df.to_csv(out_csv, index=True)
    print(f"Saved: {out_csv}")

    if need_print_latex:
        df_marked = df.copy().astype(object)
        for col in COLUMNS:
            sorted_vals = df[col].sort_values(ascending=True).unique()
            best   = sorted_vals[0]
            second = sorted_vals[1] if len(sorted_vals) > 1 else None
            df_marked[col] = df[col].apply(
                lambda x, b=best, s=second:
                f"\\first{{{x}}}" if x == b else
                (f"\\second{{{x}}}" if s is not None and x == s else f"{x}")
            )
        print()
        for idx, row in df_marked.iterrows():
            print(f"{idx} & " + " & ".join(row.astype(str)) + r" \\")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", help="Path to quality_aggregated.json")
    parser.add_argument("--no-latex", action="store_true", help="Skip LaTeX printing")
    parser.add_argument("--out", default="", help="Output CSV path (default: same dir as json)")
    args = parser.parse_args()

    convert(args.json_path, need_print_latex=not args.no_latex, out_csv=args.out)
