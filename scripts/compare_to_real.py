"""Compare the synthetic dataset with the real PaySim data and sanity-check it.

    python compare_to_real.py <real.csv> <synthetic.csv> [--report report.md]
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
FLOATS = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]


def summarize(df: pd.DataFrame) -> dict:
    hour = (df.step - 1) % 24  # same clock as PaySim: step 1 = hour 0
    legit, fraud = df[df.isFraud == 0], df[df.isFraud == 1]
    s = {
        "rows": len(df),
        "fraud rate %": 100 * df.isFraud.mean(),
        "flagged rows": int(df.isFlaggedFraud.sum()),
        "fraud types": ", ".join(sorted(fraud.type.unique())),
        "fraud: TRANSFER / CASH_OUT": f"{(fraud.type == 'TRANSFER').sum():,} / {(fraud.type == 'CASH_OUT').sum():,}",
        "fraud draining whole account %": 100 * np.isclose(fraud.amount, fraud.oldbalanceOrg).mean(),
        "fraud capped at 10M (rows)": int((fraud.amount == 1e7).sum()),
        "fraud spread over hours (min/max per hour)": "{} / {}".format(
            *(lambda c: (c.min(), c.max()))(fraud.groupby(hour[fraud.index]).size().reindex(range(24), fill_value=0))),
        "legit night share (hours 0-6) %": 100 * (hour[legit.index] <= 6).mean(),
        "distinct senders": df.nameOrig.nunique(),
        "avg transactions per sender": len(df) / df.nameOrig.nunique(),
        "sender balance adds up %": 100 * (
            np.isclose(df.oldbalanceOrg - df.amount, df.newbalanceOrig)
            | np.isclose(df.oldbalanceOrg + df.amount, df.newbalanceOrig)
            | (df.isFlaggedFraud == 1)).mean(),
        "outflow larger than balance %": 100 * ((df.type != "CASH_IN") & (df.amount > df.oldbalanceOrg + 0.01)).mean(),
    }
    for t in TYPES:
        s[f"share {t} %"] = 100 * (df.type == t).mean()
    for t in TYPES:
        a = legit.amount[legit.type == t]
        s[f"{t} amount p10 / median / p90"] = " / ".join(f"{q:,.0f}" for q in a.quantile([.1, .5, .9]))
    return s


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("real")
    p.add_argument("synthetic")
    p.add_argument("--report")
    p.add_argument("--first-step", type=int, default=1)
    p.add_argument("--last-step", type=int, default=743)
    p.add_argument("--no-overlap-with", nargs="*", default=[], help="other generated CSVs that must not share ids")
    args = p.parse_args()

    real, syn = pd.read_csv(args.real), pd.read_csv(args.synthetic)
    checks = []

    def check(name, ok, detail=""):
        checks.append((name, ok, detail))

    check("Same columns, same order as the real file", list(syn.columns) == list(real.columns))
    check("Same column types", list(syn.dtypes) == list(real.dtypes), str(dict(syn.dtypes)))
    check("No missing values", int(syn.isna().sum().sum()) == 0)
    check("No duplicate rows", int(syn.duplicated().sum()) == 0, f"{int(syn.duplicated().sum())} duplicates")
    check("No negative amounts or balances", bool((syn[FLOATS] >= 0).all().all()))
    check(f"Steps within {args.first_step}–{args.last_step} (all {args.last_step - args.first_step + 1} hours present)",
          bool(syn.step.between(args.first_step, args.last_step).all())
          and syn.step.nunique() == args.last_step - args.first_step + 1)
    check("Only the five PaySim types", set(syn.type) <= set(TYPES))
    check("IDs look like PaySim (C… customers, M… merchants)",
          bool(syn.nameOrig.str.match(r"^C\d+$").all() and syn.nameDest.str.match(r"^[CM]\d+$").all()))
    real_ids = set(real.nameOrig) | set(real.nameDest)
    collisions = len((set(syn.nameOrig) | set(syn.nameDest)) & real_ids)
    check("No ID shared with the real dataset", collisions == 0, f"{collisions} shared")
    syn_ids = set(syn.nameOrig) | set(syn.nameDest)
    for other in args.no_overlap_with:
        o = pd.read_csv(other, usecols=["nameOrig", "nameDest"])
        shared = len(syn_ids & (set(o.nameOrig) | set(o.nameDest)))
        check(f"No ID shared with {other.split('/')[-1]}", shared == 0, f"{shared} shared")
    check("Fraud only in TRANSFER / CASH_OUT", set(syn.type[syn.isFraud == 1]) <= {"TRANSFER", "CASH_OUT"})
    check("Flagged rows are always fraud", bool((syn.isFraud[syn.isFlaggedFraud == 1] == 1).all()))
    check("Fraud rate within 10% of real", abs(syn.isFraud.mean() / real.isFraud.mean() - 1) < 0.10,
          f"{100*syn.isFraud.mean():.3f}% vs {100*real.isFraud.mean():.3f}%")

    rs, ss = summarize(real), summarize(syn)
    lines = ["| Measure | Real PaySim | Synthetic |", "|---|---|---|"]
    for k in rs:
        f = lambda v: f"{v:,.2f}" if isinstance(v, float) else (f"{v:,}" if isinstance(v, int) else v)
        lines.append(f"| {k} | {f(rs[k])} | {f(ss[k])} |")
    table = "\n".join(lines)

    out = ["## Sanity checks", ""]
    out += [f"- [{'x' if ok else ' '}] {n}" + (f" ({d})" if d and not ok else "") for n, ok, d in checks]
    out += ["", f"**{sum(ok for _, ok, _ in checks)}/{len(checks)} checks passed**", "", "## Real vs synthetic", "", table]
    text = "\n".join(out)
    print(text)
    if args.report:
        open(args.report, "w").write(text + "\n")
    sys.exit(0 if all(ok for _, ok, _ in checks) else 1)


if __name__ == "__main__":
    main()
