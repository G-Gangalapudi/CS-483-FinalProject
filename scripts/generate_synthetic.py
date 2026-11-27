"""Step 2: generate a plausible synthetic transaction dataset in the PaySim format.

    python generate_synthetic.py --real PS_20174392719_1491204439457_log.csv
    python generate_synthetic.py --real ... --rows 3181310 --seed 2026 --out paysim_synthetic.csv

Same 11 columns, column order and number formatting as the Kaggle PaySim file, so
it can be combined with the real data. Statistical shape is CALIBRATED from the
real file (nothing is copied row for row):

  * transaction-type mix per hour of day
  * amount distribution per type (log-scale histogram, sampled within bins)
  * starting-balance distribution
  * daily activity cycle; fraud spread evenly over the clock, as in PaySim
  * fraud rate (~0.13%), fraud only in TRANSFER / CASH_OUT, mostly as a
    TRANSFER to a mule account followed by a CASH_OUT of the same amount
  * per-transaction fraud cap of 10,000,000 (larger thefts are split)

Deliberately MORE realistic than PaySim (documented differences):

  * customers make several transactions over the month (PaySim: almost all once)
  * balances always add up: new = old - amount (outflows) or old + amount (inflows)
  * outflows never exceed the balance
  * cash agents (CASH_IN / CASH_OUT / DEBIT counterparties) keep real balances
  * steady 31-day calendar with a weekend dip (PaySim's volume collapses after day 17)
  * ~15% of frauds take only part of the balance, so not every fraud empties the account

Identifiers never collide with any nameOrig / nameDest in the real file.
"""
from __future__ import annotations

import argparse
import csv
import json
from decimal import Decimal
from pathlib import Path

import numpy as np

COLUMNS = ["step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig", "nameDest",
           "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud"]
TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
N_STEPS = 743
FRAUD_CAP = 10_000_000.0


# ----------------------------------------------------------------- formatting
def jfmt(v: float) -> str:
    """Numbers written the way the Kaggle file writes them (Java Double.toString)."""
    v = round(float(v), 2) + 0.0
    if v == 0 or 1e-3 <= abs(v) < 1e7:
        return repr(v)
    sign, digits, exp = Decimal(repr(v)).normalize().as_tuple()
    ds = "".join(map(str, digits))
    return f"{'-' if sign else ''}{ds[0]}.{ds[1:] or '0'}E{len(ds) + exp - 1}"


# ---------------------------------------------------------------- calibration
def calibrate(real_path: Path, cache: Path) -> dict:
    if cache.exists():
        return json.loads(cache.read_text())
    import pandas as pd

    df = pd.read_csv(real_path)
    hour = (df.step - 1) % 24
    cal = {"source": str(real_path.name), "rows": len(df)}
    cal["fraud_rate"] = float(df.isFraud.mean())
    cal["hour_volume"] = (df[df.isFraud == 0].groupby(hour[df.isFraud == 0]).size()
                          .reindex(range(24), fill_value=0) / (df.isFraud == 0).sum()).tolist()
    mix = pd.crosstab(hour[df.isFraud == 0], df.type[df.isFraud == 0], normalize="index")
    cal["type_mix_by_hour"] = mix.reindex(columns=TYPES, fill_value=0).values.tolist()
    cal["amount_hist"] = {}
    for t in TYPES:
        a = np.log10(df.amount[(df.type == t) & (df.isFraud == 0) & (df.amount > 0)].values)
        counts, edges = np.histogram(a, bins=120)
        cal["amount_hist"][t] = {"p": (counts / counts.sum()).tolist(), "edges": edges.tolist()}
    # Starting balances are sized from what customers actually spend (all legit outflows);
    # PaySim's own balances are often smaller than the amount spent, which a real ledger forbids.
    out_amt = np.log10(df.amount[(df.type != "CASH_IN") & (df.isFraud == 0) & (df.amount > 0)].values)
    counts, edges = np.histogram(out_amt, bins=120)
    cal["balance_hist"] = {"p": (counts / counts.sum()).tolist(), "edges": edges.tolist()}
    cal["zero_balance_share"] = 0.05  # new / emptied accounts
    ids = pd.concat([df.nameOrig, df.nameDest]).unique()
    cal["real_id_count"] = int(len(ids))
    cache.write_text(json.dumps(cal))
    np.save(cache.with_suffix(".ids.npy"), np.array(sorted(int(i[1:]) for i in ids), dtype=np.int64))
    return cal


# ------------------------------------------------------------------ generator
class Generator:
    def __init__(self, cal: dict, real_ids: np.ndarray, rng: np.random.Generator, rows: int,
                 balance_multiplier: float = 8.0, balance_spread: float = 0.4,
                 start_step: int = 1, n_steps: int = N_STEPS):
        self.cal, self.rng, self.rows = cal, rng, rows
        self.start, self.end = start_step, start_step + n_steps - 1
        self.real_ids = real_ids
        self.used_ids: set[int] = set()
        n_customers = max(1000, rows // 3)            # ~3 outgoing transactions per customer per month
        n_agents = max(50, rows // 200)               # cash-in/cash-out agents
        n_merchants = max(100, rows // 80)
        self.customers = self._new_ids(n_customers)
        self.agents = self._new_ids(n_agents)
        self.merchants = self._new_ids(n_merchants)
        self.balance = np.zeros(n_customers)
        has_money = rng.random(n_customers) >= cal["zero_balance_share"]
        # balance = a typical spend amount x a lognormal multiple (median `balance_multiplier`)
        self.balance[has_money] = np.round(
            self._sample_hist(cal["balance_hist"], has_money.sum())
            * rng.lognormal(np.log(balance_multiplier), balance_spread, has_money.sum()), 2)
        self.shrunk = 0
        self.agent_balance = np.round(10 ** rng.uniform(6, 7.5, n_agents), 2)
        self.out: list[list] = []

    # -- helpers
    def _new_ids(self, n: int) -> np.ndarray:
        """Random 9–10 digit ids never seen in the real file or earlier in this run."""
        out = []
        while len(out) < n:
            cand = self.rng.integers(10**8, 2 * 10**9, size=(n - len(out)) * 2)
            cand = np.unique(cand)
            pos = np.searchsorted(self.real_ids, cand).clip(max=len(self.real_ids) - 1)
            cand = cand[self.real_ids[pos] != cand]           # drop ids that exist in the real file
            cand = [c for c in cand.tolist() if c not in self.used_ids][: n - len(out)]
            self.used_ids.update(cand)
            out.extend(cand)
        return np.array(out, dtype=np.int64)

    def _sample_hist(self, hist: dict, n: int) -> np.ndarray:
        edges = np.array(hist["edges"])
        b = self.rng.choice(len(hist["p"]), size=n, p=hist["p"])
        return np.round(10 ** self.rng.uniform(edges[b], edges[b + 1]), 2)

    def _emit(self, step, typ, amount, orig, ob, nb, dest, odb, ndb, fraud=0, flagged=0):
        self.out.append([step, typ, amount, orig, ob, nb, dest, odb, ndb, fraud, flagged])

    # -- legitimate activity
    def legit_plan(self, n_legit: int) -> np.ndarray:
        """How many legit transactions happen in each step (daily cycle + weekend dip + noise)."""
        n_steps = self.end - self.start + 1
        hours = (self.start - 1 + np.arange(n_steps)) % 24
        days = (self.start - 1 + np.arange(n_steps)) // 24
        weight = np.array(self.cal["hour_volume"])[hours]
        weight *= np.where(days % 7 >= 5, 0.85, 1.0)
        weight *= self.rng.lognormal(0, 0.08, n_steps)
        return self.rng.multinomial(n_legit, weight / weight.sum())

    def legit_step(self, step: int, n: int) -> None:
        rng, hour = self.rng, (step - 1) % 24
        types = rng.choice(TYPES, size=n, p=self.cal["type_mix_by_hour"][hour])
        amounts = {t: iter(self._sample_hist(self.cal["amount_hist"][t], int((types == t).sum())))
                   for t in TYPES}
        for t in types:
            amt = float(next(amounts[t]))
            if t == "CASH_IN":
                c = rng.integers(len(self.customers))
                a = rng.integers(len(self.agents))
                if self.agent_balance[a] < amt:
                    self.agent_balance[a] = round(self.agent_balance[a] + 5e6, 2)  # agent restocks cash
                ob, odb = self.balance[c], self.agent_balance[a]
                self.balance[c] = round(ob + amt, 2)
                self.agent_balance[a] = round(odb - amt, 2)
                self._emit(step, t, amt, f"C{self.customers[c]}", ob, ob + amt,
                           f"C{self.agents[a]}", odb, odb - amt)
                continue
            # outflows: find a customer who can afford it; as a last resort shrink the amount
            c, best = None, None
            for _ in range(20):
                cand = rng.integers(len(self.customers))
                if self.balance[cand] >= amt:
                    c = cand
                    break
                if best is None or self.balance[cand] > self.balance[best]:
                    best = cand
            if c is None:
                c = best
                if self.balance[c] < 1:
                    funded = np.flatnonzero(self.balance >= 1)
                    c = funded[rng.integers(len(funded))]
                amt = min(round(self.balance[c] * rng.uniform(0.3, 1.0), 2), self.balance[c])
                self.shrunk += 1
            ob = self.balance[c]
            self.balance[c] = round(ob - amt, 2)
            if t == "PAYMENT":                          # merchants' balances are not visible (as in PaySim)
                m = self.merchants[rng.integers(len(self.merchants))]
                self._emit(step, t, amt, f"C{self.customers[c]}", ob, self.balance[c], f"M{m}", 0.0, 0.0)
            elif t == "TRANSFER":
                d = rng.integers(len(self.customers))
                while d == c:
                    d = rng.integers(len(self.customers))
                odb = self.balance[d]
                self.balance[d] = round(odb + amt, 2)
                self._emit(step, t, amt, f"C{self.customers[c]}", ob, self.balance[c],
                           f"C{self.customers[d]}", odb, odb + amt)
            else:                                       # CASH_OUT / DEBIT go to an agent
                a = rng.integers(len(self.agents))
                odb = self.agent_balance[a]
                self.agent_balance[a] = round(odb + amt, 2)
                self._emit(step, t, amt, f"C{self.customers[c]}", ob, self.balance[c],
                           f"C{self.agents[a]}", odb, odb + amt)

    # -- fraud
    def fraud_event(self, step: int, budget: int) -> int:
        """Account takeover: TRANSFER to a fresh mule account, then CASH_OUT the same amount.
        Returns rows written (never more than `budget`)."""
        rng = self.rng
        rich = np.flatnonzero(self.balance > 1000)
        c = rich[rng.integers(len(rich))]
        victim = f"C{self.customers[c]}"
        total = self.balance[c] if rng.random() < 0.85 else round(self.balance[c] * rng.uniform(0.5, 0.97), 2)
        total = min(round(total, 2), self.balance[c])
        written = 0

        if budget == 1 or rng.random() < 0.04:          # direct cash-out from the victim's account
            amt = round(min(total, FRAUD_CAP), 2)
            a = rng.integers(len(self.agents))
            ob, odb = self.balance[c], self.agent_balance[a]
            self.balance[c] = round(ob - amt, 2)
            self.agent_balance[a] = round(odb + amt, 2)
            self._emit(step, "CASH_OUT", amt, victim, ob, self.balance[c], f"C{self.agents[a]}", odb, odb + amt, 1)
            return 1

        while total > 0.005 and budget - written >= 2:
            amt = round(min(total, FRAUD_CAP), 2)
            mule = f"C{self._new_ids(1)[0]}"
            ob = self.balance[c]
            if amt > 200_000 and rng.random() < 0.006:  # bank's rule blocks the transfer
                self._emit(step, "TRANSFER", amt, victim, ob, ob, mule, 0.0, 0.0, 1, 1)
                return written + 1
            self.balance[c] = round(ob - amt, 2)
            self._emit(step, "TRANSFER", amt, victim, ob, self.balance[c], mule, 0.0, amt, 1)
            a = rng.integers(len(self.agents))
            odb = self.agent_balance[a]
            self.agent_balance[a] = round(odb + amt, 2)
            self._emit(step, "CASH_OUT", amt, mule, amt, 0.0, f"C{self.agents[a]}", odb, odb + amt, 1)
            total = round(total - amt, 2)
            written += 2
        return written

    # -- whole month
    def run(self, write) -> int:
        """Simulate the month, passing each step's rows to `write` as it goes. Returns row count."""
        rng = self.rng
        written = 0
        n_fraud = round(self.rows * self.cal["fraud_rate"])
        legit_per_step = self.legit_plan(self.rows - n_fraud)
        fraud_steps = np.sort(rng.integers(self.start, self.end + 1, size=n_fraud))  # fraud ignores the clock
        fi, fraud_rows = 0, 0
        for step in range(self.start, self.end + 1):
            self.legit_step(step, int(legit_per_step[step - self.start]))
            while fi < len(fraud_steps) and fraud_steps[fi] == step and fraud_rows < n_fraud:
                fraud_rows += self.fraud_event(step, n_fraud - fraud_rows)
                fi += 1
            if step == self.end:
                while fraud_rows < n_fraud:              # top up if events ended early
                    fraud_rows += self.fraud_event(self.end, n_fraud - fraud_rows)
            write(self.out)
            written += len(self.out)
            self.out = []
            if (step - self.start + 1) % 100 == 0:
                print(f"  step {step} ({self.start}–{self.end}): {written:,} rows")
        return written


def write_rows(fh, rows: list[list]) -> None:
    for r in rows:
        fh.write(f"{r[0]},{r[1]},{jfmt(r[2])},{r[3]},{jfmt(r[4])},{jfmt(r[5])},{r[6]},"
                     f"{jfmt(r[7])},{jfmt(r[8])},{r[9]},{r[10]}\n")


class JsonArrayWriter:
    """Streams records as one JSON array, same layout as paysim_second_half.json
    (one record per line, numbers as numbers, keys in PaySim column order)."""

    def __init__(self, fh):
        self.fh, self.first = fh, True
        fh.write("[\n")

    def __call__(self, rows: list[list]) -> None:
        for r in rows:
            rec = dict(zip(COLUMNS, r))
            for c in ("amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"):
                rec[c] = round(float(rec[c]), 2) + 0.0
            for c in ("step", "isFraud", "isFlaggedFraud"):
                rec[c] = int(rec[c])
            self.fh.write(("" if self.first else ",\n") + json.dumps(rec))
            self.first = False

    def close(self) -> None:
        self.fh.write("\n]\n")


def load_id_numbers(paths: list[Path]) -> np.ndarray:
    """Numeric parts of every nameOrig / nameDest in other generated CSVs (to avoid reusing them)."""
    ids = set()
    for path in paths:
        with open(path, newline="") as fh:
            reader = csv.reader(fh)
            header = next(reader)
            io, idd = header.index("nameOrig"), header.index("nameDest")
            for row in reader:
                ids.add(int(row[io][1:]))
                ids.add(int(row[idd][1:]))
    return np.array(sorted(ids), dtype=np.int64)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--real", type=Path, required=True, help="Kaggle PaySim CSV used for calibration")
    p.add_argument("--calibration", type=Path, help="reuse an existing calibration.json (default: next to --out)")
    p.add_argument("--rows", type=int, default=3_181_310, help="rows to generate (default: same as each half)")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--balance-multiplier", type=float, default=8.0,
                   help="median starting balance as a multiple of a typical spend amount")
    p.add_argument("--balance-spread", type=float, default=0.4, help="lognormal sigma of that multiple")
    p.add_argument("--out", type=Path, default=Path("paysim_synthetic.csv"))
    p.add_argument("--format", choices=["csv", "json-array"], default="csv")
    p.add_argument("--start-step", type=int, default=1, help="first step (744 = the month after PaySim's)")
    p.add_argument("--steps", type=int, default=N_STEPS, help="number of hourly steps to simulate")
    p.add_argument("--exclude-ids-from", type=Path, nargs="*", default=[],
                   help="other generated CSVs whose customer/merchant ids must not be reused")
    args = p.parse_args()

    cache = args.calibration or args.out.with_name("calibration.json")
    cal = calibrate(args.real, cache)
    real_ids = np.load(cache.with_suffix(".ids.npy"))
    if args.exclude_ids_from:
        real_ids = np.union1d(real_ids, load_id_numbers(args.exclude_ids_from))
        print(f"avoiding {len(real_ids):,} existing ids (real file + {len(args.exclude_ids_from)} generated file(s))")
    print(f"calibrated from {cal['rows']:,} real rows; generating {args.rows:,} rows (seed {args.seed})")
    gen = Generator(cal, real_ids, np.random.default_rng(args.seed), args.rows, args.balance_multiplier,
                    args.balance_spread, args.start_step, args.steps)
    with open(args.out, "w", newline="") as fh:
        if args.format == "csv":
            fh.write(",".join(COLUMNS) + "\n")
            n = gen.run(lambda rows: write_rows(fh, rows))
        else:
            writer = JsonArrayWriter(fh)
            n = gen.run(writer)
            writer.close()
    print(f"wrote {args.out} ({n:,} rows, {args.out.stat().st_size / 1e6:,.1f} MB); "
          f"{gen.shrunk:,} outflows ({100 * gen.shrunk / n:.1f}%) reduced to fit the account balance")


if __name__ == "__main__":
    main()
