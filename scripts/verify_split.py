"""One-time integrity check: did splitting PaySim into CSV + JSON lose or change anything?

    python verify_split.py <original.csv> <first_half.csv> <second_half.json>

Streams all three files row by row (low memory). Uses only the standard library.
"""
import csv
import hashlib
import json
import sys
from collections import Counter
from decimal import Decimal

orig_path, csv_path, json_path = sys.argv[1:4]
FLOATS = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
INTS = ["step", "isFraud", "isFlaggedFraud"]
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def iter_json_array(path, chunk_size=1 << 20):
    """Parse a top-level JSON array one element at a time with the standard JSON decoder."""
    dec = json.JSONDecoder()
    with open(path) as fh:
        buf, pos, started, ended, eof = "", 0, False, False, False
        while not ended:
            if not eof and len(buf) - pos < chunk_size:
                more = fh.read(chunk_size)
                eof = not more
                buf = buf[pos:] + more
                pos = 0
            while pos < len(buf) and buf[pos] in " \t\r\n,":
                if buf[pos] == "," and not started:
                    raise ValueError("comma before array start")
                pos += 1
            if pos >= len(buf):
                if eof:
                    raise ValueError("JSON array is not closed with ]")
                continue
            ch = buf[pos]
            if not started:
                if ch != "[":
                    raise ValueError("file does not start with a JSON array")
                started, pos = True, pos + 1
            elif ch == "]":
                ended, pos = True, pos + 1
            else:
                obj, pos = dec.raw_decode(buf, pos)  # each record is far smaller than a chunk
                yield obj
        if (buf[pos:] + fh.read()).strip():
            raise ValueError("unexpected data after the JSON array")


class Profile:
    def __init__(self):
        self.rows, self.flagged, self.blank = 0, 0, 0
        self.types, self.fraud_by_type = Counter(), Counter()
        self.sums = {c: 0.0 for c in FLOATS}
        self.min, self.max = {}, {}

    def add(self, r):
        self.rows += 1
        self.types[r["type"]] += 1
        self.fraud_by_type[r["type"]] += int(r["isFraud"])
        self.flagged += int(r["isFlaggedFraud"])
        self.blank += sum(1 for v in r.values() if v is None or str(v).strip() == "")
        for c in FLOATS + INTS:
            v = float(r[c])
            if c in FLOATS:
                self.sums[c] += v
            self.min[c] = min(self.min.get(c, v), v)
            self.max[c] = max(self.max.get(c, v), v)


def fmt(v):
    """Write a number the way the original Kaggle file does (Java's Double.toString):
    plain decimals between 0.001 and 10,000,000, scientific notation like 1.04197848E7 outside."""
    if not isinstance(v, float):
        return str(v)
    if v == 0 or 1e-3 <= abs(v) < 1e7:
        return repr(v)
    sign, digits, exp = Decimal(repr(v)).normalize().as_tuple()  # shortest round-trip digits
    ds = "".join(map(str, digits))
    return f"{'-' if sign else ''}{ds[0]}.{ds[1:] or '0'}E{len(ds) + exp - 1}"


def line(row):
    return (",".join(row) + "\n").encode()


h_orig, h_rebuilt = hashlib.sha256(), hashlib.sha256()
po, pa, pb = Profile(), Profile(), Profile()

with open(orig_path, newline="") as fo, open(csv_path, newline="") as fa:
    orig_reader, csv_reader = csv.reader(fo), csv.reader(fa)
    orig_header, csv_header = next(orig_reader), next(csv_reader)
    check("CSV header identical to original (same columns, same order)", csv_header == orig_header)
    h_orig.update(line(orig_header))
    h_rebuilt.update(line(csv_header))

    # ---- first half: CSV vs original rows 1..n --------------------------------
    n_csv, csv_mismatch = 0, 0
    for a in csv_reader:
        o = next(orig_reader)
        n_csv += 1
        csv_mismatch += a != o
        h_orig.update(line(o))
        h_rebuilt.update(line(a))
        po.add(dict(zip(orig_header, o)))
        pa.add(dict(zip(orig_header, a)))

    # ---- second half: JSON vs original rows n+1..end --------------------------
    n_json, json_mismatch, extra_json, wrong_keys, wrong_types, text_diffs, first_bad = 0, 0, 0, 0, 0, 0, None
    for rec in iter_json_array(json_path):
        n_json += 1
        o = next(orig_reader, None)
        if o is None:
            extra_json += 1
            continue
        od = dict(zip(orig_header, o))
        wrong_keys += list(rec.keys()) != orig_header
        wrong_types += not (all(type(rec.get(c)) is float for c in FLOATS)
                            and all(type(rec.get(c)) is int for c in INTS))
        same = all(
            rec.get(c) == (float(od[c]) if c in FLOATS else int(od[c]) if c in INTS else od[c])
            for c in orig_header
        )
        if not same:
            json_mismatch += 1
            first_bad = first_bad or (n_csv + n_json, od, rec)
        text_diffs += sum(fmt(rec.get(c)) != od[c] for c in FLOATS)
        h_orig.update(line(o))
        h_rebuilt.update(line([fmt(rec.get(c)) for c in orig_header]))
        po.add(od)
        pb.add(rec)
    leftover = sum(1 for _ in orig_reader)  # original rows missing from both halves

total = po.rows + leftover
print(f"\noriginal {total:,} rows | CSV half {n_csv:,} | JSON half {n_json:,}\n")
check("Row count: CSV + JSON = original", n_csv + n_json == total and leftover == 0 and extra_json == 0,
      f"{n_csv:,} + {n_json:,} = {n_csv + n_json:,} vs {total:,}; missing {leftover}, extra {extra_json}")
check("Halves are equal size", n_csv == n_json)
check("JSON file is one valid JSON array (every element parsed, closed with ])", True)
check("Every JSON record has exactly the original 11 columns, in order", wrong_keys == 0, f"{wrong_keys} bad")
check("JSON numbers stored as numbers (floats/ints), not strings", wrong_types == 0, f"{wrong_types} bad")
check(f"CSV half = original rows 1–{n_csv:,} exactly (text, in order)", csv_mismatch == 0,
      f"{csv_mismatch} mismatched rows")
check(f"JSON half = original rows {n_csv + 1:,}–{total:,} value by value (in order)", json_mismatch == 0,
      f"{json_mismatch} mismatched rows" + (f"; first: {first_bad}" if first_bad else ""))
check("No numeric precision lost (every JSON number writes back to the original digits)",
      text_diffs == 0, f"{text_diffs} values differ")
check("Original file rebuilt byte-for-byte from the two halves (SHA-256 match)",
      h_orig.hexdigest() == h_rebuilt.hexdigest(), f"sha256 {h_orig.hexdigest()[:16]}…")
check("No blank / null values in either half", pa.blank == 0 and pb.blank == 0)
check("Fraud count preserved",
      sum((pa.fraud_by_type + pb.fraud_by_type).values()) == sum(po.fraud_by_type.values()),
      f"{sum(pa.fraud_by_type.values()):,} + {sum(pb.fraud_by_type.values()):,} = {sum(po.fraud_by_type.values()):,}")
check("isFlaggedFraud count preserved", pa.flagged + pb.flagged == po.flagged,
      f"{pa.flagged} + {pb.flagged} = {po.flagged}")
check("Transaction-type counts preserved", pa.types + pb.types == po.types)
check("Fraud-by-type counts preserved", pa.fraud_by_type + pb.fraud_by_type == po.fraud_by_type)
for c in FLOATS:
    s = pa.sums[c] + pb.sums[c]
    check(f"Sum of {c} preserved", abs(s - po.sums[c]) <= 1e-9 * abs(po.sums[c]), f"{s:,.2f} vs {po.sums[c]:,.2f}")
check("Min/max of every numeric column preserved", all(
    min(pa.min[c], pb.min[c]) == po.min[c] and max(pa.max[c], pb.max[c]) == po.max[c] for c in FLOATS + INTS))

print(f"\n{sum(results)}/{len(results)} checks passed")
print(f"\nOriginal:  {po.rows:,} rows · {sum(po.fraud_by_type.values()):,} frauds · {po.flagged} flagged · "
      f"types {dict(po.types)}")
print(f"CSV half:  {pa.rows:,} rows · {sum(pa.fraud_by_type.values()):,} frauds · steps "
      f"{pa.min['step']:.0f}–{pa.max['step']:.0f}")
print(f"JSON half: {pb.rows:,} rows · {sum(pb.fraud_by_type.values()):,} frauds · steps "
      f"{pb.min['step']:.0f}–{pb.max['step']:.0f}")
