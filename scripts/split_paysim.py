"""Step 1: split the PaySim dataset in half.
First half  -> CSV
Second half -> JSON

Usage
    python split_paysim.py                                   # downloads from Kaggle if needed
    python split_paysim.py --input PS_20174392719_1491204439457_log.csv
    python split_paysim.py --json-format array               # one big JSON array instead of JSON Lines

Halves follow the file's original row order (the file is sorted by `step`, the
simulation hour), so the first half is the earlier transactions.

Uses only the Python standard library and streams row by row, so it never
loads the 470 MB file into memory. Downloading needs `pip install kagglehub`
and a Kaggle API token in ~/.kaggle/kaggle.json.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

DATASET = "ealaxi/paysim1"
FILE_NAME = "PS_20174392719_1491204439457_log.csv"

INT_COLS = {"step", "isFraud", "isFlaggedFraud"}
FLOAT_COLS = {"amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"}


def download() -> Path:
    import kagglehub

    folder = Path(kagglehub.dataset_download(DATASET))
    return next(folder.rglob(FILE_NAME))


def typed(row: dict) -> dict:
    """Numbers become JSON numbers instead of strings."""
    out = {}
    for k, v in row.items():
        if k in INT_COLS:
            out[k] = int(v)
        elif k in FLOAT_COLS:
            out[k] = float(v)
        else:
            out[k] = v
    return out


def count_rows(path: Path) -> int:
    with open(path, "rb") as fh:
        return sum(1 for _ in fh) - 1  # minus header


def split(input_path: Path, out_dir: Path, json_format: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    total = count_rows(input_path)
    half = (total + 1) // 2  # if the row count is odd, the first half gets the extra row
    print(f"{total:,} rows -> first half {half:,} (CSV), second half {total - half:,} (JSON)")

    ext = "jsonl" if json_format == "jsonl" else "json"
    csv_path = out_dir / "paysim_first_half.csv"
    json_path = out_dir / f"paysim_second_half.{ext}"

    with open(input_path, newline="") as src, open(csv_path, "w", newline="") as csv_out, \
            open(json_path, "w") as json_out:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(csv_out, fieldnames=reader.fieldnames)
        writer.writeheader()

        if json_format == "array":
            json_out.write("[\n")
        first_json = True
        for i, row in enumerate(reader):
            if i < half:
                writer.writerow(row)
            else:
                record = json.dumps(typed(row))
                if json_format == "jsonl":
                    json_out.write(record + "\n")
                else:
                    json_out.write(("" if first_json else ",\n") + record)
                    first_json = False
            if (i + 1) % 1_000_000 == 0:
                print(f"  {i + 1:,} rows done")
        if json_format == "array":
            json_out.write("\n]\n")

    print(f"wrote {csv_path} ({csv_path.stat().st_size / 1e6:,.1f} MB)")
    print(f"wrote {json_path} ({json_path.stat().st_size / 1e6:,.1f} MB)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, help=f"path to {FILE_NAME} (downloaded from Kaggle if omitted)")
    p.add_argument("--out-dir", type=Path, default=Path("data/split"))
    p.add_argument("--json-format", choices=["jsonl", "array"], default="jsonl",
                   help="jsonl = one record per line (Spark's default, default here); array = one JSON array")
    args = p.parse_args()

    input_path = args.input or download()
    split(input_path, args.out_dir, args.json_format)


if __name__ == "__main__":
    main()
