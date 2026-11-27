# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Environment check (RQ 1.2: Initial Validation)
# MAGIC **CS 483 Big Data Mining · Fall 2026 · Financial Transaction Fraud Detection**
# MAGIC
# MAGIC Run this notebook top to bottom on **serverless compute, environment version 4 or later**
# MAGIC (Environment side panel → Environment version; earlier versions do not support `pyspark.ml`).
# MAGIC
# MAGIC | Part | What it proves |
# MAGIC |---|---|
# MAGIC | 1. Access log | Who ran it; every team member runs this cell once, so the log shows all four of us |
# MAGIC | 2. Spark compute | Spark executes a distributed job (100 million row aggregation) |
# MAGIC | 3. Data access | All four project datasets are readable from the shared volume, with the expected row and fraud counts |
# MAGIC | 4. Delta table | Data can be written once to a shared table and read back by everyone |
# MAGIC | 5. Spark ML | MLlib can train a model (smoke test only, not a project result) |
# MAGIC | 6. Summary | One PASS/FAIL table to screenshot for Canvas |

# COMMAND ----------

# MAGIC %md ## 0. Settings

# COMMAND ----------

import os
import sys
import time
from datetime import datetime, timezone

try:  # Databricks: notebook widgets
    dbutils.widgets.text("catalog", "workspace")
    dbutils.widgets.text("schema", "cs483_fraud")
    dbutils.widgets.text("volume", "raw_data")
    CATALOG = dbutils.widgets.get("catalog")
    SCHEMA = dbutils.widgets.get("schema")
    VOLUME = dbutils.widgets.get("volume")
    DATA_DIR = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
    TABLE_FORMAT = "delta"
    ON_DATABRICKS = True
except NameError:  # local Spark (for testing the notebook outside Databricks)
    from pyspark.sql import SparkSession

    # 4 GB: parsing two 800 MB JSON arrays at once exhausts Spark's default 1 GB driver
    spark = (SparkSession.builder.master("local[*]").appName("env-check")
             .config("spark.driver.memory", "4g").getOrCreate())
    CATALOG, SCHEMA = None, os.environ.get("CHECK_SCHEMA", "cs483_fraud")
    DATA_DIR = os.environ["CHECK_DATA_DIR"]
    TABLE_FORMAT = os.environ.get("CHECK_TABLE_FORMAT", "parquet")
    ON_DATABRICKS = False

    def display(df):  # noqa: D103 - stand-in for Databricks display()
        df.show(truncate=False)

from pyspark.sql import functions as F

def table(name: str) -> str:
    return f"{CATALOG}.{SCHEMA}.{name}" if CATALOG else f"{SCHEMA}.{name}"

if ON_DATABRICKS:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
else:
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {SCHEMA}")

SCHEMA_DDL = ("step INT, type STRING, amount DOUBLE, nameOrig STRING, oldbalanceOrg DOUBLE, "
              "newbalanceOrig DOUBLE, nameDest STRING, oldbalanceDest DOUBLE, newbalanceDest DOUBLE, "
              "isFraud INT, isFlaggedFraud INT")

# Expected values, measured when the files were created (see data_selection.md)
DATASETS = [
    # file name,                     format, source,      month, rows,      frauds, flagged, min step, max step
    ("paysim_first_half.csv",         "csv",  "real",      1,     3_181_310, 2_699,  1,       1,        239),
    ("paysim_second_half.json",       "json", "real",      1,     3_181_310, 5_514,  15,      239,      743),
    ("paysim_synthetic.csv",          "csv",  "synthetic", 1,     3_181_310, 4_106,  9,       1,        743),
    ("paysim_synthetic_month2.json",  "json", "synthetic", 2,     3_181_310, 4_106,  7,       744,      1_486),
]

checks = []  # (part, check, expected, actual, passed, seconds)

def record(part, name, expected, actual, passed, started):
    checks.append((part, name, str(expected), str(actual), "PASS" if passed else "FAIL",
                   round(time.perf_counter() - started, 1)))
    print(f"[{'PASS' if passed else 'FAIL'}] {part} · {name}: expected {expected}, got {actual}")

print("Data folder:", DATA_DIR, "| table format:", TABLE_FORMAT)

# COMMAND ----------

# MAGIC %md ## 1. Access log: every team member runs this cell once

# COMMAND ----------

t = time.perf_counter()
who = spark.sql("SELECT current_user() AS user").first()["user"]
spark_version = spark.version
access = spark.createDataFrame(
    [(who, datetime.now(timezone.utc).replace(tzinfo=None), spark_version, sys.version.split()[0])],
    "user STRING, checked_at TIMESTAMP, spark_version STRING, python_version STRING",
)
access.write.format(TABLE_FORMAT).mode("append").saveAsTable(table("access_log"))
log = spark.table(table("access_log"))
display(log.groupBy("user").agg(F.min("checked_at").alias("first_access"),
                                F.max("checked_at").alias("last_access"),
                                F.count("*").alias("runs")).orderBy("first_access"))
record("1 Access", f"logged in as {who}", "user recorded", f"{log.select('user').distinct().count()} distinct user(s) so far",
       log.filter(F.col("user") == who).count() > 0, t)
print(f"Spark {spark_version} · Python {sys.version.split()[0]}")

# COMMAND ----------

# MAGIC %md ## 2. Spark compute: a distributed aggregation over 100 million generated rows

# COMMAND ----------

t = time.perf_counter()
r = (spark.range(100_000_000)
     .select((F.col("id") % 1000).alias("k"), F.col("id"))
     .groupBy("k").agg(F.sum("id").alias("s"))
     .agg(F.count("*").alias("groups"), F.sum("s").alias("total"))
     .first())
expected_total = 100_000_000 * (100_000_000 - 1) // 2
record("2 Compute", "sum of 0..99,999,999 in 1,000 groups", f"{expected_total:,} / 1000 groups",
       f"{r['total']:,} / {r['groups']} groups", r["total"] == expected_total and r["groups"] == 1000, t)

# COMMAND ----------

# MAGIC %md ## 3. Data access: read all four datasets from the shared volume

# COMMAND ----------

frames = []
for fname, fmt, source, month, exp_rows, exp_fraud, exp_flag, exp_min, exp_max in DATASETS:
    t = time.perf_counter()
    path = f"{DATA_DIR}/{fname}"
    try:
        reader = spark.read.schema(SCHEMA_DDL).option("mode", "FAILFAST")
        df = reader.option("header", True).csv(path) if fmt == "csv" else reader.option("multiLine", True).json(path)
        s = df.agg(F.count("*").alias("rows"), F.sum("isFraud").alias("fraud"),
                   F.sum("isFlaggedFraud").alias("flagged"), F.min("step").alias("min_step"),
                   F.max("step").alias("max_step")).first()
        actual = (s["rows"], s["fraud"], s["flagged"], s["min_step"], s["max_step"])
        expected = (exp_rows, exp_fraud, exp_flag, exp_min, exp_max)
        record("3 Data", f"{fname} (rows, frauds, flagged, steps)", expected, actual, actual == expected, t)
        frames.append(df.withColumn("source", F.lit(source)).withColumn("month", F.lit(month))
                        .withColumn("source_file", F.lit(fname)))
    except Exception as exc:  # noqa: BLE001 - report every failure in the summary
        record("3 Data", f"{fname}", "file readable", f"{type(exc).__name__}: {str(exc)[:120]}", False, t)

# COMMAND ----------

# MAGIC %md ## 4. Shared table: write all four sources into one table and read it back

# COMMAND ----------

t = time.perf_counter()
if len(frames) == len(DATASETS):
    combined = frames[0]
    for f in frames[1:]:
        combined = combined.unionByName(f)
    (combined.write.format(TABLE_FORMAT).mode("overwrite")
        .partitionBy("source", "month").saveAsTable(table("transactions_raw")))
    back = spark.table(table("transactions_raw"))
    per_source = back.groupBy("source", "month").agg(F.count("*").alias("rows"),
                                                     F.sum("isFraud").alias("frauds")).orderBy("source", "month")
    display(per_source)
    n = back.count()
    record("4 Table", f"{table('transactions_raw')} row count", f"{12_725_240:,}", f"{n:,}", n == 12_725_240, t)
else:
    record("4 Table", "transactions_raw", "all 4 files read", f"{len(frames)} read", False, t)

# COMMAND ----------

# MAGIC %md ## 5. Spark ML smoke test: train a tiny model on a 2% sample
# MAGIC Only checks that MLlib works in this environment. The score below is **not** a project result.

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler

t = time.perf_counter()
try:
    sample = (spark.table(table("transactions_raw"))
              .filter(F.col("source") == "real")
              .sampleBy("isFraud", {0: 0.02, 1: 0.5}, seed=42)
              .withColumn("is_transfer", (F.col("type") == "TRANSFER").cast("double"))
              .withColumn("is_cash_out", (F.col("type") == "CASH_OUT").cast("double"))
              .withColumn("log_amount", F.log1p("amount")))
    model = Pipeline(stages=[
        VectorAssembler(inputCols=["log_amount", "is_transfer", "is_cash_out"], outputCol="features"),
        LogisticRegression(labelCol="isFraud", featuresCol="features", maxIter=20),
    ]).fit(sample)
    auc = BinaryClassificationEvaluator(labelCol="isFraud", metricName="areaUnderROC").evaluate(model.transform(sample))
    record("5 Spark ML", "LogisticRegression trains and scores", "model trained, 0 < AUC <= 1", f"AUC {auc:.3f}",
           0 < auc <= 1, t)
except Exception as exc:  # noqa: BLE001
    record("5 Spark ML", "LogisticRegression trains and scores", "model trained",
           f"{type(exc).__name__}: {str(exc)[:120]} (check environment version >= 4)", False, t)

# COMMAND ----------

# MAGIC %md ## 6. Summary: screenshot this for Canvas

# COMMAND ----------

summary = spark.createDataFrame(checks, "part STRING, check STRING, expected STRING, actual STRING, "
                                        "result STRING, seconds DOUBLE")
display(summary)
passed = sum(1 for c in checks if c[4] == "PASS")
print(f"{passed}/{len(checks)} checks passed · run by {who} at "
      f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC · Spark {spark_version}")
summary.withColumn("run_by", F.lit(who)).withColumn("run_at", F.current_timestamp()) \
    .write.format(TABLE_FORMAT).mode("append").saveAsTable(table("environment_check_runs"))
