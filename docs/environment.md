# Big Data Processing Environment (RQ 1.2)

**Team "Team Name (That is the team name)"**: Ritesh Goud Annela (ranne3) · Pruthvi Biradar (pbira2) · Joel Noble Maliekal (jjohn345) · Gnana Sai Teja Reddy Gangalapudi (ggang4)  
**Project:** Financial Transaction Fraud Detection (PaySim, 12.7 million transactions, 2.1 GB across 4 files)

## 1. Platform Selection

**Platform: Databricks Free Edition**, a cloud workspace with serverless Apache Spark compute that every team member can use at no cost.

| Technology | What we use it for |
|---|---|
| **Apache Spark (PySpark, Spark SQL)** | Distributed loading, cleaning, joining and aggregation of the 12.7M transactions |
| **Spark MLlib (`pyspark.ml`)** | Logistic Regression and Random Forest; runs on serverless environment version 4 or later |
| **Delta Lake** | One shared, partitioned table (`transactions_raw`) built from our CSV and JSON files. The JSON arrays are parsed once instead of on every run. |
| **Unity Catalog volume** | Shared storage for the four raw data files: `/Volumes/workspace/cs483_fraud/raw_data/` |
| **Spark Structured Streaming** | Replaying month 2 as newly arriving data, using incremental triggers |
| **Databricks Git folders + GitHub** | Version control for notebooks, scripts and documentation |
| **Python 3** | Data generation and verification scripts, and notebook code |
| **Local PySpark** (team laptops) | Fallback, and the core-count scalability tests that serverless compute does not let us control |
| **Trello** | Sprint task board (Agile Scrum, weekly check-ins) |

**Why this platform:**
- The data is 2.1 GB, split across CSV and non-splittable JSON arrays, and the models need repeated passes over millions of rows. Spark parallelises that work.
- Databricks gives all four of us the same Spark environment, data and tables with no installation.
- Free Edition supports team collaboration: there is no limit on how many students can share an account. Shared accounts can reach daily compute limits, so we plan heavy jobs across several days.

## 2. Team Collaboration

### a. How we access and share the environment

- **One shared team workspace.** It's created by Joel Noble Maliekal (jjohn345), who is the workspace admin. The workspace admin added the other three members by their UIC email addresses. Everyone signs in with their own account (email code, Google or Microsoft sign-in), so all activity is attributed to the person who did it.
- **Shared data.** The four dataset files are uploaded **once** to the Unity Catalog volume `workspace.cs483_fraud.raw_data`. Every notebook reads from that path, so nobody keeps a private copy.
- **Shared tables.** Prepared data is written to Delta tables in the schema `workspace.cs483_fraud` (for example `transactions_raw`), which all members can query.
- **Shared code.** Each member works in their own Git folder linked to our GitHub repository (see 2b). Finished notebooks are merged on GitHub and pulled by everyone.
- **Quota etiquette.** Heavy jobs (full-data model training, cross-validation, scalability runs) are scheduled on the Trello board, so we don't use up the shared daily compute quota at the same time.

### b. How GitHub manages our source code and documentation

**Repository:** https://github.com/G-Gangalapudi/CS-483-FinalProject

- **What goes in the repository:**
  - notebooks, in Databricks source format (`.py`)
  - data scripts (`split_paysim.py`, `verify_split.py`, `generate_synthetic.py`, `compare_to_real.py`)
  - documentation (`data_selection.md`, this page as `docs/environment.md`, the synthetic data cards)
  - test results
- **What does not:** the data. Each file is over GitHub's 100 MB limit, so `.gitignore` excludes it. The data is reproducible from the Kaggle download and our seeded scripts, and identified by SHA-256 checksums in `data_selection.md`.
- **Linking Databricks to GitHub:** each member links Databricks to GitHub with their **own** personal access token and clones the repository as a Databricks Git folder. Commits made in Databricks are therefore credited to that member's GitHub account.
- **Workflow:**
  1. Each member works on their own branch (e.g. `Joel`).
  2. Changes go into `main` only through a pull request, reviewed by another member.
  3. Each pull request is linked to its Trello card.

## 3. Initial Validation

### a. Access by every team member

Each member opened the workspace and ran the first cell of the notebook `00_environment_check`. That cell records the signed-in user in the shared table `workspace.cs483_fraud.access_log`, so the table proves which members can access the environment:

| User | First access (UTC) |
|---|---|
| jjohn345@uic.edu | 2026-11-27 |
| ⟨TO FILL: member 2 email⟩ | ⟨TO FILL⟩ |
| ⟨TO FILL: member 3 email⟩ | ⟨TO FILL⟩ |
| ⟨TO FILL: member 4 email⟩ | ⟨TO FILL⟩ |

⟨TO FILL: screenshot of the access_log output⟩

### b. Initial test

**Notebook:** `notebooks/00_environment_check.py` (in the GitHub repository)  
**Run on:** Databricks Free Edition, serverless compute, base environment Standard v6 (Python 3.12, Spark 4.2.0), 16 GB memory  
**Run by / date:** Joel Noble Maliekal (jjohn345@uic.edu), 2026-11-27 20:01 UTC

The test checks the whole path our project depends on:

1. **Access log.** Records who ran the notebook.
2. **Spark compute.** A distributed aggregation over 100 million generated rows, checked against the exact mathematical total.
3. **Data access.** Reads all four datasets from the shared volume, the CSV files and the multi-line JSON arrays, with an explicit schema and strict parsing. Each file's row, fraud and flag counts and step range are compared with the values we measured when the files were created.
4. **Shared table.** Combines the four sources into one partitioned Delta table (`transactions_raw`), reads it back and checks the total of 12,725,240 rows.
5. **Spark ML.** Trains a small Logistic Regression on a 2% sample to confirm MLlib works in this environment. This is a smoke test, not a project result.

### c. Test results

| Part | Check | Expected | Result in Databricks |
|---|---|---|---|
| 1 Access | Signed-in user recorded | user recorded | **PASS**: jjohn345@uic.edu recorded |
| 2 Compute | Sum of 0…99,999,999 in 1,000 groups | 4,999,999,950,000,000 | **PASS**: 4,999,999,950,000,000 in 1,000 groups |
| 3 Data | `paysim_first_half.csv`: rows, frauds, flagged, steps | 3,181,310 · 2,699 · 1 · 1–239 | **PASS**: 3,181,310 · 2,699 · 1 · 1–239 |
| 3 Data | `paysim_second_half.json` | 3,181,310 · 5,514 · 15 · 239–743 | **PASS**: 3,181,310 · 5,514 · 15 · 239–743 |
| 3 Data | `paysim_synthetic.csv` | 3,181,310 · 4,106 · 9 · 1–743 | **PASS**: 3,181,310 · 4,106 · 9 · 1–743 |
| 3 Data | `paysim_synthetic_month2.json` | 3,181,310 · 4,106 · 7 · 744–1,486 | **PASS**: 3,181,310 · 4,106 · 7 · 744–1,486 |
| 4 Table | `transactions_raw` total rows | 12,725,240 | **PASS**: 12,725,240 |
| 5 Spark ML | Logistic Regression trains and scores | model trained | **PASS**: trained, AUC 0.884 on its own sample (smoke test, not a result) |
| | **Overall** | **8/8 PASS** | **8/8 PASS** |

Notebook output: `8/8 checks passed · run by jjohn345@uic.edu at 2026-11-27 20:01 UTC · Spark 4.2.0`

⟨TO FILL: screenshot of the notebook's summary table⟩

**Before running on Databricks, we also validated the same notebook on a local PySpark 4.3 installation using the same four files: 8/8 checks passed.** That run also gave us evidence for our Volume analysis: with Spark's default 1 GB driver memory, writing the combined table failed with an out-of-memory error. The cause was parsing the two 800 MB JSON arrays at the same time; with 4 GB it succeeded.

**Notes for running on serverless compute** (documented for the team):
- Use environment version 4 or later; earlier versions do not support `pyspark.ml`.
- `df.cache()`, `spark.sparkContext` and RDD APIs are not available on serverless compute; our code avoids them.
- Spark ML models are limited to 100 MB each on serverless, which bounds our Random Forest size.

## 4. Sharing with Instructors

- **Databricks workspace.** On request, the workspace admin adds each instructor's email as a regular (non-admin) workspace user. Instructors can then open the shared notebooks, see the stored run outputs, and query the tables in `workspace.cs483_fraud`, including the `access_log` and `environment_check_runs` tables, which keep a history of every validation run. Free Edition does not allow users to be removed from an account once added, so we only add instructors who want access.
- **GitHub.** Instructors are invited as collaborators on https://github.com/G-Gangalapudi/CS-483-FinalProject, where all notebooks, scripts and documentation are versioned. For each milestone we also commit an HTML export of the executed notebooks to `reports/`, so results can be viewed without signing in to Databricks.
- **Task board.** The Trello board linked in our proposal shows sprint progress and who is working on what.
