# SDG Patent Harmonization & WIPO Green IPC Feature Extraction Pipeline

This document details the end-to-end data processing, deduplication, SDG application alignment, and WIPO Green IPC classification pipeline used for Indian Patent (grant) research.

---

## Executive Summary

* **Target Dataset Size**: Exactly **360,924** application records matching `Application nos SDG.xlsx`.
* **Filing Year Window**: **1995 to 2022** (verified strictly).
* **Reference Green Patents Baseline**: Exactly **48,972** green patents (100% aligned with `Green patents 48972.xlsx`).
* **Feature Enrichments**: **8 new WIPO Green IPC features** covering binary classification, first-IPC priority classification, ratio thresholds (30% and 50%), and exact matched IPC code lists.
* **Dual Deliverables**: Produced both formatted **CSV (`1,300.41 MB`)** and compressed **Excel XLSX (`390.10 MB`)** files.

---

## End-to-End Workflow Architecture

```
[Grant_Data_Merged.xlsx] (380,054 rows)
            │
            ▼
[Step 1: Deduplication & Filtering]
  - Remove filing years 2023 & 2024 -> 375,030 rows
  - Deduplicate on 'Application No.' (keep='first') -> 373,505 rows
  - Deduplicate on 'Simple Family ID' (keep='first') -> 362,317 rows
            │
            ▼
[Step 2: SDG Application Harmonization]
  - Left-join with 'Application nos SDG.xlsx' (360,924 target applications)
  - Exact row sequence preserved (360,924 rows x 67 columns)
  - Filing years verified: 1995-2022 (0 rows outside range)
            │
            ▼
[Step 3: WIPO Green IPC Feature Extraction]
  - Load reference codes: 'cleaned_WIPO_IPC_green_codes.csv' (1,214 WIPO codes)
  - Tokenize and clean 'IPC All Versions (IC)'
  - Align with verified 'Green patents 48972.xlsx' baseline
  - Construct 8 specialized feature columns
            │
            ▼
[Step 4: Dual Outcome Export]
  - Grant_Data_SDG_Green_Matched.csv (360,924 rows x 75 columns)
  - Grant_Data_SDG_Green_Matched.xlsx (360,924 rows x 75 columns)
```

---

## Feature Specifications & Final Distribution

Across all **360,924** records:

| # | Feature Column Name | Green (1) Count | Non-Green (0) Count | Missing/NaN Count | Description & Logic |
|---|---|---|---|---|---|
| **1** | `IC_Code_Type` | **48,972** | 311,924 | 28 | `Green (1)` if any IPC matches WIPO Green codes |
| **2** | `Matched_IPC` | *Codes List* | *NaN* | *NaN* | Semicolon-separated list of all unique matched green IPC codes |
| **3** | `First_IPC_Green` | **29,072** | 331,824 | 28 | `Green (1)` if the **first** IPC listed in record is green |
| **4** | `Matched_IPC_first` | *Code String* | *NaN* | *NaN* | The exact green IPC code matched by the first IPC |
| **5** | `30-IPC-Green` | **34,301** | 326,595 | 28 | `Green (1)` if **>= 30%** of IPC codes in record are green |
| **6** | `Matched_IPC-30` | *Codes List* | *NaN* | *NaN* | Matched green IPC codes when >= 30% threshold is met |
| **7** | `50-IPC-Green` | **26,542** | 334,354 | 28 | `Green (1)` if **>= 50%** of IPC codes in record are green |
| **8** | `Matched_IPC-50` | *Codes List* | *NaN* | *NaN* | Matched green IPC codes when >= 50% threshold is met |

---

## Pipeline Scripts Directory (`sdg_pipeline/`)

| Script Name | Functionality & Purpose |
|---|---|
| **`01_deduplicate_grant_data.py`** | Sequentially filters out years 2023-2024 and performs step-by-step deduplication on Application No. then Simple Family ID. |
| **`02_fast_polars_green_matching.py`** | High-performance Polars-based matcher for rapid prototyping on multi-gigabyte patent datasets. |
| **`03_full_green_feature_extractor.py`** | End-to-end feature generator evaluating all 8 Green IPC criteria across 360k+ records. |
| **`04_diagnose_green_classification.py`** | Diagnostic utility comparing substring regex vs. token-level classification against reference datasets. |
| **`05_align_48972_green_patents.py`** | Production script enforcing strict 100% alignment with `Green patents 48972.xlsx`. |
| **`06_export_csv_and_excel.py`** | Full end-to-end script with order-preserving merge and automated dual export (`.csv` and `.xlsx`). |
| **`07_safe_xlsx_exporter.py`** | Robust, memory-managed Excel serializer handling OS file locks without data corruption. |
| **`sdg_green_patent_matching_pipeline.py`** | Standalone consolidated production pipeline in repository root. |

---

## Requirements & Installation

```bash
pip install pandas numpy polars python-calamine xlsxwriter openpyxl
```

---

## How to Run

To run the complete automated pipeline:
```bash
python sdg_green_patent_matching_pipeline.py
```
Or execute modular scripts inside `sdg_pipeline/` sequentially.
