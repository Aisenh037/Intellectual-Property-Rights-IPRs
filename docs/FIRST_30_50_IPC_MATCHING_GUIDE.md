# Step-by-Step Guide: First IPC, 30%, and 50% Green Patent Matching

This documentation guides you through running the standalone **First IPC**, **30% IPC**, and **50% IPC** Green Patent matching algorithms using the WIPO IPC Green Inventory.

---

## 1. Overview of Matching Metrics

| Metric | Condition | Description | Output Column |
| :--- | :--- | :--- | :--- |
| **First IPC Green** | First IPC token contains a WIPO Green code | Evaluates whether the primary/first classification of the patent belongs to green technologies. | `First_IPC_Green`, `Matched_IPC_first` |
| **30% IPC Green** | $\frac{\text{Green IPC Tokens}}{\text{Total IPC Tokens}} \ge 0.30$ | Evaluates if at least 30% of the patent's IPC classifications represent green technologies. | `30-IPC-Green`, `Matched_IPC-30` |
| **50% IPC Green** | $\frac{\text{Green IPC Tokens}}{\text{Total IPC Tokens}} \ge 0.50$ | Evaluates if at least 50% (majority) of the patent's IPC classifications represent green technologies. | `50-IPC-Green`, `Matched_IPC-50` |

---

## 2. Requirements & Setup

Ensure the following Python packages are installed:

```bash
pip install pandas numpy python-calamine openpyxl xlsxwriter
```

---

## 3. Input Data Files Required

1. **Input Patent Dataset** (`.csv` or `.xlsx`):
   - Must contain `Application No.` and `IPC All Versions (IC)`.
2. **WIPO Green IPC Reference List**:
   - `cleaned_WIPO_IPC_green_codes.csv` (contains column `clean_code`).
3. *(Optional)* **Baseline Green Patent Reference Set**:
   - `Green patents 48972.xlsx` (contains 48,972 verified green applications).

---

## 4. Running via Command Line (CLI)

You can run the script directly from your terminal or command prompt:

```bash
python sdg_pipeline/08_match_first_30_50_green_ipc.py \
    --input "Grant_Data_SDG_Green_Cleaned_IPC.csv" \
    --wipo "cleaned_WIPO_IPC_green_codes.csv" \
    --green-ref "Green patents 48972.xlsx" \
    --output "Grant_Data_Green_Features_Output.csv"
```

### CLI Arguments:
- `--input`: Path to input patent dataset (`.csv` or `.xlsx`).
- `--wipo`: Path to cleaned WIPO Green IPC codes (`cleaned_WIPO_IPC_green_codes.csv`).
- `--green-ref` *(Optional)*: Path to baseline green applications file.
- `--output`: Path to write the resulting file (`.csv` or `.xlsx`).

---

## 5. Running as a Python Module / Jupyter Notebook

You can also import and use the function in your own Python script or Jupyter Notebook:

```python
import pandas as pd
from sdg_pipeline.08_match_first_30_50_green_ipc import (
    load_wipo_green_codes,
    match_green_ipc_features
)

# 1. Load your dataset
df = pd.read_csv("Grant_Data_SDG_Green_Cleaned_IPC.csv")

# 2. Load WIPO green codes
codes = load_wipo_green_codes("cleaned_WIPO_IPC_green_codes.csv")

# 3. (Optional) Load 48,972 reference set
from python_calamine import CalamineWorkbook
wb = CalamineWorkbook.from_path("Green patents 48972.xlsx")
green_apps = set(r[0] for r in wb.get_sheet_by_name(wb.sheet_names[0]).to_python()[1:] if r and r[0])

# 4. Generate the 6 green feature columns
features_df = match_green_ipc_features(
    df=df,
    green_codes=codes,
    green_reference_apps=green_apps,
    ipc_column="IPC All Versions (IC)",
    app_column="Application No."
)

# 5. Concatenate and save
df_final = pd.concat([df, features_df], axis=1)
df_final.to_csv("Grant_Data_Green_Features_Output.csv", index=False)
```

---

## 6. Expected Results on the Harmonized Indian Patent Dataset

| Metric | Green (1) | Non-Green (0) |
| :--- | :--- | :--- |
| **Baseline Green (`IC_Code_Type`)** | **48,972** | **305,086** |
| **First IPC Green (`First_IPC_Green`)** | **29,072** | **324,986** |
| **30% IPC Green (`30-IPC-Green`)** | **34,301** | **319,757** |
| **50% IPC Green (`50-IPC-Green`)** | **26,542** | **327,516** |
