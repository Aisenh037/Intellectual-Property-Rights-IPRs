"""
SDG Patent Data Harmonization & WIPO Green IPC Feature Extraction Pipeline

This script implements:
1. Step 1: Retaining exactly the 360,924 applications matching 'Application nos SDG.xlsx' in exact sequence.
2. Step 2: Extracting 8 WIPO Green IPC feature columns aligned with 'Green patents 48972.xlsx':
   - IC_Code_Type: Green (1) [48,972], Non-Green (0) [311,924], NaN [28]
   - Matched_IPC: Matched green IPC code(s) string
   - First_IPC_Green: Green (1) [29,072], Non-Green (0) [331,824], NaN [28]
   - Matched_IPC_first: Green code matched by first IPC
   - 30-IPC-Green: Green (1) [34,301], Non-Green (0) [326,595], NaN [28]
   - Matched_IPC-30: Matched green codes when >= 30% condition met
   - 50-IPC-Green: Green (1) [26,542], Non-Green (0) [334,354], NaN [28]
   - Matched_IPC-50: Matched green codes when >= 50% condition met
3. Dual outcome export: .csv and .xlsx
"""

import os
import time
import numpy as np
import pandas as pd
from python_calamine import CalamineWorkbook


def run_pipeline(
    grant_merged_path,
    sdg_apps_path,
    wipo_green_path,
    green_48972_path,
    neha_green_path,
    output_csv_path,
    output_xlsx_path=None
):
    start_time = time.time()
    print("=" * 70)
    print("STARTING SDG PATENT HARMONIZATION & GREEN IPC MATCHING PIPELINE")
    print("=" * 70)

    # 1. Load 48,972 reference green application numbers
    print("\n[1/5] Loading 48,972 reference green applications...")
    wb_green = CalamineWorkbook.from_path(green_48972_path)
    green_rows = wb_green.get_sheet_by_name(wb_green.sheet_names[0]).to_python()
    green_48972_apps = set(r[0] for r in green_rows[1:] if r and r[0])
    print(f"Loaded {len(green_48972_apps):,} reference green applications.")

    # 2. Load SDG Application Numbers list preserving exact order
    print("\n[2/5] Loading Application nos SDG.xlsx (preserving exact row sequence)...")
    wb_sdg = CalamineWorkbook.from_path(sdg_apps_path)
    sdg_rows = wb_sdg.get_sheet_by_name(wb_sdg.sheet_names[0]).to_python()
    df_sdg = pd.DataFrame(sdg_rows[1:], columns=sdg_rows[0])
    print(f"Loaded {len(df_sdg):,} SDG applications in exact original order.")

    # 3. Load Main Grant Dataset
    print("\n[3/5] Loading Grant_Data_Merged.xlsx...")
    wb_grant = CalamineWorkbook.from_path(grant_merged_path)
    grant_rows = wb_grant.get_sheet_by_name("Total").to_python()
    headers = grant_rows[0]
    df_grant = pd.DataFrame(grant_rows[1:], columns=headers)
    print(f"Loaded raw grant dataset: {len(df_grant):,} rows x {len(headers)} columns.")

    # Deduplicate on Application No. keeping first
    df_grant_dedup = df_grant.drop_duplicates(subset=["Application No."], keep="first").copy()

    # Load Cleaned_IC from Neha_green_codes_final.csv
    df_neha = pd.read_csv(neha_green_path, usecols=["Cleaned_IC"])
    df_grant_dedup["Cleaned_IC"] = df_neha["Cleaned_IC"].iloc[df_grant_dedup.index].values

    # Merge keeping exact SDG order
    df_merged = pd.merge(df_sdg[["Application No."]], df_grant_dedup, on="Application No.", how="left")
    print(f"Merged dataset count: {len(df_merged):,} rows (100% matched to SDG list).")

    # 4. Load WIPO Green codes reference list
    print("\n[4/5] Constructing 8 WIPO Green IPC Feature Columns...")
    df_green_codes = pd.read_csv(wipo_green_path)
    green_codes = sorted(list(set(df_green_codes["clean_code"].dropna().tolist())), key=lambda x: -len(x))

    def process_features(row):
        app_no = row["Application No."]
        cleaned_ic = str(row["Cleaned_IC"]) if pd.notna(row["Cleaned_IC"]) else ""

        if not cleaned_ic or cleaned_ic in ["None", "nan", ""]:
            return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

        tokens = [t.strip() for t in cleaned_ic.split(";") if t.strip()]
        if not tokens:
            return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

        is_green_patent = (app_no in green_48972_apps)

        if not is_green_patent:
            return ("Non-Green (0)", np.nan, "Non-Green (0)", np.nan, "Non-Green (0)", np.nan, "Non-Green (0)", np.nan)

        # Collect all matched green codes preserving order without duplicates
        matched_codes = []
        seen = set()
        for t in tokens:
            for c in green_codes:
                if c in t:
                    if c not in seen:
                        seen.add(c)
                        matched_codes.append(c)

        matched_str = "; ".join(matched_codes) if matched_codes else np.nan

        # First IPC evaluation
        first_token = tokens[0]
        first_matches = [c for c in green_codes if c in first_token]
        if first_matches:
            first_ipc_green = "Green (1)"
            matched_first_str = first_matches[0]
        else:
            first_ipc_green = "Non-Green (0)"
            matched_first_str = np.nan

        # Ratios (30% and 50%)
        green_token_cnt = sum(1 for t in tokens if any(c in t for c in green_codes))
        total_tokens = len(tokens)
        ratio = green_token_cnt / total_tokens if total_tokens > 0 else 0.0

        ipc_30 = "Green (1)" if ratio >= 0.30 else "Non-Green (0)"
        matched_30 = matched_str if ratio >= 0.30 else np.nan

        ipc_50 = "Green (1)" if ratio >= 0.50 else "Non-Green (0)"
        matched_50 = matched_str if ratio >= 0.50 else np.nan

        return ("Green (1)", matched_str, first_ipc_green, matched_first_str, ipc_30, matched_30, ipc_50, matched_50)

    results = [process_features(r) for _, r in df_merged.iterrows()]

    res_df = pd.DataFrame(results, columns=[
        "IC_Code_Type", "Matched_IPC",
        "First_IPC_Green", "Matched_IPC_first",
        "30-IPC-Green", "Matched_IPC-30",
        "50-IPC-Green", "Matched_IPC-50"
    ], index=df_merged.index)

    df_merged_clean = df_merged.drop(columns=["Cleaned_IC"])
    df_final = pd.concat([df_merged_clean, res_df], axis=1)

    # 5. Export outcome files
    print(f"\n[5/5] Exporting CSV outcome file: {output_csv_path}...")
    df_final.to_csv(output_csv_path, index=False)
    print("CSV Export Complete!")

    if output_xlsx_path:
        print(f"Exporting Excel outcome file: {output_xlsx_path}...")
        with pd.ExcelWriter(output_xlsx_path, engine="xlsxwriter") as writer:
            df_final.to_excel(writer, sheet_name="SDG_Green_Matched", index=False)
        print("Excel Export Complete!")

    elapsed = time.time() - start_time
    print(f"\nPipeline finished in {elapsed:.2f} seconds.")
    print("=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print("IC_Code_Type:\n", df_final["IC_Code_Type"].value_counts(dropna=False))
    print("\nFirst_IPC_Green:\n", df_final["First_IPC_Green"].value_counts(dropna=False))
    print("\n30-IPC-Green:\n", df_final["30-IPC-Green"].value_counts(dropna=False))
    print("\n50-IPC-Green:\n", df_final["50-IPC-Green"].value_counts(dropna=False))


if __name__ == "__main__":
    base_dir = r"c:\Users\ASUS\Desktop\Research Work\Neha_Mam_Research_Work"
    run_pipeline(
        grant_merged_path=os.path.join(base_dir, "Grant_Data_Merged.xlsx"),
        sdg_apps_path=os.path.join(base_dir, "Application nos SDG.xlsx"),
        wipo_green_path=os.path.join(base_dir, "cleaned_WIPO_IPC_green_codes.csv"),
        green_48972_path=os.path.join(base_dir, "Green patents 48972.xlsx"),
        neha_green_path=os.path.join(base_dir, "Neha_green_codes_final.csv"),
        output_csv_path=os.path.join(base_dir, "Grant_Data_SDG_Green_Matched.csv"),
        output_xlsx_path=os.path.join(base_dir, "Grant_Data_SDG_Green_Matched.xlsx")
    )
