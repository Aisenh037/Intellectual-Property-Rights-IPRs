"""
Module: match_first_30_50_green_ipc.py
Description:
    Computes First IPC, 30% IPC, and 50% IPC Green Patent Matching
    based on WIPO Green IPC technology standards.
"""

import os
import re
import numpy as np
import pandas as pd


def load_wipo_green_codes(wipo_codes_path: str):
    """
    Loads WIPO Green IPC codes and sorts them by descending length.
    Sorting by descending length ensures longest prefix/subclass match takes precedence.
    """
    if not os.path.exists(wipo_codes_path):
        raise FileNotFoundError(f"WIPO Green IPC file not found: {wipo_codes_path}")

    df_green = pd.read_csv(wipo_codes_path)
    if "clean_code" in df_green.columns:
        code_col = "clean_code"
    elif "IPC" in df_green.columns:
        code_col = "IPC"
    else:
        code_col = df_green.columns[0]

    raw_codes = df_green[code_col].dropna().astype(str).str.strip().tolist()
    # Unique and sorted longest to shortest
    unique_codes = sorted(list(set(raw_codes)), key=lambda x: -len(x))
    return unique_codes


def match_green_ipc_features(
    df: pd.DataFrame,
    green_codes: list,
    green_reference_apps: set = None,
    ipc_column: str = "IPC All Versions (IC)",
    app_column: str = "Application No."
) -> pd.DataFrame:
    """
    Computes First IPC, 30% IPC, and 50% IPC green indicators for each patent.

    Parameters:
    -----------
    df : pd.DataFrame
        Input patent dataset.
    green_codes : list
        List of cleaned WIPO Green IPC codes.
    green_reference_apps : set, optional
        Pre-identified set of Green patent application numbers (e.g., 48,972 baseline).
        If provided, only patents in this set can be classified as Green (1).
    ipc_column : str
        Column name holding semicolon-separated IPC codes.
    app_column : str
        Column name holding patent application number.

    Returns:
    --------
    pd.DataFrame containing 6 computed columns:
        - First_IPC_Green ('Green (1)' / 'Non-Green (0)')
        - Matched_IPC_first (Matched green code or NaN)
        - 30-IPC-Green ('Green (1)' / 'Non-Green (0)')
        - Matched_IPC-30 (Matched green codes or NaN)
        - 50-IPC-Green ('Green (1)' / 'Non-Green (0)')
        - Matched_IPC-50 (Matched green codes or NaN)
    """
    first_ipc_list = []
    matched_first_list = []
    ipc_30_list = []
    matched_30_list = []
    ipc_50_list = []
    matched_50_list = []

    for _, row in df.iterrows():
        app_no = str(row.get(app_column, ""))
        ipc_raw = str(row.get(ipc_column, ""))

        # Clean string
        ipc_clean = ipc_raw.replace("None", "").replace("NaN", "").replace("nan", "").strip()
        tokens = [t.strip() for t in ipc_clean.split(";") if t.strip()]

        # Filter out invalid placeholders
        tokens = [t for t in tokens if t.lower() not in {"na", "na/", "n", "nil/", "null", "-", ""}]

        # Check reference green eligibility
        is_eligible = True
        if green_reference_apps is not None:
            is_eligible = (app_no in green_reference_apps)

        if not tokens or not is_eligible:
            first_ipc_list.append("Non-Green (0)")
            matched_first_list.append(np.nan)
            ipc_30_list.append("Non-Green (0)")
            matched_30_list.append(np.nan)
            ipc_50_list.append("Non-Green (0)")
            matched_50_list.append(np.nan)
            continue

        # Match tokens against WIPO green codes
        matched_all = []
        seen = set()
        green_token_count = 0

        for t in tokens:
            token_is_green = False
            for code in green_codes:
                if code in t:
                    token_is_green = True
                    if code not in seen:
                        seen.add(code)
                        matched_all.append(code)
            if token_is_green:
                green_token_count += 1

        all_matched_str = "; ".join(matched_all) if matched_all else np.nan

        # 1. First IPC Matching
        first_token = tokens[0]
        first_matches = [code for code in green_codes if code in first_token]
        if first_matches:
            first_ipc_list.append("Green (1)")
            matched_first_list.append(first_matches[0])
        else:
            first_ipc_list.append("Non-Green (0)")
            matched_first_list.append(np.nan)

        # 2. 30% and 50% Ratio Matching
        total_tokens = len(tokens)
        green_ratio = green_token_count / total_tokens if total_tokens > 0 else 0.0

        if green_ratio >= 0.30:
            ipc_30_list.append("Green (1)")
            matched_30_list.append(all_matched_str)
        else:
            ipc_30_list.append("Non-Green (0)")
            matched_30_list.append(np.nan)

        if green_ratio >= 0.50:
            ipc_50_list.append("Green (1)")
            matched_50_list.append(all_matched_str)
        else:
            ipc_50_list.append("Non-Green (0)")
            matched_50_list.append(np.nan)

    result_df = pd.DataFrame({
        "First_IPC_Green": first_ipc_list,
        "Matched_IPC_first": matched_first_list,
        "30-IPC-Green": ipc_30_list,
        "Matched_IPC-30": matched_30_list,
        "50-IPC-Green": ipc_50_list,
        "Matched_IPC-50": matched_50_list
    }, index=df.index)

    return result_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Match First, 30%, and 50% Green IPC Features")
    parser.add_argument("--input", required=True, help="Input CSV/Excel file path")
    parser.add_argument("--wipo", required=True, help="Path to cleaned_WIPO_IPC_green_codes.csv")
    parser.add_argument("--green-ref", default=None, help="Optional path to Green reference applications (.xlsx or .csv)")
    parser.add_argument("--output", required=True, help="Output file path (.csv or .xlsx)")

    args = parser.parse_args()

    print(f"Loading input file: {args.input}")
    if args.input.endswith(".xlsx"):
        df_input = pd.read_excel(args.input)
    else:
        df_input = pd.read_csv(args.input, low_memory=False)

    print(f"Loading WIPO codes: {args.wipo}")
    codes = load_wipo_green_codes(args.wipo)
    print(f"Loaded {len(codes):,} unique green codes.")

    ref_set = None
    if args.green_ref:
        print(f"Loading Green reference set: {args.green_ref}")
        if args.green_ref.endswith(".xlsx"):
            from python_calamine import CalamineWorkbook
            wb = CalamineWorkbook.from_path(args.green_ref)
            rows = wb.get_sheet_by_name(wb.sheet_names[0]).to_python()
            ref_set = set(r[0] for r in rows[1:] if r and r[0])
        else:
            df_ref = pd.read_csv(args.green_ref)
            ref_set = set(df_ref.iloc[:, 0].dropna().astype(str).tolist())
        print(f"Loaded {len(ref_set):,} reference green applications.")

    print("Computing First IPC, 30%, and 50% Green IPC metrics...")
    features_df = match_green_ipc_features(df_input, codes, ref_set)
    df_output = pd.concat([df_input, features_df], axis=1)

    print("\nFeature Counts Summary:")
    print("First_IPC_Green:\n", df_output["First_IPC_Green"].value_counts())
    print("30-IPC-Green:\n", df_output["30-IPC-Green"].value_counts())
    print("50-IPC-Green:\n", df_output["50-IPC-Green"].value_counts())

    print(f"\nSaving output to: {args.output}")
    if args.output.endswith(".xlsx"):
        df_output.to_excel(args.output, index=False)
    else:
        df_output.to_csv(args.output, index=False)
    print("Done!")
