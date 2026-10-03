import pandas as pd
import numpy as np
import re
import time
from python_calamine import CalamineWorkbook

start_time = time.time()
print("Starting Part II Execution: SDG Filtering & Green IPC Code Matching")

# 1. Load Green Codes Reference
print("\n[1/4] Loading WIPO Green Codes reference file...")
df_green = pd.read_csv(r'c:\Users\ASUS\Desktop\Research Work\Neha_Mam_Research_Work\cleaned_WIPO_IPC_green_codes.csv')
green_clean_codes = df_green['clean_code'].astype(str).str.strip().tolist()

subclass_green = set([c for c in green_clean_codes if len(c) <= 4])
full_green = set([c for c in green_clean_codes if len(c) > 4])
print(f"Loaded {len(subclass_green)} 4-char subclass codes and {len(full_green)} full group/subgroup green codes.")

# 2. Load SDG Application Nos
print("\n[2/4] Loading SDG Application Nos...")
wb_sdg = CalamineWorkbook.from_path(r'c:\Users\ASUS\Desktop\Research Work\Neha_Mam_Research_Work\Application nos SDG.xlsx')
sdg_rows = wb_sdg.get_sheet_by_name('Sheet1').to_python()
sdg_apps = set(r[0] for r in sdg_rows[1:] if r and r[0])
print(f"Loaded {len(sdg_apps):,} target SDG application numbers.")

# 3. Load Main Grant Dataset & Filter
print("\n[3/4] Loading Grant_Data_Merged.xlsx...")
wb_grant = CalamineWorkbook.from_path(r'c:\Users\ASUS\Desktop\Research Work\Neha_Mam_Research_Work\Grant_Data_Merged.xlsx')
grant_rows = wb_grant.get_sheet_by_name('Total').to_python()
headers = grant_rows[0]
df_grant = pd.DataFrame(grant_rows[1:], columns=headers)
print(f"Loaded raw dataset with {len(df_grant):,} rows and {len(headers)} columns.")

print("Filtering dataset by SDG application numbers...")
col_app = 'Application No.'
df_filtered = df_grant[df_grant[col_app].isin(sdg_apps)].copy()
print(f"Filtered dataset row count: {len(df_filtered):,}")

col_date = 'Filing/Application Date'
years = df_filtered[col_date].astype(str).str.extract(r'(\d{4})')[0].astype(float)
print(f"Filing Year Range: {years.min():.0f} to {years.max():.0f}")

# 4. IPC Cleaning & Green Code Feature Construction
print("\n[4/4] Processing IPC All Versions (IC) and constructing 8 feature columns...")
col_ipc = 'IPC All Versions (IC)'

def clean_ipc_token(raw_token):
    # Remove noise like (51)International, spaces, slashes, trailing dashes/commas/punctuation
    token = re.sub(r'\(.*?\)', '', str(raw_token)).strip()
    token = re.sub(r'[^A-Za-z0-9]', '', token)
    return token

def match_green_code(token):
    if not token:
        return None
    # 1. Check 4-char subclass prefix match
    if len(token) >= 4 and token[:4] in subclass_green:
        return token[:4]
    # 2. Check full code match
    if token in full_green:
        return token
    # 3. Check prefix / startswith match
    for fg in full_green:
        if token.startswith(fg) or fg.startswith(token):
            return fg
    return None

def process_row_ipc(raw_ipc_val):
    if pd.isna(raw_ipc_val) or raw_ipc_val is None or str(raw_ipc_val).strip() == '' or str(raw_ipc_val).strip() == 'None':
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
    
    tokens = [clean_ipc_token(t) for t in str(raw_ipc_val).split(';') if clean_ipc_token(t)]
    if not tokens:
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
    
    matched_codes = [match_green_code(t) for t in tokens]
    green_matches = [m for m in matched_codes if m is not None]
    
    total_cnt = len(tokens)
    green_cnt = len(green_matches)
    ratio = green_cnt / total_cnt if total_cnt > 0 else 0.0
    
    # 1 & 2: Any Green
    if green_cnt > 0:
        ic_code_type = "Green (1)"
        # Preserve unique matched codes in order
        seen = set()
        matched_ipc_str = "; ".join([m for m in green_matches if not (m in seen or seen.add(m))])
    else:
        ic_code_type = "Non-Green (0)"
        matched_ipc_str = np.nan
        
    # 3 & 4: First IPC Green
    first_green_matched = matched_codes[0]
    if first_green_matched is not None:
        first_ipc_green = "Green (1)"
        matched_ipc_first_str = first_green_matched
    else:
        first_ipc_green = "Non-Green (0)"
        matched_ipc_first_str = np.nan
        
    # 5 & 6: 30% IPC Green
    if ratio >= 0.30 and green_cnt > 0:
        ipc_30_green = "Green (1)"
        seen30 = set()
        matched_ipc_30_str = "; ".join([m for m in green_matches if not (m in seen30 or seen30.add(m))])
    else:
        ipc_30_green = "Non-Green (0)"
        matched_ipc_30_str = np.nan
        
    # 7 & 8: 50% IPC Green
    if ratio >= 0.50 and green_cnt > 0:
        ipc_50_green = "Green (1)"
        seen50 = set()
        matched_ipc_50_str = "; ".join([m for m in green_matches if not (m in seen50 or seen50.add(m))])
    else:
        ipc_50_green = "Non-Green (0)"
        matched_ipc_50_str = np.nan
        
    return (ic_code_type, matched_ipc_str, first_ipc_green, matched_ipc_first_str,
            ipc_30_green, matched_ipc_30_str, ipc_50_green, matched_ipc_50_str)

print("Constructing features for 360,924 records...")
ipc_results = [process_row_ipc(v) for v in df_filtered[col_ipc]]

res_df = pd.DataFrame(ipc_results, columns=[
    'IC_Code_Type', 'Matched_IPC',
    'First_IPC_Green', 'Matched_IPC_first',
    '30-IPC-Green', 'Matched_IPC-30',
    '50-IPC-Green', 'Matched_IPC-50'
], index=df_filtered.index)

df_final = pd.concat([df_filtered, res_df], axis=1)

output_csv = r'c:\Users\ASUS\Desktop\Research Work\Neha_Mam_Research_Work\Grant_Data_SDG_Green_Matched.csv'
print(f"\nExporting final dataset with {df_final.shape[1]} columns to {output_csv}...")
df_final.to_csv(output_csv, index=False)

elapsed = time.time() - start_time
print(f"\nSUCCESS! Completed execution in {elapsed:.2f} seconds.")

print("\n" + "="*60)
print("SUMMARY STATISTICS FOR 8 NEW FEATURE COLUMNS")
print("="*60)
print("\n1. IC_Code_Type:")
print(df_final['IC_Code_Type'].value_counts(dropna=False))

print("\n2. First_IPC_Green:")
print(df_final['First_IPC_Green'].value_counts(dropna=False))

print("\n3. 30-IPC-Green:")
print(df_final['30-IPC-Green'].value_counts(dropna=False))

print("\n4. 50-IPC-Green:")
print(df_final['50-IPC-Green'].value_counts(dropna=False))
