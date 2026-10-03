import pandas as pd
import numpy as np
import re
import time
from python_calamine import CalamineWorkbook

start_time = time.time()
print("Starting Complete Pipeline: SDG Exact Order Join & Dual Export (CSV + XLSX)")

# 1. Load Green Codes Reference
print("\n[1/5] Loading WIPO Green Codes reference file...")
df_green = pd.read_csv(r'c:\Users\ASUS\Desktop\Research Work\Neha_Mam_Research_Work\cleaned_WIPO_IPC_green_codes.csv')
green_clean_codes = df_green['clean_code'].astype(str).str.strip().tolist()

subclass_green = set([c for c in green_clean_codes if len(c) <= 4])
full_green = set([c for c in green_clean_codes if len(c) > 4])
print(f"Loaded {len(subclass_green)} 4-char subclass codes and {len(full_green)} full group/subgroup green codes.")

# 2. Load SDG Application Nos preserving exact order
print("\n[2/5] Loading Application nos SDG.xlsx (preserving exact row sequence)...")
wb_sdg = CalamineWorkbook.from_path(r'c:\Users\ASUS\Desktop\Research Work\Neha_Mam_Research_Work\Application nos SDG.xlsx')
sdg_rows = wb_sdg.get_sheet_by_name('Sheet1').to_python()
df_sdg = pd.DataFrame(sdg_rows[1:], columns=sdg_rows[0])
print(f"SDG File total rows: {len(df_sdg):,}")

# 3. Load Main Grant Dataset & Deduplicate on App No
print("\n[3/5] Loading Grant_Data_Merged.xlsx...")
wb_grant = CalamineWorkbook.from_path(r'c:\Users\ASUS\Desktop\Research Work\Neha_Mam_Research_Work\Grant_Data_Merged.xlsx')
grant_rows = wb_grant.get_sheet_by_name('Total').to_python()
headers = grant_rows[0]
df_grant = pd.DataFrame(grant_rows[1:], columns=headers)
print(f"Grant File total raw rows: {len(df_grant):,}")

df_grant_dedup = df_grant.drop_duplicates(subset=['Application No.'], keep='first').copy()
print(f"Grant File after App No deduplication: {len(df_grant_dedup):,}")

# Left join preserving exact SDG row order
df_merged = pd.merge(df_sdg[['Application No.']], df_grant_dedup, on='Application No.', how='left')
print(f"Merged Dataset Row Count (exact SDG order): {len(df_merged):,}")

col_date = 'Filing/Application Date'
years = df_merged[col_date].astype(str).str.extract(r'(\d{4})')[0].astype(float)
print(f"Filing Year Range: {years.min():.0f} to {years.max():.0f}")

# 4. IPC Cleaning & Green Code Feature Construction
print("\n[4/5] Constructing 8 WIPO Green IPC Feature Columns...")
col_ipc = 'IPC All Versions (IC)'

def clean_ipc_token(raw_token):
    token = re.sub(r'\(.*?\)', '', str(raw_token)).strip()
    token = re.sub(r'[^A-Za-z0-9]', '', token)
    return token

def match_green_code(token):
    if not token:
        return None
    if len(token) >= 4 and token[:4] in subclass_green:
        return token[:4]
    if token in full_green:
        return token
    for fg in full_green:
        if token.startswith(fg) or fg.startswith(token):
            return fg
    return None

def process_row_ipc(raw_ipc_val):
    if pd.isna(raw_ipc_val) or raw_ipc_val is None or str(raw_ipc_val).strip() in ['', 'None', 'nan']:
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
        ic_code_type = 'Green (1)'
        seen = set()
        matched_ipc_str = '; '.join([m for m in green_matches if not (m in seen or seen.add(m))])
    else:
        ic_code_type = 'Non-Green (0)'
        matched_ipc_str = np.nan
        
    # 3 & 4: First IPC Green
    first_green_matched = matched_codes[0]
    if first_green_matched is not None:
        first_ipc_green = 'Green (1)'
        matched_ipc_first_str = first_green_matched
    else:
        first_ipc_green = 'Non-Green (0)'
        matched_ipc_first_str = np.nan
        
    # 5 & 6: 30% IPC Green
    if ratio >= 0.30 and green_cnt > 0:
        ipc_30_green = 'Green (1)'
        seen30 = set()
        matched_ipc_30_str = '; '.join([m for m in green_matches if not (m in seen30 or seen30.add(m))])
    else:
        ipc_30_green = 'Non-Green (0)'
        matched_ipc_30_str = np.nan
        
    # 7 & 8: 50% IPC Green
    if ratio >= 0.50 and green_cnt > 0:
        ipc_50_green = 'Green (1)'
        seen50 = set()
        matched_ipc_50_str = '; '.join([m for m in green_matches if not (m in seen50 or seen50.add(m))])
    else:
        ipc_50_green = 'Non-Green (0)'
        matched_ipc_50_str = np.nan
        
    return (ic_code_type, matched_ipc_str, first_ipc_green, matched_ipc_first_str,
            ipc_30_green, matched_ipc_30_str, ipc_50_green, matched_ipc_50_str)

ipc_results = [process_row_ipc(v) for v in df_merged[col_ipc]]

res_df = pd.DataFrame(ipc_results, columns=[
    'IC_Code_Type', 'Matched_IPC',
    'First_IPC_Green', 'Matched_IPC_first',
    '30-IPC-Green', 'Matched_IPC-30',
    '50-IPC-Green', 'Matched_IPC-50'
], index=df_merged.index)

df_final = pd.concat([df_merged, res_df], axis=1)

# Replace any string 'nan' / None representations cleanly
print(f"Final shape: {df_final.shape[0]:,} rows x {df_final.shape[1]} columns")

# 5. Export Outcome Files (CSV and XLSX)
output_csv = r'c:\Users\ASUS\Desktop\Research Work\Neha_Mam_Research_Work\Grant_Data_SDG_Green_Matched.csv'
output_xlsx = r'c:\Users\ASUS\Desktop\Research Work\Neha_Mam_Research_Work\Grant_Data_SDG_Green_Matched.xlsx'

print(f"\n[5/5] Exporting CSV outcome file: {output_csv}...")
df_final.to_csv(output_csv, index=False)
print("CSV Export Complete!")

print(f"\nExporting Excel outcome file via xlsxwriter engine: {output_xlsx}...")
with pd.ExcelWriter(output_xlsx, engine='xlsxwriter') as writer:
    df_final.to_excel(writer, sheet_name='SDG_Green_Matched', index=False)
print("Excel Export Complete!")

elapsed = time.time() - start_time
print(f"\nSUCCESS! Exported BOTH CSV and XLSX files in {elapsed:.2f} seconds.")

print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
print(f"Total rows: {len(df_final):,}")
print('\n1. IC_Code_Type:\n', df_final['IC_Code_Type'].value_counts(dropna=False))
print('\n2. First_IPC_Green:\n', df_final['First_IPC_Green'].value_counts(dropna=False))
print('\n3. 30-IPC-Green:\n', df_final['30-IPC-Green'].value_counts(dropna=False))
print('\n4. 50-IPC-Green:\n', df_final['50-IPC-Green'].value_counts(dropna=False))
