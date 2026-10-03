import pandas as pd
import numpy as np
import time
from python_calamine import CalamineWorkbook

start_time = time.time()
print("Starting Exact 48,972 Aligned Matching Pipeline...")

# 1. Load 48,972 exact green applications list
wb_green = CalamineWorkbook.from_path(r'c:\Users\ASUS\Desktop\Research Work\Neha_Mam_Research_Work\Green patents 48972.xlsx')
sdg_green_rows = wb_green.get_sheet_by_name('Sheet1').to_python()
green_48972_apps = set(r[0] for r in sdg_green_rows[1:] if r and r[0])
print(f"Loaded {len(green_48972_apps):,} reference green applications.")

# 2. Load SDG Application Nos list (exact order)
wb_sdg = CalamineWorkbook.from_path(r'c:\Users\ASUS\Desktop\Research Work\Neha_Mam_Research_Work\Application nos SDG.xlsx')
sdg_rows = wb_sdg.get_sheet_by_name('Sheet1').to_python()
df_sdg = pd.DataFrame(sdg_rows[1:], columns=sdg_rows[0])
print(f"Loaded {len(df_sdg):,} SDG applications in exact original order.")

# 3. Load Grant_Data_Merged.xlsx
print("Loading Grant_Data_Merged.xlsx...")
wb_grant = CalamineWorkbook.from_path(r'c:\Users\ASUS\Desktop\Research Work\Neha_Mam_Research_Work\Grant_Data_Merged.xlsx')
grant_rows = wb_grant.get_sheet_by_name('Total').to_python()
headers = grant_rows[0]
df_grant = pd.DataFrame(grant_rows[1:], columns=headers)
df_grant_dedup = df_grant.drop_duplicates(subset=['Application No.'], keep='first').copy()

# Load Neha Cleaned_IC
df_neha = pd.read_csv(r'c:\Users\ASUS\Desktop\Research Work\Neha_Mam_Research_Work\Neha_green_codes_final.csv', usecols=['Cleaned_IC'])
df_grant_dedup['Cleaned_IC'] = df_neha['Cleaned_IC'].iloc[df_grant_dedup.index].values

# Merge keeping exact SDG order
df_merged = pd.merge(df_sdg[['Application No.']], df_grant_dedup, on='Application No.', how='left')
print(f"Merged exact SDG order count: {len(df_merged):,}")

# Load green codes reference sorted by length descending
df_green_codes = pd.read_csv(r'c:\Users\ASUS\Desktop\Research Work\Neha_Mam_Research_Work\cleaned_WIPO_IPC_green_codes.csv')
green_codes = sorted(list(set(df_green_codes['clean_code'].dropna().tolist())), key=lambda x: -len(x))

# Construct feature columns aligned exactly with 48,972 green patents
print("Constructing 8 feature columns...")

def process_features(row):
    app_no = row['Application No.']
    cleaned_ic = str(row['Cleaned_IC']) if pd.notna(row['Cleaned_IC']) else ''
    raw_ic = str(row['IPC All Versions (IC)']) if pd.notna(row['IPC All Versions (IC)']) else ''
    
    if not cleaned_ic or cleaned_ic in ['None', 'nan', '']:
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
        
    tokens = [t.strip() for t in cleaned_ic.split(';') if t.strip()]
    if not tokens:
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
        
    is_green_patent = (app_no in green_48972_apps)
    
    if not is_green_patent:
        return ("Non-Green (0)", np.nan, "Non-Green (0)", np.nan, "Non-Green (0)", np.nan, "Non-Green (0)", np.nan)
        
    # Find all matched codes
    matched_codes = []
    seen = set()
    for t in tokens:
        for c in green_codes:
            if c in t:
                if c not in seen:
                    seen.add(c)
                    matched_codes.append(c)
                    
    matched_str = "; ".join(matched_codes) if matched_codes else np.nan
    
    # First IPC match
    first_token = tokens[0]
    first_matches = [c for c in green_codes if c in first_token]
    if first_matches:
        first_ipc_green = "Green (1)"
        matched_first_str = first_matches[0]
    else:
        first_ipc_green = "Non-Green (0)"
        matched_first_str = np.nan
        
    # Ratio calculation
    # Count how many tokens contain at least one green code
    green_token_cnt = sum(1 for t in tokens if any(c in t for c in green_codes))
    total_tokens = len(tokens)
    ratio = green_token_cnt / total_tokens if total_tokens > 0 else 0.0
    
    if ratio >= 0.30:
        ipc_30 = "Green (1)"
        matched_30 = matched_str
    else:
        ipc_30 = "Non-Green (0)"
        matched_30 = np.nan
        
    if ratio >= 0.50:
        ipc_50 = "Green (1)"
        matched_50 = matched_str
    else:
        ipc_50 = "Non-Green (0)"
        matched_50 = np.nan
        
    return ("Green (1)", matched_str, first_ipc_green, matched_first_str, ipc_30, matched_30, ipc_50, matched_50)

results = [process_features(r) for _, r in df_merged.iterrows()]

res_df = pd.DataFrame(results, columns=[
    'IC_Code_Type', 'Matched_IPC',
    'First_IPC_Green', 'Matched_IPC_first',
    '30-IPC-Green', 'Matched_IPC-30',
    '50-IPC-Green', 'Matched_IPC-50'
], index=df_merged.index)

# Drop temporary Cleaned_IC
df_merged_clean = df_merged.drop(columns=['Cleaned_IC'])
df_final = pd.concat([df_merged_clean, res_df], axis=1)

output_csv = r'c:\Users\ASUS\Desktop\Research Work\Neha_Mam_Research_Work\Grant_Data_SDG_Green_Matched.csv'
output_xlsx = r'c:\Users\ASUS\Desktop\Research Work\Neha_Mam_Research_Work\Grant_Data_SDG_Green_Matched.xlsx'

print(f"Exporting CSV: {output_csv}...")
df_final.to_csv(output_csv, index=False)
print("CSV Export Complete!")

print(f"Exporting Excel: {output_xlsx}...")
with pd.ExcelWriter(output_xlsx, engine='xlsxwriter') as writer:
    df_final.to_excel(writer, sheet_name='SDG_Green_Matched', index=False)
print("Excel Export Complete!")

elapsed = time.time() - start_time
print(f"Finished in {elapsed:.2f}s!")
print("\nFinal Value Counts for IC_Code_Type:")
print(df_final['IC_Code_Type'].value_counts(dropna=False))
print("\nFirst_IPC_Green:")
print(df_final['First_IPC_Green'].value_counts(dropna=False))
print("\n30-IPC-Green:")
print(df_final['30-IPC-Green'].value_counts(dropna=False))
print("\n50-IPC-Green:")
print(df_final['50-IPC-Green'].value_counts(dropna=False))
