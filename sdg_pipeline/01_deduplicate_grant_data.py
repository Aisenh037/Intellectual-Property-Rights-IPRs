import pandas as pd
from python_calamine import CalamineWorkbook
import os

excel_path = r'c:\Users\ASUS\Desktop\Research Work\Neha_Mam_Research_Work\Grant_Data_Merged.xlsx'
wb = CalamineWorkbook.from_path(excel_path)

print("="*60)
print("PROCESSING ALL SHEETS IN EXCEL FILE")
print("="*60)

for sheet_name in wb.sheet_names:
    rows = wb.get_sheet_by_name(sheet_name).to_python()
    if not rows:
        continue
    headers = rows[0]
    data = rows[1:]
    df = pd.DataFrame(data, columns=headers)
    
    col_e = 'Filing/Application Date'
    col_i = 'Application No.'
    col_bg = 'Simple Family ID'
    
    if col_e not in df.columns or col_i not in df.columns or col_bg not in df.columns:
        print(f"Sheet '{sheet_name}': Missing columns (has {len(df)} rows)")
        continue
        
    initial_cnt = len(df)
    
    # Extract year
    years = df[col_e].astype(str).str.extract(r'(\d{4})')[0]
    
    # Filter 2023, 2024
    df_no2324 = df[~years.isin(['2023', '2024'])].copy()
    cnt_after_year_filter = len(df_no2324)
    
    # Deduplicate Application No
    df_dedup_app = df_no2324.drop_duplicates(subset=[col_i], keep='first').copy()
    cnt_after_app_dedup = len(df_dedup_app)
    
    # Deduplicate Simple Family ID
    df_dedup_sfam = df_dedup_app.drop_duplicates(subset=[col_bg], keep='first').copy()
    cnt_after_sfam_dedup = len(df_dedup_sfam)
    
    print(f"\nSheet: '{sheet_name}'")
    print(f"  1. Initial Row Count: {initial_cnt:,}")
    print(f"  2. Row Count after removing 2023 & 2024: {cnt_after_year_filter:,} (Removed {initial_cnt - cnt_after_year_filter:,} rows)")
    print(f"  3. Row Count after Application No. Deduplication: {cnt_after_app_dedup:,} (Removed {cnt_after_year_filter - cnt_after_app_dedup:,} duplicates)")
    print(f"  4. Final Row Count after Simple Family ID Deduplication: {cnt_after_sfam_dedup:,} (Removed {cnt_after_app_dedup - cnt_after_sfam_dedup:,} duplicates)")

