import pandas as pd
import numpy as np
import time

start_time = time.time()
print("Reading updated CSV to export XLSX...")
csv_path = r'c:\Users\ASUS\Desktop\Research Work\Neha_Mam_Research_Work\Grant_Data_SDG_Green_Matched.csv'
xlsx_path = r'c:\Users\ASUS\Desktop\Research Work\Neha_Mam_Research_Work\Grant_Data_SDG_Green_Matched_v2.xlsx'

df = pd.read_csv(csv_path, low_memory=False)
print(f"Loaded {len(df):,} rows from CSV. Exporting to {xlsx_path}...")

with pd.ExcelWriter(xlsx_path, engine='xlsxwriter') as writer:
    df.to_excel(writer, sheet_name='SDG_Green_Matched', index=False)

print(f"Excel export finished in {time.time() - start_time:.2f}s!")

# Replace the original xlsx if not locked, otherwise keep v2
import os
try:
    orig = r'c:\Users\ASUS\Desktop\Research Work\Neha_Mam_Research_Work\Grant_Data_SDG_Green_Matched.xlsx'
    os.replace(xlsx_path, orig)
    print("Replaced Grant_Data_SDG_Green_Matched.xlsx successfully!")
except Exception as e:
    print(f"Original file is open in Excel; saved as Grant_Data_SDG_Green_Matched_v2.xlsx ({e})")
