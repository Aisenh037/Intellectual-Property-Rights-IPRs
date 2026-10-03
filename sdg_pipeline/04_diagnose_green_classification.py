import pandas as pd
from python_calamine import CalamineWorkbook

wb = CalamineWorkbook.from_path(r'c:\Users\ASUS\Desktop\Research Work\Neha_Mam_Research_Work\Green patents 48972.xlsx')
sdg_green_rows = wb.get_sheet_by_name('Sheet1').to_python()
green_48972_apps = set(r[0] for r in sdg_green_rows[1:] if r and r[0])
print(f"Total in Green patents 48972: {len(green_48972_apps):,}")

df_current = pd.read_csv(r'c:\Users\ASUS\Desktop\Research Work\Neha_Mam_Research_Work\Grant_Data_SDG_Green_Matched.csv', 
                         usecols=['Application No.', 'IC_Code_Type', 'Matched_IPC', 'IPC All Versions (IC)'])
our_green_apps = set(df_current[df_current['IC_Code_Type'] == 'Green (1)']['Application No.'])

print(f"Our detected green apps: {len(our_green_apps):,}")
print(f"Intersection: {len(our_green_apps.intersection(green_48972_apps)):,}")
print(f"In 48972 but NOT in our green: {len(green_48972_apps - our_green_apps):,}")
print(f"In our green but NOT in 48972: {len(our_green_apps - green_48972_apps):,}")

# Inspect difference samples
diff_apps = list(our_green_apps - green_48972_apps)[:10]
print("\nSamples in our green but NOT in 48972:")
for da in diff_apps:
    row = df_current[df_current['Application No.'] == da].iloc[0]
    print(f"App: {da} | Matched: {row['Matched_IPC']} | Raw: {row['IPC All Versions (IC)']}")

missed_apps = list(green_48972_apps - our_green_apps)[:10]
print("\nSamples in 48972 but NOT in our green:")
for ma in missed_apps:
    matches = df_current[df_current['Application No.'] == ma]
    if len(matches) > 0:
        row = matches.iloc[0]
        print(f"App: {ma} | Detected Type: {row['IC_Code_Type']} | Raw: {row['IPC All Versions (IC)']}")
    else:
        print(f"App: {ma} NOT in SDG matched file!")
