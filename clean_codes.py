import pandas as pd
from tkinter import Tk, filedialog

# Hide the root window
Tk().withdraw()

# Ask user to pick a file
input_filename = filedialog.askopenfilename(title="Select your CSV file", filetypes=[("CSV files", "*.csv")])

# Load CSV
df = pd.read_csv(input_filename, encoding='latin1')

# First column name
ipc_column = df.columns[0]

# Create cleaned column
df[ipc_column + "_cleaned"] = df[ipc_column].astype(str) \
    .str.replace("/", "") \
    .str.replace(" ", "")

# Save new CSV
output_filename = "cleaned_" + input_filename.split("/")[-1]
df.to_csv(output_filename, index=False, encoding='utf-8')

print(f"Cleaned file saved as {output_filename}")
