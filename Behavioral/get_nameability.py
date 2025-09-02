import pandas as pd
import numpy as np

# Paths to files
excel_path = "/Users/admin/Box Sync/Starling/Experiment1/Behavioral_Data/counterbalancing_all.xlsx"
metadata_path = "/Users/admin/Box Sync/Starling/Experiment1/_images-metadata_things.tsv"

df_excel = pd.read_excel(excel_path)

words_list = pd.concat([df_excel.iloc[:, 20], df_excel.iloc[:, 22]]).dropna().astype(str).str.strip("'").str.lower().unique()

df_meta = pd.read_csv(metadata_path, sep='\t')

df_meta['Word_clean'] = df_meta['Word'].astype(str).str.strip().str.lower()

nameability_values = []
words_found = []
words_not_found = []

for word in words_list:
    if word in df_meta['Word_clean'].tolist():
        row = df_meta[df_meta['Word'] == word]
        nameability_values.append(row['nameability'].values[0])
        words_found.append(word)
    else:
        words_not_found.append(word)

if len(nameability_values) > 0:
    avg_nameability = np.mean(nameability_values)
else:
    avg_nameability = np.nan

# Output results
print("Number of words found in metadata:", len(nameability_values))
print("Number of words not found:", len(words_not_found))
print("Average nameability:", avg_nameability)
print("Max nameability:", np.max(nameability_values))
print("Min nameability:", np.min(nameability_values))
print("Nameability standard deviation:", np.std(nameability_values))
