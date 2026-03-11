## Use this only if you have to combine across multirun options from various regions together. For a single region, use combine_inputs.py
import os
import pandas as pd

multirun_dir = "multirun/2025-12-06"
out_dir='tmp_data/met'

# Loop through each immediate subfolder of multirun_dir
for subfolder in os.listdir(multirun_dir):
    subfolder_path = os.path.join(multirun_dir, subfolder)
    if not os.path.isdir(subfolder_path):
        continue  # skip non-directories

    input_csvs = []

    # Walk through subfolder to find all 'input_fts.csv' files
    for root, dirs, files in os.walk(subfolder_path):
        for file in files:
            if file == "input_fts.csv":
                input_csvs.append(os.path.join(root, file))

    if not input_csvs:
        print(f"No input_fts.csv files found in {subfolder}")
        continue

    # Combine all CSVs within this subfolder
    combined_df = pd.concat([pd.read_csv(csv) for csv in input_csvs], ignore_index=True)

    # Save the combined CSV as <subfolder_name>.csv in the main directory
    output_path = os.path.join(out_dir, f"{subfolder}.csv")
    combined_df.to_csv(output_path, index=False)

    print(f"✅ Combined {len(input_csvs)} files from '{subfolder}' into {output_path}")