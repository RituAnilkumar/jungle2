# Combine annual csvs downloaded through multirun and save it in data_for_model
import pandas as pd
import os
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

def combine_annual_csvs(combine_config: DictConfig) -> None:
    # print('entered combine_annual_csvs function')
    multirun_dir=combine_config.multirun_dir
    output_dir=combine_config.output_dir
    output_name=combine_config.output_name

    print('multirun_dir',multirun_dir)
    # print('os walk', os.walk(multirun_dir))

    # Get all train.csv files within subdirectories of multirun_dir
    train_csvs = []
    for root, dirs, files in os.walk(multirun_dir):
        print('root',root)
        print('dirs',dirs)
        print('files',files)
        for file in files:
            if file == 'input_fts.csv':
                print('entered')
                # print(os.path.join(root, file))
                train_csvs.append(os.path.join(root, file))
            else:
                print('not entered')

    # Concatenate all train.csv files into a single DataFrame
    combined_df = pd.concat([pd.read_csv(csv) for csv in train_csvs], ignore_index=True)
    print(f"Combined {len(train_csvs)} files with total shape: {combined_df.shape}")

    # Save the combined DataFrame to a new CSV file
    combined_df.to_csv(os.path.join(output_dir, output_name), index=False)