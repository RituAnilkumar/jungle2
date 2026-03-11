# main_input_prep.py
import pandas as pd
import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
from src.data_prep.combine_inputs import combine_annual_csvs
import os

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print('entered main')
    combine_annual_csvs(cfg.combine)
    
if __name__ == "__main__":
    main()
