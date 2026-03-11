# main_input_prep.py
import pandas as pd
import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
from src.data_prep.input_prep import run_input_preparation #to change to this when multiple years code is being used. But present best and tested is for per_yr_dryrun which is a no frills code
from src.data_prep.inp_prep_per_yr_dryrun import run_input_preparation_one_year
import os

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    out_dir = HydraConfig.get().runtime.output_dir
    if cfg.use_precomputed_inputs:
        inp_df=pd.read_csv(cfg.inp_csv)
    else:
        # inp_df=run_input_preparation(cfg.input)
        # inp_gdf=run_input_preparation(cfg.input
        inp_gdf=run_input_preparation_one_year(cfg.input)
        # Drop geometry column and save to csv
        inp_df=pd.DataFrame(inp_gdf.drop('geometry', axis=1))
        inp_df.to_csv(os.path.join(out_dir, 'input_fts.csv'), index=False) #os.path.join(out_dir, 'test.csv', index=False)
    

    if cfg.use_precomputed_targets:
        if cfg.target_aggregation == "temporal":
            targ_df=pd.read_csv(cfg.targ_csv_list[0])
        elif cfg.target_aggregation == "spatial":
            targ_df=pd.read_csv(cfg.targ_csv_list[1])
        elif cfg.target_aggregation == "combined":
            targ_df_temp=pd.read_csv(cfg.targ_csv_list[0])
            targ_df_spat=pd.read_csv(cfg.targ_csv_list[1])
        elif cfg.target_aggregation == "no aggregation":
            targ_df=pd.read_csv(cfg.targ_csv_list[0])
    else:
        # raise ValueError(f"Unsuported target_aggregation: {cfg.target_aggregation}") # Remove this later and replace with targ_df_list=run_target_preparation(cfg.target)
        pass

if __name__ == "__main__":
    main()
