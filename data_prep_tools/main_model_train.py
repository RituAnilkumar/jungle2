# main_train.py

import hydra
from omegaconf import DictConfig
import pandas as pd

from src.model.sklearn_mlp import train_model


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.use_precomputed_inputs:
        inp_df=pd.read_csv(cfg.inp_csv)
    else:
        # inp_df=run_input_preparation(cfg.input)
        # run_input_preparation(cfg.input)
        print('entered not precomputed')
        pass

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
    
    metrics = train_model(inp_df.values, targ_df.values, inp_df.values, targ_df.values, cfg.model)

    print(metrics)


if __name__ == "__main__":
    main()
