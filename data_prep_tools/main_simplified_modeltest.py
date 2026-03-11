# main_train.py

import hydra
from omegaconf import DictConfig
import pandas as pd

# from src.model.simplified_jaxann import load_simplified_model_config
from src.model.testing_ann_all import load_simplified_model_config


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    load_simplified_model_config(cfg.model)


if __name__ == "__main__":
    main()
