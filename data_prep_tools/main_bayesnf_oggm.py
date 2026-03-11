# main_input_prep.py
import hydra
from omegaconf import DictConfig
from src.model.bayesnf_oggm import train_oggm_bayesnf

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print('entered main')
    train_oggm_bayesnf(cfg.model)
    
if __name__ == "__main__":
    main()
