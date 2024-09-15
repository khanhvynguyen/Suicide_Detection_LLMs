import os

os.environ["FSDP_CPU_RAM_EFFICIENT_LOADING"] = "1"

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from fine_tune import fine_tune_v2


@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(cfg) -> None:

    print(OmegaConf.to_yaml(cfg))
    fine_tune_v2(cfg)


if __name__ == "__main__":
    main()
