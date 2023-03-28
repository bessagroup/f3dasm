import os

import hydra
from config import Config  # NOQA
from hydra.core.config_store import ConfigStore
from hydra.utils import get_original_cwd


@hydra.main(config_path=".", config_name="config")
def main(config: Config):
    """Main script to call

    Parameters
    ----------
    config
        Configuration parameters defined in config.yaml
    """
    print("This f3dasm study works!")


cs = ConfigStore.instance()
cs.store(name="f3dasm_config", node=Config)

if __name__ == "__main__":
    main()
