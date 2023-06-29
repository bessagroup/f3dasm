import os

import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import get_original_cwd


@hydra.main(config_path=".", config_name="config")
def main(config):
    """Main script to call

    Parameters
    ----------
    config
        Configuration parameters defined in config.yaml
    """
    print("This f3dasm study works!")


cs = ConfigStore.instance()
cs.store(name="f3dasm_config")

if __name__ == "__main__":
    main()
