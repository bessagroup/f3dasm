import logging
from typing import List

import hydra
from config import Config
from hydra.core.config_store import ConfigStore

import f3dasm

log = logging.getLogger(__name__)


def create_space(config) -> List[f3dasm.design.Parameter]:
    parameters = []
    for param in config:
        param = dict(param)
        name = param.pop('class')
        parameter = f3dasm.design.Parameter.from_dict(parameter_dict=param, name=name)
        parameters.append(parameter)
    return parameters


@hydra.main(config_path=".", config_name="config")
def main(config: Config):
    # input_space = create_space(config.input_space)
    # output_space = create_space(config.output_space)
    # design = f3dasm.DesignSpace(input_space=input_space, output_space=output_space)

    design = f3dasm.DesignSpace.from_yaml(config.design)
    s = f3dasm.sampling.RandomUniform(design).get_samples(10)

    print(s.data)


cs = ConfigStore.instance()
cs.store(name="f3dasm_config", node=Config)


if __name__ == "__main__":
    main()
