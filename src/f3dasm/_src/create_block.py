from typing import Callable

from .datageneration.datagenerator_factory import DATAGENERATOR_MAPPING
from .design.samplers import SAMPLER_MAPPING
from .optimization.optimizer_factory import get_optimizer_mapping


def create_block(block: str, **parameters):
    if isinstance(block, str):
        filtered_name = block.lower().replace(
            ' ', '').replace('-', '').replace('_', '')

        # Check in the built-in samplers
        if filtered_name in SAMPLER_MAPPING:
            return SAMPLER_MAPPING[filtered_name](**parameters)

        OPTIMIZER_MAPPING = get_optimizer_mapping()

        # Check in the built-in optimizers
        if filtered_name in OPTIMIZER_MAPPING:
            return OPTIMIZER_MAPPING[filtered_name](**parameters)

        raise KeyError(f'Block {block} not found')


def create_datagenerator(data_generator: str | Callable, **parameters) -> DataGenerator:
    ...
