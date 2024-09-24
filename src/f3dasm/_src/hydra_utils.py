"""
This module defines utility functions for the Hydra configuration system.
"""
#                                                                       Modules
# =============================================================================

# Standard
from copy import deepcopy

# Third-party
from omegaconf import OmegaConf

# Local
from .experimentdata.experimentsample import ExperimentSample

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Alpha'
# =============================================================================
#
# =============================================================================


def update_config_with_experiment_sample(
        config: OmegaConf, experiment_sample: ExperimentSample,
        force_add: bool = False) -> OmegaConf:
    """
    Update the config with the values from the experiment sample

    Parameters
    ----------
    config : OmegaConf
        The configuration to update
    experiment_sample : ExperimentSample
        The experiment sample to update the configuration with
    force_add : bool, optional
        If True, the function will add keys that are not present in the
        configuration. If False, the function will ignore keys that are not
        present in the configuration. Default is False.

    Returns
    -------
    OmegaConf
        The updated configuration

    Notes
    -----
    The function will update the configuration with the values from the
    experiment sample. The function will only update the configuration with
    values that are present in the experiment sample. If the experiment sample
    contains values that are not present in the configuration, they will be
    ignored. Keys can be nested using dots, e.g. 'a.b' will update the value
    of 'c' in the configuration key 'b'.

    The function will return a new configuration object with the
    updated values. The original configuration object will not be modified.
    """
    cfg = deepcopy(config)
    for key, value in experiment_sample.to_dict().items():
        try:
            OmegaConf.update(cfg, key, value, force_add=force_add)
        except AttributeError:
            continue

    return cfg
