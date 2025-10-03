"""Load all the configuration objects that are needed from a toml file."""

from pathlib import Path
from typing import Any

import tomllib

from GalacticStochastic.iteration_config import IterationConfig, get_iteration_config
from LisaWaveformTools.lisa_config import LISAConstants, get_lisa_constants
from WaveletWaveforms.wdm_config import WDMWaveletConstants, get_wavelet_model


def get_config_objects_from_dict(config: dict[str, Any]) -> tuple[dict[str, Any], WDMWaveletConstants, LISAConstants, IterationConfig, int]:
    """
    Load configuration objects from a configuration dictionary.

    This function parses the input configuration dictionary and constructs the required
    configuration objects for the iterative fit, including wavelet model, LISA constants,
    and iteration configuration. It also extracts the instrument noise random seed.

    Parameters
    ----------
    config : dict[str, Any]
        Dictionary containing configuration parameters, typically loaded from a TOML file.

    Returns
    -------
    config : dict[str, Any]
        The original configuration dictionary.
    wc : WDMWaveletConstants
        Wavelet model constants object.
    lc : LISAConstants
        LISA instrument constants object.
    ic : IterationConfig
        Iteration configuration object.
    instrument_random_seed : int
        Random seed for instrument noise realization.

    Raises
    ------
    AssertionError
        If the number of galaxy channels exceeds the number of waveform or SNR channels.
    """
    wc = get_wavelet_model(config)

    lc = get_lisa_constants(config)

    ic = get_iteration_config(config)

    config_noise: dict[str, int] = config['noise_realization']
    instrument_random_seed = int(config_noise.get('instrument_noise_realization_seed', -1))

    assert ic.nc_galaxy <= lc.nc_waveform, 'cannot compute background for channels without intrinsic_waveform'
    assert ic.nc_galaxy <= lc.nc_snr, 'cannot compute background for channels not included in snr calculation'

    return config, wc, lc, ic, instrument_random_seed


def get_config_dict_from_file(toml_filename: str) -> dict[str, Any]:
    """
    Load a configuration dictionary from a TOML file.

    This function reads the specified TOML file and parses its contents into a Python
    dictionary. The resulting dictionary contains all configuration parameters needed
    for the pipeline. The filename is also stored in the dictionary under the key
    'toml_filename' for reference.

    Parameters
    ----------
    toml_filename : str
        Path to the TOML configuration file.

    Returns
    -------
    config : dict[str, Any]
        Dictionary containing configuration parameters loaded from the TOML file.

    Raises
    ------
    AssertionError
        If the toml filename exists in the dict but is different than expected
    """
    with Path(toml_filename).open('rb') as f:
        config: dict[str, Any] = tomllib.load(f)

    assert toml_filename == config.setdefault('toml_filename', toml_filename)
    return config


def get_config_objects(toml_filename: str) -> tuple[dict[str, Any], WDMWaveletConstants, LISAConstants, IterationConfig, int]:
    """
    Load configuration objects from a TOML file.

    This function reads the specified TOML configuration file, parses its contents,
    and constructs the required configuration objects for the iterative fit. It returns
    the configuration dictionary, wavelet model constants, LISA instrument constants,
    iteration configuration, and the instrument noise random seed.

    Parameters
    ----------
    toml_filename : str
        Path to the TOML configuration file.

    Returns
    -------
    config : dict[str, Any]
        Dictionary containing configuration parameters loaded from the TOML file.
    wc : WDMWaveletConstants
        Wavelet model constants object.
    lc : LISAConstants
        LISA instrument constants object.
    ic : IterationConfig
        Iteration configuration object.
    instrument_random_seed : int
        Random seed for instrument noise realization.
    """
    config = get_config_dict_from_file(toml_filename)
    return get_config_objects_from_dict(config)
