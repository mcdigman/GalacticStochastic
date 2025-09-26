"""Load all the configuration objects that are needed from a toml file"""

from pathlib import Path
from typing import Any

import tomllib

from GalacticStochastic.iteration_config import IterationConfig, get_iteration_config
from LisaWaveformTools.lisa_config import LISAConstants, get_lisa_constants
from WaveletWaveforms.wdm_config import WDMWaveletConstants, get_wavelet_model


def get_config_objects(toml_filename: str) -> tuple[dict[str, Any], WDMWaveletConstants, LISAConstants, IterationConfig]:
    """Load the configuration from the input toml filename
    and create some of the config objects the iterative fit will need
    """
    with Path(toml_filename).open('rb') as f:
        config: dict[str, Any] = tomllib.load(f)

    wc = get_wavelet_model(config)

    lc = get_lisa_constants(config)

    ic = get_iteration_config(config)

    assert ic.nc_galaxy <= lc.nc_waveform, 'cannot compute background for channels without intrinsic_waveform'
    assert ic.nc_galaxy <= lc.nc_snr, 'cannot compute background for channels not included in snr calculation'

    return config, wc, lc, ic
