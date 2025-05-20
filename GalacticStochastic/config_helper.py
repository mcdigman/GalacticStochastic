"""Load all the configuration objects that are needed from a toml file"""

from pathlib import Path

import tomllib

from GalacticStochastic.iteration_config import get_iteration_config
from LisaWaveformTools.lisa_config import get_lisa_constants
from WaveletWaveforms.wdm_config import get_wavelet_model


def get_config_objects(toml_filename):
    """Load the configuration from the input toml filename
    and create some of the config objects the iterative fit will need
    """
    with Path.open(toml_filename, 'rb') as f:
        config = tomllib.load(f)

    wc = get_wavelet_model(config)

    lc = get_lisa_constants(config)

    ic = get_iteration_config(config)

    assert ic.nc_galaxy <= lc.nc_waveform, 'cannot compute background for channels without waveform'
    assert ic.nc_galaxy <= lc.nc_snr, 'cannot compute background for channels not included in snr calculation'

    return config, wc, lc, ic
