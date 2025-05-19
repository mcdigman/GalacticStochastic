"""Load all the configuration objects that are needed from an ini file"""
import configparser

from GalacticStochastic.iteration_config import get_iteration_config
from LisaWaveformTools.lisa_config import get_lisa_constants
from WaveletWaveforms.wdm_config import get_wavelet_model


def get_config_objects(ini_filename):
    """Load the configuration from the input ini filename
    and create some of the config objects the iterative fit will need
    """
    config = configparser.ConfigParser()
    config.read(ini_filename)

    wc = get_wavelet_model(config)

    lc = get_lisa_constants(config)

    ic = get_iteration_config(config)

    assert ic.nc_galaxy <= lc.nc_waveform, 'cannot compute background for channels without waveform'
    assert ic.nc_galaxy <= lc.nc_snr, 'cannot compute background for channels not included in snr calculation'

    return config, wc, lc, ic
