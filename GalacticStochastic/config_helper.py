"""Load all the configuration objects that are needed from an ini file"""
import configparser

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

    return config, wc, lc
