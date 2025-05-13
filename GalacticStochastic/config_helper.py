import configparser

from LisaWaveformTools.lisa_config import get_lisa_constants
from WaveletWaveforms.wdm_config import get_wavelet_model


def get_config_objects(ini_filename):
    config = configparser.ConfigParser()
    config.read(ini_filename)

    galaxy_dir = config['files']['galaxy_dir']

    wc = get_wavelet_model(config)

    lc = get_lisa_constants(config)

    return config, wc, lc
