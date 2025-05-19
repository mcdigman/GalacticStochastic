"""run iterative processing of galactic background"""

import configparser

import numpy as np

import GalacticStochastic.global_file_index as gfi
import GalacticStochastic.plot_creation_helpers as pch
from GalacticStochastic.background_decomposition import BGDecomposition
from GalacticStochastic.inclusion_state_manager import BinaryInclusionState
from GalacticStochastic.iteration_config import get_iteration_config
from GalacticStochastic.iterative_fit_manager import IterativeFitManager
from GalacticStochastic.iterative_fit_state_machine import IterativeFitState
from GalacticStochastic.noise_manager import NoiseModelManager
from LisaWaveformTools.instrument_noise import instrument_noise_AET_wdm_m
from LisaWaveformTools.lisa_config import get_lisa_constants
from WaveletWaveforms.wdm_config import get_wavelet_model

if __name__ == '__main__':

    a = np.array([])

    config = configparser.ConfigParser()
    config.read('default_parameters.ini')

    galaxy_file = config['files']['galaxy_file']
    galaxy_dir = config['files']['galaxy_dir']

    wc = get_wavelet_model(config)

    lc = get_lisa_constants(config)

    ic = get_iteration_config(config)

    S_inst_m = instrument_noise_AET_wdm_m(lc, wc)

    stat_only = True
    preprocess_mode = True
    nt_min = 0
    nt_max = wc.Nt

    params_gb, _, _, _, _ = gfi.get_full_galactic_params(galaxy_file, galaxy_dir)

    fit_state = IterativeFitState(ic, preprocess_mode=2)

    bgd = BGDecomposition(wc, ic.NC_gal)

    noise_manager = NoiseModelManager(ic, wc, fit_state, bgd, S_inst_m, stat_only, nt_min, nt_max)

    bis = BinaryInclusionState(wc, ic, lc, params_gb, noise_manager, fit_state, ic.NC_snr)

    ifm = IterativeFitManager(ic, fit_state, noise_manager, bis)

    params_gb = None

    ifm.do_loop()

    do_hf_out = True
    if do_hf_out:
        gfi.store_preliminary_gb_file(galaxy_dir, galaxy_file, wc, lc, ic, bgd.get_galactic_below_low(), bis.n_bin_use, noise_manager.S_inst_m, bis.snrs_tot_lower)

    do_plot_noise_spectrum_ambiguity = True

    if do_plot_noise_spectrum_ambiguity:
        pch.plot_noise_spectrum_ambiguity(ifm)

    do_plot_noise_spectrum_evolve = True
    if do_plot_noise_spectrum_evolve:
        pch.plot_noise_spectrum_evolve(ifm)
