"""run iterative processing of galactic background"""

import configparser

import matplotlib.pyplot as plt
import numpy as np

import GalacticStochastic.global_file_index as gfi
import GalacticStochastic.plot_creation_helpers as pch
from GalacticStochastic.iteration_config import get_iteration_config
from GalacticStochastic.iterative_fit_manager import IterativeFitManager
from LisaWaveformTools.lisa_config import get_lisa_constants
from WaveletWaveforms.wdm_config import get_wavelet_model

if __name__ == '__main__':

    config = configparser.ConfigParser()
    config.read('default_parameters.ini')

    galaxy_file = config['files']['galaxy_file']
    galaxy_dir = config['files']['galaxy_dir']

    wc = get_wavelet_model(config)

    lc = get_lisa_constants(config)

    ic = get_iteration_config(config)

    galactic_below_in, snr_tots_in, SAET_m, _, _, snr_min_in = gfi.load_init_galactic_file(galaxy_dir, ic.snr_thresh, wc.Nf, wc.Nt, wc.dt)

    for itrm in range(0, 1):
        stat_only = False
        nt_min = 256*(7-itrm)
        nt_max = nt_min+512*(itrm+1)
        print(nt_min, nt_max, wc.Nt, wc.Nf, stat_only)

        params_gb, _, _, _, _ = gfi.get_full_galactic_params(galaxy_file, galaxy_dir)

        ifm = IterativeFitManager(lc, wc, ic, SAET_m, galactic_below_in, snr_tots_in, snr_min_in, params_gb, nt_min, nt_max, stat_only)

        params_gb = None

        ifm.do_loop()

        do_hf_out = True
        if do_hf_out:
            gfi.store_processed_gb_file(galaxy_dir, galaxy_file, ifm.wc, ifm.lc, ifm.ic, ifm.nt_min, ifm.nt_max, ifm.bgd, ic.period_list, ifm.n_bin_use, ifm.SAET_m, ifm.noise_manager.SAET_fin, ifm.stat_only, ifm.bis.snrs_tot_upper, ifm.n_full_converged, ifm.argbinmap, ifm.bis.faints_old, ifm.bis.faints_cur, ifm.bis.brights, snr_min_in)

    do_plot_noise_spectrum_ambiguity = True
    if do_plot_noise_spectrum_ambiguity:
        pch.plot_noise_spectrum_ambiguity(ifm)

    do_plot_noise_spectrum_evolve = True
    if do_plot_noise_spectrum_evolve:
        pch.plot_noise_spectrum_evolve(ifm)
