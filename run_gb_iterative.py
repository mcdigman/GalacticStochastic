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

    galactic_below_in, snrs_tot_in, SAET_m, _, _, _ = gfi.load_init_galactic_file(galaxy_dir, ic.snr_thresh, wc.Nf, wc.Nt, wc.dt)

    for itrm in range(1):
        stat_only = False
        nt_min = 256 * (7 - itrm)
        nt_max = nt_min + 512 * (itrm + 1)
        print(nt_min, nt_max, wc.Nt, wc.Nf, stat_only)

        params_gb, _, _, _, _ = gfi.get_full_galactic_params(galaxy_file, galaxy_dir)

        fit_state = IterativeFitState(ic)

        bgd = BGDecomposition(wc, ic.NC_gal, galactic_floor=galactic_below_in.copy())
        galactic_below_in = None

        noise_manager = NoiseModelManager(ic, wc, fit_state, bgd, SAET_m, stat_only, nt_min, nt_max)

        bis = BinaryInclusionState(wc, ic, lc, params_gb, noise_manager, fit_state, ic.NC_snr, snrs_tot_in)

        ifm = IterativeFitManager(ic, fit_state, noise_manager, bis)

        params_gb = None

        ifm.do_loop()

        do_hf_out = True
        if do_hf_out:
            gfi.store_processed_gb_file(galaxy_dir, galaxy_file, wc, lc, ifm.ic, ifm.noise_manager.nt_min, ifm.noise_manager.nt_max, bgd, ic.period_list, ifm.bis.n_bin_use, ifm.noise_manager.SAET_m, ifm.noise_manager.SAET_fin, ifm.noise_manager.stat_only, ifm.bis.snrs_tot_upper, ifm.n_full_converged, ifm.bis.argbinmap, ifm.bis.faints_old, ifm.bis.faints_cur, ifm.bis.brights)

    do_plot_noise_spectrum_ambiguity = True
    if do_plot_noise_spectrum_ambiguity:
        pch.plot_noise_spectrum_ambiguity(ifm)

    do_plot_noise_spectrum_evolve = True
    if do_plot_noise_spectrum_evolve:
        pch.plot_noise_spectrum_evolve(ifm)
