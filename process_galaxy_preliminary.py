"""run iterative processing of galactic background"""

import numpy as np

import GalacticStochastic.global_file_index as gfi
import GalacticStochastic.plot_creation_helpers as pch
from GalacticStochastic.background_decomposition import BGDecomposition
from GalacticStochastic.config_helper import get_config_objects
from GalacticStochastic.inclusion_state_manager import BinaryInclusionState
from GalacticStochastic.iterative_fit_manager import IterativeFitManager
from GalacticStochastic.iterative_fit_state_machine import IterativeFitState
from GalacticStochastic.noise_manager import NoiseModelManager
from LisaWaveformTools.instrument_noise import instrument_noise_AET_wdm_m
from WaveletWaveforms.sparse_waveform_functions import PixelTimeRange

if __name__ == '__main__':
    a = np.array([])

    config_file = 'default_parameters.toml'
    config, wc, lc, ic = get_config_objects(config_file)

    galaxy_file = config['files']['galaxy_file']
    galaxy_dir = config['files']['galaxy_dir']

    S_inst_m = instrument_noise_AET_wdm_m(lc, wc)

    stat_only = True
    preprocess_mode = True
    nt_min = 0
    nt_max = wc.Nt
    nt_lim_snr = PixelTimeRange(nt_min, nt_max)
    nt_lim_waveform = PixelTimeRange(nt_min, nt_max)

    params_gb, _, _, _, _ = gfi.get_full_galactic_params(galaxy_file, galaxy_dir)

    fit_state = IterativeFitState(ic, preprocess_mode=1)

    bgd = BGDecomposition(wc, ic.nc_galaxy)

    noise_manager = NoiseModelManager(ic, wc, lc, fit_state, bgd, S_inst_m, stat_only, nt_lim_snr)

    bis = BinaryInclusionState(wc, ic, lc, params_gb, noise_manager, fit_state, nt_lim_waveform)

    ifm = IterativeFitManager(ic, fit_state, noise_manager, bis)

    del params_gb

    ifm.do_loop()

    do_hf_out = True
    if do_hf_out:
        gfi.store_preliminary_gb_file(
            config_file,
            galaxy_dir,
            galaxy_file,
            wc,
            lc,
            ic,
            bgd.get_galactic_below_low(),
            noise_manager.S_inst_m,
            bis.snrs_tot_lower,
        )

    do_plot_noise_spectrum_ambiguity = True

    if do_plot_noise_spectrum_ambiguity:
        pch.plot_noise_spectrum_ambiguity(ifm)

    do_plot_noise_spectrum_evolve = True
    if do_plot_noise_spectrum_evolve:
        pch.plot_noise_spectrum_evolve(ifm)
