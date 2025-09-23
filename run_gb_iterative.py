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
from WaveletWaveforms.sparse_waveform_functions import PixelGenericRange

if __name__ == '__main__':
    a = np.array([])

    config, wc, lc, ic = get_config_objects('default_parameters.toml')

    galaxy_file = config['files']['galaxy_file']
    galaxy_dir = config['files']['galaxy_dir']

    galactic_below_in, snrs_tot_in, S_inst_m, wc, lc = gfi.load_preliminary_galactic_file(
        config, ic, wc, lc,
    )

    for itrm in range(1):
        stat_only = False
        nt_min = 256 * (7 - itrm)
        nt_max = nt_min + 512 * (itrm + 1)
        nt_lim_snr = PixelGenericRange(nt_min, nt_max, wc.DT, 0.)
        nt_lim_waveform = PixelGenericRange(0, wc.Nt, wc.DT, 0.)

        print(nt_lim_snr.nx_min, nt_lim_snr.nx_max, nt_lim_waveform.nx_min, nt_lim_waveform.nx_max, wc.Nt, wc.Nf, stat_only)

        params_gb, _, _, _, _ = gfi.get_full_galactic_params(config)

        fit_state = IterativeFitState(ic)

        bgd = BGDecomposition(wc, ic.nc_galaxy, galactic_floor=galactic_below_in.copy())
        del galactic_below_in

        noise_manager = NoiseModelManager(ic, wc, lc, fit_state, bgd, S_inst_m, stat_only, nt_lim_snr)

        bis = BinaryInclusionState(wc, ic, lc, params_gb, noise_manager, fit_state, nt_lim_waveform, snrs_tot_in)

        ifm = IterativeFitManager(ic, fit_state, noise_manager, bis)

        del params_gb

        ifm.do_loop()

        do_hf_out = True
        if do_hf_out:
            gfi.store_processed_gb_file(
                config,
                wc,
                lc,
                ifm.ic,
                ifm.noise_manager,
                bgd,
                ifm.n_full_converged,
                ifm.bis,
            )

    do_plot_noise_spectrum_ambiguity = True
    if do_plot_noise_spectrum_ambiguity:
        pch.plot_noise_spectrum_ambiguity(ifm)

    do_plot_noise_spectrum_evolve = True
    if do_plot_noise_spectrum_evolve:
        pch.plot_noise_spectrum_evolve(ifm)
