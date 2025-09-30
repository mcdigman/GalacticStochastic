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


def fetch_or_run_iterative_loop(nt_min, nt_max, config, wc, lc, ic, instrument_random_seed, stat_only, *, fetch_mode=0, output_mode=1):
    del fetch_mode
    nt_lim_snr = PixelGenericRange(nt_min, nt_max, wc.DT, 0.)
    nt_lim_waveform = PixelGenericRange(0, wc.Nt, wc.DT, 0.)

    print(nt_lim_snr.nx_min, nt_lim_snr.nx_max, nt_lim_waveform.nx_min, nt_lim_waveform.nx_max, wc.Nt, wc.Nf, stat_only)

    galactic_below_in, snrs_tot_in, _ = gfi.load_preliminary_galactic_file(
        config, ic, wc, lc,
    )

    params_gb, _ = gfi.get_full_galactic_params(config)

    fit_state = IterativeFitState(ic)

    bgd = BGDecomposition(wc, ic.nc_galaxy, galactic_floor=galactic_below_in.copy(), storage_mode=ic.background_storage_mode)
    del galactic_below_in

    noise_manager = NoiseModelManager(ic, wc, lc, fit_state, bgd, stat_only, nt_lim_snr, instrument_random_seed=instrument_random_seed)

    bis = BinaryInclusionState(wc, ic, lc, params_gb, noise_manager, fit_state, nt_lim_waveform, snrs_tot_in)

    del snrs_tot_in

    ifm = IterativeFitManager(ic, fit_state, noise_manager, bis)

    del params_gb

    ifm.do_loop()

    if output_mode == 1:
        gfi.store_processed_gb_file(
            config,
            wc,
            ic,
            ifm,
        )
    elif output_mode == 0:
        pass
    else:
        msg = 'Unrecognized option for output_mode'
        raise ValueError(msg)
    return ifm


if __name__ == '__main__':
    a = np.array([])

    config_filename = 'default_parameters.toml'
    config, wc, lc, ic, instrument_random_seed = get_config_objects(config_filename)

    for itrm in [0, 1, 3, 7]:
        stat_only = False
        nt_min = 256 * (7 - itrm)
        nt_max = nt_min + 512 * (itrm + 1)
        ifm = fetch_or_run_iterative_loop(nt_min, nt_max, config, wc, lc, ic, instrument_random_seed, stat_only=stat_only, fetch_mode=0, output_mode=1)

    do_plot_noise_spectrum_ambiguity = True
    if do_plot_noise_spectrum_ambiguity:
        pch.plot_noise_spectrum_ambiguity(ifm)

    do_plot_noise_spectrum_evolve = True
    if do_plot_noise_spectrum_evolve:
        pch.plot_noise_spectrum_evolve(ifm)
