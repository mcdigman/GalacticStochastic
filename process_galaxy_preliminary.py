"""run iterative processing of galactic background"""


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
    config_file = 'default_parameters.toml'
    config, wc, lc, ic, instrument_random_seed = get_config_objects(config_file)

    galaxy_file = config['files']['galaxy_file']

    cyclo_mode = 1
    preprocess_mode = 1
    nt_min = 0
    nt_max = wc.Nt
    nt_lim_snr = PixelGenericRange(nt_min, nt_max, wc.DT, 0.)
    nt_lim_waveform = PixelGenericRange(nt_min, nt_max, wc.DT, 0.)

    params_gb, _ = gfi.get_full_galactic_params(config)

    fit_state = IterativeFitState(ic, preprocess_mode=preprocess_mode)

    bgd = BGDecomposition(wc, ic.nc_galaxy, storage_mode=ic.background_storage_mode)

    noise_manager = NoiseModelManager(ic, wc, lc, fit_state, bgd, cyclo_mode, nt_lim_snr, instrument_random_seed)

    bis = BinaryInclusionState(wc, ic, lc, params_gb, noise_manager, fit_state, nt_lim_waveform)

    ifm = IterativeFitManager(ic, fit_state, noise_manager, bis)

    del params_gb

    ifm.do_loop()

    do_hf_out = True
    if do_hf_out:
        gfi.store_preliminary_gb_file(
            config,
            wc,
            ifm,
            write_mode=2,
        )

    do_plot_noise_spectrum_ambiguity = True

    if do_plot_noise_spectrum_ambiguity:
        pch.plot_noise_spectrum_ambiguity(ifm)

    do_plot_noise_spectrum_evolve = True
    if do_plot_noise_spectrum_evolve:
        pch.plot_noise_spectrum_evolve(ifm)
