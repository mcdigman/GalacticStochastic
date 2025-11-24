"""run iterative processing of galactic background"""

import GalacticStochastic.plot_creation_helpers as pch
from GalacticStochastic import config_helper
from GalacticStochastic.iterative_fit import fetch_or_run_iterative_loop

if __name__ == '__main__':
    # config_filename = 'default_parameters.toml'
    # config_filename = 'Galaxies/GalaxyNarrow/run_old_parameters4.toml'
    filename_config = 'parameters_default_12year.toml'
    target_directory = 'Galaxies/GalaxyFullLDC/'

    config, wc, lc, ic, instrument_random_seed = config_helper.get_config_objects(filename_config)
    config['files']['galaxy_dir'] = target_directory
    cyclo_mode = 'stationary'
    # ifm = fetch_or_run_iterative_loop(config, cyclo_mode=cyclo_mode, fetch_mode=2, output_mode=1, preprocess_mode=1)
    # ifm = fetch_or_run_iterative_loop(config, cyclo_mode=cyclo_mode, fetch_mode=0, output_mode=1, preprocess_mode=0)
    ifm = fetch_or_run_iterative_loop(config, cyclo_mode=cyclo_mode, fetch_mode='run_reprocess_only', output_mode='no_store', preprocess_mode=0)
    # ifm_alt1 = fetch_or_run_iterative_loop(config, cyclo_mode=cyclo_mode, fetch_mode=1, output_mode=1)
    # ifm_alt2 = fetch_or_run_iterative_loop(config, cyclo_mode=cyclo_mode, fetch_mode=2, output_mode=1)
    # ifm_alt3 = fetch_or_run_iterative_loop(config, cyclo_mode=cyclo_mode, fetch_mode=3, output_mode=1)
    # ifm_alt4 = fetch_or_run_iterative_loop(config, cyclo_mode=cyclo_mode, fetch_mode=4, output_mode=1)

    # for itrm in [0, 1, 3, 7]:
    # for itrm in [0]:
    #    cyclo_mode = 0
    #    nt_min = 256 * (7 - itrm)
    #    nt_max = nt_min + 512 * (itrm + 1)
    #    nt_range = (nt_min, nt_max)
    #    ifm = fetch_or_run_iterative_loop(config, cyclo_mode=cyclo_mode, nt_range=nt_range, fetch_mode=0, output_mode=1)

    do_plot_noise_spectrum_ambiguity = True
    if do_plot_noise_spectrum_ambiguity:
        pch.plot_noise_spectrum_ambiguity(ifm)

    do_plot_noise_spectrum_evolve = True
    if do_plot_noise_spectrum_evolve:
        pch.plot_noise_spectrum_evolve(ifm)
