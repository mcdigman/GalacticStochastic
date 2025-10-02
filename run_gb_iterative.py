"""run iterative processing of galactic background"""

import GalacticStochastic.plot_creation_helpers as pch
from GalacticStochastic.config_helper import get_config_dict_from_file
from GalacticStochastic.iterative_fit import fetch_or_run_iterative_loop

if __name__ == '__main__':
    config_filename = 'default_parameters.toml'
    config = get_config_dict_from_file(config_filename)
    stat_only = False
    ifm_alt = fetch_or_run_iterative_loop(config, stat_only=stat_only, fetch_mode=0, output_mode=1)
    ifm = fetch_or_run_iterative_loop(config, stat_only=stat_only, fetch_mode=1, output_mode=1)

    # for itrm in [0, 1, 3, 7]:
    for itrm in [0]:
        stat_only = False
        nt_min = 256 * (7 - itrm)
        nt_max = nt_min + 512 * (itrm + 1)
        nt_range = (nt_min, nt_max)
        ifm = fetch_or_run_iterative_loop(config, stat_only=stat_only, nt_range=nt_range, fetch_mode=0, output_mode=1)

    do_plot_noise_spectrum_ambiguity = True
    if do_plot_noise_spectrum_ambiguity:
        pch.plot_noise_spectrum_ambiguity(ifm)

    do_plot_noise_spectrum_evolve = True
    if do_plot_noise_spectrum_evolve:
        pch.plot_noise_spectrum_evolve(ifm)
