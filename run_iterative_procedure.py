"""Run entire iterative processing of a galactic binary file"""

from GalacticStochastic.config_helper import get_config_dict_from_file
from GalacticStochastic.iterative_fit import fetch_or_run_iterative_loop

if __name__ == '__main__':
    config_filename = 'Galaxies/Galaxy1/run_default_parameters.toml'
    config = get_config_dict_from_file(config_filename)
    cyclo_mode = 1  # use the cyclostationary model by default
    ifm = fetch_or_run_iterative_loop(config, cyclo_mode=cyclo_mode, fetch_mode=3, output_mode=1, preprocess_mode=0)
