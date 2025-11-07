"""Run entire iterative processing of a galactic binary file"""

import sys

from GalacticStochastic.config_helper import get_config_dict_from_file
from GalacticStochastic.iterative_fit import fetch_or_run_iterative_loop


def main() -> None:
    if len(sys.argv) != 3:
        print('run_iterative_procedure galaxy_dir config_file')
    target_directory = str(sys.argv[1])
    config_filename = str(sys.argv[2])
    config = get_config_dict_from_file(config_filename)
    target_directory_got = config['files'].get('galaxy_dir', target_directory)
    if target_directory != target_directory_got:
        msg = f'Inconsistent target directory: cli got {target_directory} but {target_directory_got} in config file'
        raise ValueError(msg)

    config['files']['galaxy_dir'] = target_directory

    cyclo_mode = 1
    ifm = fetch_or_run_iterative_loop(config, cyclo_mode=cyclo_mode, fetch_mode=3, output_mode=1, preprocess_mode=0)


if __name__ == '__main__':
    main()
