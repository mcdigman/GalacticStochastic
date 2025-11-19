# python
"""Run entire iterative processing of a galactic binary file."""

import argparse

from GalacticStochastic.config_helper import get_config_dict_from_file
from GalacticStochastic.iterative_fit import fetch_or_run_iterative_loop


def main() -> None:
    """Run iterative processing from command line."""
    parser = argparse.ArgumentParser(
        prog='run-galactic-stochastic-iterative',
        description='Run iterative processing for a galactic binary file',
    )
    parser.add_argument('galaxy_dir', help='target galaxy directory')
    parser.add_argument('config_file', help='path to config file')
    parser.add_argument(
        '--fetch-mode',
        type=str,
        default='run_all',
        help="fetch_mode (default: 'run_all')",
    )
    parser.add_argument(
        '--cyclo-mode',
        type=str,
        default='stationary',
        help="cyclo_mode (default: 'stationary')",
    )
    parser.add_argument(
        '--output-mode',
        type=str,
        default='store_always',
        help="output_mode (default: 'store_always')",
    )
    parser.add_argument(
        '--preprocess-mode',
        type=str,
        default='final',
        help="preprocess_mode (default: 'final')",
    )

    args = parser.parse_args()

    target_directory = str(args.galaxy_dir)
    config_filename = str(args.config_file)
    config = get_config_dict_from_file(config_filename)

    target_directory_got = config['files'].get('galaxy_dir', target_directory)
    if target_directory != target_directory_got:
        msg = f'Inconsistent target directory: cli got {target_directory} but {target_directory_got} in config file'
        raise ValueError(msg)

    config['files']['galaxy_dir'] = target_directory

    fetch_or_run_iterative_loop(
        config,
        cyclo_mode=args.cyclo_mode,
        fetch_mode=args.fetch_mode,
        output_mode=args.output_mode,
        preprocess_mode=args.preprocess_mode,
    )


if __name__ == '__main__':
    main()
