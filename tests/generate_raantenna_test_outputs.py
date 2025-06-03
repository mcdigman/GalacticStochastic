from pathlib import Path

import h5py
import numpy as np
import tomllib

from LisaWaveformTools.lisa_config import get_lisa_constants
from LisaWaveformTools.ra_waveform_freq import AntennaResponseChannels, ExtrinsicParams, rigid_adiabatic_antenna

# If you need a toml config, import get_lisa_constants, Path, tomllib, etc., and update generate_test_inputs accordingly.


def generate_test_inputs(seed, nt_loc=128):
    """Programmatically create varied yet deterministic test inputs for rigid_adiabatic_antenna."""
    toml_filename = 'tests/raantenna_test_config1.toml'
    with Path(toml_filename).open('rb') as f:
        config = tomllib.load(f)

    lc = get_lisa_constants(config)

    rng = np.random.default_rng(seed)

    n_channels = 3
    dt = 0.05
    dfdt = 0.01
    dtdf = 0.1
    f0 = 0.01
    DF = 0.1
    f_scatter = 0.01
    t_scatter = 0.1

    RR = np.zeros((n_channels, nt_loc))
    II = np.zeros((n_channels, nt_loc))
    dRR = np.zeros((n_channels, nt_loc))
    dII = np.zeros((n_channels, nt_loc))
    kdotx = np.zeros(nt_loc)
    t_model = rng.integers(0, 2)
    if t_model == 0:
        TTs = dt * np.arange(0, nt_loc)
        ts = TTs
        FFs = np.abs(f0 + dfdt * TTs + dfdt * rng.normal(f_scatter, size=nt_loc))
    else:
        FFs = np.arange(1, nt_loc + 1) * DF
        TTs = dtdf * FFs + rng.normal(t_scatter, size=nt_loc)
        ts = TTs

    spacecraft_channels = AntennaResponseChannels(TTs, RR.copy(), II.copy(), dRR.copy(), dII.copy())

    params_extrinsic = ExtrinsicParams(
        costh=rng.uniform(-1., 1.), phi=rng.uniform(0, 2 * np.pi), cosi=rng.uniform(-1., 1.), psi=rng.uniform(0, 2 * np.pi),
    )

    nf_low = 0
    NTs = nt_loc

    return spacecraft_channels, params_extrinsic, ts, FFs, nf_low, NTs, kdotx, lc


def main(seeds, output_path):
    with h5py.File(output_path, 'w') as f:
        realizations = f.create_group('realizations')
        for seed in seeds:
            # Generate inputs and run
            spacecraft_channels, params_extrinsic, ts, FFs, nf_low, NTs, kdotx, lc = generate_test_inputs(seed)
            kdotx_mut = kdotx.copy()  # Don't overwrite for future runs
            rigid_adiabatic_antenna(
                spacecraft_channels,
                params_extrinsic,
                ts, FFs, nf_low, NTs, kdotx_mut, lc,
            )
            # Prepare group
            realization = realizations.create_group(str(seed))
            realization.create_dataset('spacecraft_channels_RR', data=spacecraft_channels.RR)
            realization.create_dataset('spacecraft_channels_II', data=spacecraft_channels.II)
            realization.create_dataset('_kdotx', data=kdotx_mut)
    print(f'Saved results for {len(seeds)} seeds to {output_path}')


if __name__ == '__main__':
    # Edit the list of seeds and path as desired
    seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 42, 123, 2024, 31415926535]
    output_path = 'tests/known_raantenna_outputs.hdf5'
    main(seeds, output_path)
