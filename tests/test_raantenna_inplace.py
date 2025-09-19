"""Testing file to ensure rigid_adiabatic_antenna behavior is unchanged"""
from typing import Tuple

import h5py
import numpy as np
import pytest

from LisaWaveformTools.ra_waveform_freq import rigid_adiabatic_antenna
from tests.generate_raantenna_test_outputs import generate_test_inputs
from WaveletWaveforms.sparse_waveform_functions import PixelGenericRange

KNOWN_HDF5_PATH = 'tests/known_raantenna_outputs.hdf5'


def load_known_outputs(hdf5_path) -> Tuple[dict, list[int]]:
    """Load reference sc_channels and kdotx arrays from HDF5."""
    with h5py.File(hdf5_path, 'r') as f:
        seeds: list[int] = []
        results = {}

        realizations = f['realizations']
        if not isinstance(realizations, h5py.Group):
            msg = 'Unrecognized hdf5 file format'
            raise TypeError(msg)

        for seed in realizations:
            if seed is None:
                msg = 'Unrecognized hdf5 file format'
                raise TypeError(msg)
            seeds.append(int(str(seed)))

            realize_loc = realizations[seed]

            if not isinstance(realize_loc, h5py.Group):
                msg = 'Unrecognized hdf5 file format'
                raise TypeError(msg)
            ref_RR = np.array(realize_loc['spacecraft_channels_RR'])
            ref_II = np.array(realize_loc['spacecraft_channels_II'])
            ref_kdotx = np.array(realize_loc['kdotx'])
            results[int(seed)] = (ref_RR, ref_II, ref_kdotx)

    return results, seeds


_outputs_dict, _all_seeds = load_known_outputs(KNOWN_HDF5_PATH)


@pytest.mark.parametrize('seed', _all_seeds)
def test_raantenna_inplace_parametrized(seed):
    spacecraft_channels, params_extrinsic, ts, FFs, nf_low, NTs, kdotx, lc = generate_test_inputs(seed)
    ref_RR, ref_II, ref_kdotx = _outputs_dict[seed]
    kdotx_test = kdotx.copy()
    nf_lim = PixelGenericRange(nf_low, nf_low + NTs, ts[1] - ts[0], lc.t0)
    rigid_adiabatic_antenna(
        spacecraft_channels,
        params_extrinsic,
        ts, FFs, nf_lim, kdotx_test, lc)
    np.testing.assert_allclose(spacecraft_channels.RR, ref_RR, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(spacecraft_channels.II, ref_II, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(kdotx_test, ref_kdotx, rtol=1e-14, atol=1e-14)
