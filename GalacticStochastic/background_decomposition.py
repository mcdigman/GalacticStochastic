"""Classes and functions to handle the decomposition of the galactic background."""

from __future__ import annotations

from typing import TYPE_CHECKING

import h5py
import numpy as np
from numpy.testing import assert_allclose

from GalacticStochastic.galactic_fit_helpers import get_S_cyclo
from WaveletWaveforms.sparse_waveform_functions import SparseWaveletWaveform, sparse_addition_helper

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from WaveletWaveforms.wdm_config import WDMWaveletConstants


class BGDecomposition:
    """Handle the internal decomposition of the galactic background."""

    def __init__(
        self,
        wc: WDMWaveletConstants,
        nc_galaxy: int,
        *,
        galactic_floor: NDArray[np.floating] | None = None,
        galactic_below: NDArray[np.floating] | None = None,
        galactic_undecided: NDArray[np.floating] | None = None,
        galactic_above: NDArray[np.floating] | None = None,
        track_mode: int = 1,
        storage_mode: int = 0,
    ) -> None:
        """
        Initialize a BGDecomposition object for handling the galactic background decomposition.

        Parameters
        ----------
        wc : WDMWaveletConstants
            Wavelet constants describing the time-frequency grid.
        nc_galaxy : int
            Number of tdi channels in the galactic background.
        galactic_floor : ndarray of float, optional
            Array containing the faintest (floor) component of the galactic background.
            If None, initialized to zeros.
        galactic_below : ndarray of float, optional
            Array containing the faint component of the galactic background.
            If None, initialized to zeros.
        galactic_undecided : ndarray of float, optional
            Array containing the undecided component of the galactic background.
            If None, initialized to zeros.
        galactic_above : ndarray of float, optional
            Array containing the bright (above threshold) component of the galactic background.
            If None, initialized to zeros.
        track_mode : int, optional
            If nonzero, enables internal consistency checks and diagnostics. Default is 1.
        storage_mode : int, optional
            Storage mode for the background. Only 0 is supported. Default is 0.

        Raises
        ------
        ValueError
            If an unrecognized storage mode is provided.
        AssertionError
            If provided arrays do not match the expected shapes.
        """
        if storage_mode != 0:
            msg = 'Unrecognized option for storage mode'
            raise ValueError(msg)

        self._wc: WDMWaveletConstants = wc
        self._storage_mode: int = storage_mode
        self._nc_galaxy: int = nc_galaxy
        self._shape1: tuple[int, int] = (wc.Nt * wc.Nf, self.nc_galaxy)
        self._shape2: tuple[int, int, int] = (wc.Nt, wc.Nf, self.nc_galaxy)

        if galactic_floor is None:
            self.galactic_floor: NDArray[np.floating] = np.zeros(self._shape1, dtype=np.float64)
        else:
            assert galactic_floor.shape == self._shape1
            self.galactic_floor = galactic_floor

        if galactic_below is None:
            self.galactic_below: NDArray[np.floating] = np.zeros(self._shape1)
        else:
            assert galactic_below.shape == self._shape1
            self.galactic_below = galactic_below

        if galactic_undecided is None:
            self.galactic_undecided: NDArray[np.floating] = np.zeros(self._shape1, dtype=np.float64)
        else:
            assert galactic_undecided.shape == self._shape1
            self.galactic_undecided = galactic_undecided

        if galactic_above is None:
            self.galactic_above: NDArray[np.floating] = np.zeros(self._shape1, dtype=np.float64)
        else:
            assert galactic_above.shape == self._shape1
            self.galactic_above = galactic_above

        self.galactic_total_cache: NDArray[np.floating] | None = None

        self.track_mode: int = track_mode

        self.power_galactic_undecided: list[NDArray[np.floating]] = []
        self.power_galactic_above: list[NDArray[np.floating]] = []
        self.power_galactic_below_low: list[NDArray[np.floating]] = []
        self.power_galactic_below_high: list[NDArray[np.floating]] = []
        self.power_galactic_total: list[NDArray[np.floating]] = []

    @property
    def nc_galaxy(self) -> int:
        """Number of tdi channels in the galactic background."""
        return self._nc_galaxy

    def store_hdf5(self, hf_in: h5py.Group, *, group_name: str = 'background', group_mode: int = 0) -> h5py.Group:
        """Store the background to an hdf5 file."""
        if group_mode == 0:
            hf_background = hf_in.create_group(group_name)
        elif group_mode == 1:
            hf_background = hf_in
        else:
            msg = 'Unrecognized option for group mode'
            raise NotImplementedError(msg)
        hf_background.attrs['creator_name'] = self.__class__.__name__
        hf_background.attrs['storage_mode'] = self._storage_mode
        hf_background.attrs['track_mode'] = self.track_mode
        hf_background.attrs['nc_galaxy'] = self.nc_galaxy
        hf_background.attrs['shape1'] = self._shape1
        hf_background.attrs['shape2'] = self._shape2

        if self._storage_mode == 0:
            _ = hf_background.create_dataset('galactic_below_low', data=self.get_galactic_below_low(), compression='gzip')
            _ = hf_background.create_dataset('galactic_above', data=self.get_galactic_coadd_resolvable(), compression='gzip')
            _ = hf_background.create_dataset('galactic_undecided', data=self.get_galactic_coadd_undecided(), compression='gzip')
        return hf_background

    def load_hdf5(self, hf_in: h5py.Group, *, group_name: str = 'background', group_mode: int = 0) -> None:
        """Load the background from an hdf5 file."""
        if group_mode == 0:
            hf_background = hf_in[group_name]
        elif group_mode == 1:
            hf_background = hf_in
        else:
            msg = 'Unrecognized option for group mode'
            raise NotImplementedError(msg)
        if not isinstance(hf_background, h5py.Group):
            msg = 'Could not find group ' + group_name + ' in hdf5 file'
            raise TypeError(msg)

        assert hf_background.attrs['creator_name'] == self.__class__.__name__, 'incorrect creator name found in hdf5 file'

        storage_mode_temp = hf_background.attrs['storage_mode']
        assert isinstance(storage_mode_temp, (int, np.integer))
        self._storage_mode = int(storage_mode_temp)
        track_mode_temp = hf_background.attrs['track_mode']
        assert isinstance(track_mode_temp, (int, np.integer))
        self.track_mode = int(track_mode_temp)
        nc_galaxy_temp = hf_background.attrs['nc_galaxy']
        assert isinstance(nc_galaxy_temp, (int, np.integer))

        nc_galaxy_loaded = int(nc_galaxy_temp)
        if nc_galaxy_loaded != self.nc_galaxy:
            msg = f'nc_galaxy does not match: {nc_galaxy_loaded} != {self.nc_galaxy}'
            raise ValueError(msg)

        shape1_temp = hf_background.attrs['shape1']
        assert isinstance(shape1_temp, (tuple, list, np.ndarray))
        shape1_loaded = tuple(int(x) for x in shape1_temp)

        if shape1_loaded != self._shape1:
            msg = f'shape1 does not match: {shape1_loaded} != {self._shape1}'
            raise ValueError(msg)

        shape2_temp = hf_background.attrs['shape2']
        assert isinstance(shape2_temp, (tuple, list, np.ndarray))
        shape2_loaded = tuple(int(x) for x in shape2_temp)

        if shape2_loaded != self._shape2:
            msg = f'shape2 does not match: {shape2_loaded} != {self._shape2}'
            raise ValueError(msg)

        if self._storage_mode == 0:
            self.galactic_below[:] = 0.0  # reset to zero, since we cannot separate the two components

            galactic_below_low_temp = hf_background['galactic_below_low']
            assert isinstance(galactic_below_low_temp, h5py.Dataset)
            galactic_below_low = np.asarray(galactic_below_low_temp)
            assert galactic_below_low.shape == self._shape1, 'Incorrect shape for galactic_below_low in hdf5 file'
            self.galactic_floor[:] = galactic_below_low

            galactic_above_temp = hf_background['galactic_above']
            assert isinstance(galactic_above_temp, h5py.Dataset)
            galactic_above = np.asarray(galactic_above_temp)
            assert galactic_above.shape == self._shape1, 'Incorrect shape for galactic_above in hdf5 file'
            self.galactic_above[:] = galactic_above

            galactic_undecided_temp = hf_background['galactic_undecided']
            assert isinstance(galactic_undecided_temp, h5py.Dataset)
            galactic_undecided = np.asarray(galactic_undecided_temp)
            assert galactic_undecided.shape == self._shape1, 'Incorrect shape for galactic_undecided in hdf5 file'
            self.galactic_undecided[:] = galactic_undecided

            self.galactic_total_cache = None  # reset cache since we have new data
            # reset diagnostics
            self.power_galactic_above = []
            self.power_galactic_undecided = []
            self.power_galactic_below_low = []
            self.power_galactic_below_high = []
            self.power_galactic_total = []

    def get_galactic_total(self, *, bypass_check: bool = False) -> NDArray[np.floating]:
        """Get the sum of the entire galactic signal, including detectable binaries."""
        if not bypass_check:
            self.state_check()
        return self.get_galactic_below_high(bypass_check=True) + self.galactic_above

    def get_galactic_below_high(self, *, bypass_check: bool = False) -> NDArray[np.floating]:
        """Get the upper estimate of the unresolvable signal from the galactic background.
        Assume that the undecided part of the signal *is* part of the unresolvable background
        """
        if not bypass_check:
            self.state_check()
        return self.get_galactic_below_low(bypass_check=True) + self.galactic_undecided

    def get_galactic_below_low(self, *, bypass_check: bool = False) -> NDArray[np.floating]:
        """Get the lower estimate of the unresolvable signal from the galactic background.
        Assume that the undecided part of the signal *is not* part of the unresolvable background.
        """
        if not bypass_check:
            self.state_check()
        return self.galactic_floor + self.galactic_below

    def get_galactic_coadd_resolvable(self, *, bypass_check: bool = False) -> NDArray[np.floating]:
        """Get the coadded signal from only bright/resolvable galactic binaries."""
        if not bypass_check:
            self.state_check()
        return self.galactic_above

    def get_galactic_coadd_undecided(self, *, bypass_check: bool = False) -> NDArray[np.floating]:
        """Get the coadded signal from galactic binaries whose status as bright or faint has not yet been decided."""
        if not bypass_check:
            self.state_check()
        return self.galactic_undecided

    def get_galactic_coadd_floor(self, *, bypass_check: bool = False) -> NDArray[np.floating]:
        """Get the coadded signal from the faintest set of galactic binaries."""
        if not bypass_check:
            self.state_check()
        return self.galactic_floor

    def state_check(self) -> None:
        """If the total recorded galactic signal is cached, check that the total not changed much.
        Otherwise, cache the current total so future runs can check if it has changed.
        """
        if self.track_mode:
            if self.galactic_total_cache is None:
                assert np.all(self.galactic_below == 0.0)
                self.galactic_total_cache = self.get_galactic_total(bypass_check=True)
            else:
                # check all contributions to the total signal are tracked accurately
                assert_allclose(
                    self.galactic_total_cache,
                    self.get_galactic_total(bypass_check=True),
                    atol=1.0e-300,
                    rtol=1.0e-6,
                )

    def log_state(self, S_mean: NDArray[np.floating]) -> None:
        """Record any diagnostics we want to track about this iteration."""
        power_undecided = np.asarray(
            np.sum(
                np.sum((self.galactic_undecided**2).reshape(self._shape2)[:, 1:, :], axis=0) / S_mean[1:, :],
                axis=0,
            ),
            dtype=np.float64,
        )
        power_above = np.asarray(
            np.sum(
                np.sum((self.galactic_above**2).reshape(self._shape2)[:, 1:, :], axis=0) / S_mean[1:, :],
                axis=0,
            ),
            dtype=np.float64,
        )

        power_total = np.asarray(
            np.sum(
                np.sum((self.get_galactic_total(bypass_check=True) ** 2).reshape(self._shape2)[:, 1:, :], axis=0) / S_mean[1:, :],
                axis=0,
            ),
            dtype=np.float64,
        )
        power_below_high = np.asarray(
            np.sum(
                np.sum((self.get_galactic_below_high(bypass_check=True) ** 2).reshape(self._shape2)[:, 1:, :], axis=0) / S_mean[1:, :],
                axis=0,
            ),
            dtype=np.float64,
        )
        power_below_low = np.asarray(
            np.sum(
                np.sum((self.get_galactic_below_low(bypass_check=True) ** 2).reshape(self._shape2)[:, 1:, :], axis=0) / S_mean[1:, :],
                axis=0,
            ),
            dtype=np.float64,
        )

        self.power_galactic_undecided.append(power_undecided)
        self.power_galactic_above.append(power_above)

        self.power_galactic_total.append(power_total)
        self.power_galactic_below_high.append(power_below_high)
        self.power_galactic_below_low.append(power_below_low)

    def clear_undecided(self) -> None:
        """Clear the undecided part of the galactic spectrum."""
        self.galactic_undecided[:] = 0.0

    def clear_above(self) -> None:
        """Clear the bright part of the galactic spectrum."""
        self.galactic_above[:] = 0.0

    def clear_below(self) -> None:
        """Clear the faint part of the galactic spectrum."""
        self.galactic_below[:] = 0.0

    def get_S_below_high(self, S_mean: NDArray[np.floating], smooth_lengthf: float, filter_periods: int, period_list: tuple[int, ...] | tuple[np.floating, ...]) -> NDArray[np.floating]:
        """Get the upper estimate of the galactic power spectrum."""
        galactic_loc = self.get_galactic_below_high(bypass_check=True)
        S, _, _, _, _ = get_S_cyclo(galactic_loc, S_mean, self._wc.DT, smooth_lengthf, filter_periods,
                                    period_list=period_list)
        return S

    def get_S_below_low(self, S_mean: NDArray[np.floating], smooth_lengthf: float, filter_periods: int, period_list: tuple[int, ...] | tuple[np.floating, ...]) -> NDArray[np.floating]:
        """Get the lower estimate of the galactic power spectrum."""
        galactic_loc = self.get_galactic_below_low(bypass_check=True)
        S, _, _, _, _ = get_S_cyclo(galactic_loc, S_mean, self._wc.DT, smooth_lengthf, filter_periods,
                                    period_list=period_list)
        return S

    def add_undecided(self, wavelet_waveform: SparseWaveletWaveform) -> None:
        """Add a binary to the undecided component of the galactic background."""
        sparse_addition_helper(wavelet_waveform, self.galactic_undecided)

    def add_floor(self, wavelet_waveform: SparseWaveletWaveform) -> None:
        """Add a binary to the floor component of the galactic background."""
        sparse_addition_helper(wavelet_waveform, self.galactic_floor)

    def add_faint(self, wavelet_waveform: SparseWaveletWaveform) -> None:
        """Add a binary to the faint component of the galactic background."""
        sparse_addition_helper(wavelet_waveform, self.galactic_below)

    def add_bright(self, wavelet_waveform: SparseWaveletWaveform) -> None:
        """Add a binary to the bright component of the galactic background."""
        sparse_addition_helper(wavelet_waveform, self.galactic_above)


def _check_correct_component_shape(nc: int, wc: WDMWaveletConstants, galactic_component: NDArray[np.floating], *, shape_mode: int = 0) -> NDArray[np.floating]:
    """Check that the galactic component has the correct shape, and reshape if needed."""
    assert galactic_component.size == wc.Nt * wc.Nf * nc, 'Incorrectly sized galaxy component'

    shape1 = (wc.Nt * wc.Nf, nc)
    shape2 = (wc.Nt, wc.Nf, nc)
    shape3 = (wc.Nt * wc.Nf * nc,)
    shapes_allowed = (shape1, shape2, shape3)

    if shape_mode >= len(shapes_allowed):
        msg = f'Invalid shape mode {shape_mode}'
        raise ValueError(msg)

    shape_got = galactic_component.shape
    assert shape_got in shapes_allowed, 'Unrecognized shape for galactic background component'

    if shape_got != shapes_allowed[shape_mode]:
        galactic_component = galactic_component.reshape(shapes_allowed[shape_mode])

    return galactic_component
