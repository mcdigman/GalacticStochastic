"""helper functions for the iterative fit loops"""

from __future__ import annotations

from typing import TYPE_CHECKING

import h5py
import numpy as np

from GalacticStochastic.galactic_fit_helpers import get_S_cyclo
from WaveletWaveforms.sparse_waveform_functions import SparseWaveletWaveform, sparse_addition_helper

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from WaveletWaveforms.wdm_config import WDMWaveletConstants


class BGDecomposition:
    """class to handle the internal decomposition of the galactic background"""

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

        if storage_mode != 0:
            msg = 'Unrecognized option for storage mode'
            raise ValueError(msg)

        self.wc: WDMWaveletConstants = wc
        self.storage_mode: int = storage_mode
        self.nc_galaxy: int = nc_galaxy
        self.shape1: tuple[int, int] = (wc.Nt * wc.Nf, self.nc_galaxy)
        self.shape2: tuple[int, int, int] = (wc.Nt, wc.Nf, self.nc_galaxy)

        if galactic_floor is None:
            self.galactic_floor: NDArray[np.floating] = np.zeros(self.shape1, dtype=np.float64)
        else:
            assert galactic_floor.shape == self.shape1
            self.galactic_floor = galactic_floor

        if galactic_below is None:
            self.galactic_below: NDArray[np.floating] = np.zeros(self.shape1)
        else:
            assert galactic_below.shape == self.shape1
            self.galactic_below = galactic_below

        if galactic_undecided is None:
            self.galactic_undecided: NDArray[np.floating] = np.zeros(self.shape1, dtype=np.float64)
        else:
            assert galactic_undecided.shape == self.shape1
            self.galactic_undecided = galactic_undecided

        if galactic_above is None:
            self.galactic_above: NDArray[np.floating] = np.zeros(self.shape1, dtype=np.float64)
        else:
            assert galactic_above.shape == self.shape1
            self.galactic_above = galactic_above

        self.galactic_total_cache:  NDArray[np.floating] | None = None

        self.track_mode: int = track_mode

        self.power_galactic_undecided: list[NDArray[np.floating]] = []
        self.power_galactic_above: list[NDArray[np.floating]] = []
        self.power_galactic_below_low: list[NDArray[np.floating]] = []
        self.power_galactic_below_high: list[NDArray[np.floating]] = []
        self.power_galactic_total: list[NDArray[np.floating]] = []

    def store_hdf5(self, hf_in: h5py.Group, *, group_name: str = 'background') -> h5py.Group:
        """Store the background to an hdf5 file"""
        hf_background = hf_in.create_group(group_name)
        hf_background.attrs['creator_name'] = self.__class__.__name__
        hf_background.attrs['storage_mode'] = self.storage_mode
        hf_background.attrs['track_mode'] = self.track_mode
        hf_background.attrs['nc_galaxy'] = self.nc_galaxy
        hf_background.attrs['shape1'] = self.shape1
        hf_background.attrs['shape2'] = self.shape2
        return hf_background

    def get_galactic_total(self, *, bypass_check: bool = False) -> NDArray[np.floating]:
        """Get the sum of the entire galactic signal, including detectable binaries"""
        if not bypass_check:
            self.state_check()
        return self.get_galactic_below_high(bypass_check=True) + self.galactic_above

    def get_galactic_below_high(self, *, bypass_check: bool = False) -> NDArray[np.floating]:
        """Get the upper estimate of the unresolvable signal from the galactic background,
        assuming that the undecided part of the signal *is* part of the unresolvable background
        """
        if not bypass_check:
            self.state_check()
        return self.get_galactic_below_low(bypass_check=True) + self.galactic_undecided

    def get_galactic_below_low(self, *, bypass_check: bool = False) -> NDArray[np.floating]:
        """Get the lower estimate of the unresolvable signal from the galactic background,
        assuming that the undecided part of the signal *is not* part of the unresolvable background
        """
        if not bypass_check:
            self.state_check()
        return self.galactic_floor + self.galactic_below

    def get_galactic_coadd_resolvable(self, *, bypass_check: bool = False) -> NDArray[np.floating]:
        """Get the coadded signal from only bright/resolvable galactic binaries"""
        if not bypass_check:
            self.state_check()
        return self.galactic_above

    def get_galactic_coadd_undecided(self, *, bypass_check: bool = False) -> NDArray[np.floating]:
        """Get the coadded signal from galactic binaries whose status as bright or faint has not yet been decided"""
        if not bypass_check:
            self.state_check()
        return self.galactic_undecided

    def get_galactic_coadd_floor(self, *, bypass_check: bool = False) -> NDArray[np.floating]:
        """Get the coadded signal from the faintest set of galactic binaries"""
        if not bypass_check:
            self.state_check()
        return self.galactic_floor

    def state_check(self) -> None:
        """If we have previously cached the total recorded galactic signal,
        check that the total not changed much.
        Otherwise, cache the current total so future runs can check if it has changed
        """
        if self.track_mode:
            if self.galactic_total_cache is None:
                assert np.all(self.galactic_below == 0.0)
                self.galactic_total_cache = self.get_galactic_total(bypass_check=True)
            else:
                # check all contributions to the total signal are tracked accurately
                assert np.allclose(
                    self.galactic_total_cache, self.get_galactic_total(bypass_check=True), atol=1.0e-300, rtol=1.0e-6,
                )

    def log_state(self, S_mean: NDArray[np.floating]) -> None:
        """Record any diagnostics we want to track about this iteration"""
        power_undecided = np.asarray(np.sum(
            np.sum((self.galactic_undecided**2).reshape(self.shape2)[:, 1:, :], axis=0) / S_mean[1:, :], axis=0,
        ), dtype=np.float64)
        power_above = np.asarray(np.sum(
            np.sum((self.galactic_above**2).reshape(self.shape2)[:, 1:, :], axis=0) / S_mean[1:, :], axis=0,
        ), dtype=np.float64)

        power_total = np.asarray(np.sum(
            np.sum((self.get_galactic_total(bypass_check=True) ** 2).reshape(self.shape2)[:, 1:, :], axis=0)
            / S_mean[1:, :],
            axis=0,
        ), dtype=np.float64)
        power_below_high = np.asarray(np.sum(
            np.sum((self.get_galactic_below_high(bypass_check=True) ** 2).reshape(self.shape2)[:, 1:, :], axis=0)
            / S_mean[1:, :],
            axis=0,
        ), dtype=np.float64)
        power_below_low = np.asarray(np.sum(
            np.sum((self.get_galactic_below_low(bypass_check=True) ** 2).reshape(self.shape2)[:, 1:, :], axis=0)
            / S_mean[1:, :],
            axis=0,
        ), dtype=np.float64)

        self.power_galactic_undecided.append(power_undecided)
        self.power_galactic_above.append(power_above)

        self.power_galactic_total.append(power_total)
        self.power_galactic_below_high.append(power_below_high)
        self.power_galactic_below_low.append(power_below_low)

    def clear_undecided(self) -> None:
        """Clear the undecided part of the galactic spectrum"""
        self.galactic_undecided[:] = 0.0

    def clear_above(self) -> None:
        """Clear the bright part of the galactic spectrum"""
        self.galactic_above[:] = 0.0

    def clear_below(self) -> None:
        """Clear the faint part of the galactic spectrum"""
        self.galactic_below[:] = 0.0

    def get_S_below_high(self, S_mean: NDArray[np.floating], smooth_lengthf: float, filter_periods: int, period_list: tuple[int, ...] | tuple[np.floating, ...]) -> NDArray[np.floating]:
        """Get the upper estimate of the galactic power spectrum"""
        galactic_loc = self.get_galactic_below_high(bypass_check=True)
        S, _, _, _, _ = get_S_cyclo(
            galactic_loc, S_mean, self.wc, smooth_lengthf, filter_periods, period_list=period_list,
        )
        return S

    def get_S_below_low(self, S_mean: NDArray[np.floating], smooth_lengthf: float, filter_periods: int, period_list: tuple[int, ...] | tuple[np.floating, ...]) -> NDArray[np.floating]:
        """Get the lower estimate of the galactic power spectrum"""
        galactic_loc = self.get_galactic_below_low(bypass_check=True)
        S, _, _, _, _ = get_S_cyclo(
            galactic_loc, S_mean, self.wc, smooth_lengthf, filter_periods, period_list=period_list,
        )
        return S

    def add_undecided(self, wavelet_waveform: SparseWaveletWaveform) -> None:
        """Add a binary to the undecided component of the galactic background"""
        sparse_addition_helper(wavelet_waveform, self.galactic_undecided)

    def add_floor(self, wavelet_waveform: SparseWaveletWaveform) -> None:
        """Add a binary to the floor component of the galactic background"""
        sparse_addition_helper(wavelet_waveform, self.galactic_floor)

    def add_faint(self, wavelet_waveform: SparseWaveletWaveform) -> None:
        """Add a binary to the faint component of the galactic background"""
        sparse_addition_helper(wavelet_waveform, self.galactic_below)

    def add_bright(self, wavelet_waveform: SparseWaveletWaveform) -> None:
        """Add a binary to the bright component of the galactic background"""
        sparse_addition_helper(wavelet_waveform, self.galactic_above)


def _check_correct_component_shape(nc: int, wc: WDMWaveletConstants, galactic_component: NDArray[np.floating], *, shape_mode: int = 0) -> NDArray[np.floating]:
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


def load_bgd_from_hdf5(wc: WDMWaveletConstants, hf_signal: h5py.Group) -> BGDecomposition:
    nc_galaxy_attr = hf_signal.attrs['nc_galaxy']
    if isinstance(nc_galaxy_attr, (int, np.integer, str)):
        nc = int(nc_galaxy_attr)
    else:
        msg = f'Unexpected type for nc_galaxy: {type(nc_galaxy_attr)}'
        raise TypeError(msg)
    galactic_below = np.asarray(hf_signal['galactic_below'])
    assert len(galactic_below.shape) > 1
    galactic_below = _check_correct_component_shape(nc, wc, galactic_below)

    try:
        galactic_undecided = np.asarray(hf_signal['galactic_undecided'])
        galactic_undecided = _check_correct_component_shape(nc, wc, galactic_undecided)
    except KeyError:
        print('No galactic undecided component to read from file.')
        galactic_undecided = np.zeros_like(galactic_below)
    try:
        galactic_above = np.asarray(hf_signal['galactic_above'])
        galactic_above = _check_correct_component_shape(nc, wc, galactic_above)
    except KeyError:
        print('No galactic above component to read from file.')
        galactic_above = np.zeros_like(galactic_undecided)
    try:
        galactic_floor = np.asarray(hf_signal['galactic_floor'])
        galactic_floor = _check_correct_component_shape(nc, wc, galactic_floor)
    except KeyError:
        print('No galactic floor component to read from file.')
        galactic_floor = np.zeros_like(galactic_undecided)

    return BGDecomposition(wc, nc, galactic_floor=galactic_floor, galactic_below=galactic_below, galactic_undecided=galactic_undecided, galactic_above=galactic_above, track_mode=0)
