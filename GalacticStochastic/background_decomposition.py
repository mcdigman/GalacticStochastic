"""helper functions for the iterative fit loops"""

from __future__ import annotations

from typing import TYPE_CHECKING

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
    ) -> None:

        self.wc: WDMWaveletConstants = wc
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
