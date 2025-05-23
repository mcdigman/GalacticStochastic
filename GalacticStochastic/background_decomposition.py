"""helper functions for the iterative fit loops"""

import numpy as np
from numpy.typing import NDArray

from GalacticStochastic.galactic_fit_helpers import get_S_cyclo
from WaveletWaveforms.coefficientsWDM_time_helpers import SparseTaylorWaveform, sparse_addition_helper
from WaveletWaveforms.wdm_config import WDMWaveletConstants


class BGDecomposition:
    """class to handle the internal decomposition of the galactic background"""

    def __init__(
        self,
        wc: WDMWaveletConstants,
        nc_galaxy,
        *,
        galactic_floor=None,
        galactic_below=None,
        galactic_undecided=None,
        galactic_above=None,
        do_total_track=True,
    ) -> None:
        self.wc = wc
        self.nc_galaxy = nc_galaxy
        self.shape1 = (wc.Nt * wc.Nf, self.nc_galaxy)
        self.shape2 = (wc.Nt, wc.Nf, self.nc_galaxy)

        if galactic_floor is None:
            self.galactic_floor = np.zeros(self.shape1)
        else:
            self.galactic_floor = galactic_floor

        if galactic_below is None:
            self.galactic_below = np.zeros(self.shape1)
        else:
            self.galactic_below = galactic_below

        if galactic_undecided is None:
            self.galactic_undecided = np.zeros(self.shape1)
        else:
            self.galactic_undecided = galactic_undecided

        if galactic_above is None:
            self.galactic_above = np.zeros(self.shape1)
        else:
            self.galactic_above = galactic_above

        self.galactic_total_cache = None

        self.do_total_track = do_total_track

        self.power_galactic_undecided = []
        self.power_galactic_above = []
        self.power_galactic_below_low = []
        self.power_galactic_below_high = []
        self.power_galactic_total = []

    def get_galactic_total(self, *, bypass_check=False) -> NDArray[float]:
        """Get the sum of the entire galactic signal, including detectable binaries"""
        if not bypass_check:
            self.state_check()
        return self.get_galactic_below_high(bypass_check=True) + self.galactic_above

    def get_galactic_below_high(self, *, bypass_check=False) -> NDArray[float]:
        """Get the upper estimate of the unresolvable signal from the galactic background,
        assuming that the undecided part of the signal *is* part of the unresolvable background
        """
        if not bypass_check:
            self.state_check()
        return self.get_galactic_below_low(bypass_check=True) + self.galactic_undecided

    def get_galactic_below_low(self, *, bypass_check=False) -> NDArray[float]:
        """Get the lower estimate of the unresolvable signal from the galactic background,
        assuming that the undecided part of the signal *is not* part of the unresolvable background
        """
        if not bypass_check:
            self.state_check()
        return self.galactic_floor + self.galactic_below

    def get_galactic_coadd_resolvable(self, *, bypass_check=False) -> NDArray[float]:
        """Get the coadded signal from only bright/resolvable galactic binaries"""
        if not bypass_check:
            self.state_check()
        return self.galactic_above

    def get_galactic_coadd_undecided(self, *, bypass_check=False) -> NDArray[float]:
        """Get the coadded signal from galactic binaries whose status as bright or faint has not yet been decided"""
        if not bypass_check:
            self.state_check()
        return self.galactic_undecided

    def get_galactic_coadd_floor(self, *, bypass_check=False) -> NDArray[float]:
        """Get the coadded signal from the faintest set of galactic binaries"""
        if not bypass_check:
            self.state_check()
        return self.galactic_floor

    def state_check(self) -> None:
        """If we have previously cached the total recorded galactic signal,
        check that the total not changed much.
        Otherwise, cache the current total so future runs can check if it has changed
        """
        if self.do_total_track:
            if self.galactic_total_cache is None:
                assert np.all(self.galactic_below == 0.0)
                self.galactic_total_cache = self.get_galactic_total(bypass_check=True)
            else:
                # check all contributions to the total signal are tracked accurately
                assert np.allclose(
                    self.galactic_total_cache, self.get_galactic_total(bypass_check=True), atol=1.0e-300, rtol=1.0e-6
                )

    def log_state(self, S_mean: NDArray[float]) -> None:
        """Record any diagnostics we want to track about this iteration"""
        power_undecided = np.sum(
            np.sum((self.galactic_undecided**2).reshape(self.shape2)[:, 1:, :], axis=0) / S_mean[1:, :], axis=0
        )
        power_above = np.sum(
            np.sum((self.galactic_above**2).reshape(self.shape2)[:, 1:, :], axis=0) / S_mean[1:, :], axis=0
        )

        power_total = np.sum(
            np.sum((self.get_galactic_total(bypass_check=True) ** 2).reshape(self.shape2)[:, 1:, :], axis=0)
            / S_mean[1:, :],
            axis=0,
        )
        power_below_high = np.sum(
            np.sum((self.get_galactic_below_high(bypass_check=True) ** 2).reshape(self.shape2)[:, 1:, :], axis=0)
            / S_mean[1:, :],
            axis=0,
        )
        power_below_low = np.sum(
            np.sum((self.get_galactic_below_low(bypass_check=True) ** 2).reshape(self.shape2)[:, 1:, :], axis=0)
            / S_mean[1:, :],
            axis=0,
        )

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

    def get_S_below_high(self, S_mean, smooth_lengthf, filter_periods, period_list) -> NDArray[float]:
        """Get the upper estimate of the galactic power spectrum"""
        galactic_loc = self.get_galactic_below_high(bypass_check=True)
        S, _, _, _, _ = get_S_cyclo(
            galactic_loc, S_mean, self.wc, smooth_lengthf, filter_periods, period_list=period_list
        )
        return S

    def get_S_below_low(self, S_mean, smooth_lengthf, filter_periods, period_list) -> NDArray[float]:
        """Get the lower estimate of the galactic power spectrum"""
        galactic_loc = self.get_galactic_below_low(bypass_check=True)
        S, _, _, _, _ = get_S_cyclo(
            galactic_loc, S_mean, self.wc, smooth_lengthf, filter_periods, period_list=period_list
        )
        return S

    def add_undecided(self, wavelet_waveform: SparseTaylorWaveform) -> None:
        """Add a binary to the undecided component of the galactic background"""
        sparse_addition_helper(wavelet_waveform, self.galactic_undecided)

    def add_floor(self, wavelet_waveform: SparseTaylorWaveform) -> None:
        """Add a binary to the floor component of the galactic background"""
        sparse_addition_helper(wavelet_waveform, self.galactic_floor)

    def add_faint(self, wavelet_waveform: SparseTaylorWaveform) -> None:
        """Add a binary to the faint component of the galactic background"""
        sparse_addition_helper(wavelet_waveform, self.galactic_below)

    def add_bright(self, wavelet_waveform: SparseTaylorWaveform) -> None:
        """Add a binary to the bright component of the galactic background"""
        sparse_addition_helper(wavelet_waveform, self.galactic_above)
