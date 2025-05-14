"""helper functions for the iterative fit loops"""

import numpy as np

from GalacticStochastic.galactic_fit_helpers import \
    get_SAET_cyclostationary_mean


class BGDecomposition():
    """class to handle the internal decomposition of the galactic background"""
    def __init__(self, galactic_floor, galactic_below, galactic_undecided, galactic_above):
        self.galactic_floor = galactic_floor
        self.galactic_below = galactic_below
        self.galactic_undecided = galactic_undecided
        self.galactic_above = galactic_above
        self.NC_gal = galactic_below.shape[-1]
        self.galactic_total_cache = None

        self.power_galactic_undecided = []
        self.power_galactic_above = []
        self.power_galactic_below_low = []
        self.power_galactic_below_high = []
        self.power_galactic_total = []

    def get_galactic_total(self, bypass_check=False):
        """get the sum of all components of the galactic signal, detectable or not"""
        if not bypass_check:
            self.total_signal_consistency_check()
        return self.get_galactic_below_high(bypass_check=True) + self.galactic_above

    def get_galactic_below_high(self, bypass_check=False):
        """
        get the upper estimate of the unresolvable signal from the galactic background,
        assuming that the undecided part of the signal *is* part of the unresolvable background
        """
        if not bypass_check:
            self.total_signal_consistency_check()
        return self.get_galactic_below_low(bypass_check=True) + self.galactic_undecided

    def get_galactic_below_low(self, bypass_check=False):
        """
        get the lower estimate of the unresolvable signal from the galactic background,
        assuming that the undecided part of the signal *is not* part of the unresolvable background
        """
        if not bypass_check:
            self.total_signal_consistency_check()
        return self.galactic_floor + self.galactic_below

    def get_galactic_coadd_resolvable(self, bypass_check=False):
        """
        get the coadded signal from only bright/resolvable galactic binaries
        """
        if not bypass_check:
            self.total_signal_consistency_check()
        return self.galactic_above

    def get_galactic_coadd_undecided(self, bypass_check=False):
        """
        get the coadded signal from galactic binaries whose status as bright or faint has not yet been decided
        """
        if not bypass_check:
            self.total_signal_consistency_check()
        return self.galactic_undecided

    def get_galactic_coadd_floor(self, bypass_check=False):
        """
        get the coadded signal from the faintest set of galactic binaries
        """
        if not bypass_check:
            self.total_signal_consistency_check()
        return self.galactic_floor

    def total_signal_consistency_check(self):
        """
        if we have previously cached the total recorded galactic signal,
        check that the total not changed much.
        Otherwise, cache the current total so future runs can check if it has changed
        """
        if self.galactic_total_cache is None:
            assert np.all(self.galactic_below == 0.)
            self.galactic_total_cache = self.get_galactic_total(bypass_check=True)
        else:
            # check all contributions to the total signal are tracked accurately
            assert np.allclose(self.galactic_total_cache, self.get_galactic_total(bypass_check=True), atol=1.e-300, rtol=1.e-6)

    def log_tracking(self, wc, S_mean):
        """
        record diagnostics we want to track about this iteration
        """
        shape = (wc.Nt, wc.Nf, self.NC_gal)

        power_undecided = np.sum(np.sum((self.galactic_undecided**2).reshape(shape)[:, 1:, :], axis=0)/S_mean[1:,:], axis=0)
        power_above = np.sum(np.sum((self.galactic_above**2).reshape(shape)[:, 1:, :], axis=0)/S_mean[1:,:], axis=0)

        power_total = np.sum(np.sum((self.get_galactic_total(bypass_check=True)**2).reshape(shape)[:, 1:, :], axis=0)/S_mean[1:,:], axis=0)
        power_below_high = np.sum(np.sum((self.get_galactic_below_high(bypass_check=True)**2).reshape(shape)[:, 1:, :], axis=0)/S_mean[1:,:], axis=0)
        power_below_low = np.sum(np.sum((self.get_galactic_below_low(bypass_check=True)**2).reshape(shape)[:, 1:, :], axis=0)/S_mean[1:,:], axis=0)

        self.power_galactic_undecided.append(power_undecided)
        self.power_galactic_above.append(power_above)

        self.power_galactic_total.append(power_total)
        self.power_galactic_below_high.append(power_below_high)
        self.power_galactic_below_low.append(power_below_low)


    # TODO the methods below might work better in a child class
    def get_S_below_high(self, S_mean, wc, smooth_lengthf, filter_periods, period_list, bypass_check=False):
        S, _, _, _, _ = get_SAET_cyclostationary_mean(self.get_galactic_below_high(bypass_check=bypass_check), S_mean, wc, smooth_lengthf=smooth_lengthf, filter_periods=filter_periods, period_list=period_list)
        return S

    def get_S_below_low(self, S_mean, wc, smooth_lengthf, filter_periods, period_list, bypass_check=False):
        S, _, _, _, _ = get_SAET_cyclostationary_mean(self.get_galactic_below_low(bypass_check=bypass_check), S_mean, wc, smooth_lengthf=smooth_lengthf, filter_periods=filter_periods, period_list=period_list)
        return S

    def add_undecided(self, listT_temp, NUTs_temp, waveT_temp):
        for itrc in range(self.NC_gal):
            self.galactic_undecided[listT_temp[itrc, :NUTs_temp[itrc]], itrc] += waveT_temp[itrc, :NUTs_temp[itrc]]

    def add_floor(self, listT_temp, NUTs_temp, waveT_temp):
        for itrc in range(self.NC_gal):
            self.galactic_floor[listT_temp[itrc, :NUTs_temp[itrc]], itrc] += waveT_temp[itrc, :NUTs_temp[itrc]]

    def add_faint(self, listT_temp, NUTs_temp, waveT_temp):
        for itrc in range(self.NC_gal):
            self.galactic_below[listT_temp[itrc, :NUTs_temp[itrc]], itrc] += waveT_temp[itrc, :NUTs_temp[itrc]]

    def add_bright(self, listT_temp, NUTs_temp, waveT_temp):
        for itrc in range(self.NC_gal):
            self.galactic_above[listT_temp[itrc, :NUTs_temp[itrc]], itrc] += waveT_temp[itrc, :NUTs_temp[itrc]]
