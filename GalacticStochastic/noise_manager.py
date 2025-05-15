"""object to manage noise models for the iterative_fit_manager"""

import numpy as np

from GalacticStochastic.state_manager import StateManager
from GalacticStochastic.testing_tools import unit_normal_battery
from LisaWaveformTools.instrument_noise import \
    DiagonalNonstationaryDenseInstrumentNoiseModel


class NoiseModelManager(StateManager):
    """object to manage the noise models used in the iterative fit"""
    def __init__(self, ic, wc, fit_state, bgd, SAET_m, stat_only, nt_min, nt_max):
        """create the noise model manager"""
        self.ic = ic
        self.wc = wc
        self.bgd = bgd
        self.fit_state = fit_state
        self.SAET_m = SAET_m
        self.stat_only = stat_only
        self.nt_min = nt_min
        self.nt_max = nt_max


        self.itrn = 0

        self.idx_SAET_save = np.hstack([np.arange(0, min(10, ic.max_iterations)), np.arange(min(10, ic.max_iterations), 4), ic.max_iterations-1])
        self.itr_save = 0

        self.SAET_tots_upper = np.zeros((self.idx_SAET_save.size, wc.Nt, wc.Nf, 3))
        self.SAET_tots_lower = np.zeros((self.idx_SAET_save.size, wc.Nt, wc.Nf, 3))
        self.SAET_fin = np.zeros((wc.Nt, wc.Nf, 3))

        SAET_tot_upper = np.zeros((wc.Nt, wc.Nf, self.bgd.NC_gal))
        SAET_tot_upper[:] = self.SAET_m

        SAET_tot_lower = np.zeros((wc.Nt, wc.Nf, self.bgd.NC_gal))
        SAET_tot_lower[:] = self.SAET_m
        if self.idx_SAET_save[self.itr_save] == 0:
            self.SAET_tots_upper[0] = SAET_tot_upper[:, :, :]
            self.SAET_tots_lower[0] = SAET_tot_lower[:, :, :]
            self.itr_save += 1
        SAET_tot_lower = np.min([SAET_tot_lower, SAET_tot_upper], axis=0)

        self.noise_upper = DiagonalNonstationaryDenseInstrumentNoiseModel(SAET_tot_upper, wc, prune=True)
        self.noise_lower = DiagonalNonstationaryDenseInstrumentNoiseModel(SAET_tot_lower, wc, prune=True)

        SAET_tot_upper = None
        SAET_tot_lower = None

    def log_state(self):
        """Perform any internal logging that should be done after advance_state is run for all objects for the iteration"""
        if self.itr_save < self.idx_SAET_save.size and self.itrn - 1 == self.idx_SAET_save[self.itr_save]:
            self.SAET_tots_upper[self.itr_save] = self.noise_upper.SAET[:, :, :]
            self.SAET_tots_lower[self.itr_save] = self.noise_lower.SAET[:, :, :]
            self.itr_save += 1
        self.bgd.log_state(self.SAET_m)

    def loop_finalize(self):
        """Perform any logic desired after convergence has been achieved and the loop ends"""
        self.SAET_fin[:] = self.noise_upper.SAET[:, :, :]

    def state_check(self):
        """Perform any sanity checks that should be performed at the end of each iteration"""
        self.bgd.state_check()
        return

    def print_report(self):
        """Do any printing desired after convergence has been achieved and the loop ends"""
        res_mask = ((self.noise_upper.SAET[:, :, 0]-self.SAET_m[:, 0]).mean(axis=0) > 0.1*self.SAET_m[:, 0]) & (self.SAET_m[:,0] > 0.)
        galactic_below_high = self.bgd.get_galactic_below_high()
        noise_divide = np.sqrt(self.noise_upper.SAET[self.nt_min:self.nt_max, res_mask, :2] - self.SAET_m[res_mask, :2])
        points_res = galactic_below_high.reshape(self.wc.Nt, self.wc.Nf, self.bgd.NC_gal)[self.nt_min:self.nt_max, res_mask, :2] / noise_divide
        n_points = points_res.size
        noise_divide = None
        galactic_below_high = None
        res_mask = None
        unit_normal_res, a2score, mean_rat, std_rat = unit_normal_battery(points_res.flatten(), A2_cut=2.28, sig_thresh=5., do_assert=False)
        points_res = None
        if unit_normal_res:
            print('Background PASSES normality: points=%12d A2=%3.5f, mean ratio=%3.5f, std ratio=%3.5f' % (n_points, a2score, mean_rat, std_rat))
        else:
            print('Background FAILS  normality: points=%12d A2=%3.5f, mean ratio=%3.5f, std ratio=%3.5f' % (n_points, a2score, mean_rat, std_rat))

    def advance_state(self):
        """Handle any logic necessary to advance the state of the object to the next iteration"""
        noise_safe_upper = self.fit_state.get_noise_safe_upper()
        noise_safe_lower = self.fit_state.get_noise_safe_lower()

        if not self.stat_only:
            period_list = self.ic.period_list
        else:
            period_list = ()

        if not noise_safe_upper:
            #assert not noise_safe_lower

            # don't use cyclostationary model until specified iteration
            if self.itrn < self.ic.n_cyclo_switch:
                filter_periods = False
            else:
                filter_periods = not self.stat_only


            # use higher estimate of galactic bg
            SAET_tot_upper = self.bgd.get_S_below_high(self.SAET_m, self.ic.smooth_lengthf[self.itrn], filter_periods, period_list)
            self.noise_upper = DiagonalNonstationaryDenseInstrumentNoiseModel(SAET_tot_upper, self.wc, prune=True)

            SAET_tot_upper = None

        if not noise_safe_lower:
            # make sure this will always predict >= snrs to the actual spectrum in use
            # use lower estimate of galactic bg
            filter_periods = not self.stat_only
            SAET_tot_lower = self.bgd.get_S_below_low(self.SAET_m, self.ic.smooth_lengthf_fix, filter_periods, period_list)
            SAET_tot_lower = np.min([SAET_tot_lower, self.noise_upper.SAET], axis=0)
            self.noise_lower = DiagonalNonstationaryDenseInstrumentNoiseModel(SAET_tot_lower, self.wc, prune=True)
            SAET_tot_lower = None
        self.itrn += 1
