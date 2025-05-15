"""object to manage noise models for the iterative_fit_manager"""

import numpy as np

from GalacticStochastic.testing_tools import unit_normal_battery
from LisaWaveformTools.instrument_noise import \
    DiagonalNonstationaryDenseInstrumentNoiseModel


class NoiseModelManager():
    """object to manage the noise models used in the iterative fit"""
    def __init__(self, ic, wc, bgd, SAET_m):
        self.ic = ic
        self.wc = wc
        self.bgd = bgd
        self.SAET_m = SAET_m

        self.idx_SAET_save = np.hstack([np.arange(0, min(10, ic.max_iterations)), np.arange(min(10, ic.max_iterations), 4), ic.max_iterations-1])
        self.itr_save = 0

        self.SAET_tots_upper = np.zeros((self.idx_SAET_save.size, wc.Nt, wc.Nf, 3))
        self.SAET_tots_lower = np.zeros((self.idx_SAET_save.size, wc.Nt, wc.Nf, 3))
        self.SAET_fin = np.zeros((wc.Nt, wc.Nf, 3))

        SAET_tot_upper = np.zeros((wc.Nt, wc.Nf, wc.NC))
        SAET_tot_upper[:] = self.SAET_m

        SAET_tot_lower = np.zeros((wc.Nt, wc.Nf, wc.NC))
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

    def iteration_cleanup(self, itrn):
        """anything to do after an iteration of the iterative fit procedure"""
        if self.itr_save < self.idx_SAET_save.size and itrn == self.idx_SAET_save[self.itr_save]:
            self.SAET_tots_upper[self.itr_save] = self.noise_upper.SAET[:, :, :]
            self.SAET_tots_lower[self.itr_save] = self.noise_lower.SAET[:, :, :]
            self.itr_save += 1

    def loop_cleanup(self):
        """anything to do after the loop finishes running"""
        self.SAET_fin[:] = self.noise_upper.SAET[:, :, :]

    def print_report(self, nt_min, nt_max):
        """anything to print as a summary report after the loop"""
        res_mask = ((self.noise_upper.SAET[:, :, 0]-self.SAET_m[:, 0]).mean(axis=0) > 0.1*self.SAET_m[:, 0]) & (self.SAET_m[:,0] > 0.)
        galactic_below_high = self.bgd.get_galactic_below_high()
        noise_divide = np.sqrt(self.noise_upper.SAET[nt_min:nt_max, res_mask, :2] - self.SAET_m[res_mask, :2])
        points_res = galactic_below_high.reshape(self.wc.Nt, self.wc.Nf, self.wc.NC)[nt_min:nt_max, res_mask, :2] / noise_divide
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

    def advance_state(self, noise_safe_upper, noise_safe_lower, itrn, stat_only):
        if not stat_only:
            period_list = self.ic.period_list
        else:
            period_list = ()

        if not noise_safe_upper:
            #assert not noise_safe_lower

            # don't use cyclostationary model until specified iteration
            if itrn < self.ic.n_cyclo_switch:
                filter_periods = False
            else:
                filter_periods = not stat_only


            # use higher estimate of galactic bg
            SAET_tot_upper = self.bgd.get_S_below_high(self.SAET_m, self.wc, self.ic.smooth_lengthf[itrn], filter_periods, period_list)
            self.noise_upper = DiagonalNonstationaryDenseInstrumentNoiseModel(SAET_tot_upper, self.wc, prune=True)

            SAET_tot_upper = None

        if not noise_safe_lower:
            # make sure this will always predict >= snrs to the actual spectrum in use
            # use lower estimate of galactic bg
            filter_periods = not stat_only
            SAET_tot_lower = self.bgd.get_S_below_low(self.SAET_m, self.wc, self.ic.smooth_lengthf_fix, filter_periods, period_list)
            SAET_tot_lower = np.min([SAET_tot_lower, self.noise_upper.SAET], axis=0)
            self.noise_lower = DiagonalNonstationaryDenseInstrumentNoiseModel(SAET_tot_lower, self.wc, prune=True)
            SAET_tot_lower = None
