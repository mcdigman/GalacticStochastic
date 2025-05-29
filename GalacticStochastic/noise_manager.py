"""object to manage noise models for the iterative_fit_manager"""

from typing import override

import numpy as np
from numpy.typing import NDArray

from GalacticStochastic.background_decomposition import BGDecomposition
from GalacticStochastic.iteration_config import IterationConfig
from GalacticStochastic.iterative_fit_state_machine import IterativeFitState
from GalacticStochastic.state_manager import StateManager
from GalacticStochastic.testing_tools import unit_normal_battery
from LisaWaveformTools.lisa_config import LISAConstants
from LisaWaveformTools.noise_model import DiagonalNonstationaryDenseNoiseModel
from WaveletWaveforms.sparse_waveform_functions import PixelTimeRange
from WaveletWaveforms.wdm_config import WDMWaveletConstants


class NoiseModelManager(StateManager):
    """object to manage the noise models used in the iterative fit"""

    def __init__(
        self,
        ic: IterationConfig,
        wc: WDMWaveletConstants,
        lc: LISAConstants,
        fit_state: IterativeFitState,
        bgd: BGDecomposition,
        S_inst_m: NDArray[np.float64],
        stat_only,
        nt_lim_snr: PixelTimeRange
    ) -> None:
        """Create the noise model manager"""
        self.ic = ic
        self.lc = lc
        self.wc = wc
        self.bgd = bgd
        self.fit_state = fit_state
        self.S_inst_m = S_inst_m
        self.stat_only = stat_only
        self.nt_lim_snr = nt_lim_snr

        self.itrn = 0

        self.idx_S_save = np.hstack(
            [
                np.arange(0, min(10, self.fit_state.get_n_itr_cut())),
                np.arange(min(10, self.fit_state.get_n_itr_cut()), 4),
                self.fit_state.get_n_itr_cut() - 1,
            ]
        )
        self.itr_save = 0

        self.S_record_upper = np.zeros((self.idx_S_save.size, wc.Nt, wc.Nf, self.bgd.nc_galaxy))
        self.S_record_lower = np.zeros((self.idx_S_save.size, wc.Nt, wc.Nf, self.bgd.nc_galaxy))
        self.S_final = np.zeros((wc.Nt, wc.Nf, self.bgd.nc_galaxy))

        S_upper: NDArray[float] = np.zeros((wc.Nt, wc.Nf, self.bgd.nc_galaxy))
        S_upper[:] = self.S_inst_m

        S_lower: NDArray[float] = np.zeros((wc.Nt, wc.Nf, self.bgd.nc_galaxy))
        S_lower[:] = self.S_inst_m
        if self.idx_S_save[self.itr_save] == 0:
            self.S_record_upper[0] = S_upper[:, :, :]
            self.S_record_lower[0] = S_lower[:, :, :]
            self.itr_save += 1
        S_lower = np.asarray(np.min([S_lower, S_upper], axis=0), dtype=np.float64)

        self.noise_upper = DiagonalNonstationaryDenseNoiseModel(S_upper, wc, prune=True, nc_snr=lc.nc_snr)
        self.noise_lower = DiagonalNonstationaryDenseNoiseModel(S_lower, wc, prune=True, nc_snr=lc.nc_snr)

        del S_upper
        del S_lower

    @override
    def log_state(self) -> None:
        """Perform any internal logging that should be done after advance_state is run."""
        if self.itr_save < self.idx_S_save.size and self.itrn - 1 == self.idx_S_save[self.itr_save]:
            self.S_record_upper[self.itr_save] = self.noise_upper.S[:, :, :]
            self.S_record_lower[self.itr_save] = self.noise_lower.S[:, :, :]
            self.itr_save += 1
        self.bgd.log_state(self.S_inst_m)

    @override
    def loop_finalize(self) -> None:
        """Perform any logic desired after convergence has been achieved and the loop ends"""
        self.S_final[:] = self.noise_upper.S[:, :, :]

    @override
    def state_check(self) -> None:
        """Perform any sanity checks that should be performed at the end of each iteration"""
        self.bgd.state_check()

    @override
    def print_report(self) -> None:
        """Do any printing desired after convergence has been achieved and the loop ends"""
        res_mask = np.asarray(((self.noise_upper.S[:, :, 0] - self.S_inst_m[:, 0]).mean(axis=0) > 0.1 * self.S_inst_m[:, 0]) & (
            self.S_inst_m[:, 0] > 0.0
        ), dtype=np.bool_)
        galactic_below_high = self.bgd.get_galactic_below_high()
        noise_divide = np.sqrt(
            self.noise_upper.S[self.nt_lim_snr.nt_min:self.nt_lim_snr.nt_max, res_mask, :2] - self.S_inst_m[res_mask, :2]
        )
        points_res = (
            galactic_below_high.reshape(self.wc.Nt, self.wc.Nf, self.bgd.nc_galaxy)[
                self.nt_lim_snr.nt_min:self.nt_lim_snr.nt_max, res_mask, :2
            ]
            / noise_divide
        )
        n_points = points_res.size

        del noise_divide
        del galactic_below_high
        del res_mask
        unit_normal_res, a2score, mean_rat, std_rat = unit_normal_battery(
            points_res.flatten(), A2_cut=2.28, sig_thresh=5.0, do_assert=False
        )
        del points_res
        if unit_normal_res:
            print(
                'Background PASSES normality: points=%12d A2=%3.5f, mean ratio=%3.5f, std ratio=%3.5f'
                % (n_points, a2score, mean_rat, std_rat)
            )
        else:
            print(
                'Background FAILS  normality: points=%12d A2=%3.5f, mean ratio=%3.5f, std ratio=%3.5f'
                % (n_points, a2score, mean_rat, std_rat)
            )

    @override
    def advance_state(self) -> None:
        """Handle any logic necessary to advance the state of the object to the next iteration"""
        noise_safe_upper = self.fit_state.get_noise_safe_upper()
        noise_safe_lower = self.fit_state.get_noise_safe_lower()

        if not self.stat_only:
            period_list = self.ic.period_list
        else:
            period_list = ()

        if not noise_safe_upper:
            # assert not noise_safe_lower

            # don't use cyclostationary model until specified iteration
            if self.itrn < self.ic.n_cyclo_switch:
                filter_periods = False
            else:
                filter_periods = not self.stat_only

            # use higher estimate of galactic bg
            S_upper = self.bgd.get_S_below_high(
                self.S_inst_m, self.ic.smooth_lengthf[self.itrn], filter_periods, period_list
            )
            self.noise_upper = DiagonalNonstationaryDenseNoiseModel(S_upper, self.wc, prune=True, nc_snr=self.lc.nc_snr)

            del S_upper

        if not noise_safe_lower:
            # make sure this will always predict >= snrs to the actual spectrum in use
            # use lower estimate of galactic bg
            filter_periods = not self.stat_only
            S_lower = self.bgd.get_S_below_low(self.S_inst_m, self.ic.smooth_lengthf_fix, filter_periods, period_list)
            S_lower = np.asarray(np.min([S_lower, self.noise_upper.S], axis=0), dtype=np.float64)
            self.noise_lower = DiagonalNonstationaryDenseNoiseModel(S_lower, self.wc, prune=True, nc_snr=self.lc.nc_snr)
            del S_lower
        self.itrn += 1
