"""object to manage noise models for the iterative_fit_manager"""

from typing import override

import h5py
import numpy as np
from numpy.typing import NDArray

from GalacticStochastic.background_decomposition import BGDecomposition
from GalacticStochastic.iteration_config import IterationConfig
from GalacticStochastic.iterative_fit_state_machine import IterativeFitState
from GalacticStochastic.state_manager import StateManager
from GalacticStochastic.testing_tools import unit_normal_battery
from LisaWaveformTools.lisa_config import LISAConstants
from LisaWaveformTools.noise_model import DiagonalNonstationaryDenseNoiseModel
from WaveletWaveforms.sparse_waveform_functions import PixelGenericRange
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
        S_inst_m: NDArray[np.floating],
        stat_only: int,
        nt_lim_snr: PixelGenericRange,
    ) -> None:
        """Create the noise model manager"""
        if ic.noise_model_storage_mode not in (0, 1, 2):
            msg = 'Unrecognized option for storage mode'
            raise ValueError(msg)

        self.ic: IterationConfig = ic
        self.lc: LISAConstants = lc
        self.wc: WDMWaveletConstants = wc
        self.bgd: BGDecomposition = bgd
        self.fit_state = fit_state
        self.S_inst_m: NDArray[np.floating] = S_inst_m
        self.stat_only: int = stat_only
        self.nt_lim_snr: PixelGenericRange = nt_lim_snr

        self.itrn: int = 0

        self.idx_S_save: NDArray = np.hstack(
            [
                np.arange(0, min(10, self.fit_state.get_n_itr_cut())),
                np.arange(min(10, self.fit_state.get_n_itr_cut()), 4),
                self.fit_state.get_n_itr_cut() - 1,
            ],
        )
        self.itr_save = 0

        self.S_record_upper = np.zeros((self.idx_S_save.size, wc.Nt, wc.Nf, self.bgd.nc_galaxy))
        self.S_record_lower = np.zeros((self.idx_S_save.size, wc.Nt, wc.Nf, self.bgd.nc_galaxy))
        self.S_final = np.zeros((wc.Nt, wc.Nf, self.bgd.nc_galaxy))

        S_upper: NDArray[np.floating] = np.zeros((wc.Nt, wc.Nf, self.bgd.nc_galaxy))
        S_upper[:] = self.S_inst_m

        S_lower: NDArray[np.floating] = np.zeros((wc.Nt, wc.Nf, self.bgd.nc_galaxy))
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

    def store_hdf5(self, hf_in: h5py.Group) -> h5py.Group:
        storage_mode = self.ic.noise_model_storage_mode
        hf_noise = hf_in.create_group('noise_model')
        hf_noise.attrs['creator_name'] = self.__class__.__name__
        hf_noise.attrs['storage_mode'] = storage_mode
        hf_noise.attrs['itrn'] = self.itrn
        hf_noise.attrs['itr_save'] = self.itr_save
        hf_noise.attrs['nt_lim_snr'] = self.nt_lim_snr
        hf_noise.attrs['stat_only'] = self.stat_only
        hf_noise.attrs['noise_upper_name'] = self.noise_upper.__class__.__name__
        hf_noise.attrs['noise_lower_name'] = self.noise_lower.__class__.__name__
        hf_noise.attrs['bgd_name'] = self.bgd.__class__.__name__
        hf_noise.attrs['fit_state_name'] = self.fit_state.__class__.__name__

        hf_noise.create_dataset('S_inst_m', data=self.S_inst_m, compression='gzip')
        hf_noise.create_dataset('idx_S_save', data=self.idx_S_save, compression='gzip')

        # at least store the history of the mean spectrum
        hf_noise.create_dataset('S_record_upper_mean', data=np.mean(self.S_record_upper, axis=1), compression='gzip')
        hf_noise.create_dataset('S_record_lower_mean', data=np.mean(self.S_record_lower, axis=1), compression='gzip')

        if storage_mode in (1, 2):
            # can safely be rederived as long as bgd is stored
            hf_noise.create_dataset('S_record_final', data=self.S_record_lower, compression='gzip')

        if storage_mode == 2:
            # full versions of these potentially take a lot of memory, and aren't always needed so don't necessarily want to write them to disk
            hf_noise.create_dataset('S_record_upper', data=self.S_record_upper, compression='gzip')
            hf_noise.create_dataset('S_record_lower', data=self.S_record_lower, compression='gzip')

        self.bgd.store_hdf5(hf_noise)

        # storing the fit state will probably be done more than once, but it shouldn't be very large
        # and it is possibly more self-descriptive that way
        self.fit_state.store_hdf5(hf_noise)

        return hf_noise

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
            self.noise_upper.S[self.nt_lim_snr.nx_min:self.nt_lim_snr.nx_max, res_mask, :2] - self.S_inst_m[res_mask, :2],
        )
        points_res = (
            galactic_below_high.reshape(self.wc.Nt, self.wc.Nf, self.bgd.nc_galaxy)[
                self.nt_lim_snr.nx_min:self.nt_lim_snr.nx_max, res_mask, :2,
            ]
            / noise_divide
        )
        n_points = points_res.size

        del noise_divide
        del galactic_below_high
        del res_mask
        unit_normal_res, a2score, mean_rat, std_rat = unit_normal_battery(
            points_res.flatten(), A2_cut=2.28, sig_thresh=5.0, do_assert=False,
        )
        del points_res
        if unit_normal_res:
            print(
                'Background PASSES normality: points=%12d A2=%3.5f, mean ratio=%3.5f, std ratio=%3.5f'
                % (n_points, a2score, mean_rat, std_rat),
            )
        else:
            print(
                'Background FAILS  normality: points=%12d A2=%3.5f, mean ratio=%3.5f, std ratio=%3.5f'
                % (n_points, a2score, mean_rat, std_rat),
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
                self.S_inst_m, self.ic.smooth_lengthf[self.itrn], filter_periods, period_list,
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
