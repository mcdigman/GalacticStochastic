"""Class to store information about the binaries in the galactic background"""

from time import perf_counter

import numpy as np

import GalacticStochastic.global_const as gc
from GalacticStochastic.iteration_config import IterationConfig
from GalacticStochastic.iterative_fit_state_machine import IterativeFitState
from GalacticStochastic.noise_manager import NoiseModelManager
from GalacticStochastic.state_manager import StateManager
from LisaWaveformTools.lisa_config import LISAConstants
from WaveletWaveforms.wavelet_detector_waveforms import BinaryWaveletAmpFreqDT
from WaveletWaveforms.wdm_config import WDMWaveletConstants


class BinaryInclusionState(StateManager):
    """Stores all the binaries under consideration in the galaxy"""

    def __init__(self, wc: WDMWaveletConstants, ic: IterationConfig, lc: LISAConstants, params_gb_in, noise_manager: NoiseModelManager, fit_state: IterativeFitState, snrs_tot_in=None) -> None:
        """Class that stores information about the binaries in the background, and which component they are assigned to"""
        self.wc = wc
        self.ic = ic
        self.lc = lc
        self.noise_manager = noise_manager
        self.fit_state = fit_state

        self.n_tot = params_gb_in.shape[0]
        self.fmin_binary = max(0., ic.fmin_binary)
        self.fmax_binary = min((wc.Nf - 1) * wc.DF, ic.fmax_binary)

        if snrs_tot_in is not None:
            faints_in = (snrs_tot_in < ic.snr_min_preprocess) | (params_gb_in[:, 3] >= self.fmax_binary) | (params_gb_in[:, 3] < self.fmin_binary)
        else:
            faints_in = (params_gb_in[:, 3] >= self.fmax_binary) | (params_gb_in[:, 3] < self.fmin_binary)

        self.argbinmap = np.argwhere(~faints_in).flatten()
        self.faints_old = faints_in[self.argbinmap]
        assert self.faints_old.sum() == 0.
        self.params_gb = params_gb_in[self.argbinmap]
        self.n_bin_use = self.argbinmap.size

        params_gb_in = None
        faints_in = None

        self.snrs_upper = np.zeros((self.fit_state.get_n_itr_cut(), self.n_bin_use, lc.nc_snr))
        self.snrs_lower = np.zeros((self.fit_state.get_n_itr_cut(), self.n_bin_use, lc.nc_snr))
        self.snrs_tot_lower = np.zeros((self.fit_state.get_n_itr_cut(), self.n_bin_use))
        self.snrs_tot_upper = np.zeros((self.fit_state.get_n_itr_cut(), self.n_bin_use))
        self.brights = np.zeros((self.fit_state.get_n_itr_cut(), self.n_bin_use), dtype=np.bool_)
        self.decided = np.zeros((self.fit_state.get_n_itr_cut(), self.n_bin_use), dtype=np.bool_)
        self.faints_cur = np.zeros((self.fit_state.get_n_itr_cut(), self.n_bin_use), dtype=np.bool_)

        params0 = self.params_gb[0].copy()
        self.waveform_manager = BinaryWaveletAmpFreqDT(params0.copy(), wc, self.lc)

        self.itrn = 0

        self.n_faints_cur = np.zeros(self.fit_state.get_n_itr_cut() + 1, dtype=np.int64)
        self.n_brights_cur = np.zeros(self.fit_state.get_n_itr_cut() + 1, dtype=np.int64)

    def sustain_snr_helper(self) -> None:
        """Helper to carry forward any other snr values we know from a previous iteration"""
        itrn = self.itrn
        if self.fit_state.get_faint_converged():
            assert itrn > 1
            self.snrs_tot_lower[itrn, self.decided[itrn]] = self.snrs_tot_lower[itrn - 1, self.decided[itrn]]
            self.snrs_lower[itrn, self.decided[itrn]] = self.snrs_lower[itrn - 1, self.decided[itrn]]

        if self.fit_state.get_bright_converged():
            assert itrn > 1
            self.snrs_tot_upper[itrn, self.decided[itrn]] = self.snrs_tot_upper[itrn - 1, self.decided[itrn]]
            self.snrs_upper[itrn, self.decided[itrn]] = self.snrs_upper[itrn - 1, self.decided[itrn]]

    def oscillation_check_helper(self) -> (bool, bool, bool):
        """Helper used by fit_state to decide if the bright binaries are oscillating without converging"""
        osc1 = False
        osc2 = False
        osc3 = False

        if self.itrn > 1:
            osc1 = np.all(self.brights[self.itrn - 1] == self.brights[self.itrn - 2])
        if self.itrn > 2:
            osc2 = np.all(self.brights[self.itrn - 1] == self.brights[self.itrn - 3])
        if self.itrn > 3:
            osc3 = np.all(self.brights[self.itrn - 1] == self.brights[self.itrn - 4])

        converged_or_cycling = osc1 or osc2 or osc3
        old_match = osc2 or osc3
        cycling = old_match and not osc1
        return cycling, converged_or_cycling, old_match

    def delta_faint_check_helper(self) -> int:
        """Get the difference in the number of faint binaries between the last two iterations"""
        if self.itrn - 1 == 0:
            delta_faints = self.n_faints_cur[self.itrn - 1]
        else:
            delta_faints = self.n_faints_cur[self.itrn - 1] - self.n_faints_cur[self.itrn - 2]
        return delta_faints

    def delta_bright_check_helper(self) -> int:
        """Get the difference in the number of bright binaries between the last two iterations"""
        if self.itrn - 1 == 0:
            delta_brights = self.n_brights_cur[self.itrn - 1]
        else:
            delta_brights = self.n_brights_cur[self.itrn - 1] - self.n_brights_cur[self.itrn - 2]
        return delta_brights

    def run_binary_coadd(self, itrb) -> None:
        """Get the waveform for a binary, store its snr, decide whether it is faint, and coadd it to the appropriate spectrum"""
        itrn = self.itrn
        self.waveform_manager.update_params(self.params_gb[itrb].copy())

        self.snr_storage_helper(itrb)
        self.brights[itrn, itrb], self.faints_cur[itrn, itrb] = self.decision_helper(itrb)
        self.decide_coadd_helper(itrb)

    def snr_storage_helper(self, itrb) -> None:
        """Helper to store the snrs of the current binary"""
        itrn = self.itrn
        wavelet_waveform = self.waveform_manager.get_unsorted_coeffs()

        if not self.fit_state.get_faint_converged():
            self.snrs_lower[itrn, itrb] = self.noise_manager.noise_lower.get_sparse_snrs(wavelet_waveform, self.noise_manager.nt_min, self.noise_manager.nt_max)
            self.snrs_tot_lower[itrn, itrb] = np.linalg.norm(self.snrs_lower[itrn, itrb])
        else:
            assert itrn > 1
            self.snrs_lower[itrn, itrb] = self.snrs_lower[itrn - 1, itrb]
            self.snrs_tot_lower[itrn, itrb] = self.snrs_tot_lower[itrn - 1, itrb]

        if not self.fit_state.get_bright_converged():
            self.snrs_upper[itrn, itrb] = self.noise_manager.noise_upper.get_sparse_snrs(wavelet_waveform, self.noise_manager.nt_min, self.noise_manager.nt_max)
            self.snrs_tot_upper[itrn, itrb] = np.linalg.norm(self.snrs_upper[itrn, itrb])
        else:
            assert itrn > 1
            self.snrs_upper[itrn, itrb] = self.snrs_upper[itrn - 1, itrb]
            self.snrs_tot_upper[itrn, itrb] = self.snrs_tot_upper[itrn - 1, itrb]

        if np.isnan(self.snrs_tot_upper[itrn, itrb]) or np.isnan(self.snrs_tot_lower[itrn, itrb]):
            raise ValueError('nan detected in snr at ' + str(itrn) + ', ' + str(itrb))

        if ~np.isfinite(self.snrs_tot_upper[itrn, itrb]) or ~np.isfinite(self.snrs_tot_lower[itrn, itrb]):
            raise ValueError('Non-finite value detected in snr at ' + str(itrn) + ', ' + str(itrb))

    def decision_helper(self, itrb: int) -> (bool, bool):
        """Helper to decide whether a binary is bright or faint by the current noise spectrum"""
        itrn = self.itrn
        if self.fit_state.get_preprocess_mode() == 1:
            snr_cut_faint_loc = self.ic.snr_min_preprocess
        elif self.fit_state.get_preprocess_mode() == 2:
            snr_cut_faint_loc = self.ic.snr_min_reprocess
        else:
            snr_cut_faint_loc = self.ic.snr_min[itrn]

        if not self.fit_state.get_faint_converged():
            faint_candidate = self.snrs_tot_lower[itrn, itrb] < snr_cut_faint_loc
        else:
            faint_candidate = False

        if not self.fit_state.get_bright_converged():
            bright_candidate = self.snrs_tot_upper[itrn, itrb] >= self.ic.snr_cut_bright[itrn]
        else:
            bright_candidate = False

        if bright_candidate and faint_candidate:
            # satifisfied conditions to be eliminated in both directions so just keep it
            bright_loc = False
            faint_loc = False
        elif bright_candidate:
            if self.snrs_tot_upper[itrn, itrb] > self.snrs_tot_lower[itrn, itrb]:
                # handle case where snr ordering is wrong to prevent oscillation
                bright_loc = False
            else:
                bright_loc = True
            faint_loc = False
        elif faint_candidate:
            bright_loc = False
            faint_loc = True
        else:
            bright_loc = False
            faint_loc = False

        return bright_loc, faint_loc

    def decide_coadd_helper(self, itrb: int) -> None:
        """Add each binary to the correct part of the galactic spectrum, depending on whether it is bright or faint"""
        itrn = self.itrn
        # the same binary cannot be decided as both bright and faint
        assert not (self.brights[itrn, itrb] and self.faints_cur[itrn, itrb])

        # don't add to anything if the bright adaptation is already converged and this binary would not be faint
        if self.fit_state.get_bright_converged() and not self.faints_cur[itrn, itrb]:
            return

        wavelet_waveform = self.waveform_manager.get_unsorted_coeffs()

        if not self.faints_cur[itrn, itrb]:
            if self.brights[itrn, itrb]:
                # binary is bright enough to decide
                self.noise_manager.bgd.add_bright(wavelet_waveform)
            else:
                # binary neither faint nor bright enough to decide
                self.noise_manager.bgd.add_undecided(wavelet_waveform)
        else:
            # binary is faint enough to decide
            if itrn == 0:
                self.faints_cur[itrn, itrb] = False
                self.faints_old[itrb] = True
                self.noise_manager.bgd.add_floor(wavelet_waveform)
            else:
                self.noise_manager.bgd.add_faint(wavelet_waveform)

    def advance_state(self) -> None:
        """Handle any logic necessary to advance the state of the object to the next iteration"""
        if self.itrn == 0:
            self.faints_cur[self.itrn] = False
            self.brights[self.itrn] = False
            self.noise_manager.bgd.clear_undecided()
            self.noise_manager.bgd.clear_above()
        else:
            self.faints_cur[self.itrn] = self.faints_cur[self.itrn - 1]

            if self.fit_state.get_bright_converged():
                self.brights[self.itrn] = self.brights[self.itrn - 1]
            else:
                self.noise_manager.bgd.clear_undecided()
                if self.fit_state.get_do_faint_check():
                    self.noise_manager.bgd.clear_above()
                    self.brights[self.itrn] = False
                else:
                    self.brights[self.itrn] = self.brights[self.itrn - 1]

        self.decided[self.itrn] = self.brights[self.itrn] | self.faints_cur[self.itrn] | self.faints_old

        idxbs = np.argwhere(~self.decided[self.itrn]).flatten()

        tib = perf_counter()

        for itrb in idxbs:
            if itrb % 10000 == 0:
                tcb = perf_counter()
                print('Starting binary # %11d of %11d to consider at t=%9.2f s of iteration %4d' % (itrb, idxbs.size, (tcb - tib), self.itrn))

            self.run_binary_coadd(itrb)

        # copy forward prior calculations of snr calculations that were skipped in this loop iteration
        self.sustain_snr_helper()

        self.n_brights_cur[self.itrn] = self.brights[self.itrn].sum()
        self.n_faints_cur[self.itrn] = self.faints_cur[self.itrn].sum()

        self.itrn += 1

    def state_check(self) -> None:
        """Do any self consistency checks based on the current state"""
        if self.itrn > 0:
            if self.fit_state.bright_converged[self.itrn - 1]:
                assert self.itrn > 1
                assert np.all(self.brights[self.itrn - 1] == self.brights[self.itrn - 2])

            if self.fit_state.faint_converged[self.itrn - 1]:
                assert self.itrn > 1
                assert np.all(self.faints_cur[self.itrn - 1] == self.faints_cur[self.itrn - 2])

    def print_report(self) -> None:
        """Do any printing desired after convergence has been achieved and the loop ends"""
        Tobs_consider_yr = (self.noise_manager.nt_max - self.noise_manager.nt_min) * self.wc.DT / gc.SECSYEAR
        n_consider = self.n_bin_use
        n_faint = self.faints_old.sum()
        n_faint2 = self.faints_cur[self.itrn - 1].sum()
        n_bright = self.brights[self.itrn - 1].sum()
        n_ambiguous = (~(self.faints_old | self.brights[self.itrn - 1] | self.faints_cur[self.itrn - 1])).sum()
        print('Out of %10d total binaries, %10d were deemed undetectable by a previous evaluation, %10d were considered here.' % (self.n_tot, self.n_tot - n_consider, n_consider))
        print('The iterative procedure deemed (%5.3f yr observation at threshold snr=%5.3f):' % (Tobs_consider_yr, self.ic.snr_thresh))
        print('       %10d undetectable due to instrument noise' % n_faint)
        print('       %10d undetectable due to galactic confusion' % n_faint2)
        print('       %10d undecided (presumed undetectable)' % n_ambiguous)
        print('       %10d total undetectable' % (self.n_tot - n_bright))
        print('       %10d total detectable' % n_bright)

        assert n_ambiguous + n_bright + n_faint + n_faint2 == n_consider

    def log_state(self) -> None:
        """Perform any internal logging that should be done after advance_state is run for all objects for the iteration"""
        return

    def loop_finalize(self) -> None:
        """Perform any logic desired after convergence has been achieved and the loop ends"""
        return
