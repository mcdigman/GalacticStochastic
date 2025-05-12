"""run iterative processing of galactic background"""

from time import perf_counter

import numpy as np

import global_const as gc
from instrument_noise import DiagonalNonstationaryDenseInstrumentNoiseModel
from iterative_fit_helpers import (BGDecomposition,
                                   addition_convergence_decision,
                                   run_binary_coadd2,
                                   subtraction_convergence_decision,
                                   sustain_snr_helper)
from wavelet_detector_waveforms import BinaryWaveletAmpFreqDT



class IterativeFitManager():
    def __init__(self, lc, wc, ic, SAET_m, n_iterations, galactic_below_in, snr_tots_in, snr_min_in, params_gb, period_list, nt_min, nt_max, n_cyclo_switch, const_only, n_const_force, const_converge_change_thresh, smooth_lengthf_fix):

        self.wc = wc
        self.lc = lc
        self.ic = ic
        self.period_list = period_list
        self.SAET_m = SAET_m
        self.nt_min = nt_min
        self.nt_max = nt_max
        self.smooth_lengthf_fix = smooth_lengthf_fix
        self.n_cyclo_switch = n_cyclo_switch
        self.const_only = const_only
        self.n_const_force = n_const_force
        self.const_converge_change_thresh = const_converge_change_thresh
        self.n_tot = params_gb.shape[0]


        #iteration to switch to fitting spectrum fully

        faints_in = (snr_tots_in < snr_min_in[0]) | (params_gb[:, 3] >= (wc.Nf-1)*wc.DF)
        self.argbinmap = np.argwhere(~faints_in).flatten()
        self.faints_old = faints_in[self.argbinmap]
        self.params_gb = params_gb[self.argbinmap]
        self.n_bin_use = self.argbinmap.size

        params_gb = None
        faints_in = None

        self.idx_SAET_save = np.hstack([np.arange(0, min(10, n_iterations)), np.arange(min(10, n_iterations), 4), n_iterations-1])
        self.itr_save = 0

        self.SAET_tots = np.zeros((self.idx_SAET_save.size, wc.Nt, wc.Nf, 3))
        self.SAET_fin = np.zeros((wc.Nt, wc.Nf, 3))


        self.parseval_const = np.zeros(n_iterations)
        self.parseval_bg = np.zeros(n_iterations)
        self.parseval_sup = np.zeros(n_iterations)
        self.parseval_tot = np.zeros(n_iterations)


        params0 = self.params_gb[0].copy()
        self.waveform_manager = BinaryWaveletAmpFreqDT(params0.copy(), wc, self.lc)

        SAET_tot_upper = np.zeros((wc.Nt, wc.Nf, wc.NC))
        SAET_tot_upper[:] = self.SAET_m

        SAET_tot_lower = np.zeros((wc.Nt, wc.Nf, wc.NC))
        SAET_tot_lower[:] = self.SAET_m
        if self.idx_SAET_save[self.itr_save] == 0:
            self.SAET_tots[0] = SAET_tot_upper[:, :, :]
            self.itr_save += 1
        SAET_tot_lower = np.min([SAET_tot_lower, SAET_tot_upper], axis=0)

        self.noise_upper = DiagonalNonstationaryDenseInstrumentNoiseModel(SAET_tot_upper, wc, prune=True)
        self.noise_lower = DiagonalNonstationaryDenseInstrumentNoiseModel(SAET_tot_lower, wc, prune=True)

        SAET_tot_upper = None
        SAET_tot_lower = None


        self.n_full_converged = ic.n_iterations-1

        self.bis = BinaryInclusionState(ic.n_iterations, self.n_bin_use, self.wc)
        self.fit_state =  IterativeFitState(self.ic)

        self.galactic_total = np.zeros((wc.Nt*wc.Nf, wc.NC))

        galactic_floor = galactic_below_in.copy()
        galactic_below_in = None
        galactic_below = np.zeros_like(galactic_floor)
        galactic_above = np.zeros((wc.Nt*wc.Nf, wc.NC))
        galactic_undecided = np.zeros((wc.Nt*wc.Nf, wc.NC))

        self.bgd = BGDecomposition(galactic_floor, galactic_below, galactic_undecided, galactic_above)

        galactic_below = None
        galactic_floor = None
        galactic_above = None
        galactic_undecided = None


    def do_loop(self):
        print('entered loop')
        ti = perf_counter()
        for itrn in range(1, self.ic.n_iterations):

            self.do_iteration(itrn)

            if self.check_done(itrn):
                break


        self._loop_cleanup()

        tf = perf_counter()
        print('loop time = %.3es' % (tf-ti))

        self.print_report(itrn)


    def do_iteration(self, itrn):
        t0n = perf_counter()

        self._state_update(itrn)

        self._run_binaries(itrn)

        # copy forward prior calculations of snr calculations that were skipped in this loop iteration
        sustain_snr_helper(self.fit_state.faint_converged, self.bis.snrs_tot_lower, self.bis.snrs_lower, self.bis.snrs_tot_upper, self.bis.snrs_upper, itrn, self.bis.decided[itrn], self.fit_state.bright_converged)

        # sanity check that the total signal does not change regardless of what bucket the binaries are allocated to
        self.bgd.total_signal_consistency_check()


        t1n = perf_counter()

        self.noise_upper = subtraction_convergence_decision(self.bgd, self.bis, self.fit_state, itrn, self.SAET_m, self.wc, self.ic, self.period_list, self.const_only, self.noise_upper, self.n_cyclo_switch)

        self.noise_lower = addition_convergence_decision(self.bgd, self.bis, self.fit_state, itrn, self.SAET_m, self.wc, self.period_list, self.const_only, self.noise_lower, self.noise_upper, self.n_const_force, self.const_converge_change_thresh, self.smooth_lengthf_fix)

        self._state_check(itrn)

        self._parseval_store(itrn)

        self._iteration_cleanup(itrn)

        t2n = perf_counter()
        print('made bg %3d in time %7.3fs fit time %7.3fs' % (itrn, t1n-t0n, t2n-t1n))

    def _run_binaries(self,itrn):
        # do the finishing step for itrn=0 to set everything at the end of the loop as it should be

        self.bis.decided[itrn] = self.bis.brights[itrn] | self.bis.faints_cur[itrn] | self.faints_old

        idxbs = np.argwhere(~self.bis.decided[itrn]).flatten()
        for itrb in idxbs:
            if not self.bis.decided[itrn, itrb]:
                run_binary_coadd2(self.waveform_manager, self.params_gb, self.bis.brights, self.faints_old, self.bis.faints_cur, self.bis.snrs_lower, self.bis.snrs_upper, self.bis.snrs_tot_upper, self.bis.snrs_tot_lower, itrn, itrb, self.noise_upper, self.noise_lower, self.ic, self.fit_state.faint_converged, self.fit_state.bright_converged, self.nt_min, self.nt_max, self.bgd)

    def _iteration_cleanup(self, itrn):
        if self.itr_save < self.idx_SAET_save.size and itrn == self.idx_SAET_save[self.itr_save]:
            self.SAET_tots[self.itr_save] = self.noise_upper.SAET[:, :, :]
            self.itr_save += 1

    def _loop_cleanup(self):
        self.SAET_fin[:] = self.noise_upper.SAET[:, :, :]

    def check_done(self,itrn):
        if self.fit_state.bright_converged[itrn+1] and self.fit_state.faint_converged[itrn+1]:
            print('result fully converged at '+str(itrn)+', no further iterations needed')
            self.n_full_converged = itrn
            return True
        return False

    def print_report(self, itrn):
        Tobs_consider_yr = (self.nt_max - self.nt_min)*self.wc.DT/gc.SECSYEAR
        n_consider = self.n_bin_use
        n_faint = self.faints_old.sum()
        n_faint2 = self.bis.faints_cur[itrn].sum()
        n_bright = self.bis.brights[itrn].sum()
        n_ambiguous = (~(self.faints_old[itrn] | self.bis.brights[itrn])).sum()
        print('Out of %10d total binaries, %10d were deemed undetectable by a previous evaluation, %10d were considered here.' % (self.n_tot, self.n_tot - n_consider, n_consider))
        print('The iterative procedure deemed (%5.3f yr observation at threshold snr=%5.3f):' % (Tobs_consider_yr, self.ic.snr_thresh))
        print('       %10d undetectable due to instrument noise' % n_faint)
        print('       %10d undetectable dut to galactic confusion' % n_faint2)
        print('       %10d undecided (presumed undetectable)' % n_ambiguous)
        print('       %10d total undetectable' % (self.n_tot - n_bright))
        print('       %10d total detectable' % n_bright)


    def _state_update(self,itrn):
        if self.fit_state.switchf_next[itrn]:
            self.bgd.galactic_below[:] = 0.
            self.bis.faints_cur[itrn] = False
        else:
            self.bis.faints_cur[itrn] = self.bis.faints_cur[itrn-1]

        if self.fit_state.bright_converged[itrn]:
            self.bis.brights[itrn] = self.bis.brights[itrn-1]
        else:
            self.bgd.galactic_undecided[:] = 0.
            if self.fit_state.switch_next[itrn]:
                self.bgd.galactic_above[:] = 0.
                self.bis.brights[itrn] = False
            else:
                self.bis.brights[itrn] = self.bis.brights[itrn-1]

    def _state_check(self,itrn):
        if self.fit_state.bright_converged[itrn]:
            assert np.all(self.bis.brights[itrn] == self.bis.brights[itrn-1])

        if self.fit_state.faint_converged[itrn]:
            assert np.all(self.bis.faints_cur[itrn] == self.bis.faints_cur[itrn-1])

        if self.fit_state.switchf_next[itrn+1]:
            assert not self.fit_state.faint_converged[itrn+1]

        if self.fit_state.switch_next[itrn+1]:
            assert not self.fit_state.bright_converged[itrn+1]

    def _parseval_store(self,itrn):
        self.parseval_tot[itrn] = np.sum((self.bgd.get_galactic_total()).reshape((self.wc.Nt, self.wc.Nf, self.wc.NC))[:, 1:, 0:2]**2/self.SAET_m[1:, 0:2])
        self.parseval_bg[itrn] = np.sum((self.bgd.get_galactic_below_high()).reshape((self.wc.Nt, self.wc.Nf, self.wc.NC))[:, 1:, 0:2]**2/self.SAET_m[1:, 0:2])
        self.parseval_const[itrn] = np.sum((self.bgd.get_galactic_below_low()).reshape((self.wc.Nt, self.wc.Nf, self.wc.NC))[:, 1:, 0:2]**2/self.SAET_m[1:, 0:2])
        self.parseval_sup[itrn] = np.sum((self.bgd.get_galactic_coadd_resolvable()).reshape((self.wc.Nt, self.wc.Nf, self.wc.NC))[:, 1:, 0:2]**2/self.SAET_m[1:, 0:2])

class BinaryInclusionState():
    def __init__(self, n_iterations, n_bin_use, wc):
        self.snrs_upper = np.zeros((n_iterations, n_bin_use, wc.NC))
        self.snrs_lower = np.zeros((n_iterations, n_bin_use, wc.NC))
        self.snrs_tot_lower = np.zeros((n_iterations, n_bin_use))
        self.snrs_tot_upper = np.zeros((n_iterations, n_bin_use))
        self.brights = np.zeros((n_iterations, n_bin_use), dtype=np.bool_)
        self.decided = np.zeros((n_iterations, n_bin_use), dtype=np.bool_)
        self.faints_cur = np.zeros((n_iterations, n_bin_use), dtype=np.bool_)

        self.n_faints_cur = np.zeros(n_iterations+1, dtype=np.int64)
        self.n_brights_cur = np.zeros(n_iterations+1, dtype=np.int64)


class IterativeFitState():
    def __init__(self, ic):
        self.bright_converged = np.zeros(ic.n_iterations+1, dtype=np.bool_)
        self.faint_converged = np.zeros(ic.n_iterations+1, dtype=np.bool_)
        self.switch_next = np.zeros(ic.n_iterations+1, dtype=np.bool_)
        self.switchf_next = np.zeros(ic.n_iterations+1, dtype=np.bool_)
        self.force_converge = np.zeros(ic.n_iterations+1, dtype=np.bool_)
