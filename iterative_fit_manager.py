"""run iterative processing of galactic background"""

from time import perf_counter

import numpy as np

import global_const as gc
from instrument_noise import DiagonalNonstationaryDenseInstrumentNoiseModel
from iterative_fit_helpers import (BGDecomposition,
                                   addition_convergence_decision,
                                   run_binary_coadd2,
                                   subtraction_convergence_decision,
                                   sustain_snr_helper,
                                   total_signal_consistency_check)
from wavelet_detector_waveforms import BinaryWaveletAmpFreqDT



class IterativeFitManager():
    def __init__(self, lc, wc, ic, SAET_m, n_iterations, galactic_bg_const_in, snr_tots_in, snr_min_in, params_gb, period_list, nt_min, nt_max, n_cyclo_switch, const_only, n_const_force, const_converge_change_thresh, smooth_lengthf_fix):

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

        const_suppress_in = (snr_tots_in < snr_min_in[0]) | (params_gb[:, 3] >= (wc.Nf-1)*wc.DF)
        self.argbinmap = np.argwhere(~const_suppress_in).flatten()
        self.const_suppress = const_suppress_in[self.argbinmap]
        self.params_gb = params_gb[self.argbinmap]
        self.n_bin_use = self.argbinmap.size

        params_gb = None
        const_suppress_in = None

        self.idx_SAE_save = np.hstack([np.arange(0, min(10, n_iterations)), np.arange(min(10, n_iterations), 4), n_iterations-1])
        self.itr_save = 0

        self.SAE_tots = np.zeros((self.idx_SAE_save.size, wc.Nt, wc.Nf, 2))
        self.SAE_fin = np.zeros((wc.Nt, wc.Nf, 2))


        self.parseval_const = np.zeros(n_iterations)
        self.parseval_bg = np.zeros(n_iterations)
        self.parseval_sup = np.zeros(n_iterations)
        self.parseval_tot = np.zeros(n_iterations)


        params0 = self.params_gb[0].copy()
        self.waveform_manager = BinaryWaveletAmpFreqDT(params0.copy(), wc, self.lc)

        self.SAET_tot_cur = np.zeros((wc.Nt, wc.Nf, wc.NC))
        self.SAET_tot_cur[:] = self.SAET_m

        self.SAET_tot_base = np.zeros((wc.Nt, wc.Nf, wc.NC))
        self.SAET_tot_base[:] = self.SAET_m
        if self.idx_SAE_save[self.itr_save] == 0:
            self.SAE_tots[0] = self.SAET_tot_cur[:, :, :2]
            self.itr_save += 1
        self.SAET_tot_base = np.min([self.SAET_tot_base, self.SAET_tot_cur], axis=0)

        self.noise_AET_dense_base = DiagonalNonstationaryDenseInstrumentNoiseModel(self.SAET_tot_cur, wc, prune=True)
        self.noise_AET_dense_base_base = DiagonalNonstationaryDenseInstrumentNoiseModel(self.SAET_tot_base, wc, prune=True)



        self.n_full_converged = ic.n_iterations-1

        self.bis = BinaryInclusionState(ic.n_iterations, self.n_bin_use, self.wc)
        self.fit_state =  IterativeFitState(self.ic)

        self.galactic_full_signal = np.zeros((wc.Nt*wc.Nf, wc.NC))

        galactic_bg_const_base = galactic_bg_const_in.copy()
        galactic_bg_const_in = None
        galactic_bg_const = np.zeros_like(galactic_bg_const_base)
        galactic_bg_suppress = np.zeros((wc.Nt*wc.Nf, wc.NC))
        galactic_bg = np.zeros((wc.Nt*wc.Nf, wc.NC))

        self.bgd = BGDecomposition(galactic_bg_const_base, galactic_bg_const, galactic_bg, galactic_bg_suppress)
        
        galactic_bg_const = None
        galactic_bg_const_base = None
        galactic_bg_suppress = None
        galactic_bg = None


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
        sustain_snr_helper(self.fit_state.const_converged, self.bis.snrs_tot_base, self.bis.snrs_base, self.bis.snrs_tot, self.bis.snrs, itrn, self.bis.suppress[itrn], self.fit_state.var_converged)

        # sanity check that the total signal does not change regardless of what bucket the binaries are allocated to
        total_signal_consistency_check(self.galactic_full_signal, self.bgd, itrn)


        t1n = perf_counter()

        self.noise_AET_dense_base, self.SAET_tot_cur = subtraction_convergence_decision(self.bgd, self.bis.var_suppress, itrn, self.fit_state.force_converge, self.bis.n_var_suppress, self.fit_state.switch_next, self.fit_state.var_converged, self.fit_state.const_converged, self.SAET_m, self.wc, self.ic, self.period_list, self.const_only, self.noise_AET_dense_base, self.n_cyclo_switch, self.SAET_tot_cur)

        self.noise_AET_dense_base_base, self.SAET_tot_base = addition_convergence_decision(self.bgd, itrn, self.bis.n_const_suppress, self.fit_state.switch_next, self.fit_state.var_converged, self.fit_state.switchf_next, self.fit_state.const_converged, self.SAET_m, self.wc, self.period_list, self.const_only, self.noise_AET_dense_base_base, self.SAET_tot_cur, self.SAET_tot_base, self.n_const_force, self.const_converge_change_thresh, self.bis.const_suppress2, self.smooth_lengthf_fix)

        self._state_check(itrn)

        self._parseval_store(itrn)

        self._iteration_cleanup(itrn)

        t2n = perf_counter()
        print('made bg %3d in time %7.3fs fit time %7.3fs' % (itrn, t1n-t0n, t2n-t1n))

    def _run_binaries(self,itrn):
        # do the finishing step for itrn=0 to set everything at the end of the loop as it should be

        self.bis.suppress[itrn] = self.bis.var_suppress[itrn] | self.bis.const_suppress2[itrn] | self.const_suppress

        idxbs = np.argwhere(~self.bis.suppress[itrn]).flatten()
        for itrb in idxbs:
            if not self.bis.suppress[itrn, itrb]:
                run_binary_coadd2(self.waveform_manager, self.params_gb, self.bis.var_suppress, self.const_suppress, self.bis.const_suppress2, self.bis.snrs_base, self.bis.snrs, self.bis.snrs_tot, self.bis.snrs_tot_base, itrn, itrb, self.noise_AET_dense_base, self.noise_AET_dense_base_base, self.ic, self.fit_state.const_converged, self.fit_state.var_converged, self.nt_min, self.nt_max, self.bgd)

    def _iteration_cleanup(self, itrn):
        if self.itr_save < self.idx_SAE_save.size and itrn == self.idx_SAE_save[self.itr_save]:
            self.SAE_tots[self.itr_save] = self.SAET_tot_cur[:, :, :2]
            self.itr_save += 1

    def _loop_cleanup(self):
        self.SAE_fin[:] = self.SAET_tot_cur[:, :, :2]

    def check_done(self,itrn):
        if self.fit_state.var_converged[itrn+1] and self.fit_state.const_converged[itrn+1]:
            print('result fully converged at '+str(itrn)+', no further iterations needed')
            self.n_full_converged = itrn
            return True
        return False

    def print_report(self, itrn):
        Tobs_consider_yr = (self.nt_max - self.nt_min)*self.wc.DT/gc.SECSYEAR
        n_consider = self.n_bin_use
        n_faint = self.const_suppress.sum()
        n_faint2 = self.bis.const_suppress2[itrn].sum()
        n_bright = self.bis.var_suppress[itrn].sum()
        n_ambiguous = (~(self.const_suppress[itrn] | self.bis.var_suppress[itrn])).sum()
        print('Out of %10d total binaries, %10d were deemed undetectable by a previous evaluation, %10d were considered here.' % (self.n_tot, self.n_tot - n_consider, n_consider))
        print('The iterative procedure deemed (%5.3f yr observation at threshold snr=%5.3f):' % (Tobs_consider_yr, self.ic.snr_thresh))
        print('       %10d undetectable due to instrument noise' % n_faint)
        print('       %10d undetectable dut to galactic confusion' % n_faint2)
        print('       %10d undecided (presumed undetectable)' % n_ambiguous)
        print('       %10d total undetectable' % (self.n_tot - n_bright))
        print('       %10d total detectable' % n_bright)


    def _state_update(self,itrn):
        if self.fit_state.switchf_next[itrn]:
            self.bgd.galactic_bg_const[:] = 0.
            self.bis.const_suppress2[itrn] = False
        else:
            self.bis.const_suppress2[itrn] = self.bis.const_suppress2[itrn-1]

        if self.fit_state.var_converged[itrn]:
            self.bis.var_suppress[itrn] = self.bis.var_suppress[itrn-1]
        else:
            self.bgd.galactic_bg[:] = 0.
            if self.fit_state.switch_next[itrn]:
                self.bgd.galactic_bg_suppress[:] = 0.
                self.bis.var_suppress[itrn] = False
            else:
                self.bis.var_suppress[itrn] = self.bis.var_suppress[itrn-1]

    def _state_check(self,itrn):
        if self.fit_state.var_converged[itrn]:
            assert np.all(self.bis.var_suppress[itrn] == self.bis.var_suppress[itrn-1])

        if self.fit_state.const_converged[itrn]:
            assert np.all(self.bis.const_suppress2[itrn] == self.bis.const_suppress2[itrn-1])

        if self.fit_state.switchf_next[itrn+1]:
            assert not self.fit_state.const_converged[itrn+1]

        if self.fit_state.switch_next[itrn+1]:
            assert not self.fit_state.var_converged[itrn+1]

    def _parseval_store(self,itrn):
        self.parseval_tot[itrn] = np.sum((self.bgd.galactic_bg_const_base+self.bgd.galactic_bg_const+self.bgd.galactic_bg+self.bgd.galactic_bg_suppress).reshape((self.wc.Nt, self.wc.Nf, self.wc.NC))[:, 1:, 0:2]**2/self.SAET_m[1:, 0:2])
        self.parseval_bg[itrn] = np.sum((self.bgd.galactic_bg_const_base+self.bgd.galactic_bg_const+self.bgd.galactic_bg).reshape((self.wc.Nt, self.wc.Nf, self.wc.NC))[:, 1:, 0:2]**2/self.SAET_m[1:, 0:2])
        self.parseval_const[itrn] = np.sum((self.bgd.galactic_bg_const_base+self.bgd.galactic_bg_const).reshape((self.wc.Nt, self.wc.Nf, self.wc.NC))[:, 1:, 0:2]**2/self.SAET_m[1:, 0:2])
        self.parseval_sup[itrn] = np.sum((self.bgd.galactic_bg_suppress).reshape((self.wc.Nt, self.wc.Nf, self.wc.NC))[:, 1:, 0:2]**2/self.SAET_m[1:, 0:2])

class BinaryInclusionState():
    def __init__(self, n_iterations, n_bin_use, wc):
        self.snrs = np.zeros((n_iterations, n_bin_use, wc.NC))
        self.snrs_base = np.zeros((n_iterations, n_bin_use, wc.NC))
        self.snrs_tot_base = np.zeros((n_iterations, n_bin_use))
        self.snrs_tot = np.zeros((n_iterations, n_bin_use))
        self.var_suppress = np.zeros((n_iterations, n_bin_use), dtype=np.bool_)
        self.suppress = np.zeros((n_iterations, n_bin_use), dtype=np.bool_)
        self.const_suppress2 = np.zeros((n_iterations, n_bin_use), dtype=np.bool_)

        self.n_const_suppress = np.zeros(n_iterations+1, dtype=np.int64)
        #self.n_const_suppress[1] = self.const_suppress2[0].sum()
        #self.n_var_suppress[1] = self.var_suppress[0].sum()
        self.n_var_suppress = np.zeros(n_iterations+1, dtype=np.int64)


class IterativeFitState():
    def __init__(self, ic):
        self.var_converged = np.zeros(ic.n_iterations+1, dtype=np.bool_)
        self.const_converged = np.zeros(ic.n_iterations+1, dtype=np.bool_)
        self.switch_next = np.zeros(ic.n_iterations+1, dtype=np.bool_)
        self.switchf_next = np.zeros(ic.n_iterations+1, dtype=np.bool_)
        self.force_converge = np.zeros(ic.n_iterations+1, dtype=np.bool_)
