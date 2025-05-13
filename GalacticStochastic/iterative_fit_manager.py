"""run iterative processing of galactic background"""

from time import perf_counter

import numpy as np

import GalacticStochastic.global_const as gc
from GalacticStochastic.background_decomposition import BGDecomposition
from GalacticStochastic.inclusion_state_manager import BinaryInclusionState
from GalacticStochastic.iterative_fit_helpers import new_noise_helper
from GalacticStochastic.iterative_fit_state_machine import IterativeFitState
from LisaWaveformTools.instrument_noise import \
    DiagonalNonstationaryDenseInstrumentNoiseModel
from WaveletWaveforms.wavelet_detector_waveforms import BinaryWaveletAmpFreqDT


class IterativeFitManager():
    def __init__(self, lc, wc, ic, SAET_m, n_iterations, galactic_below_in, snr_tots_in, snr_min_in, params_gb, period_list, nt_min, nt_max, n_cyclo_switch, stat_only, n_min_faint_adapt, faint_converge_change_thresh, smooth_lengthf_fix):

        self.wc = wc
        self.lc = lc
        self.ic = ic
        self.period_list = period_list
        self.SAET_m = SAET_m
        self.nt_min = nt_min
        self.nt_max = nt_max
        self.smooth_lengthf_fix = smooth_lengthf_fix
        self.n_cyclo_switch = n_cyclo_switch
        self.stat_only = stat_only
        self.n_min_faint_adapt = n_min_faint_adapt
        self.faint_converge_change_thresh = faint_converge_change_thresh
        self.n_tot = params_gb.shape[0]

        # iteration to switch to fitting spectrum fully

        faints_in = (snr_tots_in < snr_min_in[0]) | (params_gb[:, 3] >= (wc.Nf-1)*wc.DF)
        self.argbinmap = np.argwhere(~faints_in).flatten()
        faints_old = faints_in[self.argbinmap]
        self.params_gb = params_gb[self.argbinmap]
        self.n_bin_use = self.argbinmap.size

        params_gb = None
        faints_in = None

        self.bis = BinaryInclusionState(ic.n_iterations, self.n_bin_use, self.wc, faints_old)

        faints_old = None

        self.idx_SAET_save = np.hstack([np.arange(0, min(10, n_iterations)), np.arange(min(10, n_iterations), 4), n_iterations-1])
        self.itr_save = 0

        self.SAET_tots = np.zeros((self.idx_SAET_save.size, wc.Nt, wc.Nf, 3))
        self.SAET_fin = np.zeros((wc.Nt, wc.Nf, 3))

        self.parseval_below_low = np.zeros(n_iterations)
        self.parseval_below_high = np.zeros(n_iterations)
        self.parseval_bright = np.zeros(n_iterations)
        self.parseval_total = np.zeros(n_iterations)

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

        self.fit_state = IterativeFitState(self.ic)

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
        (do_faint_check_in, bright_converged_in, faint_converged_in, force_converge_in) = self.fit_state.get_faint_state_request()
        assert do_faint_check_in == self.fit_state.do_faint_check[itrn]
        assert bright_converged_in == self.fit_state.bright_converged[itrn]
        assert faint_converged_in == self.fit_state.faint_converged[itrn]
        assert force_converge_in == self.fit_state.force_converge[itrn]

        (do_faint_check_in, bright_converged_in, faint_converged_in, force_converge_in) = self.fit_state.get_state()
        assert do_faint_check_in == self.fit_state.do_faint_check[itrn]
        assert bright_converged_in == self.fit_state.bright_converged[itrn]
        assert faint_converged_in == self.fit_state.faint_converged[itrn]
        assert force_converge_in == self.fit_state.force_converge[itrn]

        t0n = perf_counter()

        self._state_update(itrn)

        self._run_binaries(itrn)

        # copy forward prior calculations of snr calculations that were skipped in this loop iteration
        self.bis.sustain_snr_helper(itrn, self.fit_state)

        # sanity check that the total signal does not change regardless of what bucket the binaries are allocated to
        self.bgd.total_signal_consistency_check()

        t1n = perf_counter()

        noise_safe_upper = self.fit_state.bright_convergence_decision(self.bis, itrn)
        noise_safe_lower = self.fit_state.faint_convergence_decision(self.bis, itrn, self.n_min_faint_adapt, self.faint_converge_change_thresh)

        (do_faint_check_in, bright_converged_in, faint_converged_in, force_converge_in) = self.fit_state.get_faint_state_request()

        self.fit_state.advance_state()
        self.fit_state.log_state(itrn+1)

        assert do_faint_check_in == self.fit_state.do_faint_check[itrn+1]
        assert bright_converged_in == self.fit_state.bright_converged[itrn+1]
        assert faint_converged_in == self.fit_state.faint_converged[itrn+1]
        assert force_converge_in == self.fit_state.force_converge[itrn+1]

        self.noise_upper, self.noise_lower = new_noise_helper(noise_safe_upper, noise_safe_lower, self.noise_upper, self.noise_lower, itrn, self.n_cyclo_switch, self.stat_only, self.SAET_m, self.wc, self.ic, self.bgd, self.period_list, self.smooth_lengthf_fix)

        self._state_check(itrn)

        self._parseval_store(itrn)

        self._iteration_cleanup(itrn)

        t2n = perf_counter()
        print('made bg %3d in time %7.3fs fit time %7.3fs' % (itrn, t1n-t0n, t2n-t1n))

    def _run_binaries(self, itrn):
        # do the finishing step for itrn=0 to set everything at the end of the loop as it should be

        self.bis.decided[itrn] = self.bis.brights[itrn] | self.bis.faints_cur[itrn] | self.bis.faints_old

        idxbs = np.argwhere(~self.bis.decided[itrn]).flatten()
        for itrb in idxbs:
            if not self.bis.decided[itrn, itrb]:
                self.bis.run_binary_coadd(self.waveform_manager, self.params_gb, itrn, itrb, self.noise_upper, self.noise_lower, self.ic, self.fit_state, self.nt_min, self.nt_max, self.bgd)

    def _iteration_cleanup(self, itrn):
        if self.itr_save < self.idx_SAET_save.size and itrn == self.idx_SAET_save[self.itr_save]:
            self.SAET_tots[self.itr_save] = self.noise_upper.SAET[:, :, :]
            self.itr_save += 1

    def _loop_cleanup(self):
        self.SAET_fin[:] = self.noise_upper.SAET[:, :, :]

    def check_done(self, itrn):
        if self.fit_state.bright_converged[itrn+1] and self.fit_state.faint_converged[itrn+1]:
            print('result fully converged at '+str(itrn)+', no further iterations needed')
            self.n_full_converged = itrn
            return True
        return False

    def print_report(self, itrn):
        Tobs_consider_yr = (self.nt_max - self.nt_min)*self.wc.DT/gc.SECSYEAR
        n_consider = self.n_bin_use
        n_faint = self.bis.faints_old.sum()
        n_faint2 = self.bis.faints_cur[itrn].sum()
        n_bright = self.bis.brights[itrn].sum()
        n_ambiguous = (~(self.bis.faints_old[itrn] | self.bis.brights[itrn])).sum()
        print('Out of %10d total binaries, %10d were deemed undetectable by a previous evaluation, %10d were considered here.' % (self.n_tot, self.n_tot - n_consider, n_consider))
        print('The iterative procedure deemed (%5.3f yr observation at threshold snr=%5.3f):' % (Tobs_consider_yr, self.ic.snr_thresh))
        print('       %10d undetectable due to instrument noise' % n_faint)
        print('       %10d undetectable dut to galactic confusion' % n_faint2)
        print('       %10d undecided (presumed undetectable)' % n_ambiguous)
        print('       %10d total undetectable' % (self.n_tot - n_bright))
        print('       %10d total detectable' % n_bright)

    def _state_update(self, itrn):
        self.bis.faints_cur[itrn] = self.bis.faints_cur[itrn-1]

        if self.fit_state.bright_converged[itrn]:
            self.bis.brights[itrn] = self.bis.brights[itrn-1]
        else:
            self.bgd.galactic_undecided[:] = 0.
            if self.fit_state.do_faint_check[itrn]:
                self.bgd.galactic_above[:] = 0.
                self.bis.brights[itrn] = False
            else:
                self.bis.brights[itrn] = self.bis.brights[itrn-1]

    def _state_check(self, itrn):
        if self.fit_state.bright_converged[itrn]:
            assert np.all(self.bis.brights[itrn] == self.bis.brights[itrn-1])

        if self.fit_state.faint_converged[itrn]:
            assert np.all(self.bis.faints_cur[itrn] == self.bis.faints_cur[itrn-1])

        if self.fit_state.do_faint_check[itrn+1]:
            assert not self.fit_state.bright_converged[itrn+1]

    def _parseval_store(self, itrn):
        self.parseval_total[itrn] = np.sum((self.bgd.get_galactic_total()).reshape((self.wc.Nt, self.wc.Nf, self.wc.NC))[:, 1:, 0:2]**2/self.SAET_m[1:, 0:2])
        self.parseval_below_high[itrn] = np.sum((self.bgd.get_galactic_below_high()).reshape((self.wc.Nt, self.wc.Nf, self.wc.NC))[:, 1:, 0:2]**2/self.SAET_m[1:, 0:2])
        self.parseval_below_low[itrn] = np.sum((self.bgd.get_galactic_below_low()).reshape((self.wc.Nt, self.wc.Nf, self.wc.NC))[:, 1:, 0:2]**2/self.SAET_m[1:, 0:2])
        self.parseval_bright[itrn] = np.sum((self.bgd.get_galactic_coadd_resolvable()).reshape((self.wc.Nt, self.wc.Nf, self.wc.NC))[:, 1:, 0:2]**2/self.SAET_m[1:, 0:2])
