"""run iterative processing of galactic background"""

from time import perf_counter

import numpy as np

import GalacticStochastic.global_const as gc
from GalacticStochastic.background_decomposition import BGDecomposition
from GalacticStochastic.inclusion_state_manager import BinaryInclusionState
from GalacticStochastic.iterative_fit_state_machine import IterativeFitState
from GalacticStochastic.noise_manager import NoiseModelManager
from WaveletWaveforms.wavelet_detector_waveforms import BinaryWaveletAmpFreqDT


class IterativeFitManager():
    def __init__(self, lc, wc, ic, SAET_m, galactic_below_in, snr_tots_in, snr_min_in, params_gb, nt_min, nt_max, stat_only):

        self.wc = wc
        self.lc = lc
        self.ic = ic
        self.SAET_m = SAET_m
        self.nt_min = nt_min
        self.nt_max = nt_max
        self.stat_only = stat_only
        self.n_tot = params_gb.shape[0]

        # iteration to switch to fitting spectrum fully

        # TODO make the input snr check optional and use controllable maximum and minimum frequencies
        # TODO take BinaryInclusionState, BinaryWaveletAmpFreqDT, IterativeFitState, BGDecomposition, and DiagonalNonstationaryDenseInstrumentNoiseModel as arguments for modularity
        faints_in = (snr_tots_in < snr_min_in[0]) | (params_gb[:, 3] >= (wc.Nf-1)*wc.DF)
        self.argbinmap = np.argwhere(~faints_in).flatten()
        faints_old = faints_in[self.argbinmap]
        assert faints_old.sum() == 0.
        self.params_gb = params_gb[self.argbinmap]
        self.n_bin_use = self.argbinmap.size

        params_gb = None
        faints_in = None

        self.bis = BinaryInclusionState(ic.max_iterations, self.n_bin_use, self.wc, faints_old)

        faints_old = None


        params0 = self.params_gb[0].copy()
        self.waveform_manager = BinaryWaveletAmpFreqDT(params0.copy(), wc, self.lc)

        self.n_full_converged = ic.max_iterations-1

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

        self.noise_manager = NoiseModelManager(ic, wc, self.bgd, SAET_m)

    def do_loop(self):
        print('entered loop')
        ti = perf_counter()
        for itrn in range(0, self.ic.max_iterations):

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
        noise_safe_lower = self.fit_state.faint_convergence_decision(self.bis, itrn, self.ic)

        (do_faint_check_in, bright_converged_in, faint_converged_in, force_converge_in) = self.fit_state.get_faint_state_request()

        self.fit_state.advance_state()
        self.fit_state.log_state(itrn+1)

        assert do_faint_check_in == self.fit_state.do_faint_check[itrn+1]
        assert bright_converged_in == self.fit_state.bright_converged[itrn+1]
        assert faint_converged_in == self.fit_state.faint_converged[itrn+1]
        assert force_converge_in == self.fit_state.force_converge[itrn+1]

        #self.noise_manager.noise_upper, self.noise_manager.noise_lower = new_noise_helper(noise_safe_upper, noise_safe_lower, self.noise_manager.noise_upper, self.noise_manager.noise_lower, itrn, self.stat_only, self.noise_manager.SAET_m, self.wc, self.ic, self.bgd)
        self.noise_manager.advance_state(noise_safe_upper, noise_safe_lower, itrn, self.stat_only)

        self._state_check(itrn)

        self.bgd.log_tracking(self.wc, self.SAET_m)

        self._iteration_cleanup(itrn)

        t2n = perf_counter()
        print('made bg %3d in time %7.3fs fit time %7.3fs' % (itrn, t1n-t0n, t2n-t1n))

    def _run_binaries(self, itrn):
        # do the finishing step for itrn=0 to set everything at the end of the loop as it should be

        self.bis.decided[itrn] = self.bis.brights[itrn] | self.bis.faints_cur[itrn] | self.bis.faints_old

        idxbs = np.argwhere(~self.bis.decided[itrn]).flatten()

        tib = perf_counter()

        for itrb in idxbs:
            if itrb % 10000 == 0:
                tcb = perf_counter()
                print("Starting binary # %11d of %11d to consider at t=%9.2f s of iteration %4d" % (itrb, idxbs.size, (tcb - tib), itrn))

            self.bis.run_binary_coadd(self.waveform_manager, self.params_gb, itrn, itrb, self.noise_manager.noise_upper, self.noise_manager.noise_lower, self.ic, self.fit_state, self.nt_min, self.nt_max, self.bgd)

    def _iteration_cleanup(self, itrn):
        self.noise_manager.iteration_cleanup(itrn)

    def _loop_cleanup(self):
        self.noise_manager.loop_cleanup()

    def check_done(self, itrn):
        assert self.fit_state.get_faint_converged() == self.fit_state.faint_converged[itrn+1]
        assert self.fit_state.get_bright_converged() == self.fit_state.bright_converged[itrn+1]
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
        n_ambiguous = (~(self.bis.faints_old | self.bis.brights[itrn] | self.bis.faints_cur[itrn])).sum()
        print('Out of %10d total binaries, %10d were deemed undetectable by a previous evaluation, %10d were considered here.' % (self.n_tot, self.n_tot - n_consider, n_consider))
        print('The iterative procedure deemed (%5.3f yr observation at threshold snr=%5.3f):' % (Tobs_consider_yr, self.ic.snr_thresh))
        print('       %10d undetectable due to instrument noise' % n_faint)
        print('       %10d undetectable due to galactic confusion' % n_faint2)
        print('       %10d undecided (presumed undetectable)' % n_ambiguous)
        print('       %10d total undetectable' % (self.n_tot - n_bright))
        print('       %10d total detectable' % n_bright)

        assert n_ambiguous + n_bright + n_faint + n_faint2 == n_consider

        self.noise_manager.print_report(self.nt_min, self.nt_max)


    def _state_update(self, itrn):
        if itrn == 0:
            self.bis.faints_cur[itrn] = False
            self.bis.brights[itrn] = False
            self.bgd.galactic_undecided[:] = 0.
            self.bgd.galactic_above[:] = 0.
            #assert np.all(self.bis.faints_cur[itrn-1] == False)
            #assert np.all(self.bis.brights[itrn-1] == False)
            return
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
        self.bis.state_check(itrn, self.fit_state)

        if self.fit_state.do_faint_check[itrn+1]:
            assert not self.fit_state.bright_converged[itrn+1]
