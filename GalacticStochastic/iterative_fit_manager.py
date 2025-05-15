"""run iterative processing of galactic background"""

from time import perf_counter

import numpy as np

import GalacticStochastic.global_const as gc
from GalacticStochastic.background_decomposition import BGDecomposition
from GalacticStochastic.inclusion_state_manager import BinaryInclusionState
from GalacticStochastic.iterative_fit_state_machine import IterativeFitState
from GalacticStochastic.noise_manager import NoiseModelManager
from GalacticStochastic.state_manager import StateManager


class IterativeFitManager(StateManager):
    def __init__(self, lc, wc, ic, SAET_m, galactic_below_in, snr_tots_in, snr_min_in, params_gb, nt_min, nt_max, stat_only):

        self.wc = wc
        self.lc = lc
        self.ic = ic
        self.SAET_m = SAET_m

        self.itrn = 0  # current iteration counter

        self.n_full_converged = ic.max_iterations-1

        self.fit_state = IterativeFitState(self.ic)

        galactic_floor = galactic_below_in.copy()
        galactic_below_in = None
        galactic_below = np.zeros_like(galactic_floor)
        galactic_above = np.zeros((wc.Nt*wc.Nf, wc.NC))
        galactic_undecided = np.zeros((wc.Nt*wc.Nf, wc.NC))

        self.bgd = BGDecomposition(wc, wc.NC, galactic_floor, galactic_below, galactic_undecided, galactic_above)

        galactic_below = None
        galactic_floor = None
        galactic_above = None
        galactic_undecided = None

        self.noise_manager = NoiseModelManager(ic, wc, self.fit_state, self.bgd, SAET_m, stat_only, nt_min, nt_max)

        self.bis = BinaryInclusionState(self.wc, self.ic, self.lc, params_gb, snr_tots_in, snr_min_in, self.noise_manager, self.fit_state)


    def do_loop(self):
        print('entered loop')
        ti = perf_counter()
        for itrn_loc in range(0, self.ic.max_iterations):

            self.do_iteration()

            if self.check_done():
                break

            self.itrn += 1

        self.loop_finalize()

        tf = perf_counter()
        print('loop time = %.3es' % (tf-ti))

        self.print_report()

    def do_iteration(self):
        """advance everything one full iteration"""
        self.advance_state()
        self.log_state()
        self.state_check()

    def advance_state(self):
        """ advance the state of the iteration and determine whether convergence is achieved"""
        t0n = perf_counter()

        self.bis.advance_state()

        t1n = perf_counter()

        # decide whether convergence has been achieved before calling advance_state
        self.fit_state.bright_convergence_decision(self.bis)
        self.fit_state.faint_convergence_decision(self.bis)

        self.fit_state.advance_state()
        self.noise_manager.advance_state()

        t2n = perf_counter()

        print('made bg %3d in time %7.3fs fit time %7.3fs' % (self.itrn, t1n-t0n, t2n-t1n))

    def log_state(self):
        """Perform any internal logging that should be done after advance_state is run for all objects for the iteration"""
        self.bis.log_state()
        self.fit_state.log_state()
        self.noise_manager.log_state()

    def loop_finalize(self):
        """Perform any logic desired after convergence has been achieved and the loop ends"""
        self.bis.loop_finalize()
        self.fit_state.loop_finalize()
        self.noise_manager.loop_finalize()

    def check_done(self):
        if self.fit_state.get_bright_converged() and self.fit_state.get_faint_converged():
            print('result fully converged at '+str(self.itrn)+', no further iterations needed')
            self.n_full_converged = self.itrn
            return True
        return False

    def print_report(self):
        """Do any printing desired after convergence has been achieved and the loop ends"""
        self.bis.print_report()
        self.fit_state.print_report()
        self.noise_manager.print_report()

    def state_check(self):
        """Perform any sanity checks that should be performed at the end of each iteration"""
        self.bis.state_check()
        self.fit_state.state_check()
        self.noise_manager.state_check()
