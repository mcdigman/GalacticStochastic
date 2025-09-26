"""object to run everything related to the iterative processing of galactic background"""

from time import perf_counter
from typing import override

from GalacticStochastic.inclusion_state_manager import BinaryInclusionState
from GalacticStochastic.iteration_config import IterationConfig
from GalacticStochastic.iterative_fit_state_machine import IterativeFitState
from GalacticStochastic.noise_manager import NoiseModelManager
from GalacticStochastic.state_manager import StateManager


class IterativeFitManager(StateManager):
    """Iterative fit object that runs the iterative fitting procedure"""

    def __init__(
        self,
        ic: IterationConfig,
        fit_state: IterativeFitState,
        noise_manager: NoiseModelManager,
        bis: BinaryInclusionState,
    ) -> None:
        """Create the iterative fit object"""
        self.ic: IterationConfig = ic
        self.fit_state: IterativeFitState = fit_state
        self.noise_manager: NoiseModelManager = noise_manager
        self.bis: BinaryInclusionState = bis

        self.itrn: int = 0  # current iteration counter

        self.n_full_converged: int = self.fit_state.get_n_itr_cut() - 1

    def do_loop(self) -> None:
        """Do the entire iterative fitting loop"""
        print('entered loop')
        ti = perf_counter()
        for _ in range(self.fit_state.get_n_itr_cut()):
            self.do_iteration()

            if self.check_done():
                break

            self.itrn += 1

        self.loop_finalize()

        tf = perf_counter()
        print('loop time = %.3es' % (tf - ti))

        self.print_report()

    def do_iteration(self) -> None:
        """Advance everything one full iteration"""
        self.advance_state()
        self.log_state()
        self.state_check()

    @override
    def advance_state(self) -> None:
        """Advance the state of the iteration and determine whether convergence is achieved"""
        t0n = perf_counter()

        self.bis.advance_state()

        t1n = perf_counter()

        # decide whether convergence has been achieved before calling advance_state
        _ = self.fit_state.bright_convergence_decision(self.bis)
        _ = self.fit_state.faint_convergence_decision(self.bis)

        self.fit_state.advance_state()
        self.noise_manager.advance_state()

        t2n = perf_counter()

        print('made bg %3d in time %7.3fs fit time %7.3fs'
              % (self.itrn, t1n - t0n, t2n - t1n))

    @override
    def log_state(self) -> None:
        """Perform any internal logging that should be done after advance_state is run."""
        self.bis.log_state()
        self.fit_state.log_state()
        self.noise_manager.log_state()

    @override
    def loop_finalize(self) -> None:
        """Perform any logic desired after convergence has been achieved and the loop ends"""
        self.bis.loop_finalize()
        self.fit_state.loop_finalize()
        self.noise_manager.loop_finalize()

    def check_done(self) -> bool:
        """Check whether the fitting procedure can bet stopped"""
        if self.fit_state.get_bright_converged() and self.fit_state.get_faint_converged():
            print('result fully converged at ' + str(self.itrn) + ', no further iterations needed')
            self.n_full_converged = self.itrn
            return True
        return False

    @override
    def print_report(self) -> None:
        """Do any printing desired after convergence has been achieved and the loop ends"""
        self.bis.print_report()
        self.fit_state.print_report()
        self.noise_manager.print_report()

    @override
    def state_check(self) -> None:
        """Perform any sanity checks that should be performed at the end of each iteration"""
        self.bis.state_check()
        self.fit_state.state_check()
        self.noise_manager.state_check()
