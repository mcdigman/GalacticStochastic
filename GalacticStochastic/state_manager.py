"""abstract class for storing an object that the iteration manager can run the iterative procedure on"""

from abc import ABC, abstractmethod


class StateManager(ABC):
    """Objects to be handled by the IterativeFitManager should implement this interface"""

    @abstractmethod
    def advance_state(self):
        """Handle any logic necessary to advance the state of the object to the next iteration"""
        return

    @abstractmethod
    def log_state(self):
        """Perform any internal logging that should be done after advance_state is run for all objects for the iteration"""
        return

    @abstractmethod
    def state_check(self):
        """Perform any sanity checks that should be performed at the end of each iteration"""
        return

    @abstractmethod
    def loop_finalize(self):
        """Perform any logic desired after convergence has been achieved and the loop ends"""
        return

    @abstractmethod
    def print_report(self):
        """Do any printing desired after convergence has been achieved and the loop ends"""
        return
