"""abstract class for storing an object that the iteration manager can run the iterative procedure on"""

from abc import ABC, abstractmethod

import h5py


class StateManager(ABC):
    """Objects to be handled by the IterativeFitManager should implement this interface."""

    @abstractmethod
    def advance_state(self) -> None:
        """Handle any logic necessary to advance the state of the object to the next iteration."""
        return

    @abstractmethod
    def log_state(self) -> None:
        """Perform any internal logging that should be done after advance_state is run."""
        return

    @abstractmethod
    def state_check(self) -> None:
        """Perform any sanity checks that should be performed at the end of each iteration."""
        return

    @abstractmethod
    def loop_finalize(self) -> None:
        """Perform any logic desired after convergence has been achieved and the loop ends."""
        return

    @abstractmethod
    def print_report(self) -> None:
        """Do any printing desired after convergence has been achieved and the loop ends."""
        return

    @abstractmethod
    def store_hdf5(self, hf_in: h5py.Group, *, group_name: str = 'state_manager') -> h5py.Group:
        """Store attributes, configuration, and results to an hdf5 file."""
