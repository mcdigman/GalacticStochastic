"""abstract class for storing an object that the iteration manager can run the iterative procedure on"""

from abc import ABC, abstractmethod

import h5py


class StateManager(ABC):
    """
    Abstract base class for managing the state of a component of the iterative procedure.

    This class defines the interface that objects must implement to be compatible
    with the IterativeFitManager. It provides abstract methods for advancing the
    state, logging, performing checks, finalizing after convergence, reporting,
    and storing results to an HDF5 file.

    Methods
    -------
    advance_state()
        Handle any logic necessary to advance the state of the object to the next iteration.

    log_state()
        Perform any internal logging that should be done after `advance_state` is run.

    state_check()
        Perform any sanity checks that should be performed at the end of each iteration.

    loop_finalize()
        Perform any logic desired after convergence has been achieved and the loop ends.

    print_report()
        Do any printing desired after convergence has been achieved and the loop ends.

    store_hdf5(hf_in, *, group_name='state_manager', group_mode=0)
        Store attributes, configuration, and results to an HDF5 file.

    load_hdf5(hf_in, *, group_name='state_manager', group_mode=0)
        Load attributes, configuration, and results from an HDF5 file.

    Notes
    -----
    This is an abstract base class and should not be instantiated directly.
    Subclasses must implement all abstract methods.
    """

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
    def store_hdf5(self, hf_in: h5py.Group, *, group_name: str = 'state_manager', group_mode: int = 0) -> h5py.Group:
        """Store attributes, configuration, and results to an hdf5 file."""

    @abstractmethod
    def load_hdf5(self, hf_in: h5py.Group, *, group_name: str = 'state_manager', group_mode: int = 0) -> None:
        """Load attributes, configuration, and results from an hdf5 file."""
