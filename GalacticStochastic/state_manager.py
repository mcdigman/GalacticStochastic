"""Abstract based class for storing the state of a component of the iterative procedure."""

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
        """
        Store attributes, configuration, and results to an HDF5 file.

        This method saves the current state, including relevant attributes and results,
        to the specified HDF5 group. The data can be organized under a specific group name
        and with a chosen storage mode.

        Parameters
        ----------
        hf_in : h5py.Group
            The HDF5 group where the state will be stored.
        group_name : str
            Name of the group under which to store the state (default is 'state_manager').
        group_mode : int
            If group_mode == 1, do not create a new group, and write directly to hf_in.
            If group_mode == 0, create a new group under hf_in with name group_name (default is 0).

        Returns
        -------
        h5py.Group
            The HDF5 group containing the stored state.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a subclass.
        """

    @abstractmethod
    def load_hdf5(self, hf_in: h5py.Group, *, group_name: str = 'state_manager', group_mode: int = 0) -> None:
        """
        Load attributes, configuration, and results from an HDF5 file.

        This method loads the current state, including relevant attributes and results,
        from the specified HDF5 group, as well as possible. The data can be organized under a specific group name
        and with a chosen storage mode.

        Parameters
        ----------
        hf_in : h5py.Group
            The HDF5 group where the state was stored.
        group_name : str
            Name of the group under which to store the state (default is 'state_manager').
        group_mode : int
            If group_mode == 1, assume no new group was created, and read directly from hf_in.
            If group_mode == 0, assume a new group was created under hf_in with name group_name (default is 0).

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a subclass.
        """
