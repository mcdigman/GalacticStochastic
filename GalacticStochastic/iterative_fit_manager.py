"""Object to run everything related to the iterative processing of galactic background."""

from time import perf_counter
from typing import override

import h5py
import numpy as np

from GalacticStochastic.inclusion_state_manager import BinaryInclusionState
from GalacticStochastic.iteration_config import IterationConfig
from GalacticStochastic.iterative_fit_state_machine import IterativeFitState
from GalacticStochastic.noise_manager import NoiseModelManager
from GalacticStochastic.state_manager import StateManager


class IterativeFitManager(StateManager):
    """Iterative fit object that runs the iterative fitting procedure."""

    def __init__(
        self,
        ic: IterationConfig,
        fit_state: IterativeFitState,
        noise_manager: NoiseModelManager,
        bis: BinaryInclusionState,
    ) -> None:
        """Create the iterative fit object."""
        self.ic: IterationConfig = ic
        self.fit_state: IterativeFitState = fit_state
        self.noise_manager: NoiseModelManager = noise_manager
        self.bis: BinaryInclusionState = bis

        self._itrn: int = 0  # current iteration counter

        self._n_full_converged: int = self.fit_state.get_n_itr_cut() - 1

    def do_loop(self) -> None:
        """Do the entire iterative fitting loop."""
        print('entered loop')
        ti = perf_counter()
        for _ in range(self.fit_state.get_n_itr_cut()):
            self.do_iteration()

            if self.check_done():
                break

            self._itrn += 1

        self.loop_finalize()

        tf = perf_counter()
        print('loop time = %.3es' % (tf - ti))

        self.print_report()

    def do_iteration(self) -> None:
        """Advance everything one full iteration."""
        self.advance_state()
        self.log_state()
        self.state_check()

    @override
    def store_hdf5(self, hf_in: h5py.Group, *, group_name: str = 'iterative_manager', group_mode: int = 0) -> h5py.Group:
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
        if group_mode == 0:
            hf_manager = hf_in.create_group(group_name)
        elif group_mode == 1:
            hf_manager = hf_in
        else:
            msg = 'Unrecognized option for group mode'
            raise NotImplementedError(msg)
        hf_manager.attrs['itrn'] = self._itrn
        hf_manager.attrs['n_full_converged'] = self._n_full_converged
        hf_manager.attrs['creator_name'] = self.__class__.__name__
        hf_manager.attrs['inclusion_state_name'] = self.bis.__class__.__name__
        hf_manager.attrs['fit_state_name'] = self.fit_state.__class__.__name__
        hf_manager.attrs['noise_manager_name'] = self.noise_manager.__class__.__name__
        hf_manager.attrs['ic_name'] = self.ic.__class__.__name__

        hf_ic = hf_manager.create_group('ic')
        for key in self.ic._fields:
            hf_ic.attrs[key] = getattr(self.ic, key)

        # save the objects this class uses
        _ = self.bis.store_hdf5(hf_manager, noise_recurse=0)
        _ = self.noise_manager.store_hdf5(hf_manager)
        _ = self.fit_state.store_hdf5(hf_manager)

        return hf_manager

    @override
    def load_hdf5(self, hf_in: h5py.Group, *, group_name: str = 'iterative_manager', group_mode: int = 0) -> None:
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
        TypeError
            If the format is not as expected.
        ValueError
            If loaded attributes do not match the current object's attributes.
        """
        if group_mode == 0:
            hf_manager = hf_in['iterative_manager']
        elif group_mode == 1:
            hf_manager = hf_in
        else:
            msg = 'Unrecognized option for group mode'
            raise NotImplementedError(msg)
        if not isinstance(hf_manager, h5py.Group):
            msg = 'Could not find group ' + group_name + ' in hdf5 file'
            raise TypeError(msg)

        itrn_temp = hf_manager.attrs['itrn']
        assert isinstance(itrn_temp, (int, np.integer))
        self._itrn = int(itrn_temp)
        n_full_converged_temp = hf_manager.attrs['n_full_converged']
        assert isinstance(n_full_converged_temp, (int, np.integer))
        self._n_full_converged = int(n_full_converged_temp)

        assert hf_manager.attrs['creator_name'] == self.__class__.__name__, 'incorrect creator name found in hdf5 file'
        assert hf_manager.attrs['inclusion_state_name'] == self.bis.__class__.__name__, 'incorrect inclusion state name found in hdf5 file'
        assert hf_manager.attrs['fit_state_name'] == self.fit_state.__class__.__name__, 'incorrect fit state name found in hdf5 file'
        assert hf_manager.attrs['noise_manager_name'] == self.noise_manager.__class__.__name__, 'incorrect noise manager name found in hdf5 file'
        assert hf_manager.attrs['ic_name'] == self.ic.__class__.__name__, 'incorrect iteration config name found in hdf5 file'

        hf_ic = hf_manager['ic']
        if not isinstance(hf_ic, h5py.Group):
            msg = 'Could not find group ic in hdf5 file'
            raise TypeError(msg)

        for key in self.ic._fields:
            assert np.all(getattr(self.ic, key) == hf_ic.attrs[key]), f'ic attribute {key} does not match saved value'

        self.bis.load_hdf5(hf_manager)
        self.noise_manager.load_hdf5(hf_manager)
        self.fit_state.load_hdf5(hf_manager)

    @override
    def advance_state(self) -> None:
        """Advance the state of the iteration and determine whether convergence is achieved."""
        t0n = perf_counter()

        self.bis.advance_state()

        t1n = perf_counter()

        # decide whether convergence has been achieved before calling advance_state
        inclusion_data = self.bis.convergence_decision_helper()
        _ = self.fit_state.bright_convergence_decision(inclusion_data)
        _ = self.fit_state.faint_convergence_decision(inclusion_data)

        self.fit_state.advance_state()
        self.noise_manager.advance_state()

        t2n = perf_counter()

        print('made bg %3d in time %7.3fs fit time %7.3fs' % (self._itrn, t1n - t0n, t2n - t1n))

    @override
    def log_state(self) -> None:
        """Perform any internal logging that should be done after advance_state is run."""
        self.bis.log_state()
        self.fit_state.log_state()
        self.noise_manager.log_state()

    @override
    def loop_finalize(self) -> None:
        """Perform any logic desired after convergence has been achieved and the loop ends."""
        self.bis.loop_finalize()
        self.fit_state.loop_finalize()
        self.noise_manager.loop_finalize()

    def check_done(self) -> bool:
        """Check whether the fitting procedure can bet stopped."""
        if self.fit_state.get_bright_converged() and self.fit_state.get_faint_converged():
            print('result fully converged at ' + str(self._itrn) + ', no further iterations needed')
            self._n_full_converged = self._itrn
            return True
        return False

    @override
    def print_report(self) -> None:
        """Do any printing desired after convergence has been achieved and the loop ends."""
        self.bis.print_report()
        self.fit_state.print_report()
        self.noise_manager.print_report()

    @override
    def state_check(self) -> None:
        """Perform any sanity checks that should be performed at the end of each iteration."""
        self.bis.state_check()
        self.fit_state.state_check()
        self.noise_manager.state_check()
