"""State machine that handles the state of the iterator, deciding whether it has converged or not."""

from typing import TYPE_CHECKING, override

import h5py
import numpy as np

from GalacticStochastic.iteration_config import IterationConfig
from GalacticStochastic.state_manager import StateManager

if TYPE_CHECKING:
    from numpy.typing import NDArray


class IterativeFitState(StateManager):
    """Class to manage the overall state of the iterative fitting process."""

    def __init__(self, ic: IterationConfig, preprocess_mode: int = 0) -> None:
        """
        Initialize the IterativeFitState state machine for managing the overall state of the iterative fitting process.

        Parameters
        ----------
        ic : IterationConfig
            Configuration object specifying the parameters for the iterative fit, including maximum iterations and storage mode.
        preprocess_mode : int
            Mode for preprocessing:
            - 0: No preprocessing (default, use full number of iterations).
            - 1: Standard preprocessing (single iteration).
            - 2: Reprocess a background with a new SNR (single iteration).
            Any other value raises a ValueError.

        Raises
        ------
        ValueError
            If an unrecognized fit state storage mode or preprocessing mode is provided.
        """
        self._ic: IterationConfig = ic
        self._preprocess_mode: int = preprocess_mode

        if self._ic.fit_state_storage_mode != 0:
            msg = 'Unrecognized option for fit state storage mode'
            raise ValueError(msg)

        if self._preprocess_mode == 0:
            # do not do preprocess mode
            self._n_itr_cut: int = ic.max_iterations
        elif self._preprocess_mode == 1:
            # do standard preprocessing
            self._n_itr_cut = 1
        elif self._preprocess_mode == 2:
            # reprocess a background that has already been process with a new snr
            self._n_itr_cut = 1
        else:
            msg = 'Unrecognized option for preprocessing mode'
            raise ValueError(msg)

        # for storing past states
        self._bright_converged: NDArray[np.bool_] = np.zeros(self._n_itr_cut + 1, dtype=np.bool_)
        self._faint_converged: NDArray[np.bool_] = np.zeros(self._n_itr_cut + 1, dtype=np.bool_)
        self._do_faint_check: NDArray[np.bool_] = np.zeros(self._n_itr_cut + 1, dtype=np.bool_)
        self._force_converge: NDArray[np.bool_] = np.zeros(self._n_itr_cut + 1, dtype=np.bool_)

        self._bright_converged_bright: NDArray[np.bool_] = np.zeros(self._n_itr_cut + 1, dtype=np.bool_)
        self._faint_converged_bright: NDArray[np.bool_] = np.zeros(self._n_itr_cut + 1, dtype=np.bool_)
        self._do_faint_check_bright: NDArray[np.bool_] = np.zeros(self._n_itr_cut + 1, dtype=np.bool_)
        self._force_converge_bright: NDArray[np.bool_] = np.zeros(self._n_itr_cut + 1, dtype=np.bool_)

        self._bright_converged_faint: NDArray[np.bool_] = np.zeros(self._n_itr_cut + 1, dtype=np.bool_)
        self._faint_converged_faint: NDArray[np.bool_] = np.zeros(self._n_itr_cut + 1, dtype=np.bool_)
        self._do_faint_check_faint: NDArray[np.bool_] = np.zeros(self._n_itr_cut + 1, dtype=np.bool_)
        self._force_converge_faint: NDArray[np.bool_] = np.zeros(self._n_itr_cut + 1, dtype=np.bool_)

        self._noise_safe_lower_log: NDArray[np.bool_] = np.zeros(self._n_itr_cut + 1, dtype=np.bool_)
        self._noise_safe_upper_log: NDArray[np.bool_] = np.zeros(self._n_itr_cut + 1, dtype=np.bool_)

        self._bright_state_request: tuple[bool, bool, bool, bool] = (False, False, False, False)
        self._faint_state_request: tuple[bool, bool, bool, bool] = (False, False, False, False)
        self._current_state: tuple[bool, bool, bool, bool] = (False, False, False, False)

        self._noise_safe_lower: bool = False
        self._noise_safe_upper: bool = False

        self._itrn: int = 0

    @override
    def store_hdf5(self, hf_in: h5py.Group, *, group_name: str = 'fit_state', group_mode: int = 0) -> h5py.Group:
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
            hf_state = hf_in.create_group(group_name)
        elif group_mode == 1:
            hf_state = hf_in
        else:
            msg = 'Unrecognized option for group mode'
            raise NotImplementedError(msg)
        hf_state.attrs['storage_mode'] = self._ic.fit_state_storage_mode
        _ = hf_state.create_dataset('bright_converged', data=self._bright_converged)
        _ = hf_state.create_dataset('faint_converged', data=self._faint_converged)
        _ = hf_state.create_dataset('do_faint_check', data=self._do_faint_check)
        _ = hf_state.create_dataset('force_converge', data=self._force_converge)
        _ = hf_state.create_dataset('bright_converged_bright', data=self._bright_converged_bright)
        _ = hf_state.create_dataset('faint_converged_bright', data=self._faint_converged_bright)
        _ = hf_state.create_dataset('do_faint_check_bright', data=self._do_faint_check_bright)
        _ = hf_state.create_dataset('force_converge_bright', data=self._force_converge_bright)
        _ = hf_state.create_dataset('bright_converged_faint', data=self._bright_converged_faint)
        _ = hf_state.create_dataset('faint_converged_faint', data=self._faint_converged_faint)
        _ = hf_state.create_dataset('do_faint_check_faint', data=self._do_faint_check_faint)
        _ = hf_state.create_dataset('force_converge_faint', data=self._force_converge_faint)
        _ = hf_state.create_dataset('noise_safe_lower_log', data=self._noise_safe_lower_log)
        _ = hf_state.create_dataset('noise_safe_upper_log', data=self._noise_safe_upper_log)
        _ = hf_state.create_dataset('bright_state_request', data=self.bright_state_request)
        _ = hf_state.create_dataset('faint_state_request', data=self.faint_state_request)
        _ = hf_state.create_dataset('current_state', data=self._current_state)
        hf_state.attrs['noise_safe_lower'] = self._noise_safe_lower
        hf_state.attrs['noise_safe_upper'] = self._noise_safe_upper
        hf_state.attrs['itrn'] = self._itrn
        hf_state.attrs['n_itr_cut'] = self._n_itr_cut
        hf_state.attrs['preprocess_mode'] = self._preprocess_mode
        hf_state.attrs['creator_name'] = self.__class__.__name__
        hf_state.attrs['ic_name'] = self._ic.__class__.__name__

        # iterative fit related constants
        hf_ic = hf_state.create_group('ic')
        for key in self._ic._fields:
            hf_ic.attrs[key] = getattr(self._ic, key)

        return hf_state

    @override
    def load_hdf5(self, hf_in: h5py.Group, *, group_name: str = 'fit_state', group_mode: int = 0) -> None:
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
            hf_state = hf_in[group_name]
        elif group_mode == 1:
            hf_state = hf_in
        else:
            msg = 'Unrecognized option for group mode'
            raise NotImplementedError(msg)

        if not isinstance(hf_state, h5py.Group):
            msg = 'Could not find group ' + group_name + ' in hdf5 file'
            raise TypeError(msg)

        storage_mode_temp = hf_state.attrs['storage_mode']
        assert isinstance(storage_mode_temp, (int, np.integer))
        storage_mode = int(storage_mode_temp)
        if storage_mode != self._ic.fit_state_storage_mode:
            msg = 'fit state storage mode in hdf5 file does not match current configuration'
            raise ValueError(msg)

        assert hf_state.attrs['creator_name'] == self.__class__.__name__, 'incorrect creator name found in hdf5 file'
        assert hf_state.attrs['ic_name'] == self._ic.__class__.__name__, 'incorrect iteration config name found in hdf5 file'

        hf_ic = hf_state['ic']
        if not isinstance(hf_ic, h5py.Group):
            msg = 'Could not find group ic in hdf5 file'
            raise TypeError(msg)

        for key in self._ic._fields:
            if not np.all(getattr(self._ic, key) == hf_ic.attrs[key]):
                msg = 'iteration config in hdf5 file does not match current configuration'
                raise ValueError(msg)

        tmp_bright_converged = hf_state['bright_converged']
        assert isinstance(tmp_bright_converged, h5py.Dataset)
        self._bright_converged = tmp_bright_converged[()]

        tmp_faint_converged = hf_state['faint_converged']
        assert isinstance(tmp_faint_converged, h5py.Dataset)
        self._faint_converged = tmp_faint_converged[()]

        tmp_do_faint_check = hf_state['do_faint_check']
        assert isinstance(tmp_do_faint_check, h5py.Dataset)
        self._do_faint_check = tmp_do_faint_check[()]

        tmp_force_converge = hf_state['force_converge']
        assert isinstance(tmp_force_converge, h5py.Dataset)
        self._force_converge = tmp_force_converge[()]

        tmp_bright_converged_bright = hf_state['bright_converged_bright']
        assert isinstance(tmp_bright_converged_bright, h5py.Dataset)
        self._bright_converged_bright = tmp_bright_converged_bright[()]

        tmp_faint_converged_bright = hf_state['faint_converged_bright']
        assert isinstance(tmp_faint_converged_bright, h5py.Dataset)
        self._faint_converged_bright = tmp_faint_converged_bright[()]

        tmp_do_faint_check_bright = hf_state['do_faint_check_bright']
        assert isinstance(tmp_do_faint_check_bright, h5py.Dataset)
        self._do_faint_check_bright = tmp_do_faint_check_bright[()]

        tmp_force_converge_bright = hf_state['force_converge_bright']
        assert isinstance(tmp_force_converge_bright, h5py.Dataset)
        self._force_converge_bright = tmp_force_converge_bright[()]

        tmp_bright_converged_faint = hf_state['bright_converged_faint']
        assert isinstance(tmp_bright_converged_faint, h5py.Dataset)
        self._bright_converged_faint = tmp_bright_converged_faint[()]

        tmp_faint_converged_faint = hf_state['faint_converged_faint']
        assert isinstance(tmp_faint_converged_faint, h5py.Dataset)
        self._faint_converged_faint = tmp_faint_converged_faint[()]

        tmp_do_faint_check_faint = hf_state['do_faint_check_faint']
        assert isinstance(tmp_do_faint_check_faint, h5py.Dataset)
        self._do_faint_check_faint = tmp_do_faint_check_faint[()]

        tmp_force_converge_faint = hf_state['force_converge_faint']
        assert isinstance(tmp_force_converge_faint, h5py.Dataset)
        self._force_converge_faint = tmp_force_converge_faint[()]

        tmp_noise_safe_lower_log = hf_state['noise_safe_lower_log']
        assert isinstance(tmp_noise_safe_lower_log, h5py.Dataset)
        self._noise_safe_lower_log = tmp_noise_safe_lower_log[()]

        tmp_noise_safe_upper_log = hf_state['noise_safe_upper_log']
        assert isinstance(tmp_noise_safe_upper_log, h5py.Dataset)
        self._noise_safe_upper_log = tmp_noise_safe_upper_log[()]

        tmp_bright_state_request = hf_state['bright_state_request']
        assert isinstance(tmp_bright_state_request, h5py.Dataset)
        self.bright_state_request = tuple(tmp_bright_state_request[()])

        tmp_faint_state_request = hf_state['faint_state_request']
        assert isinstance(tmp_faint_state_request, h5py.Dataset)
        self.faint_state_request = tuple(tmp_faint_state_request[()])

        tmp_current_state = hf_state['current_state']
        assert isinstance(tmp_current_state, h5py.Dataset)
        self._current_state = tuple(tmp_current_state[()])

        self._noise_safe_lower = bool(hf_state.attrs['noise_safe_lower'])
        self._noise_safe_upper = bool(hf_state.attrs['noise_safe_upper'])
        itrn_temp = hf_state.attrs['itrn']
        assert isinstance(itrn_temp, (int, np.integer))
        self._itrn = int(itrn_temp)
        n_itr_cut_temp = hf_state.attrs['n_itr_cut']
        assert isinstance(n_itr_cut_temp, (int, np.integer))
        self._n_itr_cut = int(n_itr_cut_temp)
        preprocess_mode_temp = hf_state.attrs['preprocess_mode']
        assert isinstance(preprocess_mode_temp, (int, np.integer))
        self._preprocess_mode = int(preprocess_mode_temp)

    @property
    def preprocess_mode(self) -> int:
        """Get the preprocessing mode.

        Returns
        -------
        int
            The preprocessing mode.
        """
        return self._preprocess_mode

    @property
    def bright_state_request(self) -> tuple[bool, bool, bool, bool]:
        """Get what we wanted the state to be after bright_convergence_decision.

        Returns
        -------
        tuple[bool, bool, bool, bool]
            A tuple of four booleans: (do_faint_check, bright_converged, faint_converged, force_converge).
        """
        return self._bright_state_request

    @bright_state_request.setter
    def bright_state_request(self, value: tuple[bool, bool, bool, bool]) -> None:
        """Set what we want the state to be after bright_convergence_decision.

        Parameters
        ----------
        value : tuple[bool, bool, bool, bool]
            A tuple of four booleans: (do_faint_check, bright_converged, faint_converged, force_converge).
        """
        (do_faint_check, bright_converged, faint_converged, force_converge) = value
        self._bright_state_request = (do_faint_check, bright_converged, faint_converged, force_converge)

    @property
    def faint_state_request(self) -> tuple[bool, bool, bool, bool]:
        """Get what we want the state to be after faint_convergence_decision.

        Returns
        -------
        tuple[bool, bool, bool, bool]
            A tuple of four booleans: (do_faint_check, bright_converged, faint_converged, force_converge).
        """
        return self._faint_state_request

    @faint_state_request.setter
    def faint_state_request(self, value: tuple[bool, bool, bool, bool]) -> None:
        """Set what we want the state to be after faint_convergence_decision.

        Parameters
        ----------
        value : tuple[bool, bool, bool, bool]
            A tuple of four booleans: (do_faint_check, bright_converged, faint_converged, force_converge).
        """
        (do_faint_check, bright_converged, faint_converged, force_converge) = value
        self._faint_state_request = (do_faint_check, bright_converged, faint_converged, force_converge)

    def get_noise_safe_lower(self) -> bool:
        """Get whether the lower noise background would need to be updated to handle the most recent state change.

        Returns
        -------
        bool
            Whether the lower noise background would need to be updated.
        """
        return self._noise_safe_lower

    def get_noise_safe_upper(self) -> bool:
        """Get whether the upper noise background would need to be updated to handle the most recent state change.

        Returns
        -------
        bool
            Whether the upper noise background would need to be updated.
        """
        return self._noise_safe_upper

    def get_n_itr_cut(self) -> int:
        """Get the maximum number of iterations that are currently allowed.

        Returns
        -------
        int
            The maximum number of iterations allowed.
        """
        return self._n_itr_cut

    @override
    def advance_state(self) -> None:
        """Set the current state for the next iteration to be the state requested after faint_convergence_decision."""
        self._current_state = self.faint_state_request
        self._itrn += 1

    def get_state(self) -> tuple[bool, bool, bool, bool]:
        """Get the current state of the state machine.

        Returns
        -------
        tuple[bool, bool, bool, bool]
            A tuple of four booleans: (do_faint_check, bright_converged, faint_converged, force_converge).
        """
        return self._current_state

    def get_faint_converged(self) -> bool:
        """Get whether the faint binaries are converged.

        Returns
        -------
        bool
            Whether the faint binaries are converged.
        """
        return self._current_state[2]

    def get_bright_converged(self) -> bool:
        """Get whether the bright binaries are converged.

        Returns
        -------
        bool
            Whether the bright binaries are converged.
        """
        return self._current_state[1]

    def get_do_faint_check(self) -> bool:
        """Get whether the next iteration is a check iteration for the faint background.

        Returns
        -------
        bool
            Whether the next iteration is a check iteration for the faint background.
        """
        return self._current_state[0]

    def get_force_converge(self) -> bool:
        """Get whether we are trying to force convergence.

        Returns
        -------
        bool
            Whether we are trying to force convergence.
        """
        return self._current_state[3]

    @override
    def log_state(self) -> None:
        """Store the state in arrays for diagnostic purposes."""
        (do_faint_check, bright_converged, faint_converged, force_converge) = self.get_state()

        self._bright_converged[self._itrn] = bright_converged
        self._faint_converged[self._itrn] = faint_converged
        self._do_faint_check[self._itrn] = do_faint_check
        self._force_converge[self._itrn] = force_converge

        (do_faint_check, bright_converged, faint_converged, force_converge) = self.faint_state_request

        self._bright_converged_faint[self._itrn] = bright_converged
        self._faint_converged_faint[self._itrn] = faint_converged
        self._do_faint_check_faint[self._itrn] = do_faint_check
        self._force_converge_faint[self._itrn] = force_converge

        (do_faint_check, bright_converged, faint_converged, force_converge) = self.bright_state_request

        self._bright_converged_bright[self._itrn] = bright_converged
        self._faint_converged_bright[self._itrn] = faint_converged
        self._do_faint_check_bright[self._itrn] = do_faint_check
        self._force_converge_bright[self._itrn] = force_converge

        self._noise_safe_lower_log[self._itrn] = self._noise_safe_lower
        self._noise_safe_upper_log[self._itrn] = self._noise_safe_lower

    def bright_convergence_decision(self, inclusion_data: tuple[tuple[bool, bool, bool], int, int]) -> bool:
        """Make a decision about whether the bright binaries are converged.

        Needs outputs from a BinaryInclusionState object.

        Parameters
        ----------
        inclusion_data : tuple[tuple[bool, bool, bool], int, int]
            A tuple containing:
            - A tuple of three booleans: (cycling, converged_or_cycling, old_match).
            - An integer delta_brights: the change in the number of bright binaries.
            - An integer delta_faints: the change in the number of faint binaries.

        Returns
        -------
        bool
            Whether the bright binaries are converged.
        """
        (do_faint_check_in, bright_converged_in, faint_converged_in, _force_converge_in) = self.get_state()
        ((cycling, converged_or_cycling, old_match), _, delta_brights) = inclusion_data
        noise_safe = True

        # short circuit if we have previously decided bright adaptation is converged
        if bright_converged_in:
            self.bright_state_request = (False, bright_converged_in, faint_converged_in, False)
            self._noise_safe_upper = noise_safe
            return noise_safe

        # don't check for convergence in first iteration
        # bright adaptation is either converged or oscillating
        if self._itrn > 1 and (self.get_force_converge() or converged_or_cycling):
            assert delta_brights == 0 or self.get_force_converge() or old_match
            if do_faint_check_in:
                print('bright adaptation converged at ' + str(self._itrn))
                self.bright_state_request = (False, True, True, False)
            else:
                if cycling:
                    print('cycling detected at ' + str(self._itrn) + ', doing final check iteration aborting')
                    force_converge_loc = True
                else:
                    force_converge_loc = False
                print(
                    'bright adaptation predicted initial converged at ' + str(self._itrn) + ' next iteration will be check iteration',
                )
                self.bright_state_request = (True, False, faint_converged_in, force_converge_loc)

            self._noise_safe_upper = noise_safe
            return noise_safe

        # bright adaptation has not converged, get a new noise model
        noise_safe = False
        self.bright_state_request = (False, bright_converged_in, faint_converged_in, False)
        self._noise_safe_upper = noise_safe
        return noise_safe

    def faint_convergence_decision(self, inclusion_data: tuple[tuple[bool, bool, bool], int, int]) -> bool:
        """Make a decision about whether the faint binaries are converged.

        Needs outputs from a BinaryInclusionState object.

        Parameters
        ----------
        inclusion_data : tuple[tuple[bool, bool, bool], int, int]
            A tuple containing:
            - A tuple of three booleans: (cycling, converged_or_cycling, old_match).
            - An integer delta_brights: the change in the number of bright binaries.
            - An integer delta_faints: the change in the number of faint binaries.

        Returns
        -------
        bool
            Whether the faint binaries are converged.
        """
        (_, delta_faints, _) = inclusion_data
        (do_faint_check_in, bright_converged_in, faint_converged_in, force_converge_in) = self.bright_state_request

        if not faint_converged_in or do_faint_check_in:
            noise_safe = False
            if self._itrn < self._ic.n_min_faint_adapt:
                faint_converged_loc = faint_converged_in
            else:
                faint_converged_loc = True
                # need to disable adaption of faint component here
                # because after this point the convergence isn't guaranteed to be monotonic
                print('disabled faint component adaptation at ' + str(self._itrn))

            if do_faint_check_in and faint_converged_loc:
                print('overriding faint convergence to check background model')
                faint_converged_loc = False
                self.faint_state_request = (
                    do_faint_check_in,
                    bright_converged_in,
                    faint_converged_loc,
                    force_converge_in,
                )
            elif delta_faints < 0:
                faint_converged_loc = False
                if bright_converged_in:
                    do_faint_check_loc = True
                    bright_converged_loc = False
                    self.faint_state_request = (
                        do_faint_check_loc,
                        bright_converged_loc,
                        faint_converged_loc,
                        force_converge_in,
                    )
                else:
                    self.faint_state_request = (
                        do_faint_check_in,
                        bright_converged_in,
                        faint_converged_loc,
                        force_converge_in,
                    )
                print('faint adaptation removed values at ' + str(self._itrn) + ', repeating check iteration')

            elif self._itrn > 1 and np.abs(delta_faints) < self._ic.faint_converge_change_thresh:
                print('near convergence in faint adaption at ' + str(self._itrn), ' doing check iteration')
                faint_converged_loc = False
                self.faint_state_request = (
                    do_faint_check_in,
                    bright_converged_in,
                    faint_converged_loc,
                    force_converge_in,
                )
            elif bright_converged_in:
                print('faint adaptation convergence continuing beyond bright adaptation, try check iteration')
                faint_converged_loc = False
                self.faint_state_request = (
                    do_faint_check_in,
                    bright_converged_in,
                    faint_converged_loc,
                    force_converge_in,
                )
            else:
                self.faint_state_request = (
                    do_faint_check_in,
                    bright_converged_in,
                    faint_converged_loc,
                    force_converge_in,
                )

        else:
            noise_safe = True
            self.faint_state_request = (do_faint_check_in, bright_converged_in, faint_converged_in, force_converge_in)
        self._noise_safe_lower = noise_safe
        return noise_safe

    @override
    def state_check(self) -> None:
        """Check some things we expect to be true about the state given current rules."""
        assert self.get_do_faint_check() == self._do_faint_check[self._itrn]
        assert self.get_bright_converged() == self._bright_converged[self._itrn]
        if self._do_faint_check[self._itrn]:
            assert not self._bright_converged[self._itrn]

    @override
    def loop_finalize(self) -> None:
        """Perform any logic desired after convergence has been achieved and the loop ends."""
        return

    @override
    def print_report(self) -> None:
        """Do any printing desired after convergence has been achieved and the loop ends."""
        return

    def get_bright_converged_old(self) -> bool:
        """Get whether the bright binaries were marked converged in the previous iteration.

        Returns
        -------
        bool
            Whether the bright binaries were marked converged in the previous iteration.
        """
        assert self._itrn > 0
        return self._bright_converged[self._itrn - 1]

    def get_faint_converged_old(self) -> bool:
        """Get whether the faint binaries were marked converged in the previous iteration.

        Returns
        -------
        bool
            Whether the faint binaries were marked converged in the previous iteration.
        """
        assert self._itrn > 0
        return self._faint_converged[self._itrn - 1]
