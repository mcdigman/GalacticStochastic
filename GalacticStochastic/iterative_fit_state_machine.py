"""State machine that handles the state of the iterator, deciding whether it has converged or not"""

from typing import TYPE_CHECKING, override

import h5py
import numpy as np

from GalacticStochastic.iteration_config import IterationConfig
from GalacticStochastic.state_manager import StateManager

if TYPE_CHECKING:
    from numpy.typing import NDArray


class IterativeFitState(StateManager):
    """State machine that handles the state of the iterator"""

    def __init__(self, ic: IterationConfig, preprocess_mode: int = 0) -> None:
        """Create the state machine object"""
        self._ic: IterationConfig = ic
        self._preprocess_mode: int = preprocess_mode

        if self._ic.fit_state_storage_mode != 0:
            msg = 'Unrecognized option for fit state storage mode'
            raise ValueError(msg)

        if self._preprocess_mode == 0:
            # do not do preprocess mode
            self.n_itr_cut: int = ic.max_iterations
        elif self._preprocess_mode == 1:
            # do standard preprocessing
            self.n_itr_cut = 1
        elif self._preprocess_mode == 2:
            # reprocess a background that has already been process with a new snr
            self.n_itr_cut = 1
        else:
            msg = 'Unrecognized option for preprocessing mode'
            raise ValueError(msg)

        # for storing past states
        self._bright_converged: NDArray[np.bool_] = np.zeros(self.n_itr_cut + 1, dtype=np.bool_)
        self._faint_converged: NDArray[np.bool_] = np.zeros(self.n_itr_cut + 1, dtype=np.bool_)
        self._do_faint_check: NDArray[np.bool_] = np.zeros(self.n_itr_cut + 1, dtype=np.bool_)
        self._force_converge: NDArray[np.bool_] = np.zeros(self.n_itr_cut + 1, dtype=np.bool_)

        self._bright_converged_bright: NDArray[np.bool_] = np.zeros(self.n_itr_cut + 1, dtype=np.bool_)
        self._faint_converged_bright: NDArray[np.bool_] = np.zeros(self.n_itr_cut + 1, dtype=np.bool_)
        self._do_faint_check_bright: NDArray[np.bool_] = np.zeros(self.n_itr_cut + 1, dtype=np.bool_)
        self._force_converge_bright: NDArray[np.bool_] = np.zeros(self.n_itr_cut + 1, dtype=np.bool_)

        self._bright_converged_faint: NDArray[np.bool_] = np.zeros(self.n_itr_cut + 1, dtype=np.bool_)
        self._faint_converged_faint: NDArray[np.bool_] = np.zeros(self.n_itr_cut + 1, dtype=np.bool_)
        self._do_faint_check_faint: NDArray[np.bool_] = np.zeros(self.n_itr_cut + 1, dtype=np.bool_)
        self._force_converge_faint: NDArray[np.bool_] = np.zeros(self.n_itr_cut + 1, dtype=np.bool_)

        self._noise_safe_lower_log: NDArray[np.bool_] = np.zeros(self.n_itr_cut + 1, dtype=np.bool_)
        self._noise_safe_upper_log: NDArray[np.bool_] = np.zeros(self.n_itr_cut + 1, dtype=np.bool_)

        self._bright_state_request: tuple[bool, bool, bool, bool] = (False, False, False, False)
        self._faint_state_request: tuple[bool, bool, bool, bool] = (False, False, False, False)
        self._current_state: tuple[bool, bool, bool, bool] = (False, False, False, False)

        self._noise_safe_lower: bool = False
        self._noise_safe_upper: bool = False

        self._itrn: int = 0

    @override
    def store_hdf5(self, hf_in: h5py.Group, *, group_name: str = 'fit_state', group_mode: int = 0) -> h5py.Group:
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
        hf_state.attrs['n_itr_cut'] = self.n_itr_cut
        hf_state.attrs['preprocess_mode'] = self._preprocess_mode
        hf_state.attrs['creator_name'] = self.__class__.__name__
        hf_state.attrs['ic_name'] = self._ic.__class__.__name__

        # iterative fit related constants
        hf_ic = hf_state.create_group('ic')
        for key in self._ic._fields:
            hf_ic.attrs[key] = getattr(self._ic, key)

        return hf_state

    @property
    def bright_state_request(self) -> tuple[bool, bool, bool, bool]:
        """Get what we wanted the state to be after bright_convergence_decision"""
        return self._bright_state_request

    @bright_state_request.setter
    def bright_state_request(self, value: tuple[bool, bool, bool, bool]) -> None:
        """Set what we want the state to be after bright_convergence_decision"""
        (do_faint_check, bright_converged, faint_converged, force_converge) = value
        self._bright_state_request = (do_faint_check, bright_converged, faint_converged, force_converge)

    @property
    def faint_state_request(self) -> tuple[bool, bool, bool, bool]:
        """Get what we want the state to be after faint_convergence_decision"""
        return self._faint_state_request

    @faint_state_request.setter
    def faint_state_request(self, value: tuple[bool, bool, bool, bool]) -> None:
        """Set what we want the state to be after faint_convergence_decision"""
        (do_faint_check, bright_converged, faint_converged, force_converge) = value
        self._faint_state_request = (do_faint_check, bright_converged, faint_converged, force_converge)

    def get_noise_safe_lower(self) -> bool:
        """Get whether the lower noise background would need to be updated to handle the most recent state change"""
        return self._noise_safe_lower

    def get_noise_safe_upper(self) -> bool:
        """Get whether the upper noise background would need to be updated to handle the most recent state change"""
        return self._noise_safe_upper

    def get_n_itr_cut(self) -> int:
        """Get the maximum number of iterations that are currently allowed"""
        return self.n_itr_cut

    def get_preprocess_mode(self) -> int:
        """Get whether we are currently in pre-processing mode"""
        return self._preprocess_mode

    @override
    def advance_state(self) -> None:
        """Set the current state for the next iteration to be the state requested after faint_convergence_decision"""
        self._current_state = self.faint_state_request
        self._itrn += 1

    def get_state(self) -> tuple[bool, bool, bool, bool]:
        """Get the current state of the state machine"""
        return self._current_state

    def get_faint_converged(self) -> bool:
        """Get whether the faint binaries are converged"""
        return self._current_state[2]

    def get_bright_converged(self) -> bool:
        """Get whether the bright binaries are converged"""
        return self._current_state[1]

    def get_do_faint_check(self) -> bool:
        """Get whether the next iteration is a check iteration for the faint background"""
        return self._current_state[0]

    def get_force_converge(self) -> bool:
        """Get whether we are trying to force convergence"""
        return self._current_state[3]

    @override
    def log_state(self) -> None:
        """Store the state in arrays for diagnostic purposes"""
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
        """Make a decision about whether the bright binaries are converged; needs a BinaryInclusionState object"""
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
                    'bright adaptation predicted initial converged at '
                    + str(self._itrn)
                    + ' next iteration will be check iteration',
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
        """Make a decision about whether the faint binaries are converged; needs a BinaryInclusionState object"""
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
        """Check some things we expect to be true about the state given current rules"""
        assert self.get_do_faint_check() == self._do_faint_check[self._itrn]
        assert self.get_bright_converged() == self._bright_converged[self._itrn]
        if self._do_faint_check[self._itrn]:
            assert not self._bright_converged[self._itrn]

    @override
    def loop_finalize(self) -> None:
        """Perform any logic desired after convergence has been achieved and the loop ends"""
        return

    @override
    def print_report(self) -> None:
        """Do any printing desired after convergence has been achieved and the loop ends"""
        return

    def get_bright_converged_old(self) -> bool:
        assert self._itrn > 0
        return self._bright_converged[self._itrn - 1]

    def get_faint_converged_old(self) -> bool:
        assert self._itrn > 0
        return self._faint_converged[self._itrn - 1]
