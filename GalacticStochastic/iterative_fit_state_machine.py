"""run iterative processing of galactic background"""

import numpy as np

from GalacticStochastic.state_manager import StateManager


class IterativeFitState(StateManager):
    def __init__(self, ic):
        self.ic = ic

        self.bright_converged = np.zeros(ic.max_iterations+1, dtype=np.bool_)
        self.faint_converged = np.zeros(ic.max_iterations+1, dtype=np.bool_)
        self.do_faint_check = np.zeros(ic.max_iterations+1, dtype=np.bool_)
        self.force_converge = np.zeros(ic.max_iterations+1, dtype=np.bool_)

        self.bright_converged_bright = np.zeros(ic.max_iterations+1, dtype=np.bool_)
        self.faint_converged_bright = np.zeros(ic.max_iterations+1, dtype=np.bool_)
        self.do_faint_check_bright = np.zeros(ic.max_iterations+1, dtype=np.bool_)
        self.force_converge_bright = np.zeros(ic.max_iterations+1, dtype=np.bool_)

        self.bright_converged_faint = np.zeros(ic.max_iterations+1, dtype=np.bool_)
        self.faint_converged_faint = np.zeros(ic.max_iterations+1, dtype=np.bool_)
        self.do_faint_check_faint = np.zeros(ic.max_iterations+1, dtype=np.bool_)
        self.force_converge_faint = np.zeros(ic.max_iterations+1, dtype=np.bool_)

        self.noise_safe_lower_log = np.zeros(ic.max_iterations+1, dtype=np.bool_)
        self.noise_safe_upper_log = np.zeros(ic.max_iterations+1, dtype=np.bool_)

        self.bright_state_request = (False, False, False, False)
        self.faint_state_request = (False, False, False, False)
        self.current_state = (False, False, False, False)

        self.noise_safe_lower = False
        self.noise_safe_upper = False


        self.itrn = 0

    def set_bright_state_request(self, do_faint_check, bright_converged, faint_converged, force_converge):
        self.bright_state_request = (do_faint_check, bright_converged, faint_converged, force_converge)

    def get_bright_state_request(self):
        return self.bright_state_request

    def set_faint_state_request(self, do_faint_check, bright_converged, faint_converged, force_converge):
        self.faint_state_request = (do_faint_check, bright_converged, faint_converged, force_converge)

    def get_faint_state_request(self):
        return self.faint_state_request

    def get_noise_safe_lower(self):
        return self.noise_safe_lower

    def get_noise_safe_upper(self):
        return self.noise_safe_upper

    def advance_state(self):
        self.current_state = self.faint_state_request
        self.itrn += 1

    def get_state(self):
        return self.current_state

    def get_faint_converged(self):
        return self.current_state[2]

    def get_bright_converged(self):
        return self.current_state[1]

    def get_do_faint_check(self):
        return self.current_state[0]

    def get_force_converge(self):
        return self.current_state[3]

    def log_state(self):
        (do_faint_check, bright_converged, faint_converged, force_converge) = self.current_state

        self.bright_converged[self.itrn] = bright_converged
        self.faint_converged[self.itrn] = faint_converged
        self.do_faint_check[self.itrn] = do_faint_check
        self.force_converge[self.itrn] = force_converge

        (do_faint_check, bright_converged, faint_converged, force_converge) = self.faint_state_request

        self.bright_converged_faint[self.itrn] = bright_converged
        self.faint_converged_faint[self.itrn] = faint_converged
        self.do_faint_check_faint[self.itrn] = do_faint_check
        self.force_converge_faint[self.itrn] = force_converge

        (do_faint_check, bright_converged, faint_converged, force_converge) = self.bright_state_request

        self.bright_converged_bright[self.itrn] = bright_converged
        self.faint_converged_bright[self.itrn] = faint_converged
        self.do_faint_check_bright[self.itrn] = do_faint_check
        self.force_converge_bright[self.itrn] = force_converge

        self.noise_safe_lower_log[self.itrn] = self.noise_safe_lower
        self.noise_safe_upper_log[self.itrn] = self.noise_safe_lower

    def bright_convergence_decision(self, bis):
        (do_faint_check_in, bright_converged_in, faint_converged_in, force_converge_in) = self.get_state()
        noise_safe = True

        # short circuit if we have previously decided bright adaptation is converged
        if bright_converged_in:
            self.set_bright_state_request(False, bright_converged_in, faint_converged_in, False)
            self.noise_safe_upper = noise_safe
            return noise_safe

        # don't check for convergence in first iteration
        if self.itrn > 1:
            # bright adaptation is either converged or oscillating
            cycling, converged_or_cycling, old_match = bis.oscillation_check_helper()
            if force_converge_in or converged_or_cycling:
                delta_brights = bis.delta_bright_check_helper()
                assert delta_brights == 0 or force_converge_in or old_match
                if do_faint_check_in:
                    print('bright adaptation converged at ' + str(self.itrn))
                    self.set_bright_state_request(False, True, True, False)
                else:
                    if cycling:
                        print('cycling detected at ' + str(self.itrn) + ', doing final check iteration aborting')
                        force_converge_loc = True
                    else:
                        force_converge_loc = False
                    print('bright adaptation predicted initial converged at ' + str(self.itrn) + ' next iteration will be check iteration')
                    self.set_bright_state_request(True, False, faint_converged_in, force_converge_loc)

                self.noise_safe_upper = noise_safe
                return noise_safe

        # bright adaptation has not converged, get a new noise model
        noise_safe = False
        self.set_bright_state_request(False, bright_converged_in, faint_converged_in, False)
        self.noise_safe_upper = noise_safe
        return noise_safe

    def faint_convergence_decision(self, bis):
        (do_faint_check_in, bright_converged_in, faint_converged_in, force_converge_in) = self.get_bright_state_request()

        if not faint_converged_in or do_faint_check_in:
            noise_safe = False
            if self.itrn < self.ic.n_min_faint_adapt:
                faint_converged_loc = faint_converged_in
            else:
                faint_converged_loc = True
                # need to disable adaption of faint component here because after this point the convergence isn't guaranteed to be monotonic
                print('disabled faint component adaptation at ' + str(self.itrn))

            delta_faints = bis.delta_faint_check_helper()
            if do_faint_check_in and faint_converged_loc:
                print('overriding faint convergence to check background model')
                self.set_faint_state_request(do_faint_check_in, bright_converged_in, False, force_converge_in)
            elif delta_faints < 0:
                if bright_converged_in:
                    self.set_faint_state_request(True, False, False, force_converge_in)
                else:
                    self.set_faint_state_request(do_faint_check_in, bright_converged_in, False, force_converge_in)
                print('faint adaptation removed values at ' + str(self.itrn) + ', repeating check iteration')

            elif self.itrn > 1 and np.abs(delta_faints) < self.ic.faint_converge_change_thresh:
                print('near convergence in faint adaption at '+str(self.itrn), ' doing check iteration')
                self.set_faint_state_request(do_faint_check_in, bright_converged_in, False, force_converge_in)
            elif bright_converged_in:
                print('faint adaptation convergence continuing beyond bright adaptation, try check iteration')
                self.set_faint_state_request(do_faint_check_in, bright_converged_in, False, force_converge_in)
            else:
                self.set_faint_state_request(do_faint_check_in, bright_converged_in, faint_converged_loc, force_converge_in)

        else:
            noise_safe = True
            self.set_faint_state_request(do_faint_check_in, bright_converged_in, faint_converged_in, force_converge_in)
        self.noise_safe_lower = noise_safe
        return noise_safe

    def state_check(self):
        """anything that needs to be checked about the state"""
        assert self.get_do_faint_check() == self.do_faint_check[self.itrn]
        assert self.get_bright_converged() == self.bright_converged[self.itrn]
        if self.do_faint_check[self.itrn]:
            assert not self.bright_converged[self.itrn]

    def loop_finalize(self):
        """Perform any logic desired after convergence has been achieved and the loop ends"""
        pass

    def print_report(self):
        """Do any printing desired after convergence has been achieved and the loop ends"""
        pass
