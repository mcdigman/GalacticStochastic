"""run iterative processing of galactic background"""

import numpy as np


class IterativeFitState():
    def __init__(self, ic):
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

        self.bright_state_request = (False, False, False, False)
        self.faint_state_request = (False, False, False, False)
        self.current_state = (False, False, False, False)

    def set_bright_state_request(self, do_faint_check, bright_converged, faint_converged, force_converge):
        self.bright_state_request = (do_faint_check, bright_converged, faint_converged, force_converge)

    def get_bright_state_request(self):
        return self.bright_state_request

    def set_faint_state_request(self, do_faint_check, bright_converged, faint_converged, force_converge):
        self.faint_state_request = (do_faint_check, bright_converged, faint_converged, force_converge)

    def get_faint_state_request(self):
        return self.faint_state_request

    def advance_state(self):
        self.current_state = self.faint_state_request

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

    def log_state(self, itrn):
        (do_faint_check, bright_converged, faint_converged, force_converge) = self.current_state

        self.bright_converged[itrn] = bright_converged
        self.faint_converged[itrn] = faint_converged
        self.do_faint_check[itrn] = do_faint_check
        self.force_converge[itrn] = force_converge

        (do_faint_check, bright_converged, faint_converged, force_converge) = self.faint_state_request

        self.bright_converged_faint[itrn] = bright_converged
        self.faint_converged_faint[itrn] = faint_converged
        self.do_faint_check_faint[itrn] = do_faint_check
        self.force_converge_faint[itrn] = force_converge

        (do_faint_check, bright_converged, faint_converged, force_converge) = self.bright_state_request

        self.bright_converged_bright[itrn] = bright_converged
        self.faint_converged_bright[itrn] = faint_converged
        self.do_faint_check_bright[itrn] = do_faint_check
        self.force_converge_bright[itrn] = force_converge

    def bright_convergence_decision(self, bis, itrn):
        (do_faint_check_in, bright_converged_in, faint_converged_in, force_converge_in) = self.get_state()
        noise_safe = True

        # short circuit if we have previously decided bright adaptation is converged
        if bright_converged_in:
            self.set_bright_state_request(False, bright_converged_in, faint_converged_in, False)
            bis.n_brights_cur[itrn+1] = bis.n_brights_cur[itrn]
            return noise_safe

        bis.n_brights_cur[itrn+1] = bis.brights[itrn].sum()

        # don't check for convergence in first iteration
        if itrn > 1:
            # bright adaptation is either converged or oscillating
            cycling, converged_or_cycling, old_match = bis.oscillation_check_helper(itrn)
            if force_converge_in or converged_or_cycling:
                assert bis.n_brights_cur[itrn] == bis.n_brights_cur[itrn+1] or force_converge_in or old_match
                if do_faint_check_in:
                    print('bright adaptation converged at ' + str(itrn))
                    self.set_bright_state_request(False, True, True, False)
                else:
                    if cycling:
                        print('cycling detected at ' + str(itrn) + ', doing final check iteration aborting')
                        force_converge_loc = True
                    else:
                        force_converge_loc = False
                    print('bright adaptation predicted initial converged at ' + str(itrn) + ' next iteration will be check iteration')
                    self.set_bright_state_request(True, False, faint_converged_in, force_converge_loc)

                return noise_safe

        # bright adaptation has not converged, get a new noise model
        noise_safe = False
        self.set_bright_state_request(False, bright_converged_in, faint_converged_in, False)

        return noise_safe

    def faint_convergence_decision(self, bis, itrn, ic):
        (do_faint_check_in, bright_converged_in, faint_converged_in, force_converge_in) = self.get_bright_state_request()

        if not faint_converged_in or do_faint_check_in:
            noise_safe = False
            if itrn < ic.n_min_faint_adapt:
                faint_converged_loc = faint_converged_in
            else:
                faint_converged_loc = True
                # need to disable adaption of faint component here because after this point the convergence isn't guaranteed to be monotonic
                print('disabled faint component adaptation at ' + str(itrn))

            bis.n_faints_cur[itrn+1] = bis.faints_cur[itrn].sum()
            delta_faints = bis.n_faints_cur[itrn+1] - bis.n_faints_cur[itrn]
            if do_faint_check_in and faint_converged_loc:
                print('overriding faint convergence to check background model')
                self.set_faint_state_request(do_faint_check_in, bright_converged_in, False, force_converge_in)
            elif delta_faints < 0:
                if bright_converged_in:
                    self.set_faint_state_request(True, False, False, force_converge_in)
                else:
                    self.set_faint_state_request(do_faint_check_in, bright_converged_in, False, force_converge_in)
                print('faint adaptation removed values at ' + str(itrn) + ', repeating check iteration')

            elif itrn > 1 and np.abs(delta_faints) < ic.faint_converge_change_thresh:
                print('near convergence in faint adaption at '+str(itrn), ' doing check iteration')
                self.set_faint_state_request(do_faint_check_in, bright_converged_in, False, force_converge_in)
            elif bright_converged_in:
                print('faint adaptation convergence continuing beyond bright adaptation, try check iteration')
                self.set_faint_state_request(do_faint_check_in, bright_converged_in, False, force_converge_in)
            else:
                self.set_faint_state_request(do_faint_check_in, bright_converged_in, faint_converged_loc, force_converge_in)

        else:
            noise_safe = True
            self.set_faint_state_request(do_faint_check_in, bright_converged_in, faint_converged_in, force_converge_in)
            bis.n_faints_cur[itrn+1] = bis.n_faints_cur[itrn]

        return noise_safe
