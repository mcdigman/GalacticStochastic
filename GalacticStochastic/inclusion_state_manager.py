"""run iterative processing of galactic background"""

import numpy as np


class BinaryInclusionState():
    def __init__(self, max_iterations, n_bin_use, wc, faints_old):
        self.snrs_upper = np.zeros((max_iterations, n_bin_use, wc.NC))
        self.snrs_lower = np.zeros((max_iterations, n_bin_use, wc.NC))
        self.snrs_tot_lower = np.zeros((max_iterations, n_bin_use))
        self.snrs_tot_upper = np.zeros((max_iterations, n_bin_use))
        self.brights = np.zeros((max_iterations, n_bin_use), dtype=np.bool_)
        self.decided = np.zeros((max_iterations, n_bin_use), dtype=np.bool_)
        self.faints_cur = np.zeros((max_iterations, n_bin_use), dtype=np.bool_)

        self.n_faints_cur = np.zeros(max_iterations+1, dtype=np.int64)
        self.n_brights_cur = np.zeros(max_iterations+1, dtype=np.int64)

        self.faints_old = faints_old

    def sustain_snr_helper(self, itrn, fit_state):
        # carry forward any other snr values we still know
        if fit_state.faint_converged[itrn]:
            assert itrn > 1
            self.snrs_tot_lower[itrn, self.decided[itrn]] = self.snrs_tot_lower[itrn-1, self.decided[itrn]]
            self.snrs_lower[itrn, self.decided[itrn]] = self.snrs_lower[itrn-1, self.decided[itrn]]

        if fit_state.bright_converged[itrn]:
            assert itrn > 1
            self.snrs_tot_upper[itrn, self.decided[itrn]] = self.snrs_tot_upper[itrn-1, self.decided[itrn]]
            self.snrs_upper[itrn, self.decided[itrn]] = self.snrs_upper[itrn-1, self.decided[itrn]]

    def oscillation_check_helper(self, itrn):

        osc1 = False
        osc2 = False
        osc3 = False

        if itrn > 0:
            osc1 = np.all(self.brights[itrn] == self.brights[itrn-1])
        if itrn > 1:
            osc2 = np.all(self.brights[itrn] == self.brights[itrn-2])
        if itrn > 2:
            osc3 = np.all(self.brights[itrn] == self.brights[itrn-3])

        converged_or_cycling = osc1 or osc2 or osc3
        old_match = osc2 or osc3
        cycling = old_match and not osc1
        return cycling, converged_or_cycling, old_match

    def run_binary_coadd(self, waveform_model, params_gb, itrn, itrb, noise_upper, noise_lower, ic, fit_state, nt_min, nt_max, bgd):
        waveform_model.update_params(params_gb[itrb].copy())

        self.snr_storage_helper(itrn, itrb, waveform_model, noise_upper, noise_lower, fit_state, nt_min, nt_max)
        self.brights[itrn, itrb], self.faints_cur[itrn, itrb] = self.decision_helper(itrn, itrb, ic, fit_state)
        self.decide_coadd_helper(itrn, itrb, bgd, waveform_model, fit_state)

    def snr_storage_helper(self, itrn, itrb, waveform_model, noise_upper, noise_lower, fit_state, nt_min, nt_max):
        listT_temp, waveT_temp, NUTs_temp = waveform_model.get_unsorted_coeffs()

        if not fit_state.get_faint_converged():
            self.snrs_lower[itrn, itrb] = noise_lower.get_sparse_snrs(NUTs_temp, listT_temp, waveT_temp, nt_min, nt_max)
            self.snrs_tot_lower[itrn, itrb] = np.linalg.norm(self.snrs_lower[itrn, itrb])
        else:
            assert itrn > 1
            self.snrs_lower[itrn, itrb] = self.snrs_lower[itrn-1, itrb]
            self.snrs_tot_lower[itrn, itrb] = self.snrs_tot_lower[itrn-1, itrb]

        if not fit_state.get_bright_converged():
            self.snrs_upper[itrn, itrb] = noise_upper.get_sparse_snrs(NUTs_temp, listT_temp, waveT_temp, nt_min, nt_max)
            self.snrs_tot_upper[itrn, itrb] = np.linalg.norm(self.snrs_upper[itrn, itrb])
        else:
            assert itrn > 1
            self.snrs_upper[itrn, itrb] = self.snrs_upper[itrn-1, itrb]
            self.snrs_tot_upper[itrn, itrb] = self.snrs_tot_upper[itrn-1, itrb]

        if np.isnan(self.snrs_tot_upper[itrn, itrb]) or np.isnan(self.snrs_tot_lower[itrn, itrb]):
            raise ValueError('nan detected in snr at ' + str(itrn) + ', ' + str(itrb))

        if ~np.isfinite(self.snrs_tot_upper[itrn, itrb]) or ~np.isfinite(self.snrs_tot_lower[itrn, itrb]):
            raise ValueError('Non-finite value detected in snr at ' + str(itrn) + ', ' + str(itrb))

    def decision_helper(self, itrn, itrb, ic, fit_state):
        if not fit_state.get_faint_converged():
            faint_candidate = self.snrs_tot_lower[itrn, itrb] < ic.snr_min[itrn]
        else:
            faint_candidate = False

        if not fit_state.get_bright_converged():
            bright_candidate = self.snrs_tot_upper[itrn, itrb] >= ic.snr_cut_bright[itrn]
        else:
            bright_candidate = False

        if bright_candidate and faint_candidate:
            # satifisfied conditions to be eliminated in both directions so just keep it
            bright_loc = False
            faint_loc = False
        elif bright_candidate:
            if self.snrs_tot_upper[itrn, itrb] > self.snrs_tot_lower[itrn, itrb]:
                # handle case where snr ordering is wrong to prevent oscillation
                bright_loc = False
            else:
                bright_loc = True
            faint_loc = False
        elif faint_candidate:
            bright_loc = False
            faint_loc = True
        else:
            bright_loc = False
            faint_loc = False

        return bright_loc, faint_loc

    def decide_coadd_helper(self, itrn, itrb, bgd, waveform_model, fit_state):
        """add each binary to the correct part of the galactic spectrum, depending on whether it is bright or faint"""
        # the same binary cannot be decided as both bright and faint
        assert not (self.brights[itrn, itrb] and self.faints_cur[itrn, itrb])

        # don't add to anything if the bright adaptation is already converged and this binary would not be faint
        if fit_state.bright_converged[itrn] and not self.faints_cur[itrn, itrb]:
            return

        listT_temp, waveT_temp, NUTs_temp = waveform_model.get_unsorted_coeffs()

        if not self.faints_cur[itrn, itrb]:
            if self.brights[itrn, itrb]:
                # binary is bright enough to decide
                bgd.add_bright(listT_temp, NUTs_temp, waveT_temp)
            else:
                # binary neither faint nor bright enough to decide
                bgd.add_undecided(listT_temp, NUTs_temp, waveT_temp)
        else:
            # binary is faint enough to decide
            if itrn == 0:
                self.faints_cur[itrn, itrb] = False
                self.faints_old[itrb] = True
                bgd.add_floor(listT_temp, NUTs_temp, waveT_temp)
            else:
                bgd.add_faint(listT_temp, NUTs_temp, waveT_temp)
