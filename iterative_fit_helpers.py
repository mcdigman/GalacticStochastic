"""helper functions for the iterative fit loops"""

from collections import namedtuple
from time import perf_counter

import numpy as np
import scipy.stats

from galactic_fit_helpers import get_SAET_cyclostationary_mean
from instrument_noise import DiagonalNonstationaryDenseInstrumentNoiseModel

IterationConfig = namedtuple('IterationConfig', ['n_iterations', 'snr_thresh', 'snr_min', 'snr_cut_bright', 'smooth_lengthf'])
#BGDecomposition = namedtuple('BGDecomposition', ['galactic_floor', 'galactic_below', 'galactic_undecided', 'galactic_above'])

class BGDecomposition():
    """class to handle the internal decomposition of the galactic background"""
    def __init__(self, galactic_floor, galactic_below, galactic_undecided, galactic_above):
        self.galactic_floor = galactic_floor
        self.galactic_below = galactic_below
        self.galactic_undecided = galactic_undecided
        self.galactic_above = galactic_above
        self.galactic_total_cache = None

    def get_galactic_total(self):
        """get the sum of all components of the galactic signal, detectable or not"""
        return self.get_galactic_below_high() + self.galactic_above

    def get_galactic_below_high(self):
        """
        get the upper estimate of the unresolvable signal from the galactic background,
        assuming that the undecided part of the signal *is* part of the unresolvable background
        """
        return self.get_galactic_below_low() + self.galactic_undecided

    def get_galactic_below_low(self):
        """
        get the lower estimate of the unresolvable signal from the galactic background,
        assuming that the undecided part of the signal *is not* part of the unresolvable background
        """
        return self.galactic_floor + self.galactic_below

    def total_signal_consistency_check(self):
        """
        if we have previously cached the total recorded galactic signal,
        check that the total not changed much.
        Otherwise, cache the current total so future runs can check if it has changed
        """
        if self.galactic_total_cache is None:
            assert np.all(self.galactic_below == 0.)
            self.galactic_total_cache = self.get_galactic_total()
        else:
            #check all contributions to the total signal are tracked accurately
            assert np.allclose(self.galactic_total_cache, self.get_galactic_total(), atol=1.e-300, rtol=1.e-6)

    def get_galactic_coadd_resolvable(self):
        """
        get the coadded signal from only bright/resolvable galactic binaries
        """
        return self.galactic_above

    def get_galactic_coadd_undecided(self):
        """
        get the coadded signal from galactic binaries whose status as bright or faint has not yet been decided
        """
        return self.galactic_undecided


def do_preliminary_loop(wc, ic, SAET_tot, n_bin_use, faints_in, waveform_model, params_gb, snrs_tot_upper, galactic_below, noise_realization, SAET_m):
    # TODO make snr_cut_bright and smooth_lengthf an array as a function of iteration
    # TODO make NC controllable; probably not much point in getting T channel snrs
    snrs_upper = np.zeros((ic.n_iterations, n_bin_use, wc.NC))
    brights = np.zeros((ic.n_iterations, n_bin_use), dtype=np.bool_)

    for itrn in range(ic.n_iterations):
        galactic_undecided = np.zeros((wc.Nt*wc.Nf, wc.NC))
        noise_upper = DiagonalNonstationaryDenseInstrumentNoiseModel(SAET_tot[itrn], wc, prune=False)

        t0n = perf_counter()

        for itrb in range(n_bin_use):
            if itrb % 10000 == 0 and itrn == 0:
                tin = perf_counter()
                print("Starting binary # %11d at t=%9.2f s at iteration %4d" % (itrb, (tin - t0n), itrn))

            run_binary_coadd(itrb, faints_in, waveform_model, noise_upper, snrs_upper, snrs_tot_upper, itrn, galactic_below, galactic_undecided, brights, wc, params_gb, ic.snr_min[itrn], ic.snr_cut_bright[itrn])

        t1n = perf_counter()

        print('Finished coadd for iteration %4d at time %9.2f s' % ((itrn, t1n-t0n)))

        galactic_below_high = (galactic_undecided + galactic_below).reshape((wc.Nt, wc.Nf, wc.NC))

        signal_full = galactic_below_high + noise_realization

        SAET_tot[itrn+1], _, _, _, _ = get_SAET_cyclostationary_mean(galactic_below_high, SAET_m, wc, smooth_lengthf=ic.smooth_lengthf[itrn], filter_periods=False, period_list=np.array([]))

    return galactic_below_high, galactic_below, signal_full, SAET_tot, brights, snrs_upper, snrs_tot_upper, noise_upper


def run_binary_coadd(itrb, faints_in, waveform_model, noise_upper, snrs_upper, snrs_tot_upper, itrn, galactic_below, galactic_undecided, brights, wc, params_gb, snr_min, snr_cut_bright):
    if not faints_in[itrb]:
        waveform_model.update_params(params_gb[itrb].copy())
        listT_temp, waveT_temp, NUTs_temp = waveform_model.get_unsorted_coeffs()
        snrs_upper[itrn, itrb] = noise_upper.get_sparse_snrs(NUTs_temp, listT_temp, waveT_temp)
        snrs_tot_upper[itrn, itrb] = np.linalg.norm(snrs_upper[itrn, itrb])
        if itrn == 0 and snrs_tot_upper[0, itrb]<snr_min:
            faints_in[itrb] = True
            for itrc in range(wc.NC):
                galactic_below[listT_temp[itrc, :NUTs_temp[itrc]], itrc] += waveT_temp[itrc, :NUTs_temp[itrc]]
        elif snrs_tot_upper[itrn, itrb]<snr_cut_bright:
            for itrc in range(wc.NC):
                galactic_undecided[listT_temp[itrc, :NUTs_temp[itrc]], itrc] += waveT_temp[itrc, :NUTs_temp[itrc]]
        else:
            brights[itrn, itrb] = True


# TODO consolidate with the other run_binary_coadd
def run_binary_coadd2(waveform_model, params_gb, brights, faints_old, faints_cur, snrs_lower, snrs_upper, snrs_tot_upper, snrs_tot_lower, itrn, itrb, noise_upper, noise_lower, ic, faint_converged, bright_converged, nt_min, nt_max, bgd):
    waveform_model.update_params(params_gb[itrb].copy())

    brights[itrn, itrb], faints_cur[itrn, itrb] = decision_helper(snrs_lower, snrs_upper, snrs_tot_upper, snrs_tot_lower, itrn, itrb, waveform_model, noise_upper, noise_lower, ic, faint_converged, bright_converged, nt_min, nt_max)
    decide_coadd_helper(brights, faints_old, faints_cur, itrn, itrb, bgd, waveform_model, bright_converged)


def decision_helper(snrs_lower, snrs_upper, snrs_tot_upper, snrs_tot_lower, itrn, itrb, waveform_model, noise_upper, noise_lower, ic, faint_converged, bright_converged, nt_min, nt_max):
    listT_temp, waveT_temp, NUTs_temp = waveform_model.get_unsorted_coeffs()
    if not faint_converged[itrn]:
        snrs_lower[itrn, itrb] = noise_lower.get_sparse_snrs(NUTs_temp, listT_temp, waveT_temp, nt_min, nt_max)
        snrs_tot_lower[itrn, itrb] = np.linalg.norm(snrs_lower[itrn, itrb])
        faint_candidate = snrs_tot_lower[itrn, itrb] < ic.snr_min[itrn]
    else:
        snrs_lower[itrn, itrb] = snrs_lower[itrn-1, itrb]
        snrs_tot_lower[itrn, itrb] = snrs_tot_lower[itrn-1, itrb]
        faint_candidate = False

    if not bright_converged[itrn]:
        snrs_upper[itrn, itrb] = noise_upper.get_sparse_snrs(NUTs_temp, listT_temp, waveT_temp, nt_min, nt_max)
        snrs_tot_upper[itrn, itrb] = np.linalg.norm(snrs_upper[itrn, itrb])
        bright_candidate = snrs_tot_upper[itrn, itrb] >= ic.snr_cut_bright[itrn]
    else:
        snrs_upper[itrn, itrb] = snrs_upper[itrn-1, itrb]
        snrs_tot_upper[itrn, itrb] = snrs_tot_upper[itrn-1, itrb]
        bright_candidate = False

    if np.isnan(snrs_tot_upper[itrn, itrb]) or np.isnan(snrs_tot_lower[itrn, itrb]):
        raise ValueError('nan detected in snr at '+str(itrn)+', ' + str(itrb))
    elif bright_candidate and faint_candidate:
        # satifisfied conditions to be eliminated in both directions so just keep it
        bright_loc = False
        faint_loc = False
    elif bright_candidate:
        if snrs_tot_upper[itrn, itrb] > snrs_tot_lower[itrn, itrb]:
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

def decide_coadd_helper(brights, faints_old, faints_cur, itrn, itrb, bgd, waveform_model, bright_converged):
    """add each binary to the correct part of the galactic spectrum, depending on whether it is bright or faint"""
    # the same binary cannot be decided as both bright and faint
    assert not (brights[itrn, itrb] and  faints_cur[itrn, itrb])

    # don't add to anything if the subtraction is already converged and this binary would not require addition
    if bright_converged[itrn] and not faints_cur[itrn, itrb]:
        return

    listT_temp, waveT_temp, NUTs_temp = waveform_model.get_unsorted_coeffs()

    if not faints_cur[itrn, itrb]:
        if brights[itrn, itrb]:
            # binary is bright enough to decide
            for itrc in range(2):
                bgd.galactic_above[listT_temp[itrc, :NUTs_temp[itrc]], itrc] += waveT_temp[itrc, :NUTs_temp[itrc]]
        else:
            # binary neither faint nor bright enough to decide
            for itrc in range(2):
                bgd.galactic_undecided[listT_temp[itrc, :NUTs_temp[itrc]], itrc] += waveT_temp[itrc, :NUTs_temp[itrc]]
    else:
        # binary is faint enough to decide
        if itrn == 1:
            faints_cur[itrn, itrb] = False
            faints_old[itrb] = True
            for itrc in range(2):
                bgd.galactic_floor[listT_temp[itrc, :NUTs_temp[itrc]], itrc] += waveT_temp[itrc, :NUTs_temp[itrc]]
        else:
            for itrc in range(2):
                bgd.galactic_below[listT_temp[itrc, :NUTs_temp[itrc]], itrc] += waveT_temp[itrc, :NUTs_temp[itrc]]


def sustain_snr_helper(faint_converged, snrs_tot_lower, snrs_lower, snrs_tot_upper, snrs_upper, itrn, decided, bright_converged):
    #carry forward any other snr values we still know
    if faint_converged[itrn]:
        snrs_tot_lower[itrn, decided] = snrs_tot_lower[itrn-1, decided]
        snrs_lower[itrn, decided] = snrs_lower[itrn-1, decided]
    if bright_converged[itrn]:
        snrs_tot_upper[itrn, decided] = snrs_tot_upper[itrn-1, decided]
        snrs_upper[itrn, decided] = snrs_upper[itrn-1, decided]


def subtraction_convergence_decision(bgd, bis, fit_state, itrn, SAET_m, wc, ic, period_list, const_only, noise_upper, n_cyclo_switch):

    # short circuit if we have previously decided subtraction is converged
    if fit_state.bright_converged[itrn]:
        fit_state.switch_next[itrn+1] = False
        fit_state.bright_converged[itrn+1] = fit_state.bright_converged[itrn]
        fit_state.faint_converged[itrn+1] = fit_state.faint_converged[itrn]
        bis.n_brights_cur[itrn+1] = bis.n_brights_cur[itrn]
        return noise_upper

    bis.n_brights_cur[itrn+1] = bis.brights[itrn].sum()

    # subtraction is either converged or oscillating
    osc1 = np.all(bis.brights[itrn] == bis.brights[itrn-1])
    osc2 = np.all(bis.brights[itrn] == bis.brights[itrn-2])
    osc3 = np.all(bis.brights[itrn] == bis.brights[itrn-3])
    if itrn > 1 and (fit_state.force_converge[itrn] or (osc1 or osc2 or osc3)):
        assert bis.n_brights_cur[itrn] == bis.n_brights_cur[itrn+1] or fit_state.force_converge[itrn] or osc2 or osc3
        if fit_state.switch_next[itrn]:
            print('subtraction converged at ' + str(itrn))
            fit_state.switch_next[itrn+1] = False
            fit_state.bright_converged[itrn+1] = True
            fit_state.faint_converged[itrn+1] = True
        else:
            if (osc2 or osc3) and not osc1:
                print('cycling detected at ' + str(itrn) + ', doing final check iteration aborting')
                fit_state.force_converge[itrn+1] = True
            print('subtraction predicted initial converged at ' + str(itrn) + ' next iteration will be check iteration')
            fit_state.switch_next[itrn+1] = True
            fit_state.bright_converged[itrn+1] = False
            fit_state.faint_converged[itrn+1] = fit_state.faint_converged[itrn]

        return noise_upper


    # subtraction has not converged, get a new noise model
    fit_state.switch_next[itrn+1] = False
    fit_state.bright_converged[itrn+1] = fit_state.bright_converged[itrn]
    fit_state.faint_converged[itrn+1] = fit_state.faint_converged[itrn]

    # don't use cyclostationary model until specified iteration
    if itrn < n_cyclo_switch:
        filter_periods = False
    else:
        filter_periods = not const_only

    # use higher estimate of galactic bg
    SAET_tot_upper, _, _, _, _ = get_SAET_cyclostationary_mean(bgd.get_galactic_below_high(), SAET_m, wc, ic.smooth_lengthf[itrn], filter_periods=filter_periods, period_list=period_list)
    noise_upper = DiagonalNonstationaryDenseInstrumentNoiseModel(SAET_tot_upper, wc, prune=True)

    SAET_tot_upper = None

    return noise_upper


def addition_convergence_decision(bgd, bis, fit_state, itrn, SAET_m, wc, period_list, const_only, noise_lower, noise_upper, n_const_force, const_converge_change_thresh, smooth_lengthf_targ):
    if not fit_state.faint_converged[itrn+1] or fit_state.switch_next[itrn+1]:
        if itrn < n_const_force:
            fit_state.faint_converged[itrn+1] = fit_state.faint_converged[itrn+1]
        else:
            fit_state.faint_converged[itrn+1] = True
            # need to disable adaption of constant here because after this point the convergence isn't guaranteed to be monotonic
            print('disabled constant adaptation at ' + str(itrn))

        # make sure this will always predict >= snrs to the actual spectrum in use
        # use lower estimate of galactic bg
        SAET_tot_lower, _, _, _, _ = get_SAET_cyclostationary_mean(bgd.get_galactic_below_low(), SAET_m, wc, smooth_lengthf_targ, filter_periods=not const_only, period_list=period_list)
        SAET_tot_lower = np.min([SAET_tot_lower, noise_upper.SAET], axis=0)
        noise_lower = DiagonalNonstationaryDenseInstrumentNoiseModel(SAET_tot_lower, wc, prune=True)
        SAET_tot_lower = None

        bis.n_faints_cur[itrn+1] = bis.faints_cur[itrn].sum()
        if fit_state.switch_next[itrn+1] and fit_state.faint_converged[itrn+1]:
            print('overriding constant convergence to check background model')
            fit_state.switch_next[itrn+1] = fit_state.switch_next[itrn+1]
            fit_state.bright_converged[itrn+1] = fit_state.bright_converged[itrn+1]
            fit_state.switchf_next[itrn+1] = False
            fit_state.faint_converged[itrn+1] = False
        elif bis.n_faints_cur[itrn+1] - bis.n_faints_cur[itrn] < 0:
            if fit_state.bright_converged[itrn+1]:
                fit_state.switch_next[itrn+1] = True
                fit_state.bright_converged[itrn+1] = False
            else:
                fit_state.switch_next[itrn+1] = fit_state.switch_next[itrn+1]
                fit_state.bright_converged[itrn+1] = fit_state.bright_converged[itrn+1]
            fit_state.switchf_next[itrn+1] = False
            fit_state.faint_converged[itrn+1] = False
            print('addition removed values at ' + str(itrn) + ', repeating check iteration')

        elif itrn != 1 and np.abs(bis.n_faints_cur[itrn+1] - bis.n_faints_cur[itrn]) < const_converge_change_thresh:
            if fit_state.switchf_next[itrn+1]:
                fit_state.faint_converged[itrn+1] = True
                fit_state.switchf_next[itrn+1] = False
                print('addition converged at ' + str(itrn))
            else:
                print('near convergence in constant adaption at '+str(itrn), ' doing check iteration')
                fit_state.switchf_next[itrn+1] = False
                fit_state.faint_converged[itrn+1] = False
            fit_state.switch_next[itrn+1] = fit_state.switch_next[itrn+1]
            fit_state.bright_converged[itrn+1] = fit_state.bright_converged[itrn+1]
        else:
            if fit_state.bright_converged[itrn+1]:
                print('addition convergence continuing beyond subtraction, try check iteration')
                fit_state.switchf_next[itrn+1] = False
                fit_state.faint_converged[itrn+1] = False
            else:
                fit_state.switchf_next[itrn+1] = False
                fit_state.faint_converged[itrn+1] = fit_state.faint_converged[itrn+1]

            fit_state.switch_next[itrn+1] = fit_state.switch_next[itrn+1]
            fit_state.bright_converged[itrn+1] = fit_state.bright_converged[itrn+1]

    else:
        fit_state.switchf_next[itrn+1] = False
        fit_state.faint_converged[itrn+1] = fit_state.faint_converged[itrn+1]
        fit_state.switch_next[itrn+1] = fit_state.switch_next[itrn+1]
        fit_state.bright_converged[itrn+1] = fit_state.bright_converged[itrn+1]
        bis.n_faints_cur[itrn+1] = bis.n_faints_cur[itrn]

    return noise_lower
