"""helper functions for the iterative fit loops"""

from collections import namedtuple
from time import perf_counter

import numpy as np

from GalacticStochastic.galactic_fit_helpers import \
    get_SAET_cyclostationary_mean
from LisaWaveformTools.instrument_noise import \
    DiagonalNonstationaryDenseInstrumentNoiseModel

IterationConfig = namedtuple('IterationConfig', ['n_iterations', 'snr_thresh', 'snr_min', 'snr_cut_bright', 'smooth_lengthf'])


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
        if itrn == 0 and snrs_tot_upper[0, itrb] < snr_min:
            faints_in[itrb] = True
            for itrc in range(wc.NC):
                galactic_below[listT_temp[itrc, :NUTs_temp[itrc]], itrc] += waveT_temp[itrc, :NUTs_temp[itrc]]
        elif snrs_tot_upper[itrn, itrb] < snr_cut_bright:
            for itrc in range(wc.NC):
                galactic_undecided[listT_temp[itrc, :NUTs_temp[itrc]], itrc] += waveT_temp[itrc, :NUTs_temp[itrc]]
        else:
            brights[itrn, itrb] = True


def bright_convergence_decision(bis, fit_state, itrn):
    (do_faint_check_in, bright_converged_in, faint_converged_in, force_converge_in) = fit_state.get_state()
    noise_safe = True

    # short circuit if we have previously decided bright adaptation is converged
    if bright_converged_in:
        fit_state.set_bright_state_request(False, bright_converged_in, faint_converged_in, False)
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
                fit_state.set_bright_state_request(False, True, True, False)
            else:
                if cycling:
                    print('cycling detected at ' + str(itrn) + ', doing final check iteration aborting')
                    force_converge_loc = True
                else:
                    force_converge_loc = False
                print('bright adaptation predicted initial converged at ' + str(itrn) + ' next iteration will be check iteration')
                fit_state.set_bright_state_request(True, False, faint_converged_in, force_converge_loc)

            return noise_safe

    # bright adaptation has not converged, get a new noise model
    noise_safe = False
    fit_state.set_bright_state_request(False, bright_converged_in, faint_converged_in, False)

    return noise_safe


def faint_convergence_decision(bis, fit_state, itrn, n_min_faint_adapt, faint_converge_change_thresh):
    (do_faint_check_in, bright_converged_in, faint_converged_in, force_converge_in) = fit_state.get_bright_state_request()

    if not faint_converged_in or do_faint_check_in:
        noise_safe = False
        if itrn < n_min_faint_adapt:
            faint_converged_loc = faint_converged_in
        else:
            faint_converged_loc = True
            # need to disable adaption of faint component here because after this point the convergence isn't guaranteed to be monotonic
            print('disabled faint component adaptation at ' + str(itrn))

        bis.n_faints_cur[itrn+1] = bis.faints_cur[itrn].sum()
        delta_faints = bis.n_faints_cur[itrn+1] - bis.n_faints_cur[itrn]
        if do_faint_check_in and faint_converged_loc:
            print('overriding faint convergence to check background model')
            fit_state.set_faint_state_request(do_faint_check_in, bright_converged_in, False, force_converge_in)
        elif delta_faints < 0:
            if bright_converged_in:
                fit_state.set_faint_state_request(True, False, False, force_converge_in)
            else:
                fit_state.set_faint_state_request(do_faint_check_in, bright_converged_in, False, force_converge_in)
            print('faint adaptation removed values at ' + str(itrn) + ', repeating check iteration')

        elif itrn != 1 and np.abs(delta_faints) < faint_converge_change_thresh:
            print('near convergence in faint adaption at '+str(itrn), ' doing check iteration')
            fit_state.set_faint_state_request(do_faint_check_in, bright_converged_in, False, force_converge_in)
        elif bright_converged_in:
            print('faint adaptation convergence continuing beyond bright adaptation, try check iteration')
            fit_state.set_faint_state_request(do_faint_check_in, bright_converged_in, False, force_converge_in)
        else:
            fit_state.set_faint_state_request(do_faint_check_in, bright_converged_in, faint_converged_loc, force_converge_in)

    else:
        noise_safe = True
        fit_state.set_faint_state_request(do_faint_check_in, bright_converged_in, faint_converged_in, force_converge_in)
        bis.n_faints_cur[itrn+1] = bis.n_faints_cur[itrn]

    return noise_safe


def new_noise_helper(noise_safe_upper, noise_safe_lower, noise_upper, noise_lower, itrn, n_cyclo_switch, stat_only, SAET_m, wc, ic, bgd, period_list, smooth_lengthf_targ):
    if not noise_safe_upper:
        assert not noise_safe_lower

        # don't use cyclostationary model until specified iteration
        if itrn < n_cyclo_switch:
            filter_periods = False
        else:
            filter_periods = not stat_only

        # use higher estimate of galactic bg
        SAET_tot_upper = bgd.get_S_below_high(SAET_m, wc, ic.smooth_lengthf[itrn], filter_periods, period_list)
        noise_upper = DiagonalNonstationaryDenseInstrumentNoiseModel(SAET_tot_upper, wc, prune=True)

        SAET_tot_upper = None

    if not noise_safe_lower:
        # make sure this will always predict >= snrs to the actual spectrum in use
        # use lower estimate of galactic bg
        filter_periods = not stat_only
        SAET_tot_lower = bgd.get_S_below_low(SAET_m, wc, smooth_lengthf_targ, filter_periods, period_list)
        SAET_tot_lower = np.min([SAET_tot_lower, noise_upper.SAET], axis=0)
        noise_lower = DiagonalNonstationaryDenseInstrumentNoiseModel(SAET_tot_lower, wc, prune=True)
        SAET_tot_lower = None

    return noise_upper, noise_lower
