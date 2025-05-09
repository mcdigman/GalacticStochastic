"""helper functions for the iterative fit loops"""

from collections import namedtuple
from time import perf_counter

import numpy as np
import scipy.stats

from galactic_fit_helpers import get_SAET_cyclostationary_mean
from instrument_noise import DiagonalNonstationaryDenseInstrumentNoiseModel

IterationConfig = namedtuple('IterationConfig', ['n_iterations', 'snr_thresh', 'snr_min', 'snr_autosuppress', 'smooth_lengthf', 'smooth_lengtht'])
BGDecomposition = namedtuple('BGDecomposition', ['galactic_bg_const_base', 'galactic_bg_const', 'galactic_bg', 'galactic_bg_suppress'])

def do_preliminary_loop(wc, ic, SAET_tot, n_bin_use, const_suppress_in, waveT_ini, params_gb, snrs_tot, galactic_bg_const, noise_realization, SAET_m):
    # TODO make snr_autosuppress and smooth_lengthf an array as a function of iteration
    # TODO make NC controllable; probably not much point in getting T channel snrs
    snrs = np.zeros((ic.n_iterations, n_bin_use, wc.NC))
    var_suppress = np.zeros((ic.n_iterations, n_bin_use), dtype=np.bool_)

    for itrn in range(ic.n_iterations):
        galactic_bg = np.zeros((wc.Nt*wc.Nf, wc.NC))
        noise_AET_dense = DiagonalNonstationaryDenseInstrumentNoiseModel(SAET_tot[itrn], wc, prune=False)

        t0n = perf_counter()

        for itrb in range(n_bin_use):
            if itrb % 10000 == 0 and itrn == 0:
                tin = perf_counter()
                print("Starting binary # %11d at t=%9.2f s at iteration %4d" % (itrb, (tin - t0n), itrn))

            run_binary_coadd(itrb, const_suppress_in, waveT_ini, noise_AET_dense, snrs, snrs_tot, itrn, galactic_bg_const, galactic_bg, var_suppress, wc, params_gb, ic.snr_min[itrn], ic.snr_autosuppress[itrn])

        t1n = perf_counter()

        print('Finished coadd for iteration %4d at time %9.2f s' % ((itrn, t1n-t0n)))

        galactic_bg_full = (galactic_bg + galactic_bg_const).reshape((wc.Nt, wc.Nf, wc.NC))

        signal_full = galactic_bg_full + noise_realization

        #SAET_tot[itrn+1] = get_smoothed_timevarying_spectrum(wc, galactic_bg_full, signal_full, SAET_m, ic.smooth_lengthf[itrn], ic.smooth_lengtht[itrn])
        SAET_tot[itrn+1], _, _, _, _ = get_SAET_cyclostationary_mean(galactic_bg_full, SAET_m, wc, smooth_lengthf=ic.smooth_lengthf[itrn], filter_periods=False, period_list=np.array([]))

    return galactic_bg_full, galactic_bg_const, signal_full, SAET_tot, var_suppress, snrs, snrs_tot, noise_AET_dense

#def get_smoothed_timevarying_spectrum(wc, galactic_bg_full, signal_full, SAET_m, smooth_lengthf, smooth_lengtht):
#    SAET_galactic_bg_smoothf_white = np.zeros((wc.Nt, wc.Nf, wc.NC))
#    SAET_galactic_bg_smoothft_white = np.zeros((wc.Nt, wc.Nf, wc.NC))
#    SAET_galactic_bg_smooth = np.zeros((wc.Nt, wc.Nf, wc.NC))
#
#    for itrc in range(wc.NC):
#        SAET_galactic_bg_white = signal_full[:, :, itrc]**2/SAET_m[:, itrc]
#        for itrf in range(wc.Nf):
#            rreach = smooth_lengthf//2 - max(itrf-wc.Nf+smooth_lengthf//2+1, 0)
#            lreach = smooth_lengthf//2 - max(smooth_lengthf//2-itrf, 0)
#            SAET_galactic_bg_smoothf_white[:, itrf, itrc] = np.mean(SAET_galactic_bg_white[:, itrf-lreach:itrf+rreach+1], axis=1)
#        for itrt in range(wc.Nt):
#            rreach = smooth_lengtht//2 - max(itrt-wc.Nt+smooth_lengtht//2+1, 0)
#            lreach = smooth_lengtht//2 - max(smooth_lengtht//2-itrt, 0)
#            SAET_galactic_bg_smoothft_white[itrt, :, itrc] = np.mean(SAET_galactic_bg_smoothf_white[itrt-lreach:itrt+rreach+1, :, itrc], axis=0)
#        SAET_galactic_bg_smooth[:, :, itrc] = SAET_galactic_bg_smoothft_white[:, :, itrc]*SAET_m[:, itrc]
#
#    #SAET_alt, _, _ = get_SAET_cyclostationary_mean(galactic_bg_full, SAET_m, wc, smooth_lengthf=smooth_lengthf/10., filter_periods=True, period_list=None)
#    #import matplotlib.pyplot as plt
#    #plt.semilogy(SAET_alt[0, 1:, 0])
#    #plt.semilogy(np.mean(SAET_galactic_bg_smooth[:, 1:, 0], axis=0))
#    #plt.show()
#
#
#    return SAET_galactic_bg_smooth

def run_binary_coadd(itrb, const_suppress_in, waveT_ini, noise_AET_dense, snrs, snrs_tot, itrn, galactic_bg_const, galactic_bg, var_suppress, wc, params_gb, snr_min, snr_autosuppress):
    if not const_suppress_in[itrb]:
        waveT_ini.update_params(params_gb[itrb].copy())
        listT_temp, waveT_temp, NUTs_temp = waveT_ini.get_unsorted_coeffs()
        snrs[itrn, itrb] = noise_AET_dense.get_sparse_snrs(NUTs_temp, listT_temp, waveT_temp)
        snrs_tot[itrn, itrb] = np.linalg.norm(snrs[itrn, itrb])
        if itrn == 0 and snrs_tot[0, itrb]<snr_min:
            const_suppress_in[itrb] = True
            for itrc in range(wc.NC):
                galactic_bg_const[listT_temp[itrc, :NUTs_temp[itrc]], itrc] += waveT_temp[itrc, :NUTs_temp[itrc]]
        elif snrs_tot[itrn, itrb]<snr_autosuppress:
            for itrc in range(wc.NC):
                galactic_bg[listT_temp[itrc, :NUTs_temp[itrc]], itrc] += waveT_temp[itrc, :NUTs_temp[itrc]]
        else:
            var_suppress[itrn, itrb] = True


# TODO consolidate with the other run_binary_coadd
def run_binary_coadd2(waveT_ini, params_gb, var_suppress, const_suppress, const_suppress2, snrs_base, snrs, snrs_tot, snrs_tot_base, itrn, itrb, noise_AET_dense, noise_AET_dense_base, ic, const_converged, var_converged, nt_min, nt_max, bgd):
    waveT_ini.update_params(params_gb[itrb].copy())
    listT_temp, waveT_temp, NUTs_temp = waveT_ini.get_unsorted_coeffs()

    var_suppress[itrn, itrb], const_suppress2[itrn, itrb] = suppress_decision_helper(snrs_base, snrs, snrs_tot, snrs_tot_base, itrn, itrb, listT_temp, NUTs_temp, waveT_temp, noise_AET_dense, noise_AET_dense_base, ic, const_converged, var_converged, nt_min, nt_max)
    suppress_coadd_helper(var_suppress, const_suppress, const_suppress2, itrn, itrb, bgd, listT_temp, NUTs_temp, waveT_temp, var_converged)


def suppress_decision_helper(snrs_base, snrs, snrs_tot, snrs_tot_base, itrn, itrb, listT_temp, NUTs_temp, waveT_temp, noise_AET_dense, noise_AET_dense_base, ic, const_converged, var_converged, nt_min, nt_max):
    if not const_converged[itrn]:
        snrs_base[itrn, itrb] = noise_AET_dense_base.get_sparse_snrs(NUTs_temp, listT_temp, waveT_temp, nt_min, nt_max)
        snrs_tot_base[itrn, itrb] = np.linalg.norm(snrs_base[itrn, itrb])
        thresh_base = snrs_tot_base[itrn, itrb] < ic.snr_min[itrn]
    else:
        snrs_base[itrn, itrb] = snrs_base[itrn-1, itrb]
        snrs_tot_base[itrn, itrb] = snrs_tot_base[itrn-1, itrb]
        thresh_base = False

    if not var_converged[itrn]:
        snrs[itrn, itrb] = noise_AET_dense.get_sparse_snrs(NUTs_temp, listT_temp, waveT_temp, nt_min, nt_max)
        snrs_tot[itrn, itrb] = np.linalg.norm(snrs[itrn, itrb])
        thresh_var = snrs_tot[itrn, itrb] >= ic.snr_autosuppress[itrn]
    else:
        snrs[itrn, itrb] = snrs[itrn-1, itrb]
        snrs_tot[itrn, itrb] = snrs_tot[itrn-1, itrb]
        thresh_var = False

    if np.isnan(snrs_tot[itrn, itrb]) or np.isnan(snrs_tot_base[itrn, itrb]):
        raise ValueError('nan detected in snr at '+str(itrn)+', ' + str(itrb))
    elif thresh_var and thresh_base:
        # satifisfied conditions to be eliminated in both directions so just keep it
        var_suppress_loc = False
        const_suppress_loc = False
    elif thresh_var:
        if snrs_tot[itrn, itrb] > snrs_tot_base[itrn, itrb]:
            # handle case where snr ordering is wrong to prevent oscillation
            var_suppress_loc = False
        else:
            var_suppress_loc = True
        const_suppress_loc = False
    elif thresh_base:
        var_suppress_loc = False
        const_suppress_loc = True
    else:
        var_suppress_loc = False
        const_suppress_loc = False

    return var_suppress_loc, const_suppress_loc

def suppress_coadd_helper(var_suppress, const_suppress, const_suppress2, itrn, itrb, bgd, listT_temp, NUTs_temp, waveT_temp, var_converged):
    """add each binary to the correct part of the galactic spectrum, depending on whether it is bright or faint"""
    # the same binary cannot be suppressed as both bright and faint
    assert not (var_suppress[itrn, itrb] and  const_suppress2[itrn, itrb])

    # don't add to anything if the subtraction is already converged and this binary would not require addition
    if var_converged[itrn] and not const_suppress2[itrn, itrb]:
        return

    if not const_suppress2[itrn, itrb]:
        if var_suppress[itrn, itrb]:
            # binary is bright enough to suppress
            for itrc in range(2):
                bgd.galactic_bg_suppress[listT_temp[itrc, :NUTs_temp[itrc]], itrc] += waveT_temp[itrc, :NUTs_temp[itrc]]
        else:
            # binary neither faint nor bright enough to suppress
            for itrc in range(2):
                bgd.galactic_bg[listT_temp[itrc, :NUTs_temp[itrc]], itrc] += waveT_temp[itrc, :NUTs_temp[itrc]]
    else:
        # binary is faint enough to suppress
        if itrn == 1:
            const_suppress2[itrn, itrb] = False
            const_suppress[itrb] = True
            for itrc in range(2):
                bgd.galactic_bg_const_base[listT_temp[itrc, :NUTs_temp[itrc]], itrc] += waveT_temp[itrc, :NUTs_temp[itrc]]
        else:
            for itrc in range(2):
                bgd.galactic_bg_const[listT_temp[itrc, :NUTs_temp[itrc]], itrc] += waveT_temp[itrc, :NUTs_temp[itrc]]


def sustain_snr_helper(const_converged, snrs_tot_base, snrs_base, snrs_tot, snrs, itrn, suppressed, var_converged):
    #carry forward any other snr values we still know
    if const_converged[itrn]:
        snrs_tot_base[itrn, suppressed] = snrs_tot_base[itrn-1, suppressed]
        snrs_base[itrn, suppressed] = snrs_base[itrn-1, suppressed]
    if var_converged[itrn]:
        snrs_tot[itrn, suppressed] = snrs_tot[itrn-1, suppressed]
        snrs[itrn, suppressed] = snrs[itrn-1, suppressed]

def total_signal_consistency_check(galactic_full_signal, bgd, itrn):
    if itrn == 1:
        assert np.all(bgd.galactic_bg_const == 0.)
        galactic_full_signal[:] = bgd.galactic_bg_const_base + bgd.galactic_bg_const + bgd.galactic_bg + bgd.galactic_bg_suppress
    else:
        #check all contributions to the total signal are tracked accurately
        assert np.allclose(galactic_full_signal, bgd.galactic_bg_const_base + bgd.galactic_bg_const + bgd.galactic_bg + bgd.galactic_bg_suppress, atol=1.e-300, rtol=1.e-6)

def subtraction_convergence_decision(bgd, var_suppress, itrn, force_converge, n_var_suppressed, switch_next, var_converged, const_converged, SAET_m, wc, ic, period_list1, const_only, noise_AET_dense, n_cyclo_switch, SAET_tot_cur):

    # short circuit if we have previously decided subtraction is converged
    if var_converged[itrn]:
        switch_next[itrn+1] = False
        var_converged[itrn+1] = var_converged[itrn]
        const_converged[itrn+1] = const_converged[itrn]
        n_var_suppressed[itrn+1] = n_var_suppressed[itrn]
        return noise_AET_dense, SAET_tot_cur

    galactic_bg_res = bgd.galactic_bg + bgd.galactic_bg_const + bgd.galactic_bg_const_base
    n_var_suppressed[itrn+1] = var_suppress[itrn].sum()

    # subtraction is either converged or oscillating
    if itrn > 1 and (force_converge[itrn] or (np.all(var_suppress[itrn] == var_suppress[itrn-1]) or np.all(var_suppress[itrn] == var_suppress[itrn-2]) or np.all(var_suppress[itrn] == var_suppress[itrn-3]))):
        assert n_var_suppressed[itrn] == n_var_suppressed[itrn+1] or force_converge[itrn] or np.all(var_suppress[itrn] == var_suppress[itrn-2]) or np.all(var_suppress[itrn] == var_suppress[itrn-3])
        if switch_next[itrn]:
            print('subtraction converged at ' + str(itrn))
            switch_next[itrn+1] = False
            var_converged[itrn+1] = True
            const_converged[itrn+1] = True
        else:
            if (np.all(var_suppress[itrn] == var_suppress[itrn-2]) or np.all(var_suppress[itrn] == var_suppress[itrn-3])) and not np.all(var_suppress[itrn] == var_suppress[itrn-1]):
                print('cycling detected at ' + str(itrn) + ', doing final check iteration aborting')
                force_converge[itrn+1] = True
            print('subtraction predicted initial converged at ' + str(itrn) + ' next iteration will be check iteration')
            switch_next[itrn+1] = True
            var_converged[itrn+1] = False
            const_converged[itrn+1] = const_converged[itrn]

        return noise_AET_dense, SAET_tot_cur


    # subtraction has not converged, get a new noise model
    switch_next[itrn+1] = False
    var_converged[itrn+1] = var_converged[itrn]
    const_converged[itrn+1] = const_converged[itrn]

    # don't use cyclostationary model until specified iteration
    if itrn < n_cyclo_switch:
        SAET_tot_cur, _, _, _, _ = get_SAET_cyclostationary_mean(galactic_bg_res, SAET_m, wc, ic.smooth_lengthf[itrn], filter_periods=False, period_list=period_list1)
    else:
        SAET_tot_cur, _, _, _, _ = get_SAET_cyclostationary_mean(galactic_bg_res, SAET_m, wc, ic.smooth_lengthf[itrn], filter_periods=not const_only, period_list=period_list1)

    noise_AET_dense = DiagonalNonstationaryDenseInstrumentNoiseModel(SAET_tot_cur, wc, prune=True)

    return noise_AET_dense, SAET_tot_cur


def addition_convergence_decision(bgd, itrn, n_const_suppressed, switch_next, var_converged, switchf_next, const_converged, SAET_m, wc, period_list1, const_only, noise_AET_dense_base, SAET_tot_cur, SAET_tot_base, n_const_force, const_converge_change_thresh, const_suppress2, smooth_lengthf_targ):
    if not const_converged[itrn+1] or switch_next[itrn+1]:
        if itrn < n_const_force:
            #TODO should use smooth_lengthf or smooth_lengthf_targ
            SAET_tot_base, _, _, _, _ = get_SAET_cyclostationary_mean(bgd.galactic_bg_const + bgd.galactic_bg_const_base, SAET_m, wc, smooth_lengthf_targ, filter_periods=not const_only, period_list=period_list1)
            const_converged[itrn+1] = const_converged[itrn+1]
        else:
            SAET_tot_base, _, _, _, _ = get_SAET_cyclostationary_mean(bgd.galactic_bg_const + bgd.galactic_bg_const_base, SAET_m, wc, smooth_lengthf_targ, filter_periods=not const_only, period_list=period_list1)
            const_converged[itrn+1] = True
            # need to disable adaption of constant here because after this point the convergence isn't guaranteed to be monotonic
            print('disabled constant adaptation at ' + str(itrn))

        # make sure this will always predict >= snrs to the actual spectrum in use
        SAET_tot_base = np.min([SAET_tot_base, SAET_tot_cur], axis=0)
        noise_AET_dense_base = DiagonalNonstationaryDenseInstrumentNoiseModel(SAET_tot_base, wc, prune=True)

        n_const_suppressed[itrn+1] = const_suppress2[itrn].sum()
        if switch_next[itrn+1] and const_converged[itrn+1]:
            print('overriding constant convergence to check background model')
            switch_next[itrn+1] = switch_next[itrn+1]
            var_converged[itrn+1] = var_converged[itrn+1]
            switchf_next[itrn+1] = False
            const_converged[itrn+1] = False
        elif n_const_suppressed[itrn+1] - n_const_suppressed[itrn] < 0:
            if var_converged[itrn+1]:
                switch_next[itrn+1] = True
                var_converged[itrn+1] = False
            else:
                switch_next[itrn+1] = switch_next[itrn+1]
                var_converged[itrn+1] = var_converged[itrn+1]
            switchf_next[itrn+1] = False
            const_converged[itrn+1] = False
            print('addition removed values at ' + str(itrn) + ', repeating check iteration')

        elif itrn != 1 and np.abs(n_const_suppressed[itrn+1] - n_const_suppressed[itrn]) < const_converge_change_thresh:
            if switchf_next[itrn+1]:
                const_converged[itrn+1] = True
                switchf_next[itrn+1] = False
                print('addition converged at ' + str(itrn))
            else:
                print('near convergence in constant adaption at '+str(itrn), ' doing check iteration')
                switchf_next[itrn+1] = False
                const_converged[itrn+1] = False
            switch_next[itrn+1] = switch_next[itrn+1]
            var_converged[itrn+1] = var_converged[itrn+1]
        else:
            if var_converged[itrn+1]:
                print('addition convergence continuing beyond subtraction, try check iteration')
                switchf_next[itrn+1] = False
                const_converged[itrn+1] = False
            else:
                switchf_next[itrn+1] = False
                const_converged[itrn+1] = const_converged[itrn+1]

            switch_next[itrn+1] = switch_next[itrn+1]
            var_converged[itrn+1] = var_converged[itrn+1]

    else:
        switchf_next[itrn+1] = False
        const_converged[itrn+1] = const_converged[itrn+1]
        switch_next[itrn+1] = switch_next[itrn+1]
        var_converged[itrn+1] = var_converged[itrn+1]
        n_const_suppressed[itrn+1] = n_const_suppressed[itrn]

    return noise_AET_dense_base, SAET_tot_base
