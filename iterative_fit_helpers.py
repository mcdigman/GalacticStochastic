"""helper functions for the iterative fit loops"""

from collections import namedtuple

from time import perf_counter

import numpy as np

from instrument_noise import DiagonalNonstationaryDenseInstrumentNoiseModel


IterationConfig = namedtuple('IterationConfig', ['n_iterations', 'snr_thresh', 'snr_min', 'snr_autosuppress', 'smooth_lengthf', 'smooth_lengtht'])

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

        SAET_tot[itrn+1] = get_smoothed_timevarying_spectrum(wc, signal_full, SAET_m, ic.smooth_lengthf[itrn], ic.smooth_lengtht[itrn])

    return galactic_bg_full, galactic_bg_const, signal_full, SAET_tot, var_suppress, snrs, snrs_tot, noise_AET_dense

def get_smoothed_timevarying_spectrum(wc, signal_full, SAET_m, smooth_lengthf, smooth_lengtht):
    SAET_galactic_bg_smoothf_white = np.zeros((wc.Nt, wc.Nf, wc.NC))
    SAET_galactic_bg_smoothft_white = np.zeros((wc.Nt, wc.Nf, wc.NC))
    SAET_galactic_bg_smooth = np.zeros((wc.Nt, wc.Nf, wc.NC))

    for itrc in range(wc.NC):
        SAET_galactic_bg_white = signal_full[:, :, itrc]**2/SAET_m[:, itrc]
        for itrf in range(wc.Nf):
            rreach = smooth_lengthf//2 - max(itrf-wc.Nf+smooth_lengthf//2+1, 0)
            lreach = smooth_lengthf//2 - max(smooth_lengthf//2-itrf, 0)
            SAET_galactic_bg_smoothf_white[:, itrf, itrc] = np.mean(SAET_galactic_bg_white[:, itrf-lreach:itrf+rreach+1], axis=1)
        for itrt in range(wc.Nt):
            rreach = smooth_lengtht//2 - max(itrt-wc.Nt+smooth_lengtht//2+1, 0)
            lreach = smooth_lengtht//2 - max(smooth_lengtht//2-itrt, 0)
            SAET_galactic_bg_smoothft_white[itrt, :, itrc] = np.mean(SAET_galactic_bg_smoothf_white[itrt-lreach:itrt+rreach+1, :, itrc], axis=0)
        SAET_galactic_bg_smooth[:, :, itrc] = SAET_galactic_bg_smoothft_white[:, :, itrc]*SAET_m[:, itrc]

    return SAET_galactic_bg_smooth

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

# TODO move this function to a different file
def unit_normal_battery(signal, mult=1., sig_thresh=5., A2_cut=2.28, do_assert=True):
    """battery of tests for checking if signal is unit normal white noise"""
    #default anderson darling cutoff of 2.28 is hand selected to
    #give ~1 in 1e5 empirical probablity of false positive for n=64
    #calibration looks about same for n=32 could probably choose better way
    #with current defaults that should make it the most sensitive test
    n_sig = signal.size
    if n_sig == 0:
        return False, 0., 0., 0.

    sig_adjust = signal/mult
    mean_wave = np.mean(sig_adjust)
    std_wave = np.std(sig_adjust)
    std_std_wave = np.std(sig_adjust)*np.sqrt(2/n_sig)

    #anderson darling test statistic assuming true mean and variance are unknown
    sig_sort = np.sort((sig_adjust-mean_wave)/std_wave)
    phis = scipy.stats.norm.cdf(sig_sort)
    A2 = -n_sig-1/n_sig*np.sum((2*np.arange(1, n_sig+1)-1)*np.log(phis)+(2*(n_sig-np.arange(1, n_sig+1))+1)*np.log(1-phis))
    A2Star = A2*(1+4/n_sig-25/n_sig**2)
    print(A2Star, A2_cut)

    test1 = np.abs(mean_wave)/std_wave < sig_thresh
    test2 = np.abs(std_wave-1.)/std_std_wave < sig_thresh
    test3 = A2Star < A2_cut #should be less than cutoff value

    #check mean and variance
    if do_assert:
        assert test1
        assert test2
        assert test3

    return test1 and test2 and test3, A2Star, np.abs(mean_wave)/std_wave, np.abs(std_wave-1.)/std_std_wave

