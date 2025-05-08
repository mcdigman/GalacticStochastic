"""process fits to galactic background"""

from warnings import warn

import numpy as np
import scipy.ndimage
import WDMWaveletTransforms.fft_funcs as fft
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import dual_annealing


import global_const as gc


def SAE_gal_model(f, log10A, log10f2, log10f1, log10fknee, alpha):
    """model from arXiv:2103.14598 for galactic binary confusion noise amplitude"""
    return 10**log10A/2*f**(5/3)*np.exp(-(f/10**log10f1)**alpha)*(1+np.tanh((10**log10fknee-f)/10**log10f2))


def SAE_gal_model_alt(f, A, alpha, beta, kappa, gamma, fknee):
    """model from arXiv:1703.09858 for galactic binary confusion noise amplitude"""
    return A*f**(7/3)*np.exp(-f**alpha+beta*f*np.sin(kappa*f))*(1+np.tanh(gamma*(fknee-f)))


def filter_periods_fft(r_got1, Nt_loc, period_list, wc):
    ts = np.arange(0, Nt_loc)*wc.DT
    wts = 2*np.pi/gc.SECSYEAR*ts
    r_fft1 = np.zeros((wc.Nt, wc.NC))
    amp_got = np.zeros((period_list.size, wc.NC))
    angle_got = np.zeros((period_list.size, wc.NC))
    for itrc in range(0, 2):
        res_fft = fft.rfft(r_got1[:, itrc]-1.)*2/Nt_loc
        abs_fft = np.abs(res_fft)
        angle_fft = -np.angle(res_fft)

        # handle signs for highest and lowest frequency components
        if angle_fft[0] < -0.1:
            abs_fft[0] = - abs_fft[0]

        if angle_fft[-1] < -0.1:
            abs_fft[-1] = - abs_fft[-1]

        rec = 1.+abs_fft[0]/2+np.zeros(Nt_loc)

        for itrk, k in enumerate(period_list):
            idx = np.int64(wc.Tobs/gc.SECSYEAR*k) # TODO handle non-integer k/ not exact mult of Tobs/SECSYEAR
            if np.abs(idx - wc.Tobs/gc.SECSYEAR*k) > 0.01:
                warn('fft based filtering expects periods that are an integer fraction of the observing time: got %10.8f for %10.8f' % (wc.Tobs/gc.SECSYEAR*k, k))
            if k == 0:
                # already adding the constant case above, whether or not it is requested
                amp_got[itrk, itrc] = abs_fft[0]/2
                angle_got[itrk, itrc] = 0.
            elif k == np.int64(gc.SECSYEAR//wc.DT)//2:
                amp_got[itrk,itrc] = abs_fft[-1]/np.sqrt(2.)
                angle_got[itrk, itrc] = np.pi/4.
                rec += amp_got[itrk,itrc]*np.cos(k*wts - angle_got[itrk,itrc])
            else:
                rec += abs_fft[idx]*np.cos(k*wts - angle_fft[idx])
                amp_got[itrk, itrc] = abs_fft[idx]
                angle_got[itrk, itrc] = angle_fft[idx] % (2*np.pi)
        angle_fftm = angle_fft % (2*np.pi)
        # TODO this print statement assumes 8 years always
        print("%5.3f & %5.2f & %5.3f & %5.2f & %5.3f & %5.2f & %5.3f & %5.2f & %5.3f & %5.2f"%(abs_fft[1*8], angle_fftm[1*8], abs_fft[2*8], angle_fftm[2*8], abs_fft[3*8], angle_fftm[3*8], abs_fft[4*8], angle_fftm[4*8], abs_fft[5*8], angle_fftm[5*8]))
        r_fft1[:, itrc] = rec
    return r_fft1, amp_got, angle_got


#TODO needs unit tests
def get_SAET_cyclostationary_mean(galactic_bg, SAET_m, wc, smooth_lengthf=4, filter_periods=False, period_list=None, Nt_loc=-1):
    """ note the smoothing length is the length in *log* frequency, and the input is assumed spaced linearly in frequency"""
    if Nt_loc == -1:
        Nt_loc = wc.Nt
    SAET_pure_in = (galactic_bg.reshape((wc.Nt, wc.Nf, wc.NC)))**2
    SAET_pure_mean = np.mean(SAET_pure_in, axis=0)

    amp_got = None
    angle_got = None

    if not filter_periods:
        rec_use = np.zeros((wc.Nt, wc.NC))+1.
    else:
        r_got1 = np.zeros((wc.Nt, wc.NC))
        SAET_pure_white = SAET_pure_mean/SAET_m

        for itrc in range(0, 2):
            r_eval_mask = SAET_pure_white[:, itrc] > 0.1*np.max(SAET_pure_white[:, itrc])
            r_got1[:, itrc] = np.mean(SAET_pure_in[:, r_eval_mask, itrc]/(SAET_pure_mean[r_eval_mask, itrc] + 1.e-13*np.max(SAET_pure_mean[r_eval_mask, itrc])), axis=1)

        if period_list is None:
            # if no period list is given, do every possible period
            period_list = np.arange(0, np.int64(gc.SECSYEAR//wc.DT)//2+1/int(wc.Tobs/gc.SECSYEAR), 1/int(wc.Tobs/gc.SECSYEAR))


        r_fft1, amp_got, angle_got = filter_periods_fft(r_got1, Nt_loc, period_list, wc)

        rec_use = r_fft1

    SAET_pure_mod = SAET_pure_in.copy()
    for itrc in range(0, 2):
        SAET_pure_mod[:, :, itrc] = (SAET_pure_mod[:, :, itrc].T/rec_use[:, itrc]).T


    SAET_pures = np.mean(SAET_pure_mod, axis=0)

    SAET_pures_smooth2 = np.zeros((wc.Nf, 3))
    SAET_pures_smooth2[0, :] = SAET_pures[0, :]

    log_fs = np.log10(np.arange(1, wc.Nf)*wc.DF)

    interp_mult = 10

    n_f_interp = interp_mult*wc.Nf
    log_fs_interp = np.linspace(np.log10(wc.DF), np.log10(wc.DF*(wc.Nf-1)), n_f_interp)

    for itrc in range(0, 3):
        log_SAE_pure_loc_smooth1 = np.log10(SAET_pures[1:, itrc]+1.e-50)
        log_SAE_interp_loc = InterpolatedUnivariateSpline(log_fs, log_SAE_pure_loc_smooth1, k=3, ext=2)(log_fs_interp)
        log_SAE_interp_loc_smooth = scipy.ndimage.gaussian_filter(log_SAE_interp_loc, smooth_lengthf*interp_mult)
        SAET_pures_smooth2[1:, itrc] = 10**InterpolatedUnivariateSpline(log_fs_interp, log_SAE_interp_loc_smooth, k=3, ext=2)(log_fs)-1.e-50

    SAET_res = np.zeros((wc.Nt, wc.Nf, wc.NC))+SAET_m
    for itrc in range(0, 2):
        SAET_res[:, :, itrc] += np.outer(rec_use[:, itrc], SAET_pures_smooth2[:, itrc])

    assert np.all(np.isfinite(SAET_res))

    return SAET_res, rec_use, SAET_pures_smooth2, amp_got, angle_got


def fit_gb_spectrum_evolve(SAET_goals, fs, fs_report, nt_ranges, offset, wc):
    a1 = -0.25#-0.15
    b1 = -2.70#-0.37
    ak = -0.27#-2.72
    bk = -2.47#-2.49
    log10A = np.log10(7.e-39)
    log10f2 = np.log10(0.00051)#1.0292637e-4#0.00067
    alpha = 1.6#1.56

    TobsYEAR_locs = nt_ranges*wc.DT/gc.SECSYEAR
    n_spect = SAET_goals.shape[0]

    log_SAE_goals = np.log10(SAET_goals[:, :, 0:2])


    def SAE_func_temp(tpl):
        resid = 0.
        a1 = tpl[0]
        ak = tpl[1]
        b1 = tpl[2]
        bk = tpl[3]
        log10A = tpl[4]
        log10f2 = tpl[5]
        alpha = tpl[6]
        for itry in range(n_spect):
            log10f1 = a1*np.log10(TobsYEAR_locs[itry]) + b1
            log10fknee = ak*np.log10(TobsYEAR_locs[itry]) + bk
            resid += np.sum((np.log10(np.abs(SAE_gal_model(fs, log10A, log10f2, log10f1, log10fknee, alpha))+offset)-log_SAE_goals[itry, :, :].T).flatten()**2)
        return resid

    bounds = np.zeros((7, 2))
    bounds[0, 0] = a1-0.2#-0.35
    bounds[0, 1] = a1+0.2#-0.05
    bounds[1, 0] = ak-0.2
    bounds[1, 1] = ak+0.2
    bounds[2, 0] = b1-0.4
    bounds[2, 1] = b1+0.4
    bounds[3, 0] = bk-0.4
    bounds[3, 1] = bk+0.4
    bounds[4, 0] = log10A-0.5
    bounds[4, 1] = log10A+0.5
    bounds[5, 0] = log10f2-1.5
    bounds[5, 1] = log10f2+1.5
    bounds[6, 0] = 1.35
    bounds[6, 1] = 2.25

    res_found = dual_annealing(SAE_func_temp, bounds, maxiter=2000)


    res = res_found['x']
    print(res_found)

    a1 = res[0]
    ak = res[1]
    b1 = res[2]
    bk = res[3]
    log10A = res[4]
    log10f2 = res[5]
    alpha = res[6]

    SAE_base_res = np.zeros((n_spect, fs_report.size))
    for itry in range(n_spect):
        log10f1 = a1*np.log10(TobsYEAR_locs[itry]) + b1
        log10fknee = ak*np.log10(TobsYEAR_locs[itry]) + bk
        SAE_base_res[itry, :] = SAE_gal_model(fs_report, log10A, log10f2, log10f1, log10fknee, alpha)

    return SAE_base_res, res
