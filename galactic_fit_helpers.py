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
    """filter to a specific set of periods using an fft, period_list is in multiples of wc.Tobs/gc.SECSYEAR"""

    # get the same number of frequencies as the input r
    NC = r_got1.shape[1]

    # time and angular frequency grids
    ts = np.arange(0, Nt_loc)*wc.DT
    wts = 2*np.pi/gc.SECSYEAR*ts

    # places to store results
    r_fft1 = np.zeros((wc.Nt, wc.NC))
    amp_got = np.zeros((period_list.size, wc.NC))
    angle_got = np.zeros((period_list.size, wc.NC))

    # iterate over input frequencies
    for itrc in range(NC):
        res_fft = fft.rfft(r_got1[:, itrc]-1.)*2/Nt_loc
        abs_fft = np.abs(res_fft)
        angle_fft = -np.angle(res_fft)

        # highest and lowest frequency components with signs instead of angles
        if angle_fft[0] < -0.1:
            abs_fft[0] = - abs_fft[0]

        if angle_fft[-1] < -0.1:
            abs_fft[-1] = - abs_fft[-1]

        rec = 1.+abs_fft[0]/2+np.zeros(Nt_loc)

        # iterate over the periods we want to restrict to
        for itrk, k in enumerate(period_list):
            idx = np.int64(wc.Tobs/gc.SECSYEAR*k)
            if np.abs(idx - wc.Tobs/gc.SECSYEAR*k) > 0.01:
                warn('fft based filtering expects periods that are an integer fraction of the observing time: got %10.8f for %10.8f' % (wc.Tobs/gc.SECSYEAR*k, k))
            if k == 0:
                # already adding the constant case above, whether or not it is requested
                amp_got[itrk, itrc] = abs_fft[0]/2
                angle_got[itrk, itrc] = 0.
            elif k == np.int64(gc.SECSYEAR//wc.DT)//2:
                # set amplitude and phase in highest frequency case
                amp_got[itrk,itrc] = abs_fft[-1]/np.sqrt(2.)
                angle_got[itrk, itrc] = np.pi/4.
                rec += amp_got[itrk,itrc]*np.cos(k*wts - angle_got[itrk,itrc])
            else:
                # set amplitude and phase in other cases
                rec += abs_fft[idx]*np.cos(k*wts - angle_fft[idx])
                amp_got[itrk, itrc] = abs_fft[idx]
                angle_got[itrk, itrc] = angle_fft[idx] % (2*np.pi)

        angle_fftm = angle_fft % (2*np.pi)
        mult = np.int64(wc.Tobs/gc.SECSYEAR)

        print("%3d & %5.3f & %5.2f & %5.3f & %5.2f & %5.3f & %5.2f & %5.3f & %5.2f & %5.3f & %5.2f"%(itrc, abs_fft[1*mult], angle_fftm[1*mult], abs_fft[2*mult], angle_fftm[2*mult], abs_fft[3*mult], angle_fftm[3*mult], abs_fft[4*mult], angle_fftm[4*mult], abs_fft[5*mult], angle_fftm[5*mult]))
        r_fft1[:, itrc] = rec
    return r_fft1, amp_got, angle_got


def get_SAET_cyclostationary_mean(galactic_bg, SAET_m, wc, smooth_lengthf=4, filter_periods=False, period_list=None, Nt_loc=-1):
    """ note the smoothing length is the length in *log* frequency, and the input is assumed spaced linearly in frequency"""
    if Nt_loc == -1:
        Nt_loc = wc.Nt

    NC_loc = SAET_m.shape[1]

    SAET_pure_in = (galactic_bg[...,:NC_loc].reshape((wc.Nt, wc.Nf, NC_loc)))**2
    SAET_pure_mean = np.mean(SAET_pure_in, axis=0)

    amp_got = None
    angle_got = None

    if not filter_periods:
        rec_use = np.zeros((wc.Nt, NC_loc))+1.
    else:
        r_got1 = np.zeros((wc.Nt, NC_loc))
        SAET_pure_white = np.abs(SAET_pure_mean/SAET_m)

        for itrc in range(NC_loc):
            # completely cutting out faint frequencies completely seemed to work better for calculating the envelope modulation
            # than a weighted average, although there might be other ways
            r_eval_mask = SAET_pure_white[:, itrc] > 0.1*np.max(SAET_pure_white[:, itrc])
            r_got1[:, itrc] = np.mean(SAET_pure_in[:, r_eval_mask, itrc]/(SAET_pure_mean[r_eval_mask, itrc] + 1.e-13*np.max(SAET_pure_mean[r_eval_mask, itrc])), axis=1)

            assert np.all(r_got1[:,itrc] >= 0.)

        # input ratio can't be negative except due to numerical noise (will enforce nonzero later)
        r_got1 = np.abs(r_got1)

        if period_list is None:
            # if no period list is given, do every possible period
            period_list = np.arange(0, np.int64(gc.SECSYEAR//wc.DT)//2+1/int(wc.Tobs/gc.SECSYEAR), 1/int(wc.Tobs/gc.SECSYEAR))


        r_fft1, amp_got, angle_got = filter_periods_fft(r_got1, Nt_loc, period_list, wc)

        # the multiplier must be strictly positive
        # but due to noise/numerical inaccuracy in the fft it could be slightly negative
        # it also appears in a division, so we will lose numerical stability if it is too small.
        # add a numerical cutoff to small values scaled to the largest
        rec_use = r_fft1.copy()

        rec_use[rec_use < 1.e-6*np.max(rec_use)] = 1.e-6*np.max(rec_use)

        # absolute value should have no effect here unless rec_use was entirely negative
        rec_use = np.abs(rec_use)

    # get spectrum as a function of time with time variation removed
    SAET_pure_mod = SAET_pure_in.copy()
    for itrc in range(NC_loc):
        SAET_pure_mod[:, :, itrc] = np.abs((SAET_pure_mod[:, :, itrc].T/rec_use[:, itrc]).T)

    # get mean spectrum with time variation removed
    SAET_pures = np.abs(np.mean(SAET_pure_mod, axis=0))

    SAET_pures_smooth2 = np.zeros((wc.Nf, NC_loc))
    SAET_pures_smooth2[0, :] = SAET_pures[0, :]

    log_fs = np.log10(np.arange(1, wc.Nf)*wc.DF)

    interp_mult = 10

    # interpolate from the evenly spaced input frequency bins to a finer set of log frequency bins
    # so we can apply the smoothing in log frequency instead of frequency
    n_f_interp = interp_mult*wc.Nf
    log_fs_interp = np.linspace(np.log10(wc.DF), np.log10(wc.DF*(wc.Nf-1)), n_f_interp)

    for itrc in range(NC_loc):
        # add and later remove a small numerical stabilizer for cases where the SAET is zero
        # better behaved to interpolate in log(SAE) as well
        log_SAE_pure_loc_smooth1 = np.log10(SAET_pures[1:, itrc] + 1.e-50)
        log_SAE_interp_loc = InterpolatedUnivariateSpline(log_fs, log_SAE_pure_loc_smooth1, k=3, ext=2)(log_fs_interp)
        log_SAE_interp_loc_smooth = scipy.ndimage.gaussian_filter(log_SAE_interp_loc, smooth_lengthf*interp_mult)
        SAET_pures_smooth2[1:, itrc] = 10**InterpolatedUnivariateSpline(log_fs_interp, log_SAE_interp_loc_smooth, k=3, ext=2)(log_fs) - 1.e-50

    # enforce positive just in case subtraction misbheaved
    SAET_pures_smooth2 = np.abs(SAET_pures_smooth2)

    SAET_res = np.zeros((wc.Nt, wc.Nf, NC_loc))+SAET_m
    for itrc in range(NC_loc):
        SAET_res[:, :, itrc] += np.outer(rec_use[:, itrc], SAET_pures_smooth2[:, itrc])

    SAET_res = np.abs(SAET_res)

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
