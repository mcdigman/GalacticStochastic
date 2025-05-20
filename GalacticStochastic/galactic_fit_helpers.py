"""This module contains several functions to fit
realizations of the galactic background from the iterative fit.
The functions use a cyclostationary model with controllable periodicities.
get_S_cyclo uses an FFT-based filtering to extract a fit to a smoothed background,
without using any particular spectral model. fit_gb_spectrum_evolve
uses a fit to a standard shape for the galactic background spectrum.
"""

from warnings import warn

import numpy as np
import scipy.ndimage
import WDMWaveletTransforms.fft_funcs as fft
from numpy.typing import NDArray
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import dual_annealing

import GalacticStochastic.global_const as gc
from WaveletWaveforms.wdm_config import WDMWaveletConstants


def S_gal_model(f, log10A, log10f2, log10f1, log10fknee, alpha) -> NDArray[float]:
    """Model from arXiv:2103.14598 for galactic binary confusion noise amplitude"""
    return 10**log10A / 2 * f**(5 / 3) * np.exp(-(f / 10**log10f1)**alpha) * (1 + np.tanh((10**log10fknee - f) / 10**log10f2))


def S_gal_model_alt(f, A, alpha, beta, kappa, gamma, fknee) -> NDArray[float]:
    """Model from arXiv:1703.09858 for galactic binary confusion noise amplitude"""
    return A * f**(7 / 3) * np.exp(-f**alpha + beta * f * np.sin(kappa * f)) * (1 + np.tanh(gamma * (fknee - f)))


def filter_periods_fft(r_mean: NDArray[float], Nt_loc, period_list, wc: WDMWaveletConstants) -> (NDArray[float], NDArray[float], NDArray[float]):
    """Filter to a specific set of periods using an fft.
    period_list is in multiples of wc.Tobs/gc.SECSYEAR
    """
    # get the same number of frequencies as the input r
    nc_loc = r_mean.shape[1]

    # time and angular frequency grids
    ts = np.arange(0, Nt_loc) * wc.DT
    wts = 2 * np.pi / gc.SECSYEAR * ts

    # places to store results
    r = np.zeros((wc.Nt, nc_loc))
    amp_got = np.zeros((len(period_list), nc_loc))
    angle_got = np.zeros((len(period_list), nc_loc))

    # iterate over input frequencies
    for itrc in range(nc_loc):
        res_fft = fft.rfft(r_mean[:, itrc] - 1.) * 2 / Nt_loc
        abs_fft = np.abs(res_fft)
        angle_fft = -np.angle(res_fft)

        # highest and lowest frequency components with signs instead of angles
        if angle_fft[0] < -0.1:
            abs_fft[0] = - abs_fft[0]

        if angle_fft[-1] < -0.1:
            abs_fft[-1] = - abs_fft[-1]

        rec = 1. + abs_fft[0] / 2 + np.zeros(Nt_loc)

        # iterate over the periods we want to restrict to
        for itrk, k in enumerate(period_list):
            idx = int(wc.Tobs / gc.SECSYEAR * k)
            if np.abs(idx - wc.Tobs / gc.SECSYEAR * k) > 0.01:
                warn('fft filtering expects periods to be integer fraction of total time: got %10.8f for %10.8f' %
                     (wc.Tobs / gc.SECSYEAR * k, k), stacklevel=2
                     )
            if k == 0:
                # already adding the constant case above, whether or not it is requested
                amp_got[itrk, itrc] = abs_fft[0] / 2
                angle_got[itrk, itrc] = 0.
            elif k == int(gc.SECSYEAR // wc.DT) // 2:
                # set amplitude and phase in highest frequency case
                amp_got[itrk, itrc] = abs_fft[-1] / np.sqrt(2.)
                angle_got[itrk, itrc] = np.pi / 4.
                rec += amp_got[itrk, itrc] * np.cos(k * wts - angle_got[itrk, itrc])
            else:
                # set amplitude and phase in other cases
                rec += abs_fft[idx] * np.cos(k * wts - angle_fft[idx])
                amp_got[itrk, itrc] = abs_fft[idx]
                angle_got[itrk, itrc] = angle_fft[idx] % (2 * np.pi)

        angle_fftm = angle_fft % (2 * np.pi)
        mult = int(wc.Tobs / gc.SECSYEAR)

        print("%3d & %5.3f & %5.2f & %5.3f & %5.2f & %5.3f & %5.2f & %5.3f & %5.2f & %5.3f & %5.2f" %
              (
               itrc,
               abs_fft[1 * mult], angle_fftm[1 * mult],
               abs_fft[2 * mult], angle_fftm[2 * mult],
               abs_fft[3 * mult], angle_fftm[3 * mult],
               abs_fft[4 * mult], angle_fftm[4 * mult],
               abs_fft[5 * mult], angle_fftm[5 * mult]
              )
              )
        r[:, itrc] = rec
    return r, amp_got, angle_got


def get_S_cyclo(
        galactic_below: NDArray[float],
        S_inst_m: NDArray[float],
        wc: WDMWaveletConstants,
        smooth_lengthf,
        filter_periods,
        period_list=None,
        *,
        Nt_loc=-1,
        faint_cutoff_thresh=0.1,
        t_stabilizer_mult=1.e-13,
        r_cutoff_mult=1.e-6,
        log_S_stabilizer=1.e-50,
        )-> (NDArray[float], NDArray[float], NDArray[float], NDArray[float], NDArray[float]):
    """Note the smoothing length is the length in *log* frequency,
    and the input is assumed spaced linearly in frequency
    """
    if Nt_loc == -1:
        Nt_loc = wc.Nt

    nc_s = S_inst_m.shape[1]

    S_in = (galactic_below[..., :nc_s].reshape((wc.Nt, wc.Nf, nc_s)))**2
    S_in_mean = np.mean(S_in, axis=0)

    amp_got = None
    angle_got = None

    if not filter_periods:
        r_smooth = np.zeros((wc.Nt, nc_s)) + 1.
    else:
        r_mean = np.zeros((wc.Nt, nc_s))
        # whitened mean galaxy power
        Sw_in_mean = np.zeros_like(S_in_mean)
        Sw_in_mean[S_inst_m > 0.] = np.abs(S_in_mean[S_inst_m > 0.] / S_inst_m[S_inst_m > 0.])

        for itrc in range(nc_s):
            # completely cut out faint frequencies for calculating the envelope modulation
            # faint frequencies are different and noisier, so just weighting may not work
            mask = Sw_in_mean[:, itrc] > faint_cutoff_thresh * np.max(Sw_in_mean[:, itrc])
            stabilizer = t_stabilizer_mult * np.max(S_in_mean[mask, itrc])
            Sw_in = S_in[:, mask, itrc] / (S_in_mean[mask, itrc] + stabilizer)
            r_mean[:, itrc] = np.mean(Sw_in, axis=1)

            Sw_in = None
            stabilizer = None
            mask = None

            assert np.all(r_mean[:, itrc] >= 0.)

        Sw_in_mean = None

        # input ratio can't be negative except due to numerical noise (will enforce nonzero later)
        r_mean = np.abs(r_mean)

        if period_list is None:
            # if no period list is given, do every possible period
            min_step = 1 / int(wc.Tobs / gc.SECSYEAR)
            period_list = tuple(np.arange(0, int(gc.SECSYEAR // wc.DT) // 2 + min_step, min_step))

        r_smooth, amp_got, angle_got = filter_periods_fft(r_mean, Nt_loc, period_list, wc)

        # the multiplier must be strictly positive
        # but due to noise/numerical inaccuracy in the fft it could be slightly negative
        # it also appears in a division, so we will lose numerical stability if it is too small.
        # add a numerical cutoff to small values scaled to the largest
        r_smooth[r_smooth < r_cutoff_mult * np.max(r_smooth)] = r_cutoff_mult * np.max(r_smooth)

        # absolute value should have no effect here unless r_smooth was entirely negative
        r_smooth = np.abs(r_smooth)

    # get mean of demodulated spectrum as a function of time with time variation removed
    S_demod_mean = np.zeros((wc.Nf, nc_s))

    for itrc in range(nc_s):
        S_demod_mean[:, itrc] = np.mean(np.abs(S_in[:, :, itrc].T / r_smooth[:, itrc]), axis=1)

    S_in = None

    S_demod_mean = np.abs(S_demod_mean)

    S_demod_smooth = np.zeros((wc.Nf, nc_s))
    S_demod_smooth[0, :] = S_demod_mean[0, :]

    log_f = np.log10(np.arange(1, wc.Nf) * wc.DF)

    interp_mult = 10
    smooth_sigma = smooth_lengthf * interp_mult

    # interpolate from the evenly spaced input frequency bins to a finer set of log frequency bins
    # so we can apply the smoothing in log frequency instead of frequency
    n_f_interp = interp_mult * wc.Nf
    log_f_interp = np.linspace(np.log10(wc.DF), np.log10(wc.DF * (wc.Nf - 1)), n_f_interp)

    for itrc in range(nc_s):
        # add and later remove a small numerical stabilizer for cases where the S is zero
        # better behaved to interpolate in log(S) as well
        log_S_demod_mean = np.log10(S_demod_mean[1:, itrc] + log_S_stabilizer)
        log_S_interp = InterpolatedUnivariateSpline(log_f, log_S_demod_mean, ext=2)(log_f_interp)
        log_S_interp_smooth = scipy.ndimage.gaussian_filter(log_S_interp, smooth_sigma)
        log_S_smooth = InterpolatedUnivariateSpline(log_f_interp, log_S_interp_smooth, ext=2)(log_f)
        # remove the numerical stabilizer
        S_demod_smooth[1:, itrc] = 10**log_S_smooth - log_S_stabilizer

    # enforce positive just in case subtraction misbehaved
    S_demod_smooth = np.abs(S_demod_smooth)

    S_res = np.zeros((wc.Nt, wc.Nf, nc_s)) + S_inst_m
    for itrc in range(nc_s):
        S_res[:, :, itrc] += np.outer(r_smooth[:, itrc], S_demod_smooth[:, itrc])

    S_res = np.abs(S_res)

    assert np.all(np.isfinite(S_res))

    return S_res, r_smooth, S_demod_smooth, amp_got, angle_got


def fit_gb_spectrum_evolve(S_goals: NDArray[float], fs: NDArray[float], fs_report: NDArray[float], nt_ranges, offset, wc: WDMWaveletConstants) -> (NDArray[float], NDArray[float]):
    a1 = -0.25
    b1 = -2.70
    ak = -0.27
    bk = -2.47
    log10A = np.log10(7.e-39)
    log10f2 = np.log10(0.00051)
    alpha = 1.6

    TobsYEAR_locs = nt_ranges * wc.DT / gc.SECSYEAR
    n_spect = S_goals.shape[0]

    log_S_goals = np.log10(S_goals[:, :, 0:2])

    def S_func_temp(tpl: (float, float, float, float, float, float, float)) -> float:
        resid = 0.
        a1 = tpl[0]
        ak = tpl[1]
        b1 = tpl[2]
        bk = tpl[3]
        log10A = tpl[4]
        log10f2 = tpl[5]
        alpha = tpl[6]
        for itry in range(n_spect):
            log10f1 = a1 * np.log10(TobsYEAR_locs[itry]) + b1
            log10fknee = ak * np.log10(TobsYEAR_locs[itry]) + bk
            resid += np.sum((
                             np.log10(np.abs(S_gal_model(fs, log10A, log10f2, log10f1, log10fknee, alpha)) + offset)
                             - log_S_goals[itry, :, :].T
                            ).flatten()**2
                            )
        return resid

    bounds = np.zeros((7, 2))
    bounds[0, 0] = a1 - 0.2
    bounds[0, 1] = a1 + 0.2
    bounds[1, 0] = ak - 0.2
    bounds[1, 1] = ak + 0.2
    bounds[2, 0] = b1 - 0.4
    bounds[2, 1] = b1 + 0.4
    bounds[3, 0] = bk - 0.4
    bounds[3, 1] = bk + 0.4
    bounds[4, 0] = log10A - 0.5
    bounds[4, 1] = log10A + 0.5
    bounds[5, 0] = log10f2 - 1.5
    bounds[5, 1] = log10f2 + 1.5
    bounds[6, 0] = 1.35
    bounds[6, 1] = 2.25

    res_found = dual_annealing(S_func_temp, bounds, maxiter=2000)

    res = res_found['x']
    print(res_found)

    a1 = res[0]
    ak = res[1]
    b1 = res[2]
    bk = res[3]
    log10A = res[4]
    log10f2 = res[5]
    alpha = res[6]

    S_res = np.zeros((n_spect, fs_report.size))
    for itry in range(n_spect):
        log10f1 = a1 * np.log10(TobsYEAR_locs[itry]) + b1
        log10fknee = ak * np.log10(TobsYEAR_locs[itry]) + bk
        S_res[itry, :] = S_gal_model(fs_report, log10A, log10f2, log10f1, log10fknee, alpha)

    return S_res, res
