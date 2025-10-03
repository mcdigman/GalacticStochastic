"""This module contains several functions to fit
realizations of the galactic background from the iterative fit.
The functions use a cyclostationary model with controllable periodicities.
get_S_cyclo uses an FFT-based filtering to extract a fit to a smoothed background,
without using any particular spectral model. fit_gb_spectrum_evolve
uses a fit to a standard shape for the galactic background spectrum.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from warnings import warn

import numpy as np
import scipy.ndimage
import WDMWaveletTransforms.fft_funcs as fft
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import dual_annealing

import GalacticStochastic.global_const as gc
from WaveletWaveforms.sparse_waveform_functions import PixelGenericRange

if TYPE_CHECKING:
    from numpy.typing import NDArray


def S_gal_model_5param(
    f: NDArray[np.floating] | float,
    log10A: NDArray[np.floating] | float,
    log10f2: NDArray[np.floating] | float,
    log10f1: NDArray[np.floating] | float,
    log10fknee: NDArray[np.floating] | float,
    alpha: NDArray[np.floating] | float,
) -> NDArray[np.floating] | float:
    """
    Compute an analytic model of the galactic binary confusion noise amplitude.

    The model used is described in arXiv:2103.14598.

    Parameters
    ----------
    f : float or NDArray[np.floating]
        Frequency (Hz) at which to evaluate the model.
    log10A : float or NDArray[np.floating]
        Base-10 logarithm of the amplitude normalization.
    log10f2 : float or NDArray[np.floating]
        Base-10 logarithm of the frequency scale in Hz for the tanh transition.
    log10f1 : float or NDArray[np.floating]
        Base-10 logarithm of the exponential cutoff frequency in Hz
    log10fknee : float or NDArray[np.floating]
        Base-10 logarithm of the knee frequency in Hz for the tanh transition.
    alpha : float or NDArray[np.floating]
        Exponent controlling the sharpness of the exponential cutoff.

    Returns
    -------
    S_gal : float or NDArray[np.floating]
        The modeled galactic binary confusion noise amplitude at the specified frequency.
    """
    return 10**log10A / 2 * f ** (5 / 3) * np.exp(-((f / 10**log10f1) ** alpha)) * (1 + np.tanh((10**log10fknee - f) / 10**log10f2))


def S_gal_model_7param(f: NDArray[np.floating] | float, log10A: NDArray[np.floating] | float, log10f2: NDArray[np.floating] | float, log10f1: NDArray[np.floating] | float, log10fknee: NDArray[np.floating] | float, alpha: NDArray[np.floating] | float, beta: NDArray[np.floating] | float, kappa: NDArray[np.floating] | float) -> NDArray[np.floating] | float:
    """
    Compute an analytic model of the galactic binary confusion noise amplitude.

    The model used is described in arXiv:1703.09858

    Parameters
    ----------
    f : float or NDArray[np.floating]
        Frequency (Hz) at which to evaluate the model.
    log10A : float or NDArray[np.floating]
        Base-10 logarithm of the amplitude normalization.
    log10f2 : float or NDArray[np.floating]
        Base-10 logarithm of the frequency scale in Hz for the tanh transition.
    log10f1 : float | NDArray[np.floating]
        Base-10 logarithm of the exponential cutoff frequency in Hz, set to 0. in the referenced paper
    log10fknee : float or NDArray[np.floating]
        Base-10 logarithm of the knee frequency in Hz for the tanh transition.
    alpha : float or NDArray[np.floating]
        Exponent controlling the sharpness of the exponential cutoff.
    beta : float or NDArray[np.floating]
        Scale parameter for the oscillatory part of the exponent, in Hz^-1
    kappa : float or NDArray[np.floating]
        1/f, argument for the oscillatory part of the exponent, in Hz^-1.

    Returns
    -------
    S_gal : float or NDArray[np.floating]
        The modeled galactic binary confusion noise amplitude at the specified frequency.

    Note
    ----
    The frequency shape used is different by a factor of f**(2/3) and the amplitude is different by a factor of 2
    """
    return 10**log10A * f ** (7 / 3) * np.exp(-((f / 10**log10f1) ** alpha) + beta * f * np.sin(kappa * f)) * (1 + np.tanh((10**log10fknee - f) / 10**log10f2))


def filter_periods_fft(r_mean: NDArray[np.floating], period_list: tuple[int, ...] | tuple[float, ...] | tuple[np.floating, ...], nt_lim: PixelGenericRange, *, period_tolerance: float = 0.01, angle_small: float = -0.1) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """
    Filter a time series to a specific set of periods using FFT decomposition.

    This function decomposes the input time series into its Fourier components and reconstructs
    the signal using only the specified periods. It returns the reconstructed time series,
    as well as the amplitudes and phases of the selected periodic components.

    Parameters
    ----------
    r_mean : NDArray[np.floating]
        Input time series to be filtered. Array of shape (Nt, n_channels), where Nt is the number of time samples.
    period_list : tuple of int or float
        Periods (in multiples of `Tobs/gc.SECSYEAR`) to retain in the filtered signal.
    nt_lim : PixelGenericRange
        Time range object defining the time sampling and limits of the input time series.
    period_tolerance : float, optional
        Allowed tolerance for matching requested periods to FFT bins (default is 0.01).
    angle_small : float, optional
        Threshold for determining the sign of the DC and Nyquist components (default is -0.1).

    Returns
    -------
    r : NDArray[np.floating]
        Reconstructed time series array of shape (Nt, n_channels) containing only the selected periods.
    amp_got : NDArray[np.floating]
        Array of shape (len(period_list), n_channels) with the amplitudes of the selected periodic components.
    angle_got : NDArray[np.floating]
        Array of shape (len(period_list), n_channels) with the phases (in radians) of the selected periodic components.

    Notes
    -----
    The periods in `period_list` are specified as multiples of the observation time in years.
    The function assumes the input is evenly sampled and uses the FFT to isolate the requested periodicities.
    If a requested period does not correspond exactly to an FFT bin, a warning is issued.
    """
    # get the same number of frequencies as the input r
    nc_loc = r_mean.shape[1]
    Nt_loc = r_mean.shape[0]
    assert nt_lim.nx_max - nt_lim.nx_min == Nt_loc
    dt = nt_lim.dx
    Tobs = Nt_loc * dt

    # time and angular frequency grids
    ts = np.arange(0, Nt_loc) * dt
    wts = 2 * np.pi / gc.SECSYEAR * ts

    # places to store results
    r = np.zeros((Nt_loc, nc_loc), dtype=np.float64)
    amp_got = np.zeros((len(period_list), nc_loc), dtype=np.float64)
    angle_got = np.zeros((len(period_list), nc_loc), dtype=np.float64)

    # iterate over input frequencies
    for itrc in range(nc_loc):
        res_fft: NDArray[np.complexfloating] = fft.rfft(r_mean[:, itrc] - 1.0) * 2 / Nt_loc
        abs_fft: NDArray[np.floating] = np.abs(res_fft)
        angle_fft: NDArray[np.floating] = -np.angle(res_fft)

        # highest and lowest frequency components with signs instead of angles
        if angle_fft[0] < angle_small:
            abs_fft[0] = -abs_fft[0]

        if angle_fft[-1] < angle_small:
            abs_fft[-1] = -abs_fft[-1]

        rec: NDArray[np.floating] = 1.0 + abs_fft[0] / 2.0 + np.zeros(Nt_loc, dtype=np.float64)

        # iterate over the periods we want to restrict to
        for itrk, k in enumerate(period_list):
            assert isinstance(k, (int, float))
            idx = int(Tobs / gc.SECSYEAR * k)
            if np.abs(idx - Tobs / gc.SECSYEAR * k) > period_tolerance:
                warn(
                    'fft filtering expects periods to be integer fraction of total time: got %10.8f for %10.8f' % (Tobs / gc.SECSYEAR * k, k),
                    stacklevel=2,
                )
            if k == 0:
                # already adding the constant case above, whether or not it is requested
                amp_got[itrk, itrc] = abs_fft[0] / 2
                angle_got[itrk, itrc] = 0.0
            elif k == int(gc.SECSYEAR // dt) // 2:
                # set amplitude and phase in highest frequency case
                amp_got[itrk, itrc] = abs_fft[-1] / np.sqrt(2.0)
                angle_got[itrk, itrc] = np.pi / 4.0
                rec += amp_got[itrk, itrc] * np.cos(k * wts - angle_got[itrk, itrc])
            else:
                # set amplitude and phase in other cases
                rec += abs_fft[idx] * np.cos(k * wts - angle_fft[idx])
                amp_got[itrk, itrc] = abs_fft[idx]
                angle_got[itrk, itrc] = angle_fft[idx] % (2 * np.pi)

        angle_fftm = angle_fft % (2 * np.pi)
        mult = int(Tobs / gc.SECSYEAR)

        print(
            '%3d & %5.3f & %5.2f & %5.3f & %5.2f & %5.3f & %5.2f & %5.3f & %5.2f & %5.3f & %5.2f'
            % (
                itrc,
                abs_fft[1 * mult],
                angle_fftm[1 * mult],
                abs_fft[2 * mult],
                angle_fftm[2 * mult],
                abs_fft[3 * mult],
                angle_fftm[3 * mult],
                abs_fft[4 * mult],
                angle_fftm[4 * mult],
                abs_fft[5 * mult],
                angle_fftm[5 * mult],
            ),
        )
        r[:, itrc] = rec
    return r, amp_got, angle_got


def get_S_cyclo(galactic_below: NDArray[np.floating], S_inst_m: NDArray[np.floating], dt: float, smooth_lengthf: float, filter_periods: int, *, period_list: tuple[int, ...] | tuple[np.floating, ...] | None = None, faint_cutoff_thresh: float = 0.1, t_stabilizer_mult: float = 1.0e-13, r_cutoff_mult: float = 1.0e-6, log_S_stabilizer: float = 1.0e-50) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Note the smoothing length is the length in *log* frequency,
    and the input is assumed spaced linearly in frequency starting from 0.
    """
    nf_loc = S_inst_m.shape[0]
    assert len(galactic_below.shape) == 3
    assert S_inst_m.shape[0] == galactic_below.shape[1]
    Nt_loc = galactic_below.shape[0]

    # assert Nt_loc*nf_loc == galactic_below.shape[0]
    nt_lim = PixelGenericRange(0, Nt_loc, dt, 0)
    Tobs = nt_lim.dx * (nt_lim.nx_max - nt_lim.nx_min)

    nc_s: int = S_inst_m.shape[1]

    S_in: NDArray[np.floating] = (galactic_below[..., :nc_s].reshape((Nt_loc, nf_loc, nc_s))) ** 2
    del galactic_below

    S_in_mean: NDArray[np.floating] = np.mean(S_in, axis=0)

    if filter_periods == 0:
        r_smooth: NDArray[np.floating] = np.zeros((Nt_loc, nc_s), dtype=np.float64) + 1.0
        amp_got: NDArray[np.floating] = np.zeros((0, nc_s), dtype=np.float64)
        angle_got: NDArray[np.floating] = np.zeros((0, nc_s), dtype=np.float64)
    else:
        r_mean: NDArray[np.floating] = np.zeros((Nt_loc, nc_s), dtype=np.float64)
        # whitened mean galaxy power
        Sw_in_mean: NDArray[np.floating] = np.zeros_like(S_in_mean)
        Sw_in_mean[S_inst_m > 0.0] = np.abs(S_in_mean[S_inst_m > 0.0] / S_inst_m[S_inst_m > 0.0])

        for itrc in range(nc_s):
            # completely cut out faint frequencies for calculating the envelope modulation
            # faint frequencies are different and noisier, so just weighting may not work
            mask: NDArray[np.bool_] = Sw_in_mean[:, itrc] > faint_cutoff_thresh * np.max(Sw_in_mean[:, itrc])
            stabilizer: float = t_stabilizer_mult * float(np.max(S_in_mean[mask, itrc]))
            Sw_in: NDArray[np.floating] = S_in[:, mask, itrc] / (S_in_mean[mask, itrc] + stabilizer)
            r_mean[:, itrc] = np.mean(Sw_in, axis=1)

            del Sw_in
            del stabilizer
            del mask

            assert np.all(r_mean[:, itrc] >= 0.0)

        del Sw_in_mean

        # input ratio can't be negative except due to numerical noise (will enforce nonzero later)
        r_mean_abs: NDArray[np.floating] = np.abs(r_mean)

        del r_mean

        if period_list is None:
            # if no period list is given, do every possible period
            min_step = 1 / int(Tobs / gc.SECSYEAR)
            period_list = tuple(np.arange(0, int(gc.SECSYEAR // nt_lim.dx) // 2 + min_step, min_step))

        r_smooth, amp_got, angle_got = filter_periods_fft(r_mean_abs, period_list, nt_lim)

        # the multiplier must be strictly positive
        # but due to noise/numerical inaccuracy in the fft it could be slightly negative
        # it also appears in a division, so we will lose numerical stability if it is too small.
        # add a numerical cutoff to small values scaled to the largest
        r_smooth[r_smooth < r_cutoff_mult * np.max(r_smooth)] = r_cutoff_mult * np.max(r_smooth)

        # absolute value should have no effect here unless r_smooth was entirely negative
        r_smooth = np.abs(r_smooth)

    # get mean of demodulated spectrum as a function of time with time variation removed
    S_demod_mean: NDArray[np.floating] = np.zeros((nf_loc, nc_s))

    for itrc in range(nc_s):
        S_demod_mean[:, itrc] = np.mean(np.abs(S_in[:, :, itrc].T / r_smooth[:, itrc]), axis=1)

    del S_in

    S_demod_mean_abs = np.abs(S_demod_mean)

    del S_demod_mean

    S_demod_smooth = np.zeros((nf_loc, nc_s))
    S_demod_smooth[0, :] = S_demod_mean_abs[0, :]

    log_f = np.log10(np.arange(1, nf_loc))  # set DF to 1 everywhere since it just shifts the log

    interp_mult = 10
    smooth_sigma = smooth_lengthf * interp_mult

    # interpolate from the evenly spaced input frequency bins to a finer set of log frequency bins
    # so we can apply the smoothing in log frequency instead of frequency
    n_f_interp = interp_mult * nf_loc
    log_f_interp = np.linspace(0.0, np.log10(nf_loc - 1), n_f_interp)

    for itrc in range(nc_s):
        # add and later remove a small numerical stabilizer for cases where the S is zero
        # better behaved to interpolate in log(S) as well
        log_S_demod_mean = np.log10(S_demod_mean_abs[1:, itrc] + log_S_stabilizer)
        log_S_interp = InterpolatedUnivariateSpline(log_f, log_S_demod_mean, ext=2)(log_f_interp)
        log_S_interp_smooth = scipy.ndimage.gaussian_filter(log_S_interp, smooth_sigma)
        log_S_smooth = InterpolatedUnivariateSpline(log_f_interp, log_S_interp_smooth, ext=2)(log_f)
        # remove the numerical stabilizer
        S_demod_smooth[1:, itrc] = 10**log_S_smooth - log_S_stabilizer

    # enforce positive just in case subtraction misbehaved
    S_demod_smooth_abs = np.abs(S_demod_smooth)

    del S_demod_smooth

    S_res = np.zeros((Nt_loc, nf_loc, nc_s)) + S_inst_m
    for itrc in range(nc_s):
        S_res[:, :, itrc] += np.outer(r_smooth[:, itrc], S_demod_smooth_abs[:, itrc])

    S_res = np.abs(S_res)

    assert np.all(np.isfinite(S_res))

    return S_res, r_smooth, S_demod_smooth_abs, amp_got, angle_got


def fit_gb_spectrum_evolve(
    S_goals: NDArray[np.floating],
    fs: NDArray[np.floating],
    fs_report: NDArray[np.floating],
    nt_ranges: NDArray[np.integer],
    offset: NDArray[np.floating],
    dt: float,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    a1 = -0.25
    b1 = -2.70
    ak = -0.27
    bk = -2.47
    log10A = np.log10(7.0e-39)
    log10f2 = np.log10(0.00051)

    TobsYEAR_locs = nt_ranges * dt / gc.SECSYEAR
    n_spect = S_goals.shape[0]

    log_S_goals = np.log10(S_goals[:, :, 0:2])

    def S_func_temp(tpl: NDArray[np.floating]) -> float:
        resid = 0.0
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
            resid += np.sum(
                (np.log10(np.abs(S_gal_model_5param(fs, log10A, log10f2, log10f1, log10fknee, alpha)) + offset) - log_S_goals[itry, :, :].T).flatten() ** 2,
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
        S_res[itry, :] = S_gal_model_5param(fs_report, log10A, log10f2, log10f1, log10fknee, alpha)

    return S_res, res
