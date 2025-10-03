"""Various functions to fit the galactic background for the iterative fit.

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
    log10_a: NDArray[np.floating] | float,
    log10_f2: NDArray[np.floating] | float,
    log10_f1: NDArray[np.floating] | float,
    log10_fknee: NDArray[np.floating] | float,
    alpha: NDArray[np.floating] | float,
) -> NDArray[np.floating] | float:
    """
    Compute an analytic model of the galactic binary confusion noise amplitude.

    The model used is described in arXiv:2103.14598.

    Parameters
    ----------
    f : float or NDArray[np.floating]
        Frequency (Hz) at which to evaluate the model.
    log10_a : float or NDArray[np.floating]
        Base-10 logarithm of the amplitude normalization.
    log10_f2 : float or NDArray[np.floating]
        Base-10 logarithm of the frequency scale in Hz for the tanh transition.
    log10_f1 : float or NDArray[np.floating]
        Base-10 logarithm of the exponential cutoff frequency in Hz
    log10_fknee : float or NDArray[np.floating]
        Base-10 logarithm of the knee frequency in Hz for the tanh transition.
    alpha : float or NDArray[np.floating]
        Exponent controlling the sharpness of the exponential cutoff.

    Returns
    -------
    S_gal : float or NDArray[np.floating]
        The modeled galactic binary confusion noise amplitude at the specified frequency.
    """
    return (
        10**log10_a
        / 2
        * f ** (5 / 3)
        * np.exp(-((f / 10**log10_f1) ** alpha))
        * (1 + np.tanh((10**log10_fknee - f) / 10**log10_f2))
    )


def S_gal_model_7param(
    f: NDArray[np.floating] | float,
    log10_a: NDArray[np.floating] | float,
    log10_f2: NDArray[np.floating] | float,
    log10_f1: NDArray[np.floating] | float,
    log10_fknee: NDArray[np.floating] | float,
    alpha: NDArray[np.floating] | float,
    beta: NDArray[np.floating] | float,
    kappa: NDArray[np.floating] | float,
) -> NDArray[np.floating] | float:
    """
    Compute an analytic model of the galactic binary confusion noise amplitude.

    The model used is described in arXiv:1703.09858

    Parameters
    ----------
    f : float or NDArray[np.floating]
        Frequency (Hz) at which to evaluate the model.
    log10_a : float or NDArray[np.floating]
        Base-10 logarithm of the amplitude normalization.
    log10_f2 : float or NDArray[np.floating]
        Base-10 logarithm of the frequency scale in Hz for the tanh transition.
    log10_f1 : float | NDArray[np.floating]
        Base-10 logarithm of the exponential cutoff frequency in Hz, set to 0. in the referenced paper
    log10_fknee : float or NDArray[np.floating]
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
    return (
        10**log10_a
        * f ** (7 / 3)
        * np.exp(-((f / 10**log10_f1) ** alpha) + beta * f * np.sin(kappa * f))
        * (1 + np.tanh((10**log10_fknee - f) / 10**log10_f2))
    )


def filter_periods_fft(
    r_mean: NDArray[np.floating],
    period_list: tuple[int, ...] | tuple[float, ...] | tuple[np.floating, ...],
    nt_lim: PixelGenericRange,
    *,
    period_tolerance: float = 0.01,
    angle_small: float = -0.1,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
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
        Periods (in multiples of `t_obs/gc.SECSYEAR`) to retain in the filtered signal.
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
    nt_loc = r_mean.shape[0]
    assert nt_lim.nx_max - nt_lim.nx_min == nt_loc
    dt = nt_lim.dx
    t_obs = nt_loc * dt

    # time and angular frequency grids
    ts = np.arange(0, nt_loc) * dt
    wts = 2 * np.pi / gc.SECSYEAR * ts

    # places to store results
    r = np.zeros((nt_loc, nc_loc), dtype=np.float64)
    amp_got = np.zeros((len(period_list), nc_loc), dtype=np.float64)
    angle_got = np.zeros((len(period_list), nc_loc), dtype=np.float64)

    # iterate over input frequencies
    for itrc in range(nc_loc):
        res_fft: NDArray[np.complexfloating] = fft.rfft(r_mean[:, itrc] - 1.0) * 2 / nt_loc
        abs_fft: NDArray[np.floating] = np.abs(res_fft)
        angle_fft: NDArray[np.floating] = -np.angle(res_fft)

        # highest and lowest frequency components with signs instead of angles
        if angle_fft[0] < angle_small:
            abs_fft[0] = -abs_fft[0]

        if angle_fft[-1] < angle_small:
            abs_fft[-1] = -abs_fft[-1]

        rec: NDArray[np.floating] = 1.0 + abs_fft[0] / 2.0 + np.zeros(nt_loc, dtype=np.float64)

        # iterate over the periods we want to restrict to
        for itrk, k in enumerate(period_list):
            assert isinstance(k, (int, float))
            idx = int(t_obs / gc.SECSYEAR * k)
            if np.abs(idx - t_obs / gc.SECSYEAR * k) > period_tolerance:
                warn(
                    'fft filtering expects periods to be integer fraction of total time: got %10.8f for %10.8f'
                    % (t_obs / gc.SECSYEAR * k, k),
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
        mult = int(t_obs / gc.SECSYEAR)

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


def _mean_smooth_helper(
    x_mean: NDArray[np.floating], log_stabilizer: float, smooth_lengthf: float, interp_mult: int,
) -> NDArray[np.floating]:
    """
    Smooth a mean spectrum in log-frequency space using Gaussian filtering.

    This function interpolates the input mean spectrum onto a finer log-frequency grid,
    applies a Gaussian filter to smooth the spectrum in log-frequency, and then interpolates
    back to the original frequency grid. Smoothing is performed in log-amplitude to improve
    numerical stability, with a small stabilizer added to avoid issues with zeros.

    Parameters
    ----------
    x_mean : NDArray[np.floating]
        Input mean spectrum array of shape (n_freq, n_channels).
    log_stabilizer : float
        Small positive value added to the spectrum before taking the logarithm to avoid log(0).
    smooth_lengthf : float
        Smoothing length (standard deviation) in log-frequency space.
    interp_mult : int
        Interpolation factor for increasing the resolution of the log-frequency grid.

    Returns
    -------
    x_smooth : NDArray[np.floating]
        Smoothed mean spectrum array of shape (n_freq, n_channels).

    Notes
    -----
    The smoothing is performed in log-frequency and log-amplitude space for improved stability.
    The function enforces non-negativity on the output to avoid artifacts from numerical errors.
    """
    x_mean_abs = np.abs(x_mean)

    nc_s = x_mean_abs.shape[1]
    nf_loc = x_mean_abs.shape[0]

    log_f = np.log10(np.arange(1, nf_loc))  # set DF to 1 everywhere since it just shifts the log

    smooth_sigma = smooth_lengthf * interp_mult

    # interpolate from the evenly spaced input frequency bins to a finer set of log frequency bins
    # so we can apply the smoothing in log frequency instead of frequency
    n_f_interp = interp_mult * nf_loc
    log_f_interp = np.linspace(0.0, np.log10(nf_loc - 1), n_f_interp)

    x_smooth = np.zeros((nf_loc, nc_s))
    x_smooth[0, :] = x_mean_abs[0, :]

    for itrc in range(nc_s):
        # add and later remove a small numerical stabilizer for cases where the S is zero
        # better behaved to interpolate in log(S) as well
        log_x_mean = np.log10(x_mean_abs[1:, itrc] + log_stabilizer)
        log_x_interp = InterpolatedUnivariateSpline(log_f, log_x_mean, ext=2)(log_f_interp)
        log_x_interp_smooth = scipy.ndimage.gaussian_filter(log_x_interp, smooth_sigma)
        log_x_smooth = InterpolatedUnivariateSpline(log_f_interp, log_x_interp_smooth, ext=2)(log_f)
        # remove the numerical stabilizer
        x_smooth[1:, itrc] = 10**log_x_smooth - log_stabilizer

    # enforce positive just in case subtraction misbehaved
    return np.abs(x_smooth)


def get_S_cyclo(
    galactic_below: NDArray[np.floating],
    S_inst_m: NDArray[np.floating],
    dt: float,
    smooth_lengthf: float,
    filter_periods: int,
    *,
    period_list: tuple[int, ...] | tuple[np.floating, ...] | None = None,
    faint_cutoff_thresh: float = 0.1,
    t_stabilizer_mult: float = 1.0e-13,
    r_cutoff_mult: float = 1.0e-6,
    log_stabilizer: float = 1.0e-50,
    interp_mult: int = 10,
) -> tuple[
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
]:
    """
    Estimate the cyclostationary spectrum of the galactic background using FFT-based filtering and smoothing.

    This function computes a time- and frequency-dependent estimate of the galactic background spectrum
    by removing the mean time variation in the spectrum,
    optionally filtering for specific periodicities of known physical interest, and smoothing the result in log-frequency space.
    The output cyclostationary spectrum is given by the product of the smoothed mean spectrum and the smoothed time-dependent modulation ratio.
    The method is independent of fiting functions for the frequency spectrum and uses the observed data to extract the cyclostationary structure.

    Parameters
    ----------
    galactic_below : NDArray[np.floating]
        Input array of galactic background data, shape (Nt, Nf, Nchannels).
    S_inst_m : NDArray[np.floating]
        Instrumental noise spectrum, shape (Nf, Nchannels).
    dt : float
        Time step between slices of the grid in galactic_below (seconds).
    smooth_lengthf : float
        Smoothing length (standard deviation) in log-frequency space.
    filter_periods : int
        If nonzero, apply FFT-based filtering to restrict to specific periods.
    period_list : tuple of int or float, optional
        Periods (in multiples of observation time in years) to retain in the filtered signal.
        If None and filtering is enabled, all possible periods are used.
    faint_cutoff_thresh : float, optional
        Threshold (fraction of max) for masking faint frequencies in envelope modulation (default 0.1).
    t_stabilizer_mult : float, optional
        Multiplier for numerical stabilizer added to avoid division by zero (default 1.0e-13).
    r_cutoff_mult : float, optional
        Multiplier for minimum allowed value of the demodulation ratio (default 1.0e-6).
    log_stabilizer : float, optional
        Small positive value added before taking logarithms to avoid log(0) (default 1.0e-50).
    interp_mult : int, optional
        Interpolation factor for increasing the resolution of the log-frequency grid (default 10).

    Returns
    -------
    S_res : NDArray[np.floating]
        Cyclostationary spectrum estimate, shape (Nt, Nf, Nchannels).
    r_smooth : NDArray[np.floating]
        Smoothed time-dependent modulation ratio, shape (Nt, Nchannels).
    S_demod_smooth : NDArray[np.floating]
        Smoothed mean demodulated spectrum, shape (Nf, Nchannels).
    amp_got : NDArray[np.floating]
        Amplitudes of selected periodic components (if filtering), shape (n_periods, Nchannels).
    angle_got : NDArray[np.floating]
        Phases (in radians) of selected periodic components (if filtering), shape (n_periods, Nchannels).

    Notes
    -----
    - Smoothing is performed in log-frequency and log-amplitude space for numerical stability.
    - If `filter_periods` is zero, no periodic filtering is applied and the modulation ratio is set to unity.
    - The function enforces non-negativity and applies numerical cutoffs to avoid artifacts from noise.
    - The input is assumed to be linearly spaced in frequency starting from zero.
    """
    nf_loc = S_inst_m.shape[0]
    assert len(galactic_below.shape) == 3
    assert S_inst_m.shape[0] == galactic_below.shape[1]
    nt_loc = galactic_below.shape[0]

    nt_lim = PixelGenericRange(0, nt_loc, dt, 0)
    t_obs = nt_lim.dx * (nt_lim.nx_max - nt_lim.nx_min)

    nc_s: int = S_inst_m.shape[1]

    S_in: NDArray[np.floating] = (galactic_below[..., :nc_s].reshape((nt_loc, nf_loc, nc_s))) ** 2
    del galactic_below

    S_in_mean: NDArray[np.floating] = np.mean(S_in, axis=0)

    if filter_periods == 0:
        r_smooth: NDArray[np.floating] = np.zeros((nt_loc, nc_s), dtype=np.float64) + 1.0
        amp_got: NDArray[np.floating] = np.zeros((0, nc_s), dtype=np.float64)
        angle_got: NDArray[np.floating] = np.zeros((0, nc_s), dtype=np.float64)
    else:
        r_mean: NDArray[np.floating] = np.zeros((nt_loc, nc_s), dtype=np.float64)
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
            min_step = 1 / int(t_obs / gc.SECSYEAR)
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

    S_demod_smooth = _mean_smooth_helper(S_demod_mean, log_stabilizer, smooth_lengthf, interp_mult)

    S_res = np.zeros((nt_loc, nf_loc, nc_s)) + S_inst_m
    for itrc in range(nc_s):
        S_res[:, :, itrc] += np.outer(r_smooth[:, itrc], S_demod_smooth[:, itrc])

    S_res = np.abs(S_res)

    assert np.all(np.isfinite(S_res))

    return S_res, r_smooth, S_demod_smooth, amp_got, angle_got


def fit_gb_spectrum_evolve(
    S_goals: NDArray[np.floating],
    fs: NDArray[np.floating],
    fs_report: NDArray[np.floating],
    nt_ranges: NDArray[np.integer],
    offset: NDArray[np.floating],
    dt: float,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:

    t_obs_yrs = nt_ranges * dt / gc.SECSYEAR
    n_spect = S_goals.shape[0]

    log_S_goals = np.log10(S_goals[:, :, 0:2])

    def S_func_temp(tpl: NDArray[np.floating]) -> float:
        resid: float = 0.0
        a1: float = tpl[0]
        ak: float = tpl[1]
        b1: float = tpl[2]
        bk: float = tpl[3]
        log10_a: float = tpl[4]
        log10_f2: float = tpl[5]
        alpha = tpl[6]
        for itry in range(n_spect):
            log10_f1: float = a1 * np.log10(t_obs_yrs[itry]) + b1
            log10_fknee: float = ak * np.log10(t_obs_yrs[itry]) + bk
            resid += np.sum(
                (
                    np.log10(np.abs(S_gal_model_5param(fs, log10_a, log10_f2, log10_f1, log10_fknee, alpha)) + offset)
                    - log_S_goals[itry, :, :].T
                ).flatten()
                ** 2,
            )
        return resid

    a1_0: float = -0.25
    b1_0: float = -2.70
    ak_0: float = -0.27
    bk_0: float = -2.47
    log10_a_0 = float(np.log10(7.0e-39))
    log10_f2_0 = float(np.log10(0.00051))

    bounds = np.zeros((7, 2))
    bounds[0, 0] = a1_0 - 0.2
    bounds[0, 1] = a1_0 + 0.2
    bounds[1, 0] = ak_0 - 0.2
    bounds[1, 1] = ak_0 + 0.2
    bounds[2, 0] = b1_0 - 0.4
    bounds[2, 1] = b1_0 + 0.4
    bounds[3, 0] = bk_0 - 0.4
    bounds[3, 1] = bk_0 + 0.4
    bounds[4, 0] = log10_a_0 - 0.5
    bounds[4, 1] = log10_a_0 + 0.5
    bounds[5, 0] = log10_f2_0 - 1.5
    bounds[5, 1] = log10_f2_0 + 1.5
    bounds[6, 0] = 1.35
    bounds[6, 1] = 2.25

    res_found = dual_annealing(S_func_temp, bounds, maxiter=2000)

    res = res_found['x']
    print(res_found)

    a1_res = float(res[0])
    ak_res = float(res[1])
    b1_res = float(res[2])
    bk_res = float(res[3])
    log10_a_res = float(res[4])
    log10_f2_res = float(res[5])
    alpha_res = float(res[6])

    S_res = np.zeros((n_spect, fs_report.size))
    for itry in range(n_spect):
        log10_f1 = a1_res * np.log10(t_obs_yrs[itry]) + b1_res
        log10_fknee = ak_res * np.log10(t_obs_yrs[itry]) + bk_res
        S_res[itry, :] = S_gal_model_5param(fs_report, log10_a_res, log10_f2_res, log10_f1, log10_fknee, alpha_res)

    return S_res, res
