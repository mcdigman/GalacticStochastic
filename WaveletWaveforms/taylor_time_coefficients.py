"""
Get Taylor time-domain coefficients for wavelet transforms.

C 2025 Matthew C. Digman
"""
from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING, NamedTuple
from warnings import warn

import h5py
import numpy as np
import scipy.fft as spf
from numba import njit, prange
from numpy.testing import assert_allclose
from WDMWaveletTransforms.transform_freq_funcs import phitilde_vec

from WaveletWaveforms.sparse_waveform_functions import SparseWaveletWaveform

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from WaveletWaveforms.wdm_config import WDMWaveletConstants

SECSYEAR = 24 * 365 * 3600  # Number of seconds in a calendar year


class WaveletTaylorTimeCoeffs(NamedTuple):
    Nfsam: NDArray[np.integer]
    evc: NDArray[np.floating]
    evs: NDArray[np.floating]
    wavelet_norm: NDArray[np.floating]


def get_empty_sparse_taylor_time_waveform(nc_waveform: int, wc: WDMWaveletConstants) -> SparseWaveletWaveform:
    """
    Create an empty SparseWaveletWaveform data structure for Taylor time-domain methods.

    This function initializes and returns a blank sparse wavelet object, sized according
    to the expected maximum number of nonzero wavelet coefficients for a Taylor-based
    time-domain decomposition. The returned object includes arrays for coefficient
    values, corresponding pixel indices, and counters for the number of coefficients
    set per intrinsic_waveform channel.

    Parameters
    ----------
    nc_waveform : int
        Number of intrinsic_waveform channels (e.g., polarization states, detectors, or modes).
    wc : WDMWaveletConstants
        Wavelet transform configuration, used to determine necessary array sizes
        and structure for proper allocation.

    Returns
    -------
    SparseWaveletWaveform
        An empty, preallocated object ready to be filled with Taylor time-domain
        wavelet coefficients and their pixel indices during intrinsic_waveform synthesis
        or analysis.

    Notes
    -----
    This helper ensures efficient memory allocation and organization
    for sparse operations in the Taylor time-domain framework, accounting for
    the maximum number of time pixels based on the frequency derivative bins.
    """
    # need the frequency derivatives to calculate the maximum possible size
    fds = wc.dfd * np.arange(-wc.Nfd_negative, wc.Nfd - wc.Nfd_negative)
    # calculate maximum possible number of pixels
    n_pixel_max = int(np.ceil((wc.BW + np.max(np.abs(fds)) * wc.Tw) / wc.DF)) * wc.Nt + wc.n_f_null_extend
    # array of wavelet coefficients
    wave_value = np.zeros((nc_waveform, n_pixel_max))
    # aray of pixel indices
    pixel_index = np.full((nc_waveform, n_pixel_max), -1, dtype=np.int64)
    # number of pixel indices that are set
    n_set = np.zeros(nc_waveform, dtype=np.int64)

    return SparseWaveletWaveform(wave_value, pixel_index, n_set, n_pixel_max)


def wavelet(wc: WDMWaveletConstants, m: int | float, nrm: float, *, n_in: int = -1) -> NDArray[np.floating]:
    """
    Construct and normalize a wavelet basis function for the specified frequency bin.

    This function generates a real-valued wavelet basis function in the time domain, corresponding to
    a chosen frequency bin index `m`, using the configuration provided in `_wc`.

    Parameters
    ----------
    wc : WDMWaveletConstants
        Wavelet decomposition configuration, specifying time sampling,
        frequency grid, windowing, and FFT parameters.
    m : int | float
        The index of the frequency bin around which to center the wavelet.
        Although it has units of an index, it is actually a frequency and so need not necessarily be an integer

    nrm : float
        Normalization factor to scale the resulting wavelet.
    Returns
    -------
    wave : np.ndarray
        The real-valued, normalized wavelet as a 1D NumPy array.
        The length matches the total number of time samples specified in `_wc`.

    Notes
    -----
    The wavelet is constructed in the frequency domain and transformed back to the
    time domain using the FFT, then normalized. This function is used to generate
    reference wavelets for coefficient tables and intrinsic_waveform synthesis.
    """
    if n_in == -1:
        n_use: int = wc.K
    else:
        n_use = n_in

    assert n_use % 2 == 0

    wave = np.zeros(n_use)
    half_n_use = int(n_use // 2)

    DE = np.zeros(n_use, dtype=np.complex128)

    om: NDArray[np.floating] = wc.dom * np.hstack([np.arange(0, half_n_use + 1), -np.arange(half_n_use - 1, 0, -1)])
    DE[:] = (
        np.sqrt(wc.dt)
        / np.sqrt(2.0)
        * (
            phitilde_vec(wc.dt * (om + m * wc.DOM), wc.Nf, wc.nx)
            + phitilde_vec(wc.dt * (om - m * wc.DOM), wc.Nf, wc.nx)
        )
    )

    DE_fft: NDArray[np.complex128] = spf.fft(DE, n_use, overwrite_x=True)

    del DE

    wave[half_n_use:] = np.real(DE_fft[0:half_n_use])
    wave[0:half_n_use] = np.real(DE_fft[half_n_use:])
    return 1.0 / nrm * wave


@njit(parallel=True)
def get_taylor_table_time_helper(
    wavelet_norm: NDArray[np.floating], wc: WDMWaveletConstants
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Compute cosine and sine Taylor expansion coefficient tables for a grid of frequency derivative values.

    This function leverages numba's Just-In-Time (JIT) compilation for efficient parallelized computation
    of the Taylor coefficient tables needed in the time-domain wavelet transform. For each
    frequency-derivative layer, it calculates arrays of cosine and sine coefficients
    for linear interpolation in the first-order Taylor expansion.

    Parameters
    ----------
    wavelet_norm : np.ndarray
        The normalized reference wavelet sampled on the relevant time grid.
    wc : WDMWaveletConstants
        Wavelet decomposition configuration, specifying time step,
        bandwidth, number of frequency derivatives, and related parameters.

    Returns
    -------
    evcs : np.ndarray
        Table of cosine coefficients in the first order Taylor expansion, shaped
        (number of frequency derivative layers, number of samples per layer).
    evss : np.ndarray
        Table of sine coefficients in the first order Taylor explansion, matching
        the shape of `evcs`. Used for reconstructing the wavelet basis
        functions and interpolating waveforms efficiently.

    Notes
    -----
    The returned tables are used for lookup and linear interpolation of Taylor-expanded wavelet
    coefficients, improving the speed of intrinsic_waveform computation for time-domain waveforms.

    This function can be time consuming for large tables, and so exploits parallel loops where feasible.
    """
    fd: NDArray[np.floating] = wc.dfd * np.arange(-wc.Nfd_negative, wc.Nfd - wc.Nfd_negative)  # set f-dot increments
    Nfsam: NDArray[np.integer] = ((wc.BW + np.abs(fd) * wc.Tw) / wc.df_bw).astype(np.int64)
    Nfsam[Nfsam % 2 == 1] += 1

    evcs = np.zeros((wc.Nfd, np.max(Nfsam)))
    evss = np.zeros((wc.Nfd, np.max(Nfsam)))
    z_add: float = np.pi / 16
    z_pre: float = np.pi * wc.df_bw * wc.dt
    z_quads = np.zeros(wc.K)
    for jj in range(wc.Nfd):
        for k in range(wc.K):
            z_quads[k] = np.pi * wc.dt**2 * fd[jj] * (k - wc.K // 2) ** 2
        for i in range(int(Nfsam[jj])):
            z_mult: float = z_add + z_pre * (1 - Nfsam[jj] + 2 * i)
            evc: float = 0.0
            evs: float = 0.0
            for k in prange(0, wc.K):
                z: float = z_mult * (k - wc.K // 2) + z_quads[k]
                evc += wavelet_norm[k] * np.cos(z)
                evs += wavelet_norm[k] * np.sin(z)
            assert evc != 0.0
            evcs[jj, i] = evc
            evss[jj, i] = evs
    return evcs, evss


def get_wavelet_norm(wc: WDMWaveletConstants) -> NDArray[np.floating]:
    """
    Compute a normalized reference wavelet needed for the time-domain Taylor coefficient table.

    This function constructs and normalizes a single reference WDM wavelet according
    to the configuration specified by `_wc`. The normalized wavelet is required both for
    assembling the Taylor time coefficients table and for certain getting the pixel directly.

    Parameters
    ----------
    wc : WDMWaveletConstants
        Wavelet decomposition configuration, specifying time step,
        frequency grid, windowing, and shape parameters required
        to construct the standard wavelet basis.

    Returns
    -------
    wavelet_norm : np.ndarray
        The normalized reference wavelet, as a 1D array, matching
        the configuration in `_wc`.
    """
    wave = wavelet(wc, 0, 1.0, n_in=wc.K)

    nrm: float = (1.0 / np.sqrt(2.0)) * float(np.linalg.norm(wave))
    # TODO check if this is the right way of getting the relevant m
    # kwave = int(wc.Nf // 16)
    kwave: float = wc.Nf / 16
    return wavelet(wc, kwave, nrm)


@njit()
def get_taylor_time_pixel_direct(
    fa: float, fda: float, k_in: int, wavelet_norm: NDArray[np.floating], wc: WDMWaveletConstants
) -> tuple[float, float]:
    """
    Compute Taylor expansion coefficients for a single time pixel in the wavelet domain.

    This function performs a direct, on-the-fly calculation of the Taylor expansion
    coefficients required for wavelet-domain interpolation at the given time pixel.
    The computation is carried out using the wavelet configuration parameters in `_wc`. This pixel-by-pixel
    approach provides somewhat higher accuracy and can be used for validation,
    testing, or situations where precomputed tables would be excessively large or slow to compute.
    Note while more accurate than the table-based method, it is still only the first-order Taylor approximation,
    so the overall reconstruction accuracy may not improve significantly.

    Parameters
    ----------
    fa: float
        The frequency at which to compute the Taylor expansion coefficients.
    fda: float
        The frequency derivative at which to compute the Taylor expansion coefficients.
    k_in: int
        The frequency index of the wavelet pixel for which to compute the coefficients.
    wavelet_norm: NDArray[np.floating]
        A single precomputed and normalized wavelet for reference.
    wc : WDMWaveletConstants
        Wavelet decomposition configuration containing resolution, frequency grid,
        and windowing information.

    Returns
    -------
    evc: float
        The cosine component of the Taylor expansion coefficient for the given time pixel.
    evs: float
        The sine component of the Taylor expansion coefficient for the given time pixel.

    Notes
    -----
    This method does not use or require precomputed interpolation tables. It is best suited
    for use cases demanding somewhat higher precision.

    See Also
    --------
    get_taylor_table_time : Compute or retrieve Taylor coefficient tables for a grid of points.
    wavemaket_direct : Construct sparse wavelet representations using direct coefficient evaluation.
    """
    dfa: float = fa - wc.DF * k_in
    # df_bw technically depends on Nsf, but the dependence cancels immediately
    xk: float = float(np.abs(dfa)) / wc.df_bw
    fd_mid: float = fda

    z_add: float = np.pi / 16
    z_pre: float = 2 * np.pi * wc.df_bw * wc.dt * xk
    z_quads_mid = np.zeros(wc.K)
    for k in range(wc.K):
        z_quads_mid[k] = np.pi * wc.dt**2 * fd_mid * (k - wc.K // 2) ** 2

    z_mult: float = z_add + z_pre
    evc_mid: float = 0.0
    evs_mid: float = 0.0

    for k in prange(wc.K):
        z_mid: float = z_mult * (k - wc.K // 2) + z_quads_mid[k]
        evc_mid += wavelet_norm[k] * np.cos(z_mid)
        evs_mid += wavelet_norm[k] * np.sin(z_mid)

    assert evc_mid != 0.0
    assert evs_mid != 0.0
    return evc_mid, evs_mid


def dfd_grid_spacing_check_helper(wc: WDMWaveletConstants, f_target: float, fd_target: float = 0., k_range: int = 1, atol_mult: float = 1.e-6, rtol_pix: float = 1.e-7, assert_mode: int = 1, dfd_mult: float = 1.0, wavelet_norm: NDArray[np.floating] | None = None) -> float:
    """Do some checks using the exact formula to deterimine if the dfd grid spacing appears adequate."""
    if wavelet_norm is None:
        wavelet_norm = get_wavelet_norm(wc)

    # determine the scale to set
    abs_scale: float = float(np.sum(np.abs(wavelet_norm)))
    atol_pix: float = atol_mult * abs_scale
    assert atol_pix > 0.

    # check if the dfd seems small enough for numerical stability
    k_base = int(np.floor(f_target / wc.DF))
    k_min: int = max(k_base - k_range, 1)
    k_max: int = min(k_base + k_range, wc.Nf - 1)
    dfd_mult_need: float = np.inf
    for k in range(k_min, k_max + 1):
        dfd_loc: float = dfd_mult * wc.dfd
        y_1, z_1 = get_taylor_time_pixel_direct(f_target, fd_target - dfd_loc, k, wavelet_norm, wc)
        y_2, z_2 = get_taylor_time_pixel_direct(f_target, fd_target, k, wavelet_norm, wc)
        y_3, z_3 = get_taylor_time_pixel_direct(f_target, fd_target + dfd_loc, k, wavelet_norm, wc)

        # central finite difference first and second derivative of y
        ypp: float = (y_1 - 2 * y_2 + y_3) / dfd_loc**2
        y_abs_error: float = float(np.abs(ypp)) * dfd_loc**2 / 2.0

        zpp: float = (z_1 - 2 * z_2 + z_3) / dfd_loc**2
        z_abs_error: float = float(np.abs(zpp)) * dfd_loc**2 / 2.0

        error_use: float = max(y_abs_error, z_abs_error)
        if error_use > 0:
            dfd_mult_need = min(dfd_mult_need, float(np.sqrt(atol_pix / error_use)))
        else:
            dfd_mult_need = min(dfd_mult_need, 1.)

        if assert_mode == 1:
            msg = f'The requested grid will not achieve target precision. Try decreasing dfdot by ~{dfd_mult_need}'
            assert_allclose(y_2, y_2 + y_abs_error, atol=atol_pix, rtol=rtol_pix, err_msg=msg)
            assert_allclose(z_2, z_2 + z_abs_error, atol=atol_pix, rtol=rtol_pix, err_msg=msg)
    return dfd_mult_need


@njit()
def _get_dtaylor_table_di_di(jj: float, i: float, wavelet_norm: NDArray[np.floating], wc: WDMWaveletConstants) -> tuple[float, float, float, float]:
    """Get derivative of taylor table with respect to indices."""
    fd: float = wc.dfd * (-wc.Nfd_negative + jj)  # set f-dot increments
    Nfsam: int = int((wc.BW + np.abs(fd) * wc.Tw) / wc.df_bw)
    if Nfsam % 2 == 1:
        Nfsam += 1

    z_add: float = np.pi / 16
    z_pre: float = np.pi * wc.df_bw * wc.dt
    z_quads = np.zeros(wc.K)
    for k in range(wc.K):
        z_quads[k] = np.pi * wc.dt**2 * fd * (k - wc.K // 2) ** 2

    z_mult: float = z_add + z_pre * (1 - Nfsam + 2 * i)
    # devc_di: float = 0.0
    # devs_di: float = 0.0
    evc: float = 0.0
    evs: float = 0.0
    devc_di_di: float = 0.0
    devs_di_di: float = 0.0
    for k in range(wc.K):
        z: float = z_mult * (k - wc.K // 2) + z_quads[k]
        dz_di: float = 2 * z_pre * (k - wc.K // 2)
        evc += wavelet_norm[k] * np.cos(z)
        evs += wavelet_norm[k] * np.sin(z)
        # devc_di -= wavelet_norm[k] * np.sin(z) * dz_di
        # devs_di += wavelet_norm[k] * np.cos(z) * dz_di
        devc_di_di -= wavelet_norm[k] * np.cos(z) * dz_di ** 2
        devs_di_di -= wavelet_norm[k] * np.sin(z) * dz_di ** 2
    return evc, evs, devc_di_di, devs_di_di


def df_bw_grid_spacing_check_helper(wc: WDMWaveletConstants, f_target: float, fd_target: float = 0., atol_mult: float = 1.e-6, rtol_pix: float = 1.e-7, assert_mode: int = 1, df_bw_mult: float = 1.0, wavelet_norm: NDArray[np.floating] | None = None) -> float:
    """Do some checks using the exact formula to deterimine if the df_bw grid spacing appears adequate"""
    if wavelet_norm is None:
        wavelet_norm = get_wavelet_norm(wc)

    # determine the scale to set
    abs_scale: float = float(np.sum(np.abs(wavelet_norm)))
    atol_pix: float = atol_mult * abs_scale
    assert atol_pix > 0.

    # check if the df_bw seems small enough for numerical stability
    y_2, z_2, ypp, zpp = _get_dtaylor_table_di_di(fd_target / wc.dfd, f_target / wc.df_bw, wavelet_norm, wc)

    # central finite difference first and second derivative of y
    y_abs_error: float = float(np.abs(ypp)) / 2.0 * df_bw_mult ** 2

    z_abs_error: float = float(np.abs(zpp)) / 2.0 * df_bw_mult ** 2

    error_use = max(y_abs_error, z_abs_error)

    if error_use <= 0.:
        df_bw_mult_need: float = 1.
    else:
        df_bw_mult_need = float(np.sqrt(atol_pix / error_use))

    assert df_bw_mult_need > 0.

    nsf_mult_need: float = 1. / df_bw_mult_need
    if assert_mode == 1:
        msg = f'The requested grid will not achieve target precision. Try increasing Nsf by ~{nsf_mult_need}'
        assert_allclose(y_2, y_2 + y_abs_error, atol=atol_pix, rtol=rtol_pix, err_msg=msg)
        assert_allclose(z_2, z_2 + z_abs_error, atol=atol_pix, rtol=rtol_pix, err_msg=msg)
    return df_bw_mult_need


def grid_check_helper(wc: WDMWaveletConstants, grid_check_mode: int, target_precision: float = 1.e-5, wavelet_norm: NDArray[np.floating] | None = None, assert_mode: int = 1) -> None:
    if wavelet_norm is None:
        wavelet_norm = get_wavelet_norm(wc)
    if grid_check_mode == 2:
        df_bw_mult_need: float = df_bw_grid_spacing_check_helper(wc, f_target=wc.DF * (wc.Nf - 2.), fd_target=0., atol_mult=target_precision, wavelet_norm=wavelet_norm, assert_mode=0)
        dfd_mult_need: float = dfd_grid_spacing_check_helper(wc, f_target=wc.DF * (wc.Nf - 2.), fd_target=0., atol_mult=target_precision, wavelet_norm=wavelet_norm, assert_mode=0)
    elif grid_check_mode == 1:
        fds = wc.dfd * np.arange(-wc.Nfd_negative, wc.Nfd - wc.Nfd_negative)
        df_bw_mult_need = df_bw_grid_spacing_check_helper(wc, f_target=wc.DF * (wc.Nf - 1.), fd_target=0., atol_mult=target_precision, wavelet_norm=wavelet_norm, assert_mode=0)
        t0_test_df_bw = perf_counter()
        for fd_target in [fds[-1], -wc.dfd / 2., 0., wc.dfd / 2., fds[1]]:
            for f_target in [1.5 * wc.DF, 2 * wc.DF, 2.5 * wc.DF, wc.DF * wc.Nf / 2 - wc.df_bw, wc.DF * wc.Nf / 2 - wc.df_bw / 2, (wc.DF * wc.Nf) / 2, wc.DF * wc.Nf / 2 + wc.df_bw / 2., wc.DF * wc.Nf / 2 + wc.df_bw, wc.DF * (wc.Nf - 1.5), wc.DF * (wc.Nf - 2), wc.DF * (wc.Nf - 2.5), wc.DF * (wc.Nf - 3)]:
                df_bw_mult_need = min(df_bw_mult_need, df_bw_grid_spacing_check_helper(wc, f_target=f_target, fd_target=fd_target, atol_mult=target_precision, wavelet_norm=wavelet_norm, assert_mode=0))
        tf_test_df_bw = perf_counter()
        print(f'Grid check for df_bw completed in {tf_test_df_bw - t0_test_df_bw} s')
        dfd_mult_need = dfd_grid_spacing_check_helper(wc, f_target=wc.DF * (wc.Nf - 1.), fd_target=0., atol_mult=target_precision, wavelet_norm=wavelet_norm, assert_mode=0)
        t0_test_dfd = perf_counter()
        for fd_target in [fds[-1], -wc.dfd / 2., 0., wc.dfd / 2., fds[1]]:
            for f_target in [1.5 * wc.DF, 2 * wc.DF, 2.5 * wc.DF, (wc.DF * wc.Nf) / 2, wc.DF * (wc.Nf - 1.5), wc.DF * (wc.Nf - 2), wc.DF * (wc.Nf - 2.5), wc.DF * (wc.Nf - 3)]:
                dfd_mult_need = min(dfd_mult_need, dfd_grid_spacing_check_helper(wc, f_target=f_target, fd_target=fd_target, atol_mult=target_precision, wavelet_norm=wavelet_norm, assert_mode=0))
        tf_test_dfd = perf_counter()
        print(f'Grid check for dfd completed in {tf_test_dfd - t0_test_dfd} s')
    elif grid_check_mode == 0:
        df_bw_mult_need = 1.
        dfd_mult_need = 1.
    else:
        msg = f'Unrecognized option for grid_check_mode {grid_check_mode}'
        raise ValueError(msg)

    failed = False
    if df_bw_mult_need < 0.95:
        failed = True
        assert df_bw_mult_need > 0.
        nsf_mult_need = 1. / df_bw_mult_need
        warn(f'The requested grid will not achieve target precision {target_precision}. Try increasing Nsf by ~{nsf_mult_need}', stacklevel=2)
    elif df_bw_mult_need > 2.:
        nsf_mult_need = 1. / df_bw_mult_need
        print(f'Nsf could safely be decreased by a factor of ~{nsf_mult_need}')

    if dfd_mult_need < 0.95:
        failed = True
        warn(f'The requested grid will not achieve target precision {target_precision}. Try decreasing dfd by ~{dfd_mult_need}', stacklevel=2)
    elif dfd_mult_need > 2.:
        print(f'dfd could safely be increased by ~{dfd_mult_need}')

    if failed and assert_mode == 1:
        msg = f'Taylor time interpolation will not achieve required precision {target_precision}'
        raise ValueError(msg)
    print(f'Taylor time interpolation expected to achieve specified precision {target_precision}')


def get_taylor_table_time(
    wc: WDMWaveletConstants,
    *,
    cache_mode: str = 'skip',
    output_mode: str = 'skip',
    cache_dir: str = 'coeffs/',
    filename_base: str = 'taylor_time_table_',
    grid_check_mode: int = 1,
    assert_mode: int = 1,
) -> WaveletTaylorTimeCoeffs:
    """
    Construct or retrieve the precomputed table of Taylor-expansion coefficients.

    Depending on the specified caching and output modes, this function either loads the
    coefficient table from cache (if available) or computes it on the fly according to the
    configuration in `_wc`. Optionally, it can write the newly generated table to disk for
    future reuse. The resulting interpolation table is organized over a grid of frequency
    derivative layers and sampled frequencies, enabling efficient evaluation of Taylor-expanded
    wavelet coefficients used in time-domain to wavelet-domain transforms.

    Parameters
    ----------
    wc : WDMWaveletConstants
        Configuration parameters for the wavelet decomposition; these
        define the frequency range, resolution, and grid structure
        used to generate the Taylor coefficient table.

    cache_mode : str, optional
        Specifies caching behavior for the coefficient table:
        - 'skip': Always compute a new table, do not check for a cache.
        - 'check': Attempt to load from a previously computed cached table; if not found, compute a new table.

    output_mode : str, optional
        Specifies behavior for saving the resulting table:
        - 'skip': Do not write any output file.
        - 'hf': Write the output to an HDF5 file in the specified cache directory.

    cache_dir : str, optional
        Directory in which to look for or write cached HDF5 files.
        (default is 'coeffs/')

    filename_base : str, optional
        Base file name for the cache file (before appending grid parameters).
        (default is 'taylor_time_table_')

    Returns
    -------
    WaveletTaylorTimeCoeffs
        Precomputed table of Taylor expansion coefficients and normalization values,
        indexed according to the frequency and frequency-derivative grid defined by `_wc`.

    Raises
    ------
    ValueError
        If the requested interpolation grid parameters exceed the valid range
        for the Taylor approximation, or if grid settings are inconsistent.

    Notes
    -----
    The precomputed coefficient table accelerates wavelet-domain interpolation of
    time-domain waveforms and is used by `wavemaket` for fast evaluation.
    Use direct Taylor evaluation when higher accuracy is required at individual pixels
    or when the table grid is insufficiently dense.

    See Also
    --------
    wavemaket : Use the precomputed Taylor table for sparse wavelet construction.
    wavemaket_direct : Direct Taylor evaluation without table lookup.
    """
    t0 = float(perf_counter())

    print('Filter length (seconds) %e' % wc.Tw)
    print('dt=' + str(wc.dt) + 's Tobs=' + str(wc.Tobs / SECSYEAR))

    print('full filter bandwidth %e  samples %d' % ((wc.A + wc.B) / np.pi, (wc.A + wc.B) / np.pi * wc.Tw))
    cache_good = False

    filename_cache = (
        cache_dir
        + filename_base
        + 'Nsf='
        + str(wc.Nsf)
        + '_mult='
        + str(wc.mult)
        + '_Nfd='
        + str(wc.Nfd)
        + '_Nf='
        + str(wc.Nf)
        + '_Nt='
        + str(wc.Nt)
        + '_dt='
        + str(wc.dt)
        + '_dfdot='
        + str(wc.dfdot)
        + '_Nfd_negative='
        + str(wc.Nfd_negative)
        + '_nx='
        + str(wc.nx)
        + '.h5'
    )

    fds: NDArray[np.floating] = wc.dfd * np.arange(-wc.Nfd_negative, wc.Nfd - wc.Nfd_negative)

    # number of samples for each frequency derivative layer (grow with increasing BW)
    Nfsam: NDArray[np.integer] = ((wc.BW + np.abs(fds) * wc.Tw) / wc.df_bw).astype(np.int64)
    Nfsam[Nfsam % 2 != 0] += 1  # makes sure it is an even number

    wavelet_norm = get_wavelet_norm(wc)

    max_shape = np.max(Nfsam)
    evcs = np.zeros((wc.Nfd, max_shape))
    evss = np.zeros((wc.Nfd, max_shape))

    if cache_mode == 'check':
        try:
            hf_in = h5py.File(filename_cache, 'r')
            wavelet_norm = np.asarray(hf_in['wavelet_norm'])
            evcs[:] = np.asarray(hf_in['evcs'])
            evss[:] = np.asarray(hf_in['evss'])
            hf_in.close()
            cache_good = True
        except OSError:
            print('Cache target: ' + str(filename_cache))
            print('Cache checked and missed')
    elif cache_mode == 'skip':
        pass
    else:
        msg = f'Unrecognized option for cache_mode {cache_mode}'
        raise NotImplementedError(msg)

    grid_check_helper(wc, grid_check_mode, target_precision=wc.taylor_time_interpolation_target_precision, wavelet_norm=wavelet_norm, assert_mode=assert_mode)

    if not cache_good:
        fd = wc.DF / wc.Tw * wc.dfdot * np.arange(-wc.Nfd_negative, wc.Nfd - wc.Nfd_negative)  # set f-dot increments

        max_fd = np.max(np.abs(fd))
        max_allow = 8 * wc.DF / wc.Tw * (1. + wc.max_freq_tol_time_interpolation)  # small factor to account for numerical inexactness
        if max_fd > max_allow:
            msg = f'Requested interpolation grid max {max_fd} exceeds valid range of time domain taylor approximation {max_allow}'
            if assert_mode:
                raise ValueError(msg)
            warn(msg, stacklevel=2)

        if not np.any(fd == 0.0):
            msg = 'Requested frequency derivative grid does not contain zero; results may be unexpected'
            if assert_mode:
                raise ValueError(msg)
            warn(msg, stacklevel=2)

        print(
            'DT=%e DF=%.14e DOM/2pi=%.14e fd1=%e fd-1=%e' % (wc.DT, wc.DF, wc.DOM / (2 * np.pi), fd[1], fd[wc.Nfd - 1])
        )

        Nfsam = ((wc.BW + np.abs(fd) * wc.Tw) / wc.df_bw).astype(np.int64)
        odd_mask = np.mod(Nfsam, 2) != 0
        Nfsam[odd_mask] += 1

        # The odd wavelets coefficiens can be obtained from the even.
        # odd cosine = -even sine, odd sine = even cosine

        # each wavelet covers a frequency band of width DW
        # except for the first and last wavelets
        # there is some overlap. The wavelet pixels are of width
        # DOM/PI, except for the first and last which have width
        # half that

        t1 = float(perf_counter())
        print('Taylor Time Table Loop start time ', t1 - t0, 's')
        evcs, evss = get_taylor_table_time_helper(wavelet_norm, wc)
        tf = perf_counter()
        print('Got Time Taylor Table in %f s' % (tf - t1))
        if output_mode == 'hf':
            hf = h5py.File(filename_cache, 'w')

            wc_group = hf.create_group('_wc')
            for key in wc._fields:
                wc_group.attrs[key] = getattr(wc, key)

            _ = hf.create_dataset('wavelet_norm', data=wavelet_norm, compression='gzip')
            _ = hf.create_dataset('fd', data=fd, compression='gzip')
            _ = hf.create_dataset('Nfsam', data=Nfsam, compression='gzip')
            _ = hf.create_dataset('evcs', data=evcs, compression='gzip')
            _ = hf.create_dataset('evss', data=evss, compression='gzip')
            hf.close()
            t3 = perf_counter()
            print('Taylor Time Table output time', t3 - tf, 's')
        elif output_mode == 'skip':
            pass
        else:
            msg = 'unrecognized option for output_mode'
            raise NotImplementedError(msg)

    return WaveletTaylorTimeCoeffs(Nfsam, evcs, evss, wavelet_norm)
