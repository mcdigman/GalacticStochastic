"""Numba-compiled dense-stripe coaddition alternatives to ``wavemaket``.

This module provides three supported ``njit`` functions that coadd a single
time-domain waveform directly into a narrow, dense wavelet-domain frequency
stripe, reproducing the ``force_nulls=0``, ``amplitude_order=0`` behavior of
:func:`WaveletWaveforms.taylor_time_wavelet_funcs.wavemaket` over the stripe.

Unlike ``wavemaket``, which builds a sparse representation over the full
wavelet band, these functions write only into the contiguous global
frequency-layer window ``[nf_start, nf_start + stripe_height)`` and never
materialize a full-band ``(wc.Nt, wc.Nf, Nc)`` intermediate. They are intended
for future higher-level parallel coaddition over independent frequency stripes.

The functions are:

- :func:`wavemaket_stripe_sparse` -- preserves the variable-``k`` inner-loop
  structure of ``wavemaket`` while writing directly into the stripe.
- :func:`wavemaket_stripe_dense` -- fixed-``k`` inner loop over the stripe
  window using the original :class:`WaveletTaylorTimeCoeffs` table.
- :func:`wavemaket_stripe_dense_aligned` -- fixed-``k`` inner loop using a
  zero-padded, integer-aligned table (:class:`WaveletTaylorTimeCoeffsAligned`)
  that supports constant-stride ``R`` reads when the input grid is integer
  aligned (``2 * wc.Nsf % 3 == 0`` and ``wc.DF / wc.df_bw == 2 * wc.Nsf // 3``).

All three accumulate into ``wavelet_stripe`` in place with ``+=``.
"""

from typing import NamedTuple

import numpy as np
from numba import njit
from numpy.typing import NDArray

from LisaWaveformTools.stationary_source_waveform import StationaryWaveformTime
from WaveletWaveforms.sparse_waveform_functions import PixelGenericRange
from WaveletWaveforms.taylor_time_coefficients import WaveletTaylorTimeCoeffs
from WaveletWaveforms.wdm_config import WDMWaveletConstants


class WaveletTaylorTimeCoeffsAligned(NamedTuple):
    """Zero-padded, integer-aligned Taylor time interpolation table.

    The coefficient values are reused unchanged from a
    :class:`WaveletTaylorTimeCoeffs` table on the original ``wc.df_bw`` grid,
    but each frequency-derivative row is copied into a wider array with ``pad``
    zeros on both sides of its original valid coefficient range. The original
    valid coefficients for row ``n`` occupy padded columns
    ``[pad, pad + Nfsam[n])``; everything outside that interval is zero padding.

    This layout supports the integer constant-stride ``R`` reads used by
    :func:`wavemaket_stripe_dense_aligned`, where ``R == 2 * wc.Nsf // 3`` equals
    the integer ratio ``wc.DF / wc.df_bw`` for integer-aligned configurations.

    Fields
    ------
    Nfsam : NDArray[np.integer]
        Number of original valid coefficients per frequency-derivative row;
        together with ``pad`` this is the valid-region metadata distinguishing
        original coefficients from left and right padding.
    evc : NDArray[np.floating]
        Two-sided zero-padded cosine coefficients, shape ``(Nfd, width)``.
    evs : NDArray[np.floating]
        Two-sided zero-padded sine coefficients, shape ``(Nfd, width)``.
    wavelet_norm : NDArray[np.floating]
        Normalized reference wavelet, preserved from the original table.
    pad : int
        Width of the zero padding on each side of every row's valid range.
    R : int
        Integer frequency-layer stride, equal to ``2 * Nsf // 3`` and to the
        integer ratio ``wc.DF / wc.df_bw`` for integer-aligned configurations.
    Nfd_negative : int
        Number of negative frequency-derivative layers, preserving the original
        frequency-derivative grid origin.
    """

    Nfsam: NDArray[np.integer]
    evc: NDArray[np.floating]
    evs: NDArray[np.floating]
    wavelet_norm: NDArray[np.floating]
    pad: int
    R: int
    Nfd_negative: int


def build_aligned_taylor_time_table(
    taylor_table: WaveletTaylorTimeCoeffs, wc: WDMWaveletConstants
) -> WaveletTaylorTimeCoeffsAligned:
    """Build a zero-padded, integer-aligned table from an original Taylor time table.

    The aligned table reuses the original coefficient values bitwise on the
    original ``wc.df_bw`` grid; it only relocates each row's valid coefficients
    into a wider array with two-sided zero padding and records the integer
    stride ``R`` and valid-region metadata. The padding width ``pad`` is chosen
    as ``max(Nfsam)`` so that every table index reachable in the supported
    stripe-height domain (whose constant-stride reach is at most
    ``R * (stripe_height - 1)``) stays in bounds.

    Parameters
    ----------
    taylor_table : WaveletTaylorTimeCoeffs
        Original Taylor time interpolation table on the ``wc.df_bw`` grid.
    wc : WDMWaveletConstants
        Wavelet decomposition configuration. Must be integer aligned:
        ``2 * wc.Nsf % 3 == 0`` and ``wc.DF / wc.df_bw == 2 * wc.Nsf // 3``.

    Returns
    -------
    WaveletTaylorTimeCoeffsAligned
        The zero-padded, integer-aligned table.
    """
    assert 2 * wc.Nsf % 3 == 0, 'Aligned table requires 2 * wc.Nsf divisible by 3'
    r_stride = 2 * wc.Nsf // 3
    assert abs(wc.DF / wc.df_bw - r_stride) <= 1e-15 * max(1.0, abs(r_stride)), 'wc.DF / wc.df_bw must equal 2 * wc.Nsf // 3'

    nfsam = taylor_table.Nfsam
    assert nfsam.ndim == 1
    assert nfsam.size == wc.Nfd
    assert taylor_table.evc.shape == taylor_table.evs.shape
    assert taylor_table.evc.shape[0] == wc.Nfd

    max_nfsam = int(np.max(nfsam))
    pad = max_nfsam
    width = 2 * pad + max_nfsam

    evc_padded: NDArray[np.floating] = np.zeros((wc.Nfd, width))
    evs_padded: NDArray[np.floating] = np.zeros((wc.Nfd, width))
    for n in range(wc.Nfd):
        n_valid = int(nfsam[n])
        evc_padded[n, pad:pad + n_valid] = taylor_table.evc[n, :n_valid]
        evs_padded[n, pad:pad + n_valid] = taylor_table.evs[n, :n_valid]

    return WaveletTaylorTimeCoeffsAligned(
        nfsam.copy(),
        evc_padded,
        evs_padded,
        taylor_table.wavelet_norm.copy(),
        pad,
        r_stride,
        wc.Nfd_negative,
    )


@njit(fastmath=True)
def wavemaket_stripe_sparse(
    wavelet_stripe: NDArray[np.floating],
    waveform: StationaryWaveformTime,
    nf_start: int,
    stripe_height: int,
    nt_lim_waveform: PixelGenericRange,
    wc: WDMWaveletConstants,
    taylor_table: WaveletTaylorTimeCoeffs,
) -> None:
    """Coadd one waveform into a dense frequency stripe, preserving ``wavemaket``'s loop.

    This reproduces the ``force_nulls=0``, ``amplitude_order=0`` behavior of
    :func:`WaveletWaveforms.taylor_time_wavelet_funcs.wavemaket` over the global
    frequency-layer window ``[nf_start, nf_start + stripe_height)``, keeping the
    original variable-``k`` inner loop (bounded by the per-pixel bandwidth) and
    restricting writes to the stripe window. Contributions are accumulated in
    place: ``wavelet_stripe[j, k - nf_start, c] += value``.

    Parameters
    ----------
    wavelet_stripe : NDArray[np.floating]
        Output stripe of shape ``(wc.Nt, stripe_height, waveform.AT.shape[0])``,
        C- or F-contiguous. Accumulated into in place.
    waveform : StationaryWaveformTime
        Time-domain waveform with amplitude, phase, frequency, and frequency
        derivative arrays of shape ``(Nc, wc.Nt)``.
    nf_start : int
        Lowest global frequency layer of the stripe window.
    stripe_height : int
        Number of frequency layers in the stripe; one of ``2, 3, 4, 5``.
    nt_lim_waveform : PixelGenericRange
        Time-pixel range to process.
    wc : WDMWaveletConstants
        Wavelet decomposition configuration.
    taylor_table : WaveletTaylorTimeCoeffs
        Original Taylor time interpolation table.

    Returns
    -------
    None
        ``wavelet_stripe`` is modified in place.
    """
    assert wavelet_stripe.ndim == 3
    assert wavelet_stripe.shape[0] == wc.Nt
    assert wavelet_stripe.shape[1] == stripe_height
    assert wavelet_stripe.shape[2] == waveform.AT.shape[0]
    assert stripe_height in (2, 3, 4, 5)
    assert wc.Nt % 2 == 0
    assert 0 <= nf_start
    assert nf_start + stripe_height <= wc.Nf
    assert wavelet_stripe.flags.c_contiguous or wavelet_stripe.flags.f_contiguous
    assert nt_lim_waveform.nx_min >= 0
    assert nt_lim_waveform.nx_max <= wc.Nt
    assert nt_lim_waveform.nx_min <= nt_lim_waveform.nx_max
    assert taylor_table.Nfsam.size == wc.Nfd

    nc_waveform: int = wavelet_stripe.shape[2]
    nf_top: int = nf_start + stripe_height - 1

    for itrc in range(nc_waveform):
        for j in range(nt_lim_waveform.nx_min, nt_lim_waveform.nx_max):
            j_ind: int = j

            y0: float = waveform.FTd[itrc, j] / wc.dfd
            ny: int = int(np.floor(y0))
            n_ind: int = ny + wc.Nfd_negative

            if 0 <= n_ind < wc.Nfd - 1:
                cval: float = np.cos(waveform.PT[itrc, j])
                sval: float = np.sin(waveform.PT[itrc, j])

                dy: float = y0 - ny
                fa: float = waveform.FT[itrc, j]
                za: float = fa / wc.df_bw

                nfsam1_loc: int = int(taylor_table.Nfsam[n_ind])
                nfsam2_loc: int = int(taylor_table.Nfsam[n_ind + 1])
                half_bandwidth: float = (min(nfsam1_loc, nfsam2_loc) - 1) * wc.df_bw / 2

                kmin: int = max(nf_start, int(np.ceil((fa - half_bandwidth) / wc.DF)))
                kmax: int = min(nf_top, int(np.floor((fa + half_bandwidth) / wc.DF)))

                for k in range(kmin, kmax + 1):
                    zmid: float = (wc.DF / wc.df_bw) * k

                    if za < zmid:
                        zmid = za - np.abs(za - zmid)

                    kk_float: float = np.floor(za - zmid - 0.5)
                    zsam: float = zmid + kk_float + 0.5
                    kk: int = int(kk_float)
                    dx: float = za - zsam

                    jj1: int = kk + nfsam1_loc // 2
                    jj2: int = kk + nfsam2_loc // 2

                    if (0 <= jj1 < nfsam1_loc - 1) and (0 <= jj2 < nfsam2_loc - 1):
                        y: float = (1.0 - dx) * taylor_table.evc[n_ind, jj1] + dx * taylor_table.evc[n_ind, jj1 + 1]
                        yy: float = (1.0 - dx) * taylor_table.evc[n_ind + 1, jj2] + dx * taylor_table.evc[n_ind + 1, jj2 + 1]
                        z: float = (1.0 - dx) * taylor_table.evs[n_ind, jj1] + dx * taylor_table.evs[n_ind, jj1 + 1]
                        zz: float = (1.0 - dx) * taylor_table.evs[n_ind + 1, jj2] + dx * taylor_table.evs[n_ind + 1, jj2 + 1]

                        mult1: float = waveform.AT[itrc, j]
                        y1: float = ((1.0 - dy) * y + dy * yy) * mult1
                        z1: float = ((1.0 - dy) * z + dy * zz) * mult1

                        if (j_ind + k) % 2:
                            value: float = -(cval * z1 + sval * y1)
                        else:
                            value = cval * y1 - sval * z1

                        wavelet_stripe[j, k - nf_start, itrc] += value


@njit(fastmath=True)
def wavemaket_stripe_dense(
    wavelet_stripe: NDArray[np.floating],
    waveform: StationaryWaveformTime,
    nf_start: int,
    stripe_height: int,
    nt_lim_waveform: PixelGenericRange,
    wc: WDMWaveletConstants,
    taylor_table: WaveletTaylorTimeCoeffs,
) -> None:
    """Coadd one waveform into a dense frequency stripe with a fixed-``k`` loop.

    This reproduces the ``force_nulls=0``, ``amplitude_order=0`` behavior of
    :func:`WaveletWaveforms.taylor_time_wavelet_funcs.wavemaket` over the stripe
    window ``[nf_start, nf_start + stripe_height)`` using a fixed-``k`` inner
    loop over the whole window and the original
    :class:`WaveletTaylorTimeCoeffs` table. Frequency layers ``k`` failing the
    oracle bandwidth or table-overflow guards contribute exactly zero.
    Contributions are accumulated in place with ``+=``.

    Parameters
    ----------
    wavelet_stripe : NDArray[np.floating]
        Output stripe of shape ``(wc.Nt, stripe_height, waveform.AT.shape[0])``,
        C- or F-contiguous. Accumulated into in place.
    waveform : StationaryWaveformTime
        Time-domain waveform with amplitude, phase, frequency, and frequency
        derivative arrays of shape ``(Nc, wc.Nt)``.
    nf_start : int
        Lowest global frequency layer of the stripe window.
    stripe_height : int
        Number of frequency layers in the stripe; one of ``2, 3, 4, 5``.
    nt_lim_waveform : PixelGenericRange
        Time-pixel range to process.
    wc : WDMWaveletConstants
        Wavelet decomposition configuration.
    taylor_table : WaveletTaylorTimeCoeffs
        Original Taylor time interpolation table.

    Returns
    -------
    None
        ``wavelet_stripe`` is modified in place.
    """
    assert wavelet_stripe.ndim == 3
    assert wavelet_stripe.shape[0] == wc.Nt
    assert wavelet_stripe.shape[1] == stripe_height
    assert wavelet_stripe.shape[2] == waveform.AT.shape[0]
    assert stripe_height in (2, 3, 4, 5)
    assert wc.Nt % 2 == 0
    assert 0 <= nf_start
    assert nf_start + stripe_height <= wc.Nf
    assert wavelet_stripe.flags.c_contiguous or wavelet_stripe.flags.f_contiguous
    assert nt_lim_waveform.nx_min >= 0
    assert nt_lim_waveform.nx_max <= wc.Nt
    assert nt_lim_waveform.nx_min <= nt_lim_waveform.nx_max
    assert taylor_table.Nfsam.size == wc.Nfd

    nc_waveform: int = wavelet_stripe.shape[2]

    for itrc in range(nc_waveform):
        for j in range(nt_lim_waveform.nx_min, nt_lim_waveform.nx_max):
            j_ind: int = j

            y0: float = waveform.FTd[itrc, j] / wc.dfd
            ny: int = int(np.floor(y0))
            n_ind: int = ny + wc.Nfd_negative

            if 0 <= n_ind < wc.Nfd - 1:
                cval: float = np.cos(waveform.PT[itrc, j])
                sval: float = np.sin(waveform.PT[itrc, j])

                dy: float = y0 - ny
                fa: float = waveform.FT[itrc, j]
                za: float = fa / wc.df_bw

                nfsam1_loc: int = int(taylor_table.Nfsam[n_ind])
                nfsam2_loc: int = int(taylor_table.Nfsam[n_ind + 1])
                half_bandwidth: float = (min(nfsam1_loc, nfsam2_loc) - 1) * wc.df_bw / 2

                kmin: int = max(nf_start, int(np.ceil((fa - half_bandwidth) / wc.DF)))
                kmax: int = min(nf_start + stripe_height - 1, int(np.floor((fa + half_bandwidth) / wc.DF)))

                mult1: float = waveform.AT[itrc, j]

                for k in range(nf_start, nf_start + stripe_height):
                    zmid: float = (wc.DF / wc.df_bw) * k

                    if za < zmid:
                        zmid = za - np.abs(za - zmid)

                    kk_float: float = np.floor(za - zmid - 0.5)
                    zsam: float = zmid + kk_float + 0.5
                    kk: int = int(kk_float)
                    dx: float = za - zsam

                    jj1: int = kk + nfsam1_loc // 2
                    jj2: int = kk + nfsam2_loc // 2

                    if (kmin <= k <= kmax) and (0 <= jj1 < nfsam1_loc - 1) and (0 <= jj2 < nfsam2_loc - 1):
                        y: float = (1.0 - dx) * taylor_table.evc[n_ind, jj1] + dx * taylor_table.evc[n_ind, jj1 + 1]
                        yy: float = (1.0 - dx) * taylor_table.evc[n_ind + 1, jj2] + dx * taylor_table.evc[n_ind + 1, jj2 + 1]
                        z: float = (1.0 - dx) * taylor_table.evs[n_ind, jj1] + dx * taylor_table.evs[n_ind, jj1 + 1]
                        zz: float = (1.0 - dx) * taylor_table.evs[n_ind + 1, jj2] + dx * taylor_table.evs[n_ind + 1, jj2 + 1]

                        y1: float = ((1.0 - dy) * y + dy * yy) * mult1
                        z1: float = ((1.0 - dy) * z + dy * zz) * mult1

                        if (j_ind + k) % 2:
                            value: float = -(cval * z1 + sval * y1)
                        else:
                            value = cval * y1 - sval * z1

                        wavelet_stripe[j, k - nf_start, itrc] += value


@njit(fastmath=True)
def wavemaket_stripe_dense_aligned(
    wavelet_stripe: NDArray[np.floating],
    waveform: StationaryWaveformTime,
    nf_start: int,
    stripe_height: int,
    nt_lim_waveform: PixelGenericRange,
    wc: WDMWaveletConstants,
    taylor_table_aligned: WaveletTaylorTimeCoeffsAligned,
) -> None:
    """Coadd one waveform into a dense frequency stripe using the aligned table.

    This reproduces the ``force_nulls=0``, ``amplitude_order=0`` behavior of
    :func:`WaveletWaveforms.taylor_time_wavelet_funcs.wavemaket` over the stripe
    window ``[nf_start, nf_start + stripe_height)`` using a fixed-``k`` inner
    loop and the zero-padded, integer-aligned
    :class:`WaveletTaylorTimeCoeffsAligned` table. When the input grid is integer
    aligned, ``wc.DF / wc.df_bw`` equals the integer stride ``R``, so the table
    index advances by the constant stride ``R`` per frequency layer, and the
    two-sided zero padding keeps those constant-stride reads in bounds.

    The bandwidth and table-overflow drops are reproduced by the same per-pixel
    bandwidth bounds and ``jj1``/``jj2`` overflow guard as ``wavemaket``, so
    dropped cells contribute exactly zero rather than relying on zero padding to
    blend away non-negligible edge coefficients. The drop arithmetic mirrors
    ``wavemaket`` exactly so the table-overflow drop decision is reproduced
    bit-for-bit under ``fastmath``. Contributions are accumulated in place with
    ``+=``.

    Parameters
    ----------
    wavelet_stripe : NDArray[np.floating]
        Output stripe of shape ``(wc.Nt, stripe_height, waveform.AT.shape[0])``,
        C- or F-contiguous. Accumulated into in place.
    waveform : StationaryWaveformTime
        Time-domain waveform with amplitude, phase, frequency, and frequency
        derivative arrays of shape ``(Nc, wc.Nt)``.
    nf_start : int
        Lowest global frequency layer of the stripe window.
    stripe_height : int
        Number of frequency layers in the stripe; one of ``2, 3, 4, 5``.
    nt_lim_waveform : PixelGenericRange
        Time-pixel range to process.
    wc : WDMWaveletConstants
        Wavelet decomposition configuration. Must be integer aligned:
        ``2 * wc.Nsf % 3 == 0`` and ``wc.DF / wc.df_bw == 2 * wc.Nsf // 3``.
    taylor_table_aligned : WaveletTaylorTimeCoeffsAligned
        Zero-padded, integer-aligned Taylor time interpolation table.

    Returns
    -------
    None
        ``wavelet_stripe`` is modified in place.
    """
    assert wavelet_stripe.ndim == 3
    assert wavelet_stripe.shape[0] == wc.Nt
    assert wavelet_stripe.shape[1] == stripe_height
    assert wavelet_stripe.shape[2] == waveform.AT.shape[0]
    assert stripe_height in (2, 3, 4, 5)
    assert wc.Nt % 2 == 0
    assert 0 <= nf_start
    assert nf_start + stripe_height <= wc.Nf
    assert wavelet_stripe.flags.c_contiguous or wavelet_stripe.flags.f_contiguous
    assert nt_lim_waveform.nx_min >= 0
    assert nt_lim_waveform.nx_max <= wc.Nt
    assert nt_lim_waveform.nx_min <= nt_lim_waveform.nx_max

    # integer-alignment preconditions. The ratio bound abs(wc.DF / wc.df_bw - r_stride)
    # <= 1e-15 * max(1.0, abs(r_stride)) is asserted in the algebraically identical
    # cross-multiplied form (wc.df_bw > 0). Writing wc.DF / wc.df_bw here would let
    # fastmath common-subexpression-eliminate it with the loop's (wc.DF / wc.df_bw) * k,
    # shifting zmid by one ULP and flipping the oracle's table-overflow drop decision.
    assert 2 * wc.Nsf % 3 == 0
    r_stride: int = 2 * wc.Nsf // 3
    assert abs(wc.DF - r_stride * wc.df_bw) <= 1e-15 * max(1.0, abs(r_stride)) * wc.df_bw

    # aligned-table consistency with wc and the supported stripe-height domain
    assert taylor_table_aligned.R == r_stride
    assert taylor_table_aligned.Nfd_negative == wc.Nfd_negative
    assert taylor_table_aligned.Nfsam.size == wc.Nfd
    assert taylor_table_aligned.evc.shape[0] == wc.Nfd
    assert taylor_table_aligned.evc.shape == taylor_table_aligned.evs.shape
    assert taylor_table_aligned.pad >= 1
    assert taylor_table_aligned.evc.shape[1] >= taylor_table_aligned.pad + int(np.max(taylor_table_aligned.Nfsam)) + 1

    nc_waveform: int = wavelet_stripe.shape[2]
    pad: int = taylor_table_aligned.pad

    for itrc in range(nc_waveform):
        for j in range(nt_lim_waveform.nx_min, nt_lim_waveform.nx_max):
            j_ind: int = j

            y0: float = waveform.FTd[itrc, j] / wc.dfd
            ny: int = int(np.floor(y0))
            n_ind: int = ny + wc.Nfd_negative

            if 0 <= n_ind < wc.Nfd - 1:
                cval: float = np.cos(waveform.PT[itrc, j])
                sval: float = np.sin(waveform.PT[itrc, j])

                dy: float = y0 - ny
                fa: float = waveform.FT[itrc, j]
                za: float = fa / wc.df_bw

                nfsam1_loc: int = int(taylor_table_aligned.Nfsam[n_ind])
                nfsam2_loc: int = int(taylor_table_aligned.Nfsam[n_ind + 1])
                half_bandwidth: float = (min(nfsam1_loc, nfsam2_loc) - 1) * wc.df_bw / 2

                kmin: int = max(nf_start, int(np.ceil((fa - half_bandwidth) / wc.DF)))
                kmax: int = min(nf_start + stripe_height - 1, int(np.floor((fa + half_bandwidth) / wc.DF)))

                mult1: float = waveform.AT[itrc, j]

                for k in range(nf_start, nf_start + stripe_height):
                    # For an integer-aligned grid wc.DF / wc.df_bw == r_stride, so this
                    # quantity advances by the integer stride r_stride per layer and jj1
                    # reads at a constant stride r_stride from the padded layout. The ratio
                    # is evaluated exactly as in wavemaket (not as the integer r_stride * k)
                    # because wavemaket runs under fastmath=True, where reassociation puts
                    # the product up to one ULP from r_stride * k; matching the expression
                    # and the per-pixel bandwidth bounds reproduces the bandwidth and
                    # table-overflow drop decisions bit-for-bit.
                    zmid: float = (wc.DF / wc.df_bw) * k

                    if za < zmid:
                        zmid = za - np.abs(za - zmid)

                    kk_float: float = np.floor(za - zmid - 0.5)
                    zsam: float = zmid + kk_float + 0.5
                    kk: int = int(kk_float)
                    dx: float = za - zsam

                    jj1: int = kk + nfsam1_loc // 2
                    jj2: int = kk + nfsam2_loc // 2

                    if (kmin <= k <= kmax) and (0 <= jj1 < nfsam1_loc - 1) and (0 <= jj2 < nfsam2_loc - 1):
                        idx1: int = jj1 + pad
                        idx2: int = jj2 + pad

                        y: float = (1.0 - dx) * taylor_table_aligned.evc[n_ind, idx1] + dx * taylor_table_aligned.evc[n_ind, idx1 + 1]
                        yy: float = (1.0 - dx) * taylor_table_aligned.evc[n_ind + 1, idx2] + dx * taylor_table_aligned.evc[n_ind + 1, idx2 + 1]
                        z: float = (1.0 - dx) * taylor_table_aligned.evs[n_ind, idx1] + dx * taylor_table_aligned.evs[n_ind, idx1 + 1]
                        zz: float = (1.0 - dx) * taylor_table_aligned.evs[n_ind + 1, idx2] + dx * taylor_table_aligned.evs[n_ind + 1, idx2 + 1]

                        y1: float = ((1.0 - dy) * y + dy * yy) * mult1
                        z1: float = ((1.0 - dy) * z + dy * zz) * mult1

                        if (j_ind + k) % 2:
                            value: float = -(cval * z1 + sval * y1)
                        else:
                            value = cval * y1 - sval * z1

                        wavelet_stripe[j, k - nf_start, itrc] += value
