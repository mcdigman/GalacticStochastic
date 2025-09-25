"""Functions for getting the wavelet representation using the sparse method"""

from typing import NamedTuple

import numpy as np
import scipy.fft as spf
from numba import njit
from numpy.typing import NDArray

from LisaWaveformTools.stationary_source_waveform import StationarySourceWaveform, StationaryWaveformTime
from WaveletWaveforms.sparse_waveform_functions import PixelGenericRange, SparseWaveletWaveform
from WaveletWaveforms.taylor_time_coefficients import wavelet
from WaveletWaveforms.wdm_config import WDMWaveletConstants


class SparseTimeCoefficientTable(NamedTuple):
    cM: NDArray[np.complexfloating]
    sM: NDArray[np.complexfloating]


def get_sparse_source_t_grid(wc: WDMWaveletConstants, t0: float) -> PixelGenericRange:
    """Get the time grid needed to evaluate the intrinsic waveform for the sparse source time transform."""
    assert wc.K % wc.L == 0, 'K currently needs to be an integer multiple of L'
    sparse_thin = int(wc.K // wc.L)
    assert wc.Nf % sparse_thin == 0, 'Nf currently needs to be an integer multiple of sparse thinning factor'

    M = int(wc.Nf * wc.Nt / sparse_thin)

    dtd = float(wc.dt * sparse_thin)  # downsampled data spacing in time
    return PixelGenericRange(0, M, dtd, t0)


@njit()
def _get_sparse_m_pixel_helper(waveform: StationaryWaveformTime, wc: WDMWaveletConstants) -> NDArray[np.integer]:
    """Get the frequency indices of the pixels specified by the frequencies in the source waveform."""
    assert wc.K % wc.L == 0, 'K currently needs to be an integer multiple of L'
    sparse_thin = int(wc.K // wc.L)
    assert wc.Nf % sparse_thin == 0, 'Nf currently needs to be an integer multiple of sparse thinning factor'

    thin_factor = int(wc.Nf // sparse_thin)

    return (waveform.FT[:, ::thin_factor] / wc.DF).astype(np.int64)


def _update_bounds_helper(m_pixel: NDArray[np.integer], wc: WDMWaveletConstants) -> PixelGenericRange:
    """Update the minimum and maximum time pixels that could be needed."""
    # currently assumes frequency is generally increasing
    nt_mins: NDArray[np.integer] = (np.argmax(m_pixel > -1, axis=-1)).astype(np.int64)
    nt_maxs: NDArray[np.integer] = (wc.Nt - np.argmax(m_pixel[:, ::-1] < wc.Nf, axis=-1)).astype(np.int64)
    return PixelGenericRange(int(np.max(nt_mins)), int(np.min(nt_maxs)), wc.DT, 0.)


@njit(fastmath=True)
def _sparse_time_assign_loop(m_pixel: NDArray[np.integer], nt_lim: PixelGenericRange, waveform: StationaryWaveformTime, sparse_table: SparseTimeCoefficientTable, wc: WDMWaveletConstants) -> NDArray[np.complexfloating]:
    """Pack the array that will be fourier transformed."""
    nc_waveform = m_pixel.shape[0]
    assert wc.L % 2 == 0

    half_L = int(wc.L // 2)

    p = float(wc.K / wc.L)
    M = int(wc.Nf * wc.Nt / p)

    # TODO: it would be better to have DX stored and replaced, rather than be reallocated with each loop
    DX = np.zeros((nc_waveform, nt_lim.nx_max - nt_lim.nx_min, wc.L), dtype=np.complex128)

    nn = int((wc.Nf * wc.L) / wc.K)

    AT = waveform.AT
    PT = waveform.PT

    for itrc in range(nc_waveform):
        # cos_p: NDArray[np.floating] = AT[itrc] * np.cos(PT[itrc])
        # sin_p: NDArray[np.floating] = AT[itrc] * np.sin(PT[itrc])
        exp_p: NDArray[np.floating] = AT[itrc] * np.exp(1j * PT[itrc])
        for n in range(nt_lim.nx_min, nt_lim.nx_max):
            n_itr: int = n - nt_lim.nx_min
            ii: int = nn * n
            mc: int = int(m_pixel[itrc, n])
            i_min: int = max(half_L - ii, 0)
            i_max: int = min(wc.L, M + half_L - ii)

            for i in range(i_min, i_max):
                k: int = i + ii - half_L
                # TODO: why doesn't the second set of coefficients matter? c_m = 1j * s_m, currently
                # DX[itrc, n_itr, i] = cos_p[k] * sparse_table.cM[mc][i] + sin_p[k] * sparse_table.sM[mc][i]
                # DX[itrc, n_itr, i] = cos_p[k] * sparse_table.cM[mc][i] + 1j * sin_p[k] * sparse_table.cM[mc][i]
                # DX[itrc, n_itr, i] = (cos_p[k] + 1j * sin_p[k]) * sparse_table.cM[mc][i]
                DX[itrc, n_itr, i] = exp_p[k] * sparse_table.cM[mc][i]

            for i in range(i_min):
                DX[itrc, n_itr, i] = 0.0

            for i in range(i_max, wc.L):
                DX[itrc, n_itr, i] = 0.0
    return DX


@njit(fastmath=True)
def _sparse_time_unpack_loop(wavelet_waveform: SparseWaveletWaveform, m_pixel: NDArray[np.integer], nt_lim: PixelGenericRange, DX_trans: NDArray[np.complexfloating], wc: WDMWaveletConstants) -> None:
    """Unpack the array after it has been fourier transformed."""
    assert len(m_pixel.shape) == 2
    assert m_pixel.shape[1] <= nt_lim.nx_max - nt_lim.nx_min
    nc_waveform = m_pixel.shape[0]
    assert DX_trans.shape == (nc_waveform, nt_lim.nx_max - nt_lim.nx_min, wc.L)
    sparse_thin = int(wc.K // wc.L)
    assert 2 * wc.Nf % sparse_thin == 0, 'Nf currently needs to be an integer multiple of sparse thin factor'

    kx = int(2 * wc.Nf // sparse_thin)
    half_kx = int(kx // 2)

    wave_value = wavelet_waveform.wave_value
    pixel_index = wavelet_waveform.pixel_index
    n_set = wavelet_waveform.n_set
    n_set_old = n_set.copy()

    for itrc in range(nc_waveform):
        mm = 0
        for n in range(nt_lim.nx_min, nt_lim.nx_max):
            n_itr = n - nt_lim.nx_min
            mc: int = int(m_pixel[itrc, n])

            # NOTE The positive and negative loops can be fused but it changes the order of the pixel indices.
            # Fusion is mainly an advantage if we don't have to track mm for each iteration (for parallelization/vectorization)
            # Such as by using mm = n_itr * kx + j for neg and mm = n_itr*kx + j + half_kx for pos, as currently written
            # but if done in fixed order some pixels will not be set
            # so we need to ensure all later methods can efficiently handle -1 at pixel_index
            # DX_trans should be contiguous along the last axis

            # negative frequencies
            for j in range(half_kx):
                # mm_neg = n_itr * kx + j
                j_neg: int = j + half_kx
                m_neg: int = j_neg - kx + mc
                if -1 < m_neg < wc.Nf:
                    if (n + m_neg) % 2 == 0:
                        wave_value[itrc, mm] = DX_trans[itrc, n_itr, j_neg * wc.mult].real
                    else:
                        wave_value[itrc, mm] = -DX_trans[itrc, n_itr, j_neg * wc.mult].imag
                    pixel_index[itrc, mm] = n * wc.Nf + m_neg
                    mm += 1

            # postive frequencies
            for j in range(half_kx):
                # mm_pos = n_itr*kx + j + half_kx
                j_pos: int = j
                m_pos: int = j_pos + mc
                if -1 < m_pos < wc.Nf:
                    if (n + m_pos) % 2 == 0:
                        wave_value[itrc, mm] = DX_trans[itrc, n_itr, j_pos * wc.mult].real
                    else:
                        wave_value[itrc, mm] = -DX_trans[itrc, n_itr, j_pos * wc.mult].imag
                    pixel_index[itrc, mm] = n * wc.Nf + m_pos
                    mm += 1
        n_set[itrc] = mm
        # n_set[itrc] = wavelet_waveform.N_max

    # clean up any pixels that were set in the old intrinsic_waveform but aren't anymore
    for itrc in range(nc_waveform):
        for itrm in range(n_set[itrc], n_set_old[itrc]):
            pixel_index[itrc, itrm] = -1
            wave_value[itrc, itrm] = 0.0


def get_empty_sparse_sparse_wavelet_time_waveform(nc: int, wc: WDMWaveletConstants) -> SparseWaveletWaveform:
    """Get an empty SparseWaveletWaveform object with the dimensions needed for the sparse time transform."""
    assert wc.L % 2 == 0
    assert wc.K > 0
    assert wc.L > 0
    assert wc.K % wc.L == 0, 'K currently needs to be an integer multiple of L'
    sparse_thin: int = int(wc.K // wc.L)
    assert sparse_thin > 0
    assert 2 * wc.Nf % sparse_thin == 0, 'Nf currently needs to be an integer multiple of sparse thin factor'

    kx: int = int(2 * wc.Nf // sparse_thin)
    N_max: int = kx * wc.Nt
    wave_value = np.zeros((nc, N_max))
    pixel_index = np.zeros((nc, N_max), np.int64)
    n_set = np.zeros(nc, dtype=np.int64)
    return SparseWaveletWaveform(wave_value, pixel_index, n_set, N_max)


def make_sparse_wavelet_time(wave: StationarySourceWaveform[StationaryWaveformTime], wavelet_waveform: SparseWaveletWaveform, sparse_table: SparseTimeCoefficientTable, wc: WDMWaveletConstants) -> None:
    """Calculate the wavelet waveform using the sparse time domain method."""
    # have sped this up by computing cos(Phase[k]), sin(Phase[k])
    # and passing in pre-computed arrays for cos(2.*np.pi*(double)(j*mq)/(double)(L))
    # and sin(2.*np.pi*(double)(j*mq)/(double)(L)). The array is L*Nf in size
    # seems to spend about equal amounts of time in each of the 3 operations, currently

    m_pixel = _get_sparse_m_pixel_helper(wave.tdi_waveform, wc)
    nt_lim_restrict = _update_bounds_helper(m_pixel, wc)
    DX = _sparse_time_assign_loop(m_pixel, nt_lim_restrict, wave.tdi_waveform, sparse_table, wc)
    # scipy fft is faster because it can overrite the input array
    # DX_trans: NDArray[np.complexfloating] = fft.fft(DX, wc.L)
    DX_trans: NDArray[np.complexfloating] = spf.fft(DX, wc.L, overwrite_x=True)
    del DX

    _sparse_time_unpack_loop(wavelet_waveform, m_pixel, nt_lim_restrict, DX_trans, wc)


def get_sparse_table_helper(wc: WDMWaveletConstants) -> SparseTimeCoefficientTable:
    """Get the coefficients table needed for the sparse time wavelet transforms."""
    assert wc.L % 2 == 0
    assert wc.K > 0
    assert wc.L > 0
    assert wc.K % wc.L == 0, 'K currently needs to be an integer multiple of L'
    sparse_thin = int(wc.K // wc.L)
    assert sparse_thin > 0
    assert wc.Nf % sparse_thin == 0, 'Nf currently needs to be an integer multiple of sparse thinning factor'

    half_L = int(wc.L // 2)
    wave = wavelet(wc, 0, 1.0, n_in=wc.L)

    nrm = float(np.sqrt(2.0 / sparse_thin)) * float(np.linalg.norm(wave))
    wave = wave / nrm

    # pre-computed phase coefficients
    x = np.outer(np.arange(0, wc.Nf), (2.0 * np.pi * wc.mult / wc.L) * np.arange(-half_L, half_L))
    cos_h = wave * np.cos(x)
    sin_h = wave * np.sin(x)

    del wave
    del x

    c_m = cos_h - 1j * sin_h
    s_m = sin_h + 1j * cos_h
    return SparseTimeCoefficientTable(c_m, s_m)
