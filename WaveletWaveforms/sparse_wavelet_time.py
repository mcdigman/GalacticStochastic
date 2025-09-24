"""helper functions for Chirp_WDM"""

from typing import NamedTuple

import numpy as np
import WDMWaveletTransforms.fft_funcs as fft
from numba import njit
from numpy.typing import NDArray

from LisaWaveformTools.stationary_source_waveform import StationaryWaveformTime
from WaveletWaveforms.chirplet_funcs import LinearChirpletIntrinsicParams, amp_phase_t
from WaveletWaveforms.sparse_waveform_functions import PixelGenericRange, SparseWaveletWaveform
from WaveletWaveforms.taylor_time_coefficients import wavelet
from WaveletWaveforms.wdm_config import WDMWaveletConstants


class SparseCoefficientTable(NamedTuple):
    cM: NDArray[np.complexfloating]
    sM: NDArray[np.complexfloating]


@njit()
def _get_sparse_intrinsic_helper(params: LinearChirpletIntrinsicParams, wc: WDMWaveletConstants) -> tuple[StationaryWaveformTime, NDArray[np.integer]]:
    assert wc.K % wc.L == 0, 'K currently needs to be an integer multiple of L'
    sparse_thin = int(wc.K // wc.L)
    assert wc.Nf % sparse_thin == 0, 'Nf currently needs to be an integer multiple of sparse thinning factor'
    thin_factor = int(wc.Nf // sparse_thin)

    N_tot = wc.Nf * wc.Nt
    if N_tot % sparse_thin != 0:
        msg = 'Total pixels ' + str(N_tot) + ' needs to be an ingeter multiple of the thinning factor ' + str(sparse_thin)
        raise NotImplementedError(msg)

    M = int(wc.Nf * wc.Nt / sparse_thin)
    print(wc.Nf / sparse_thin)

    dtd = float(wc.dt * sparse_thin)  # downsampled data spacing in time
    T = dtd * np.arange(0, M)
    Phases, Amps, FT, FTd = amp_phase_t(T, params)

    m_pixel = (FT[::thin_factor] / wc.DF).astype(np.int64)

    return StationaryWaveformTime(T, Phases, FT, FTd, Amps), m_pixel


@njit()
def _update_bounds_helper(m_pixel, wc: WDMWaveletConstants) -> PixelGenericRange:
    # currently assumes frequency is generally increasing
    nt_min = int(np.argmax(m_pixel > -1))
    nt_max = int(wc.Nt - np.argmax(m_pixel[::-1] < wc.Nf))
    return PixelGenericRange(nt_min, nt_max, wc.DT, 0.0)


@njit()
def _sparse_time_DX_assign_loop(m_pixel, nt_lim: PixelGenericRange, waveform_intrinsic: StationaryWaveformTime, sparse_table: SparseCoefficientTable, wc: WDMWaveletConstants) -> NDArray[np.complexfloating]:
    """Helper to start loop for sparse_wavelet_time"""
    assert wc.L % 2 == 0

    half_L = int(wc.L // 2)

    p = float(wc.K / wc.L)
    M = int(wc.Nf * wc.Nt / p)

    DX = np.zeros((nt_lim.nx_max - nt_lim.nx_min, wc.L), dtype=np.complex128)

    nn = int((wc.Nf * wc.L) / wc.K)
    AT = waveform_intrinsic.AT
    PT = waveform_intrinsic.PT
    cP = AT * np.cos(PT)
    sP = AT * np.sin(PT)

    for n in range(nt_lim.nx_min, nt_lim.nx_max):
        n_itr = n - nt_lim.nx_min
        ii = nn * n
        mc = m_pixel[n]
        i_min = max(half_L - ii, 0)
        i_max = min(wc.L, M + half_L - ii)
        for i in range(i_min, i_max):
            k = i + ii - half_L
            DX[n_itr, i] = cP[k] * sparse_table.cM[mc][i] + sP[k] * sparse_table.sM[mc][i]

        if i_min > 0:
            DX[n_itr, 0:i_min] = 0.0
        if i_max < wc.L:
            DX[n_itr, i_max:] = 0.0
    return DX


@njit()
def _sparse_time_DX_unpack_loop(m_pixel: NDArray[np.integer], DX_trans: NDArray[np.complexfloating], nt_lim: PixelGenericRange, wc: WDMWaveletConstants) -> tuple[NDArray[np.int64], NDArray[np.floating], int, int]:
    """Helper to start unpack fft results for sparse_wavelet_time"""
    # indicates this pixel not used
    p = wc.K / wc.L
    kx = int(2 * wc.Nf / p)
    half_kx = int(kx // 2)
    N_max = kx * wc.Nt
    waveT = np.zeros(N_max)
    Tlist = np.full(N_max, -1, dtype=np.int64)
    mm = 0

    for n in range(nt_lim.nx_min, nt_lim.nx_max):
        n_itr = n - nt_lim.nx_min
        mc = m_pixel[n]
        # negative frequencies
        for j in range(half_kx, kx):
            m = j - kx + mc
            if -1 < m < wc.Nf:
                if (n + m) % 2 == 0:
                    waveT[mm] = np.real(DX_trans[n_itr, j * wc.mult])
                else:
                    waveT[mm] = -np.imag(DX_trans[n_itr, j * wc.mult])
                Tlist[mm] = n * wc.Nf + m
                mm += 1

        # postive frequencies
        for j in range(half_kx):
            m = j + mc
            if -1 < m < wc.Nf:
                if (n + m) % 2 == 0:
                    waveT[mm] = np.real(DX_trans[n_itr, j * wc.mult])
                else:
                    waveT[mm] = -np.imag(DX_trans[n_itr, j * wc.mult])
                Tlist[mm] = n * wc.Nf + m
                mm += 1
    return Tlist, waveT, mm, N_max


def sparse_wavelet_time(params: LinearChirpletIntrinsicParams, sparse_table: SparseCoefficientTable, wc: WDMWaveletConstants) -> SparseWaveletWaveform:
    """Calculate time domain sparse wavelet method"""
    # TODO should not be chirplet specific
    # have sped this up by computing cos(Phase[k]), sin(Phase[k])
    # and passing in pre-computed arrays for cos(2.*np.pi*(double)(j*mq)/(double)(L))
    # and sin(2.*np.pi*(double)(j*mq)/(double)(L)). The array is L*Nf in size

    waveform_intrinsic, m_pixel = _get_sparse_intrinsic_helper(params, wc)
    nt_lim = _update_bounds_helper(m_pixel, wc)
    DX = _sparse_time_DX_assign_loop(m_pixel, nt_lim, waveform_intrinsic, sparse_table, wc)
    DX_trans: NDArray[np.complexfloating] = fft.fft(DX, wc.L)
    del DX
    Tlist, waveT, n_set, N_max = _sparse_time_DX_unpack_loop(m_pixel, DX_trans, nt_lim, wc)
    return SparseWaveletWaveform(np.array([waveT]), np.array([Tlist]), np.array([n_set]), N_max)


def _sparse_table_helper(wc: WDMWaveletConstants) -> SparseCoefficientTable:
    # TODO does this function already exist somewhere else?
    assert wc.L % 2 == 0
    half_L = int(wc.L // 2)
    wave = wavelet(wc, 0, 1.0, n_in=wc.L)

    p = wc.K / wc.L
    nrm = np.sqrt(2.0 / p) * np.linalg.norm(wave)
    wave /= nrm

    # pre-computed phase coefficients
    x = np.outer(np.arange(0, wc.Nf), (2.0 * np.pi * wc.mult / wc.L) * np.arange(-half_L, half_L))
    cH = wave * np.cos(x)
    sH = wave * np.sin(x)

    del wave
    del x

    cM = cH - 1j * sH
    sM = sH + 1j * cH
    return SparseCoefficientTable(cM, sM)


def wavelet_SparseT(params: LinearChirpletIntrinsicParams, wc: WDMWaveletConstants) -> SparseWaveletWaveform:
    """Compute the time domain wavelet filter and normalize"""
    sparse_table = _sparse_table_helper(wc)

    return sparse_wavelet_time(params, sparse_table, wc)
