"""helper functions for Chirp_WDM"""

import numpy as np
import WDMWaveletTransforms.fft_funcs as fft
from numba import njit
from numpy.typing import NDArray
from WDMWaveletTransforms.transform_freq_funcs import phitilde_vec

from WaveletWaveforms.chirplet_funcs import LinearChirpletIntrinsicParams, amp_phase_t, foft
from WaveletWaveforms.sparse_waveform_functions import PixelGenericRange, SparseWaveletWaveform
from WaveletWaveforms.wdm_config import WDMWaveletConstants


@njit()
def sparse_time_DX_assign_loop(params: LinearChirpletIntrinsicParams, cM: NDArray[np.complexfloating], sM: NDArray[np.complexfloating], wc: WDMWaveletConstants) -> tuple[NDArray[np.complexfloating], NDArray[np.int64], PixelGenericRange]:
    """Helper to start loop for sparse_wavelet_time"""
    T = wc.DT * np.arange(0, wc.Nt)
    FT = foft(T, params)
    mcs = (FT / wc.DF).astype(np.int64)
    nt_min = int(np.argmax(mcs > -1))
    nt_max = int(wc.Nt - np.argmax(mcs[::-1] < wc.Nf))
    nt_lim = PixelGenericRange(nt_min, nt_max, wc.DT, 0.)

    DX = np.zeros((nt_lim.nx_max - nt_lim.nx_min, wc.L), dtype=np.complex128)

    nn = int((wc.Nf * wc.L) / wc.K)
    p = float(wc.K / wc.L)
    M = int(wc.Nf * wc.Nt / p)
    dtd = float(wc.dt * p)  # downsampled data spacing in time

    Phases, Amps, _, _ = amp_phase_t(dtd * np.arange(0, M), params)
    cP = Amps * np.cos(Phases)
    sP = Amps * np.sin(Phases)

    for n in range(nt_lim.nx_min, nt_lim.nx_max):
        n_itr = n - nt_lim.nx_min
        ii = nn * n
        mc = mcs[n]
        half_L = int(wc.L / 2)
        i_min = max(half_L - ii, 0)
        i_max = min(wc.L, M + half_L - ii)
        for i in range(i_min, i_max):
            k = i + ii - half_L
            DX[n_itr, i] = cP[k] * cM[mc][i] + sP[k] * sM[mc][i]

        if i_min > 0:
            DX[n_itr, 0:i_min] = 0.
        if i_max < wc.L:
            DX[n_itr, i_max:] = 0.
    return DX, mcs, nt_lim


@njit()
def sparse_time_DX_unpack_loop(mcs: NDArray[np.integer], DX_trans: NDArray[np.complexfloating], nt_lim: PixelGenericRange, wc: WDMWaveletConstants) -> tuple[NDArray[np.int64], NDArray[np.floating], int, int]:
    """Helper to start unpack fft results for sparse_wavelet_time"""
    # indicates this pixel not used
    p = wc.K / wc.L
    kx = int(2 * wc.Nf / p)
    N_max = kx * wc.Nt
    waveT = np.zeros(N_max)
    Tlist = np.full(N_max, -1, dtype=np.int64)
    mm = 0

    for n in range(nt_lim.nx_min, nt_lim.nx_max):
        n_itr = n - nt_lim.nx_min
        mc = mcs[n]
        # negative frequencies
        for j in range(np.int64(kx / 2), kx):
            m = j - kx + mc
            if -1 < m < wc.Nf:
                if (n + m) % 2 == 0:
                    waveT[mm] = np.real(DX_trans[n_itr, j * wc.mult])
                else:
                    waveT[mm] = -np.imag(DX_trans[n_itr, j * wc.mult])
                Tlist[mm] = n * wc.Nf + m
                mm += 1

        # postive frequencies
        for j in range(np.int64(kx / 2)):
            m = j + mc
            if -1 < m < wc.Nf:
                if (n + m) % 2 == 0:
                    waveT[mm] = np.real(DX_trans[n_itr, j * wc.mult])
                else:
                    waveT[mm] = -np.imag(DX_trans[n_itr, j * wc.mult])
                Tlist[mm] = n * wc.Nf + m
                mm += 1
    return Tlist, waveT, mm, N_max


def sparse_wavelet_time(params: LinearChirpletIntrinsicParams, cM: NDArray[np.complexfloating], sM: NDArray[np.complexfloating], wc: WDMWaveletConstants) -> SparseWaveletWaveform:
    """Calculate time domain sparse wavelet method"""
    # TODO should not be chirplet specific
    # have sped this up by computing cos(Phase[k]), sin(Phase[k])
    # and passing in pre-computed arrays for cos(2.*np.pi*(double)(j*mq)/(double)(L))
    # and sin(2.*np.pi*(double)(j*mq)/(double)(L)). The array is L*Nf in size
    DX, mcs, nt_lim = sparse_time_DX_assign_loop(params, cM, sM, wc)
    DX_trans: NDArray[np.complexfloating] = fft.fft(DX, wc.L)
    del DX
    Tlist, waveT, n_set, N_max = sparse_time_DX_unpack_loop(mcs, DX_trans, nt_lim, wc)
    return SparseWaveletWaveform(np.array([waveT]), np.array([Tlist]), np.array([n_set]), N_max)


def wavelet_SparseT(params: LinearChirpletIntrinsicParams, wc: WDMWaveletConstants) -> SparseWaveletWaveform:
    """Compute the time domain wavelet filter and normalize"""
    DX = np.zeros(wc.L, dtype=np.complex128)
    half_L = np.int64(wc.L / 2)
    # negative frequencies
    oms = wc.dom * np.arange(1, half_L)
    DX[-1:-1 - oms.size:-1] = np.sqrt(wc.dt) * phitilde_vec(-oms * wc.dt, wc.Nf, wc.nx)

    # zero frequency
    DX[0] = wc.insDOM

    # postive frequencies
    DX[1:1 + oms.size] = np.sqrt(wc.dt) * phitilde_vec(oms * wc.dt, wc.Nf, wc.nx)

    DX_real = wc.L * np.real(fft.ifft(DX, wc.L))

    phis = np.zeros(wc.L)
    phis[0:half_L] = DX_real[half_L:]
    phis[half_L:] = DX_real[0:half_L]

    p = wc.K / wc.L
    nrm = np.sqrt(2. / p) * np.linalg.norm(phis)
    phis /= nrm

    # pre-computed phase coefficients
    x = np.outer(np.arange(0, wc.Nf), (2. * np.pi * wc.mult / wc.L) * np.arange(-half_L, half_L))
    cH = phis * np.cos(x)
    sH = phis * np.sin(x)

    del x

    cM = cH - 1j * sH
    sM = sH + 1j * cH

    return sparse_wavelet_time(params, cM, sM, wc)
