"""helper functions for Chirp_WDM"""


import numpy as np
import WDMWaveletTransforms.fft_funcs as fft
from numpy.typing import NDArray
from WDMWaveletTransforms.transform_freq_funcs import phitilde_vec

from WaveletWaveforms.chirplet_funcs import ChirpWaveletT, LinearChirpletIntrinsicParams, amp_phase_t, foft
from WaveletWaveforms.sparse_waveform_functions import PixelGenericRange, SparseWaveletWaveform, wavelet_sparse_to_dense
from WaveletWaveforms.taylor_time_coefficients import get_empty_sparse_taylor_time_waveform, get_taylor_table_time
from WaveletWaveforms.taylor_time_wavelet_funcs import wavemaket, wavemaket_direct
from WaveletWaveforms.wdm_config import WDMWaveletConstants


# @njit(fastmath=True)
def sparse_time_DX_assign_loop(params: LinearChirpletIntrinsicParams, cM: NDArray[np.complexfloating], sM: NDArray[np.complexfloating], wc: WDMWaveletConstants) -> tuple[NDArray[np.complexfloating], NDArray[np.int64], PixelGenericRange]:
    """Helper to start loop for sparse_wavelet_time"""
    ts = wc.DT * np.arange(0, wc.Nt)
    fs = foft(ts, params)
    mcs = (fs / wc.DF).astype(np.int64)
    nt_min = np.argmax(mcs > -1)
    nt_max = wc.Nt - np.argmax(mcs[::-1] < wc.Nf)
    nt_lim = PixelGenericRange(nt_min, nt_max, wc.DT, 0.)

    DX = np.zeros((nt_lim.nx_max - nt_lim.nx_min, wc.L), dtype=np.complex128)

    nn = np.int64((wc.Nf * wc.L) / wc.K)
    p = wc.K / wc.L
    M = np.int64(wc.Nf * wc.Nt / p)
    dtds = wc.dt * p  # downsampled data spacing in time

    Phases, Amps, _, _ = amp_phase_t(dtds * np.arange(0, M), params)
    cP = Amps * np.cos(Phases)
    sP = Amps * np.sin(Phases)
    iis = nn * np.arange(0, wc.Nt)

    for n in range(nt_lim.nx_min, nt_lim.nx_max):
        n_itr = n - nt_lim.nx_min
        ii = iis[n]
        mc = mcs[n]
        half_L = np.int64(wc.L / 2)
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


# @njit(fastmath=True)
def sparse_time_DX_unpack_loop(mcs: NDArray[np.integer], DX_trans: NDArray[np.complexfloating], nt_lim: PixelGenericRange, wc: WDMWaveletConstants) -> tuple[NDArray[np.int64], NDArray[np.float64], int, int]:
    """Helper to start unpack fft results for sparse_wavelet_time"""
    # indicates this pixel not used
    p = wc.K / wc.L
    kx = np.int64(2 * wc.Nf / p)
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
    Tlist, waveT, n_set, N_max = sparse_time_DX_unpack_loop(mcs, DX_trans, nt_lim, wc)
    return SparseWaveletWaveform(np.array([waveT]), np.array([Tlist]), np.array([n_set]), N_max)


def wavelet_SparseT(params: LinearChirpletIntrinsicParams, wc: WDMWaveletConstants) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
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

    DX = wc.L * fft.ifft(DX, wc.L)

    phis = np.zeros(wc.L)
    phis[0:half_L] = np.real(DX[half_L:])
    phis[half_L:] = np.real(DX[0:half_L])

    p = wc.K / wc.L
    nrm = np.sqrt(2. / p) * np.linalg.norm(phis)
    phis /= nrm

    # pre-computed phase coefficients
    x = np.outer(np.arange(0, wc.Nf), (2. * np.pi * wc.mult / wc.L) * np.arange(-half_L, half_L))
    cH = phis * np.cos(x)
    sH = phis * np.sin(x)
    cM = cH - 1j * sH
    sM = sH + 1j * cH

    del x

    wavelet_waveform = sparse_wavelet_time(params, cM, sM, wc)

    wave = wavelet_sparse_to_dense(wavelet_waveform, wc)[:, :, 0]
    return wavelet_waveform.pixel_index, wave


def TaylorTime(params: LinearChirpletIntrinsicParams, wc: WDMWaveletConstants, approximation: int = 1) -> SparseWaveletWaveform:
    """Calculate wavelet signal using time taylor method, waveT contains signal, Tlist flags of pixels in use.
    If approximation=0, use direct method without interpolating taylor coefficients.
    If approximation is positive, use approximation - 1 as the value for force_nulls in wavemaket.
    """
    # Note: with the simple analytic model we don't have to evaluate the
    # frequencies and frequency derivatives numerically. Only doing so
    # here to illustrate the general method that can be used with any
    # intrinsic_waveform model
    waveform = ChirpWaveletT(params, wc)
    nc_waveform = 1

    wavelet_waveform = get_empty_sparse_taylor_time_waveform(nc_waveform, wc)
    taylor_time_table = get_taylor_table_time(wc, cache_mode='check', output_mode='hf')

    nt_lim_waveform = PixelGenericRange(0, wc.Nt, wc.DT, 0.)

    if approximation == 0:
        wavemaket_direct(wavelet_waveform, waveform, nt_lim_waveform, wc, taylor_time_table)
    elif approximation > 0:
        force_nulls = approximation - 1
        wavemaket(wavelet_waveform, waveform, nt_lim_waveform, wc, taylor_time_table, force_nulls=force_nulls)
    else:
        msg = 'Unrecognized approximation: {}. No handling for negative values.'.format(approximation)
        raise NotImplementedError(msg)

    return wavelet_waveform


def wavelet_TaylorT(params: LinearChirpletIntrinsicParams, wc: WDMWaveletConstants, approximation: int = 1) -> tuple[SparseWaveletWaveform, NDArray[np.float64]]:
    """Get wavelet transform using taylor time method"""
    # frequency spacing

    wavelet_waveform = TaylorTime(params, wc, approximation=approximation)
    wave = wavelet_sparse_to_dense(wavelet_waveform, wc)[:, :, 0]
    return wavelet_waveform, wave
