"""C 2023 Matthew C. Digman
python port of coefficients WDM_time.c
"""

from typing import Tuple

import numpy as np
import scipy.fft as spf
from numba import njit, prange
from numpy.typing import NDArray
from WDMWaveletTransforms.transform_freq_funcs import phitilde_vec

from WaveletWaveforms.wdm_config import WDMWaveletConstants


def wavelet(wc: WDMWaveletConstants, m: int, nrm: np.floating) -> NDArray[np.float64]:
    """Computes wavelet, modifies wave in place"""
    wave = np.zeros(wc.K)
    halfN = np.int64(wc.K / 2)

    DE = np.zeros(wc.K, dtype=np.complex128)

    om = wc.dom * np.hstack([np.arange(0, halfN + 1), -np.arange(halfN - 1, 0, -1)])
    DE[:] = (
        np.sqrt(wc.dt)
        / np.sqrt(2.0)
        * (
            phitilde_vec(wc.dt * (om + m * wc.DOM), wc.Nf, wc.nx)
            + phitilde_vec(wc.dt * (om - m * wc.DOM), wc.Nf, wc.nx)
        )
    )
    DE = spf.fft(DE, wc.K, overwrite_x=True)

    wave[halfN:] = np.real(DE[0:halfN])
    wave[0:halfN] = np.real(DE[halfN:])
    return 1.0 / nrm * wave


@njit(parallel=True)
def get_taylor_table_time_helper(wavelet_norm: NDArray[np.float64], wc: WDMWaveletConstants) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Helper function to take advantage of jit compiler in t calculation"""
    fd = wc.dfd * np.arange(-wc.Nfd_negative, wc.Nfd - wc.Nfd_negative)  # set f-dot increments
    Nfsam = ((wc.BW + np.abs(fd) * wc.Tw) / wc.df).astype(np.int64)
    Nfsam[Nfsam % 2 == 1] += 1

    evcs = np.zeros((wc.Nfd, np.max(Nfsam)))
    evss = np.zeros((wc.Nfd, np.max(Nfsam)))
    zadd = np.pi / 16
    zpre = np.pi * wc.df * wc.dt
    zquads = np.zeros(wc.K)
    for jj in range(wc.Nfd):
        for k in range(wc.K):
            zquads[k] = np.pi * wc.dt**2 * fd[jj] * (k - wc.K // 2) ** 2
        for i in range(Nfsam[jj]):
            zmults = zadd + zpre * (1 - Nfsam[jj] + 2 * i)
            evc = 0.0
            evs = 0.0
            for k in prange(0, wc.K):
                zs = zmults * (k - wc.K // 2) + zquads[k]
                evc += wavelet_norm[k] * np.cos(zs)
                evs += wavelet_norm[k] * np.sin(zs)
            assert evc != 0.0
            #assert evs != 0.0  # the sin component can be extremely close to zero, perhaps exactly by coincidence
            evcs[jj, i] = evc
            evss[jj, i] = evs
    return evcs, evss

@njit()
def get_taylor_pixel_direct(fa: float, fda: float, k_in: int, wavelet_norm: NDArray[np.float64], wc: WDMWaveletConstants) -> Tuple[float, float]:
    """Helper function to take advantage of jit compiler in t calculation"""
    dfa = fa - wc.DF*k_in
    xk = np.abs(dfa)/wc.df
    fd_mid = fda

    zadd = np.pi / 16
    zpre = 2 * np.pi * wc.df * wc.dt * xk
    zquads_mid = np.zeros(wc.K)
    for k in range(wc.K):
        zquads_mid[k] = np.pi * wc.dt**2 * fd_mid * (k - wc.K // 2) ** 2

    zmults = zadd + zpre
    evc_mid = 0.0
    evs_mid = 0.0

    for k in prange(wc.K):
        zs_mid = zmults * (k - wc.K // 2) + zquads_mid[k]
        evc_mid += wavelet_norm[k] * np.cos(zs_mid)
        evs_mid += wavelet_norm[k] * np.sin(zs_mid)

    assert evc_mid != 0.0
    assert evs_mid != 0.0
    return evc_mid, evs_mid

def get_wavelet_norm(wc: WDMWaveletConstants) -> NDArray[np.float64]:
    """Get the normalized wavelet needed for the taylor time coefficients table.
    Also used in the exact version of wavemaket.
    """
    phi = np.zeros(wc.K)
    DX = np.zeros(wc.K, dtype=np.complex128)
    DX[0] = wc.insDOM

    DX[1:np.int64(wc.K / 2) + 1] = np.sqrt(wc.dt) * phitilde_vec(
        wc.dom * wc.dt * np.arange(1, np.int64(wc.K / 2) + 1), wc.Nf, wc.nx
    )
    DX[np.int64(wc.K / 2) + 1:] = np.sqrt(wc.dt) * phitilde_vec(
        -wc.dom * wc.dt * np.arange(np.int64(wc.K / 2) - 1, 0, -1), wc.Nf, wc.nx
    )

    DX = spf.fft(DX, wc.K, overwrite_x=True)

    for i in range(np.int64(wc.K / 2)):
        phi[i] = np.real(DX[np.int64(wc.K / 2) + i])
        phi[np.int64(wc.K / 2) + i] = np.real(DX[i])

    nrm = np.linalg.norm(phi)
    kwave = wc.Nf / 16
    return wavelet(wc, kwave, nrm)
