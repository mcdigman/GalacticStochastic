"""C 2023 Matthew C. Digman
python port of coefficients WDM_time.c
"""

import numpy as np
import scipy.fft as spf
from numba import njit, prange
from WDMWaveletTransforms.transform_freq_funcs import phitilde_vec


def wavelet(m, N, nrm, dom, DOM, Nf, dt, nx=4.):
    """Computes wavelet, modifies wave in place"""
    wave = np.zeros(N)
    halfN = np.int64(N / 2)

    DE = np.zeros(N, dtype=np.complex128)

    om = dom * np.hstack([np.arange(0, halfN + 1), -np.arange(halfN - 1, 0, -1)])
    DE = np.sqrt(dt) / np.sqrt(2.) * (phitilde_vec(dt * (om + m * DOM), Nf, nx) + phitilde_vec(dt * (om - m * DOM), Nf, nx))
    DE = spf.fft(DE, N, overwrite_x=True)

    wave[halfN:] = np.real(DE[0:halfN])
    wave[0:halfN] = np.real(DE[halfN:])
    return 1. / nrm * wave


@njit(parallel=True)
def get_ev_t_full(wave, wc):
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
            zquads[k] = np.pi * wc.dt**2 * fd[jj] * (k - wc.K // 2)**2
        for i in range(Nfsam[jj]):
            zmults = zadd + zpre * (1 - Nfsam[jj] + 2 * i)
            evc = 0.
            evs = 0.
            for k in prange(wc.K):
                zs = zmults * (k - wc.K // 2) + zquads[k]
                evc += wave[k] * np.cos(zs)
                evs += wave[k] * np.sin(zs)
            assert evc != 0.
            assert evs != 0.
            evcs[jj, i] = evc
            evss[jj, i] = evs
    return evcs, evss
