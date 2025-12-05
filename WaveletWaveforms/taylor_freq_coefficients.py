"""
Get Taylor frequency domain coefficients for wavelet transforms.

C 2025 Matthew C. Digman
"""
from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from WaveletWaveforms.wdm_config import WDMWaveletConstants


from time import perf_counter

import h5py
from numba import njit, prange
from WDMWaveletTransforms.transform_freq_funcs import phitilde_vec

from WaveletWaveforms.sparse_waveform_functions import SparseWaveletWaveform

SECSYEAR = 24 * 365 * 3600  # Number of seconds in a calendar year
# from coefficientsWDM_time_funcs import get_ev_f_point


class WaveletTaylorFreqCoeffs(NamedTuple):
    Nfsam: NDArray[np.integer]
    evc: NDArray[np.floating]
    evs: NDArray[np.floating]
    wavelet_norm: NDArray[np.floating]


def get_empty_sparse_taylor_freq_waveform(nc_waveform: int, wc: WDMWaveletConstants) -> SparseWaveletWaveform:
    # td = wc.dtd * np.arange(-wc.Ntd_negative, wc.Ntd - wc.Ntd_negative)
    # Nfsam = ((wc.Tw / 2 + np.abs(td) * wc.DF / 2) / wc.delt).astype(np.int64)
    # max_shape = 2 * np.max(Nfsam) + 1

    # TODO check maximum pixel calculation
    n_pixel_max: int = int(np.int64(
        2 * np.int64(wc.Tw / wc.delt / 2 + wc.DF * wc.dtd / wc.delt / 2 * (wc.Ntd - wc.Ntd_negative - 1)) * (
                    wc.delt / wc.DT)) * wc.Nf)
    n_pixel_max += int(np.int64(2 * np.int64(wc.Tw / wc.delt / 2 + wc.DF * wc.dtd / wc.delt / 2 * (wc.Ntd_negative - 1)) * (
                wc.delt / wc.DT)) * wc.Nf)

    n_pixel_max += wc.n_f_null_extend
    # TODO maybe allow different null extension in time an frequency methods
    # array of wavelet coefficients
    wave_value = np.zeros((nc_waveform, n_pixel_max))
    # aray of pixel indices
    pixel_index = np.full((nc_waveform, n_pixel_max), -1, dtype=np.int64)
    # number of pixel indices that are set
    n_set = np.zeros(nc_waveform, dtype=np.int64)

    return SparseWaveletWaveform(wave_value, pixel_index, n_set, n_pixel_max)


@njit(parallel=True)
def get_taylor_table_freq_helper(wavelet_norm: NDArray[np.floating], wc: WDMWaveletConstants) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Helper frunction to take advantage of jit compiler in f calculation"""
    Np = np.int64(wc.BW * wc.Tobs) + 1
    td = wc.dtd * np.arange(-wc.Ntd_negative, wc.Ntd - wc.Ntd_negative)
    Nfsam = ((wc.Tw / 2 + np.abs(td) * wc.DF / 2) / wc.delt).astype(np.int64)
    # Nfsam = (wc.Tw/(2.*wc.delt)+0.5*wc.DF/wc.delt*np.abs(td)).astype(np.int64)
    max_shape = 2 * np.max(Nfsam) + 1
    evcs = np.zeros((wc.Ntd, max_shape))
    evss = np.zeros((wc.Ntd, max_shape))
    z_add = -np.pi * wc.BW * wc.delt
    z_pre = 2. * np.pi / wc.Tobs * wc.delt
    z_mult = z_add + z_pre * np.arange(0, Np)
    zzq = np.pi * wc.dtd * (-wc.BW / 2. + np.arange(0, Np) / wc.Tobs)**2
    # z_quads = np.outer(np.pi*td,(-wc.BW/2.+np.arange(0,Np)/wc.Tobs)**2)

    for k in range(wc.Ntd):
        z_quads = (k - wc.Ntd_negative) * zzq
        for j in prange(0, 2 * Nfsam[k] + 1):
            zs = (j - Nfsam[k]) * z_mult + z_quads
            evcs[k, j] = np.dot(np.cos(zs), wavelet_norm)
            evss[k, j] = np.dot(np.sin(zs), wavelet_norm)
    return evcs, evss


def get_taylor_table_freq(
    wc: WDMWaveletConstants,
    *,
    cache_mode: str = 'skip',
    output_mode: str = 'skip',
    cache_dir: str = 'coeffs/',
    filename_base: str = 'taylor_time_table_',
    grid_check_mode: int = 1,
    assert_mode: int = 1,
) -> WaveletTaylorFreqCoeffs:
    del assert_mode
    del grid_check_mode
    print('Filter length (seconds) %e' % wc.Tw)
    cache_good = False

    filename_cache = (
        cache_dir
        + filename_base
        + 'Nst='
        + str(wc.Nst)
        + '_mult='
        + str(wc.mult)
        + '_Ntd='
        + str(wc.Ntd)
        + '_Nf='
        + str(wc.Nf)
        + '_Nt='
        + str(wc.Nt)
        + '_dt='
        + str(wc.dt)
        + '_dtdf='
        + str(wc.dtdf)
        + '_Ntd_negative='
        + str(wc.Ntd_negative)
        + '.h5'
    )

    # The wavelet is computed at the sample frequency 1/Tobs
    # check why this differs from n_p and why dom is not the same dom from everywhere else
    Np = np.int64(wc.BW * wc.Tobs) + 1
    # TODO why is this dom different from everywhere else?

    f1 = -wc.BW / 2. + 1. / wc.Tobs * np.arange(0, Np)

    print('dt=' + str(wc.dt) + 's Tobs=' + str(wc.Tobs / SECSYEAR))

    # time spacing

    # TODO looks like Ntd must be odd
    # TODO check no even odd behavior
    td = wc.dtd * np.arange(-wc.Ntd_negative, wc.Ntd - wc.Ntd_negative)

    # Nfsam = ((wc.Tw/2+np.abs(td)*wc.DF/2)/wc.delt).astype(np.int64)
    Nfsam = ((wc.Tw / 2 + np.abs(td) * wc.DF / 2) / wc.delt).astype(np.int64)
    # Nfsam = (wc.Tw/(2.*wc.delt)+0.5*wc.DF/wc.delt*td).astype(np.int64)

    max_shape: int = 2 * int(np.max(Nfsam)) + 1

    evcs: NDArray[np.floating] = np.zeros((wc.Ntd, max_shape))
    evss: NDArray[np.floating] = np.zeros((wc.Ntd, max_shape))

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

    # TODO create a grid check function for frequency grid

    if not cache_good:
        phi = np.sqrt(wc.dt) * phitilde_vec(2. * np.pi * f1 * wc.dt, wc.Nf, wc.nx)

        nrm = np.sqrt(2.) * np.sqrt(wc.Nt * wc.Nf) * wc.dt * np.linalg.norm(phi)  # multiplying by another sqrt(2) fixes for Nt=4096,Nf=2048,dividing by sqrt(2) fixes Nt=2048,Nf=4096
        print('norm', nrm)
        wavelet_norm = phi / nrm
        t3 = perf_counter()
        evcs, evss = get_taylor_table_freq_helper(wavelet_norm, wc)
        t4 = perf_counter()
        print('Got frequency interpolation table in t=' + str(t4 - t3) + 's')

    # TODO I'm not sure theres a real advantage to cutting of evcs/evss jaggedly and it might simplify memory access patterns not to do that

        if output_mode == 'hf':
            print('Writing hdf5 frequency interpolation table to ' + str(filename_cache))
            t0_write = perf_counter()
            hf = h5py.File(filename_cache, 'w')

            wc_group = hf.create_group('_wc')
            for key in wc._fields:
                wc_group.attrs[key] = getattr(wc, key)

            _ = hf.create_dataset('wavelet_norm', data=wavelet_norm, compression='gzip')
            _ = hf.create_dataset('td', data=td, compression='gzip')
            _ = hf.create_dataset('Nfsam', data=Nfsam, compression='gzip')
            _ = hf.create_dataset('evcs', data=evcs, compression='gzip')
            _ = hf.create_dataset('evss', data=evss, compression='gzip')
            hf.close()
            tf_write = perf_counter()
            print('Finished writing frequency interpolation table in t=' + str(tf_write - t0_write) + 's')
        elif output_mode == 'skip':
            pass
        else:
            msg = f'Unrecognized option for output_mode {output_mode}'
            raise NotImplementedError(msg)

    return WaveletTaylorFreqCoeffs(Nfsam, evcs, evss, wavelet_norm)
