"""C 2023 Matthew C. Digman
test the wdm time functions
"""

from collections import namedtuple
from time import time

import h5py
import numpy as np

from WaveletWaveforms.coefficientsWDM_time_funcs import get_taylor_table_time_helper, get_wavelet_norm
from WaveletWaveforms.sparse_waveform_functions import SparseWaveletWaveform
from WaveletWaveforms.wdm_config import WDMWaveletConstants

SECSYEAR = 24 * 365 * 3600  # Number of seconds in a calendar year

WaveletTaylorTimeCoeffs = namedtuple('WaveletTaylorTimeCoeffs', ['Nfsam', 'evcs', 'evss', 'wavelet_norm'])



def get_empty_sparse_taylor_time_waveform(nc_waveform, wc: WDMWaveletConstants) -> SparseWaveletWaveform:
    """Get a blank SparseWaveletWaveform object for the Taylor time approximation methods."""
    # need the frequency derivatives to calculate the maximum possible size
    fds = wc.dfd * np.arange(-wc.Nfd_negative, wc.Nfd - wc.Nfd_negative)
    # calculate maximum possible number of pixels
    N_max = np.int64(np.ceil((wc.BW + np.max(np.abs(fds)) * wc.Tw) / wc.DF)) * wc.Nt
    # array of wavelet coefficients
    wave_value = np.zeros((nc_waveform, N_max))
    # aray of pixel indices
    pixel_index = np.full((nc_waveform, N_max), -1, dtype=np.int64)
    # number of pixel indices that are set
    n_set = np.zeros(nc_waveform, dtype=np.int64)

    return SparseWaveletWaveform(wave_value, pixel_index, n_set, N_max)




def get_taylor_table_time(wc: WDMWaveletConstants, cache_mode='skip', output_mode='skip') -> WaveletTaylorTimeCoeffs:
    """Helper to get the ev matrices
    cache_mode:
        'skip':
            do not check if a cached coefficient table is available
        'check':
            check if a cached coefficient table is available
    output_mode:
        'skip': just return, do not output to a file
        'hf': output to an hdf5 file in the coeffs/ directory
    """
    t0 = time()

    print('Filter length (seconds) %e' % wc.Tw)
    print('dt=' + str(wc.dt) + 's Tobs=' + str(wc.Tobs / SECSYEAR))

    print('full filter bandwidth %e  samples %d' % ((wc.A + wc.B) / np.pi, (wc.A + wc.B) / np.pi * wc.Tw))
    cache_good = False

    filename_cache = (
        'coeffs/WDMcoeffs_Nsf='
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
        + '_Nfd_neg'
        + str(wc.Nfd_negative)
        + '_fast.h5'
    )

    if cache_mode == 'check':
        fds = wc.dfd * np.arange(-wc.Nfd_negative, wc.Nfd - wc.Nfd_negative)

        # number of samples for each frequency derivative layer (grow with increasing BW)
        Nfsam = ((wc.BW + np.abs(fds) * wc.Tw) / wc.df).astype(np.int64)
        Nfsam[Nfsam % 2 != 0] += 1  # makes sure it is an even number

        max_shape = np.max(Nfsam)
        evcs = np.zeros((wc.Nfd, max_shape))
        evss = np.zeros((wc.Nfd, max_shape))

        try:
            hf_in = h5py.File(filename_cache, 'r')
            wavelet_norm = np.asarray(hf_in['wavelet_norm'])
            for i in range(wc.Nfd):
                evcs[i, : Nfsam[i]] = np.asarray(hf_in['evcs'][str(i)])
                evss[i, : Nfsam[i]] = np.asarray(hf_in['evss'][str(i)])
            hf_in.close()
            cache_good = True
        except OSError:
            print(filename_cache)
            print('cache checked and missed')
    elif cache_mode == 'skip':
        pass
    else:
        msg = 'Unrecognized option for cache_mode'
        raise NotImplementedError(msg)

    if not cache_good:
        wavelet_norm = get_wavelet_norm(wc)

        fd = wc.DF / wc.Tw * wc.dfdot * np.arange(-wc.Nfd_negative, wc.Nfd - wc.Nfd_negative)  # set f-dot increments

        print('%e %.14e %.14e %e %e' % (wc.DT, wc.DF, wc.DOM / (2 * np.pi), fd[1], fd[wc.Nfd - 1]))

        Nfsam = ((wc.BW + np.abs(fd) * wc.Tw) / wc.df).astype(np.int64)
        odd_mask = np.mod(Nfsam, 2) != 0
        Nfsam[odd_mask] += 1

        # The odd wavelets coefficienst can be obtained from the even.
        # odd cosine = -even sine, odd sine = even cosine

        # each wavelet covers a frequency band of width DW
        # except for the first and last wavelets
        # there is some overlap. The wavelet pixels are of width
        # DOM/PI, except for the first and last which have width
        # half that

        t1 = time()
        print('loop start time ', t1 - t0, 's')
        evcs, evss = get_taylor_table_time_helper(wavelet_norm, wc)
        tf = time()
        print('Got Time Taylor Table in %f s' % (tf - t1))

        if output_mode == 'hf':
            hf = h5py.File(filename_cache, 'w')
            hf.create_group('inds')
            hf.create_group('evcs')
            hf.create_group('evss')
            hf.create_dataset('wavelet_norm', data=wavelet_norm)
            for jj in range(wc.Nfd):
                hf['inds'].create_dataset(str(jj), data=np.arange(0, Nfsam[jj]))
                hf['evcs'].create_dataset(str(jj), data=evcs[jj, : Nfsam[jj]])
                hf['evss'].create_dataset(str(jj), data=evss[jj, : Nfsam[jj]])
            hf.close()
            t3 = time()
            print('output time', t3 - tf, 's')
        elif output_mode == 'skip':
            pass
        else:
            msg = 'unrecognized option for output_mode'
            raise NotImplementedError(msg)

    return WaveletTaylorTimeCoeffs(Nfsam, evcs, evss, wavelet_norm)
