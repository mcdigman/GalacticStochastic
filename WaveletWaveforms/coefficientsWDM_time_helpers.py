"""C 2023 Matthew C. Digman
test the wdm time functions
"""
from collections import namedtuple
from time import time

import h5py
import numpy as np
import scipy as sp
from WDMWaveletTransforms.transform_freq_funcs import phitilde_vec

from WaveletWaveforms.coefficientsWDM_time_funcs import get_ev_t_full, wavelet
from WaveletWaveforms.wdm_config import WDMWaveletConstants

SECSYEAR = 24 * 365 * 3600    # Number of seconds in a calendar year

WaveletTaylorTimeCoeffs = namedtuple('WaveletTaylorTimeCoeffs', ['Nfsam', 'evcs', 'evss'])

SparseTaylorWaveform = namedtuple('SparseTaylorTimeWaveform', ['wave_value', 'pixel_index', 'n_set', 'N_max'])

SparseTaylorWaveform.__doc__ = """
namedtuple object that contains a sparse representation of a wavelet waveform,
as computed using the taylor approximation in e.g. wavemaket_multi_inplace
because the actual array size needed isn't known in advance,
it internally stores arrays that are the maximum size possible in
the approximation, and records the number of pixels that are actually set in
n_set. In pixel_index, it additionally uses indicator values of -1 to indicate pixels
which are not set. The pixel indices don't need to be listed in any particular order,
Although for some applications it could be convenient for them to be sorted.
wave_value : numpy.ndarray
    stores the actual waveform as the values of the wavelet pixels
    at the pixel indices specified by lists_pixels.
    All values with index >= n_set should be set to 0.
    Shape is the same as lists_pixels: shape: (nc_waveform, N_max)
pixel_index : numpy.ndarray
    stores the indices of x,y coordinates of all pixels that are
    currently set. All values with index >= n_set should be set to -1
    shape: (nc_waveform, N_max) number of TDI channels x maximum number
    of pixels possible in sparse representation.
n_set : numpy.ndarray of integers
    number of wavelet coefficients that are *currently* set
    all values must be <= N_max.
    shape: number of TDI channels
N_max: integer
    the maximum number of wavelet pixels that could possibly
    be set in the sparse representation,
    which is determined by the shape of the interpolation table
"""


def get_empty_sparse_taylor_time_waveform(nc_waveform, wc: WDMWaveletConstants) -> SparseTaylorWaveform:
    """Get a blank SparseTaylorTimeWaveform object with arrays of the correct sizes"""
    # need the frequency derivatives to calculate the maximum possible size
    fds = wc.dfd * np.arange(-wc.Nfd_negative, wc.Nfd - wc.Nfd_negative)
    # calculate maximum possible number of pixels
    N_max = np.int64(np.ceil((wc.BW + fds[wc.Nfd - 1] * wc.Tw) / wc.DF)) * wc.Nt
    # array of wavelet coefficients
    wave_value = np.zeros((nc_waveform, N_max))
    # aray of pixel indices
    pixel_index = np.full((nc_waveform, N_max), -1, dtype=np.int64)
    # number of pixel indices that are set
    n_set = np.zeros(nc_waveform, dtype=np.int64)

    return SparseTaylorWaveform(wave_value, pixel_index, n_set, N_max)


def sparse_addition_helper(sparse_waveform, dense_representation) -> None:
    """Take a sparse wavelet representation from e.g. SparseTaylorTimeWaveform
    and add it to a dense wavelet representation
    sparse_waveform: namedtuple SparseTaylorTimeWaveform
        the sparse representation of waveform, with
        at least as many channels as dense_representation
    dense_representation: nd.ndarray
        the dense wavelet representation
        shape: (Nf*Nt, N channels)
    """
    nc_waveform = sparse_waveform.wave_value.shape[0]
    nc_representation = dense_representation.shape[-1]
    pixel_index = sparse_waveform.pixel_index
    n_set = sparse_waveform.n_set
    wave_value = sparse_waveform.wave_value

    # write to all the channels that we know how to write to
    nc_loc = min(nc_representation, nc_waveform)

    for itrc in range(nc_loc):
        dense_representation[pixel_index[itrc, :n_set[itrc]], itrc] += wave_value[itrc, :n_set[itrc]]


def get_evTs(wc: WDMWaveletConstants, cache_mode='skip', output_mode='skip') -> WaveletTaylorTimeCoeffs:
    """Helper to get the ev matrices
    cache_mode:
        'skip':
            do not check if a cached coefficient table is available
        'check':
            check if a cached coefficiient table is available
    output_mode:
        'skip': just return, do not output to a file
        'hf': output to an hdf5 file in the coeffs/ directory
    """
    t0 = time()

    print("Filter length (seconds) %e" % wc.Tw)
    print("dt=" + str(wc.dt) + "s Tobs=" + str(wc.Tobs / SECSYEAR))

    print("full filter bandwidth %e  samples %d" % ((wc.A + wc.B) / np.pi, (wc.A + wc.B) / np.pi * wc.Tw))
    cache_good = False

    if cache_mode == 'check':
        coeffTs_in = 'coeffs/WDMcoeffs_Nsf=' + str(wc.Nsf) + '_mult=' + str(wc.mult) + '_Nfd=' + str(wc.Nfd) + '_Nf=' + str(wc.Nf)\
            + '_Nt=' + str(wc.Nt) + '_dt=' + str(wc.dt) + '_dfdot=' + str(wc.dfdot) + '_Nfd_neg' + str(wc.Nfd_negative) + '_fast.h5'

        fds = wc.dfd * np.arange(-wc.Nfd_negative, wc.Nfd - wc.Nfd_negative)

        # number of samples for each frequency derivative layer (grow with increasing BW)
        Nfsam = ((wc.BW + np.abs(fds) * wc.Tw) / wc.df).astype(np.int64)
        Nfsam[Nfsam % 2 != 0] += 1  # makes sure it is an even number

        max_shape = np.max(Nfsam)
        evcs = np.zeros((wc.Nfd, max_shape))
        evss = np.zeros((wc.Nfd, max_shape))

        try:
            hf_in = h5py.File(coeffTs_in, 'r')
            for i in range(wc.Nfd):
                evcs[i, :Nfsam[i]] = np.asarray(hf_in['evcs'][str(i)])
                evss[i, :Nfsam[i]] = np.asarray(hf_in['evss'][str(i)])
            hf_in.close()
            cache_good = True
        except OSError:
            print(coeffTs_in)
            print('cache checked and missed')
    elif cache_mode == 'skip':
        pass
    else:
        msg = 'Unrecognized option for cache_mode'
        raise NotImplementedError(msg)

    if not cache_good:
        phi = np.zeros(wc.K)
        DX = np.zeros(wc.K, dtype=np.complex128)
        DX[0] = wc.insDOM

        DX[1:np.int64(wc.K / 2) + 1] = np.sqrt(wc.dt) * phitilde_vec(wc.dom * wc.dt * np.arange(1, np.int64(wc.K / 2) + 1), wc.Nf, wc.nx)
        DX[np.int64(wc.K / 2) + 1:] = np.sqrt(wc.dt) * phitilde_vec(-wc.dom * wc.dt * np.arange(np.int64(wc.K / 2) - 1, 0, -1), wc.Nf, wc.nx)

        DX = sp.fft.fft(DX, wc.K, overwrite_x=True)

        for i in range(np.int64(wc.K / 2)):
            phi[i] = np.real(DX[np.int64(wc.K / 2) + i])
            phi[np.int64(wc.K / 2) + i] = np.real(DX[i])

        nrm = np.linalg.norm(phi)
        print("norm=" + str(nrm))

        # it turns out that all the wavelet layers are the same modulo a
        # shift in the reference frequency. Just have to do a single layer
        # we pick one far from the boundaries to avoid edge effects

        k = wc.Nf / 16

        wave = wavelet(k, wc.K, nrm, wc.dom, wc.DOM, wc.Nf, wc.dt, wc.nx)

        fd = wc.DF / wc.Tw * wc.dfdot * np.arange(-wc.Nfd_negative, wc.Nfd - wc.Nfd_negative)  # set f-dot increments

        print("%e %.14e %.14e %e %e" % (wc.DT, wc.DF, wc.DOM / (2 * np.pi), fd[1], fd[wc.Nfd - 1]))

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
        print("loop start time ", t1 - t0, "s")
        evcs, evss = get_ev_t_full(wave, wc)
        tf = time()
        print("got full evcs in %f s" % (tf - t1))
        t1 = time()

    if output_mode == 'hf':
        hf = h5py.File('coeffs/WDMcoeffs_Nsf=' + str(wc.Nsf) + '_mult=' + str(wc.mult) + '_Nfd=' + str(wc.Nfd) + '_Nf=' + str(wc.Nf) +
                       '_Nt=' + str(wc.Nt) + '_dt=' + str(wc.dt) + '_dfdot=' + str(wc.dfdot) + '_Nfd_neg' + str(wc.Nfd_negative) + '_fast.h5', 'w')
        hf.create_group('inds')
        hf.create_group('evcs')
        hf.create_group('evss')
        for jj in range(wc.Nfd):
            hf['inds'].create_dataset(str(jj), data=np.arange(0, Nfsam[jj]))
            hf['evcs'].create_dataset(str(jj), data=evcs[jj, :Nfsam[jj]])
            hf['evss'].create_dataset(str(jj), data=evss[jj, :Nfsam[jj]])
        hf.close()
        t3 = time()
        print("output time", t3 - tf, "s")
    elif output_mode == 'skip':
        pass
    else:
        msg = 'unrecognized option for output_mode'
        raise NotImplementedError(msg)

    return WaveletTaylorTimeCoeffs(Nfsam, evcs, evss)
