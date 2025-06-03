"""Functions that relate the dense and sparse representations of waveforms."""
from collections import namedtuple

import numpy as np
from numba import njit
from numpy.typing import NDArray

from WaveletWaveforms.wdm_config import WDMWaveletConstants

SparseWaveletWaveform = namedtuple('SparseWaveletWaveform', ['wave_value', 'pixel_index', 'n_set', 'N_max'])

SparseWaveletWaveform.__doc__ = """
namedtuple object that contains a sparse representation of a wavelet waveform,
as computed using the taylor approximation in e.g. wavemaket
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

PixelTimeRange = namedtuple('PixelTimeRange', ['nt_min', 'nt_max'])

PixelTimeRange.__doc__ = """
namedtuple object to contain range of time pixels for analysis.
"""

def sparse_addition_helper(sparse_waveform: SparseWaveletWaveform, dense_representation: NDArray[np.float64]) -> None:
    """Take a sparse wavelet representation from SparseWaveletWaveform
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
        dense_representation[pixel_index[itrc, : n_set[itrc]], itrc] += wave_value[itrc, : n_set[itrc]]

@njit()
def wavelet_sparse_to_dense(wavelet_waveform: SparseWaveletWaveform, wc: WDMWaveletConstants) -> NDArray[np.float64]:
    """Unpack a sparse wavelet representation to a dense wavelet representation."""
    # initialize the array
    n_set = wavelet_waveform.n_set
    pixel_index = wavelet_waveform.pixel_index
    wave_value = wavelet_waveform.wave_value
    nc_waveform = wavelet_waveform.n_set.size

    # input size validation
    assert len(pixel_index.shape) == 2
    assert len(n_set.shape) == 1
    assert pixel_index.shape == wave_value.shape
    assert pixel_index.shape[1] == wavelet_waveform.N_max
    assert pixel_index.shape[0] == nc_waveform

    result = np.zeros((wc.Nt, wc.Nf, nc_waveform))

    #unpack the signal
    for itrc in range(nc_waveform):
        assert n_set[itrc] <= wavelet_waveform.N_max
        for itrp in range(n_set[itrc]):
            assert pixel_index[itrc, itrp] != -1
            i = pixel_index[itrc, itrp] % wc.Nf
            j = (pixel_index[itrc, itrp] - i) // wc.Nf
            result[j, i, itrc] = wave_value[itrc, itrp]
        # i = np.mod(pixel_index[itrc, :n_set[itrc]], wc.Nf).astype(np.int64)
        # j = ((pixel_index[itrc, :n_set[itrc]] - i) // wc.Nf).astype(np.int64)
        # result[j, i, itrc] = wave_value[itrc, :n_set[itrc]]

    return result
