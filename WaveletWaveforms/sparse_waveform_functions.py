"""Functions that relate the dense and sparse representations of waveforms."""

from typing import NamedTuple

import numpy as np
from numba import njit
from numpy.typing import NDArray

from WaveletWaveforms.wdm_config import WDMWaveletConstants


class SparseWaveletWaveform(NamedTuple):
    """
    NamedTuple object that contains a sparse representation of a wavelet intrinsic_waveform,
    as computed using the taylor approximation in e.g. wavemaket
    because the actual array size needed isn't known in advance,
    it internally stores arrays that are the maximum size possible in
    the approximation, and records the number of pixels that are actually set in
    n_set. In pixel_index, it additionally uses indicator values of -1 to indicate pixels
    which are not set. The pixel indices don't need to be listed in any particular order,
    Although for some applications it could be convenient for them to be sorted.
    wave_value : numpy.ndarray
        stores the actual intrinsic_waveform as the values of the wavelet pixels
        at the pixel indices specified by lists_pixels.
        All values with index >= n_set should be set to 0.
        Shape is the same as lists_pixels: shape: (_nc_waveform, n_max)
    pixel_index : numpy.ndarray
        stores the indices of x,y coordinates of all pixels that are
        currently set. All values with index >= n_set should be set to -1
        shape: (_nc_waveform, n_max) number of TDI channels x maximum number
        of pixels possible in sparse representation.
    n_set : numpy.ndarray of integers
        number of wavelet coefficients that are *currently* set
        all values must be <= n_pixel_max.
        shape: number of TDI channels
    n_pixel_max: integer
        the maximum number of wavelet pixels that could possibly
        be set in the sparse representation,
        which is determined by the shape of the interpolation table
    """

    wave_value: NDArray[np.floating]
    pixel_index: NDArray[np.int64]
    n_set: NDArray[np.integer]
    n_pixel_max: int


class PixelGenericRange(NamedTuple):
    """
    NamedTuple object to contain range of pixels for analysis.
    """

    nx_min: int
    nx_max: int
    dx: float
    x_min: float


def sparse_addition_helper(sparse_waveform: SparseWaveletWaveform, dense_representation: NDArray[np.floating]) -> None:
    """Take a sparse wavelet representation from SparseWaveletWaveform
    and add it to a dense wavelet representation
    sparse_waveform: NamedTuple SparseTaylorTimeWaveform
        the sparse representation of intrinsic_waveform, with
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
def wavelet_sparse_to_dense(wavelet_waveform: SparseWaveletWaveform, wc: WDMWaveletConstants) -> NDArray[np.floating]:
    """Unpack a sparse wavelet representation to a dense wavelet representation."""
    # initialize the array
    n_set = wavelet_waveform.n_set
    pixel_index = wavelet_waveform.pixel_index
    wave_value = wavelet_waveform.wave_value
    nc_waveform = wavelet_waveform.n_set.size
    n_pixel_max = wavelet_waveform.n_pixel_max

    # input size validation
    assert len(pixel_index.shape) == 2
    assert len(n_set.shape) == 1
    assert pixel_index.shape == wave_value.shape
    assert pixel_index.shape[1] == n_pixel_max
    assert pixel_index.shape[0] == nc_waveform
    assert np.all(n_set >= 0)
    assert np.all(n_set <= n_pixel_max)

    result = np.zeros((wc.Nt, wc.Nf, nc_waveform))

    # unpack the signal
    for itrc in range(nc_waveform):
        assert n_set[itrc] <= n_pixel_max
        for itrp in range(n_set[itrc]):
            assert pixel_index[itrc, itrp] != -1
            i: int = int(pixel_index[itrc, itrp] % wc.Nf)
            j: int = int((pixel_index[itrc, itrp] - i) // wc.Nf)
            result[j, i, itrc] = wave_value[itrc, itrp]

    return result


@njit()
def whiten_sparse_data(
    wavelet_waveform: SparseWaveletWaveform, inv_chol_S: NDArray[np.floating], wc: WDMWaveletConstants
) -> SparseWaveletWaveform:
    # initialize the array
    n_set = wavelet_waveform.n_set
    pixel_index = wavelet_waveform.pixel_index
    wave_value = wavelet_waveform.wave_value
    nc_waveform = wavelet_waveform.n_set.size
    n_pixel_max = wavelet_waveform.n_pixel_max

    # input size validation
    assert len(pixel_index.shape) == 2
    assert len(n_set.shape) == 1
    assert pixel_index.shape == wave_value.shape
    assert pixel_index.shape[1] == n_pixel_max
    assert pixel_index.shape[0] == nc_waveform
    print(inv_chol_S.shape)
    print(wc.Nt, wc.Nf)
    assert len(inv_chol_S.shape) == 3
    assert inv_chol_S.shape[0] == wc.Nt
    assert inv_chol_S.shape[1] == wc.Nf
    assert inv_chol_S.shape[2] == nc_waveform
    assert np.all(n_set >= 0)
    assert np.all(n_set <= n_pixel_max)

    wave_value_new = np.zeros_like(wave_value)

    # unpack the signal
    for itrc in range(nc_waveform):
        assert n_set[itrc] <= n_pixel_max
        for itrp in range(n_set[itrc]):
            assert pixel_index[itrc, itrp] != -1
            i: int = int(pixel_index[itrc, itrp] % wc.Nf)
            j: int = int((pixel_index[itrc, itrp] - i) // wc.Nf)
            wave_value_new[itrc, itrp] = wave_value[itrc, itrp] * inv_chol_S[j, i, itrc]

    return SparseWaveletWaveform(wave_value_new, pixel_index, n_set, n_pixel_max)


@njit()
def wavelet_dense_select_sparse(
    dense_waveform: NDArray[np.floating],
    wavelet_waveform: SparseWaveletWaveform,
    wc: WDMWaveletConstants,
    *,
    inplace_mode: int = 0,
) -> SparseWaveletWaveform:
    """Select sparse elements with matching indices from a dense waveform.

    Output is written to a new object if inplace_mode=0, if inplace_mode=1 a new object is created.
    """
    # initialize the array
    n_set = wavelet_waveform.n_set
    pixel_index = wavelet_waveform.pixel_index
    wave_value = wavelet_waveform.wave_value
    nc_waveform = wavelet_waveform.n_set.size
    n_pixel_max = wavelet_waveform.n_pixel_max

    # input size validation
    assert len(pixel_index.shape) == 2
    assert len(n_set.shape) == 1
    assert pixel_index.shape == wave_value.shape
    assert pixel_index.shape[1] == n_pixel_max
    assert pixel_index.shape[0] == nc_waveform
    assert len(dense_waveform.shape) == 3
    assert dense_waveform.shape[0] == wc.Nt
    assert dense_waveform.shape[1] == wc.Nf
    assert dense_waveform.shape[2] == nc_waveform
    assert np.all(n_set >= 0)
    assert np.all(n_set <= n_pixel_max)

    # input parameter validation
    assert inplace_mode in (0, 1), 'Use inplace_mode=0 to create a new waveform objects, inplace_mode=1 to overwrite'

    if inplace_mode == 0:
        # create a new sparse output array
        waveform_out = SparseWaveletWaveform(np.zeros_like(wave_value), pixel_index, n_set, n_pixel_max)
    else:
        # reuse the sparse input array
        waveform_out = wavelet_waveform

    # ensure that the write happens in place on the array that was stored in the output array
    wave_value_new = waveform_out.wave_value

    # unpack the signal
    for itrc in range(nc_waveform):
        assert n_set[itrc] <= n_pixel_max
        for itrp in range(n_set[itrc]):
            assert pixel_index[itrc, itrp] != -1
            i: int = int(pixel_index[itrc, itrp] % wc.Nf)
            j: int = int((pixel_index[itrc, itrp] - i) // wc.Nf)
            wave_value_new[itrc, itrp] = dense_waveform[j, i, itrc]

    return waveform_out
