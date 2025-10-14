"""Functions for constructing wavelet-domain representations of time-domain waveforms.

The functions use the WDM wavelet basis.
"""

from typing import TYPE_CHECKING

import numpy as np
from numba import njit

from LisaWaveformTools.stationary_source_waveform import StationaryWaveformTime
from WaveletWaveforms.sparse_waveform_functions import PixelGenericRange, SparseWaveletWaveform
from WaveletWaveforms.taylor_time_coefficients import WaveletTaylorTimeCoeffs, get_taylor_time_pixel_direct
from WaveletWaveforms.wdm_config import WDMWaveletConstants

if TYPE_CHECKING:
    from numpy.typing import NDArray


@njit(fastmath=True)
def wavemaket(
    wavelet_waveform: SparseWaveletWaveform,
    waveform: StationaryWaveformTime,
    nt_lim_waveform: PixelGenericRange,
    wc: WDMWaveletConstants,
    taylor_table: WaveletTaylorTimeCoeffs,
    *,
    force_nulls: int = 0,
) -> None:
    """
    Construct a sparse wavelet-domain representation of a time-domain intrinsic_waveform using interpolation tables.

    This function updates the `wavelet_waveform` object in place with values derived from the
    supplied time-domain intrinsic_waveform over the pixel range set in
    `_nt_lim_waveform`. `_wc` is an object with configuration parameters for the wavelet decomposition.
    Precomputed Taylor expansion coefficients, used to interpolate intrinsic_waveform values, are looked up from `taylor_table`.

    Parameters
    ----------
    wavelet_waveform : SparseWaveletWaveform
        The output object that will be populated with the sparse wavelet-domain coefficients.
        Modified in place.
    waveform : StationaryWaveformTime
        The input time-domain intrinsic_waveform data with amplitude and phase arrays for each pixel.
    nt_lim_waveform : PixelGenericRange
        Limits and properties of the time pixel range to be processed.
    wc : WDMWaveletConstants
        Configuration parameters for the wavelet decomposition, including:
        - `Nf`: Number of frequency layers in the wavelet decomposition.
        - `DF`: Frequency resolution of the wavelet decomposition.
        - `dfd`: Spacing of frequency derivative layers in the interpolation table.
        - `dfdot`: Fractional spacing frequency derivative layers increment.
        - `Nfd`: Total number of frequency derivative layers.
        - `Nfd_negative`: Number of negative frequency derivative layers.
        - `df_bw`: Spacing of frequency layers in the interpolation table.
        - `Nsf`: Number of frequency samples in the interpolation table.


    taylor_table : WaveletTaylorTimeCoeffs
        Precomputed Taylor expansion coefficients for linear interpolation of the intrinsic_waveform.
    force_nulls : int, optional
        If 1, sets the wavelet coefficients to zero outside the precomputed table.
        If 0, pixels outside the precomputed table are dropped from the sparse representation.
        If the table is large, pixels outside the table typically represent nulls where the
        amplitude is zero or very small, and thus dropping them is acceptable for some applications.
        For snr calculations, plotting, or computing coadded galactic backgrounds, force_nulls=1 has little benefit.
        However, for likelihood calculations, force_nulls=1 appropriately penalizes data that has power in nulls.
        force_nulls=2 is reserved for calculating wavelet coefficients outside the precomputed table directly,
        but currently returns a NotImplementedError


    Returns
    -------
    None
        This function modifies `wavelet_waveform` in place and does not return a value.

    See Also
    --------
    wavemaket_direct : Direct computation of wavelet coefficients without table lookup.
    """
    n_set_old = wavelet_waveform.n_set.copy()

    nc_waveform = wavelet_waveform.wave_value.shape[0]

    for itrc in range(nc_waveform):
        mm: int = 0
        nf_min: int = 0
        nf_max: int = wc.Nf - 1
        for j in range(nt_lim_waveform.nx_min, nt_lim_waveform.nx_max):
            # keep j_ind separate in case we want to apply a time offset in the future
            j_ind: int = j

            y0: float = waveform.FTd[itrc, j] / wc.dfd
            ny: int = int(np.floor(y0))
            n_ind: int = ny + wc.Nfd_negative

            assert taylor_table.Nfsam.size == wc.Nfd
            if 0 <= n_ind < wc.Nfd - 1:
                c: float = waveform.AT[itrc, j] * np.cos(waveform.PT[itrc, j])
                s: float = waveform.AT[itrc, j] * np.sin(waveform.PT[itrc, j])

                dy: float = y0 - ny
                assert 0.0 <= dy <= 1.0
                fa: float = waveform.FT[itrc, j]
                za: float = fa / wc.df_bw

                Nfsam1_loc = int(taylor_table.Nfsam[n_ind])
                # assert Nfsam1_loc == int(_wc.Nsf + 2 / 3 * np.abs(ny) * _wc.dfdot * _wc.Nsf)
                Nfsam2_loc = int(taylor_table.Nfsam[n_ind + 1])
                # assert Nfsam2_loc == int(_wc.Nsf + 2 / 3 * np.abs(ny + 1) * _wc.dfdot * _wc.Nsf)
                half_bandwidth: float = (min(Nfsam1_loc, Nfsam2_loc) - 1) * wc.df_bw / 2

                # lowest frequency layer
                kmin: int = max(nf_min, int(np.ceil((fa - half_bandwidth) / wc.DF)))

                # highest frequency layer
                kmax: int = min(nf_max, int(np.floor((fa + half_bandwidth) / wc.DF)))

                for k in range(kmin, kmax + 1):
                    zmid: float = (wc.DF / wc.df_bw) * k

                    # we apparently need za - zmid to be positive
                    if za < zmid:
                        zmid = za - np.abs(za - zmid)

                    kk_float: float = np.floor(za - zmid - 0.5)
                    zsam: float = zmid + kk_float + 0.5
                    kk = int(kk_float)
                    dx: float = za - zsam  # used for linear interpolation
                    assert 0.0 <= dx <= 1.0

                    # interpolate over frequency
                    jj1 = kk + Nfsam1_loc // 2
                    jj2 = kk + Nfsam2_loc // 2

                    # prevent case where we would overflow the table
                    if (0 <= jj1 < Nfsam1_loc - 1) and (0 <= jj2 < Nfsam2_loc - 1):
                        wavelet_waveform.pixel_index[itrc, mm] = j_ind * wc.Nf + k

                        assert taylor_table.evc[n_ind, jj1] != 0.0
                        assert taylor_table.evc[n_ind, jj1 + 1] != 0.0
                        assert taylor_table.evc[n_ind + 1, jj2] != 0.0
                        assert taylor_table.evc[n_ind + 1, jj2 + 1] != 0.0

                        y: float = (1.0 - dx) * taylor_table.evc[n_ind, jj1] + dx * taylor_table.evc[n_ind, jj1 + 1]
                        yy: float = (1.0 - dx) * taylor_table.evc[n_ind + 1, jj2] + dx * taylor_table.evc[n_ind + 1, jj2 + 1]

                        z: float = (1.0 - dx) * taylor_table.evs[n_ind, jj1] + dx * taylor_table.evs[n_ind, jj1 + 1]
                        zz: float = (1.0 - dx) * taylor_table.evs[n_ind + 1, jj2] + dx * taylor_table.evs[n_ind + 1, jj2 + 1]

                        # interpolate over fdot
                        y = (1.0 - dy) * y + dy * yy
                        z = (1.0 - dy) * z + dy * zz

                        if (j_ind + k) % 2:
                            wavelet_waveform.wave_value[itrc, mm] = -(c * z + s * y)
                        else:
                            wavelet_waveform.wave_value[itrc, mm] = c * y - s * z

                        mm += 1
                        # end loop over frequency layers
            elif force_nulls == 1:
                # we know what the indices would be for values not precomputed in the table
                # so force values outside the range of the table to 0 instead of dropping,
                # in order to get likelihoods right, which is particularly important around total nulls,
                # which can be quite constraining on the parameters but have large spikes in frequency derivative
                # note that if this happens only very rarely
                # we could just actually calculate the non-precomputed coefficient
                fa = waveform.FT[itrc, j]

                Nfsam1_loc = int((wc.BW + wc.dfd * wc.Tw * ny) / wc.df_bw)
                if Nfsam1_loc % 2 == 1:
                    Nfsam1_loc += 1

                half_bandwidth = (Nfsam1_loc - 1) * wc.df_bw / 2
                # lowest frequency layer
                kmin = max(0, int(np.ceil((fa - half_bandwidth) / wc.DF)))

                # highest frequency layer
                kmax = min(wc.Nf - 1, int(np.floor((fa + half_bandwidth) / wc.DF)))

                for k in range(kmin, kmax + 1):
                    wavelet_waveform.pixel_index[itrc, mm] = j_ind * wc.Nf + k
                    wavelet_waveform.wave_value[itrc, mm] = 0.0
                    mm += 1
            elif force_nulls == 2:
                msg1 = 'Direct computation of values outside the table is not implemented yet'
                raise NotImplementedError(msg1)
            elif force_nulls == 0:
                pass
            else:
                msg2 = 'Unrecognized force_nulls option. Valid options are 0, 1, or 2.'
                raise NotImplementedError(msg2)

        wavelet_waveform.n_set[itrc] = mm

    # clean up any pixels that were set in the old intrinsic_waveform but aren't anymore
    for itrc in range(nc_waveform):
        for itrm in range(wavelet_waveform.n_set[itrc], n_set_old[itrc]):
            wavelet_waveform.pixel_index[itrc, itrm] = -1
            wavelet_waveform.wave_value[itrc, itrm] = 0.0


@njit()
def wavemaket_direct(
    wavelet_waveform: SparseWaveletWaveform,
    waveform: StationaryWaveformTime,
    nt_lim_waveform: PixelGenericRange,
    wc: WDMWaveletConstants,
    taylor_table: WaveletTaylorTimeCoeffs,
) -> None:
    """
    Construct a sparse wavelet-domain representation of a time-domain intrinsic_waveform without interpolation.

    This function updates the `wavelet_waveform` object in place with values derived from the
    supplied time-domain intrinsic_waveform over the pixel range specified by `_nt_lim_waveform`.
    Unlike `wavemaket`, which uses precomputed interpolation tables,
    `wavemaket_direct` evaluates Taylor coefficients directly at each pixel, providing a more accurate
    calculation when the interpolation table is insufficiently dense or at points where rapid
    parameter variation occurs, as can happen near nulls in the intrinsic_waveform.

    Parameters
    ----------
    wavelet_waveform : SparseWaveletWaveform
        The output object that will be populated with the sparse wavelet-domain coefficients.
        Modified in place.
    waveform : StationaryWaveformTime
        The input time-domain intrinsic_waveform data with amplitude and phase arrays for each pixel.
    nt_lim_waveform : PixelGenericRange
        Limits and properties of the time pixel range to be processed.
    wc : WDMWaveletConstants
        Configuration parameters for the wavelet decomposition, including:
        - `Nf`: Number of frequency layers in the wavelet decomposition.
        - `DF`: Frequency resolution of the wavelet decomposition.
        - `dfd`: Spacing of frequency derivative layers.
        - `dfdot`: Fractional spacing increment for frequency derivative layers.
        - `Nsf`: Number of frequency samples per table entry.
        - `df_bw`: Spacing of frequency layers (bandwidth) in frequency direction.
    taylor_table : WaveletTaylorTimeCoeffs
        Taylor expansion coefficients and normalization factors required for direct evaluation.

    Returns
    -------
    None
        This function modifies `wavelet_waveform` in place and does not return a value.

    Notes
    -----
    This direct evaluation is particularly useful for:
        - Testing the accuracy of the table-based approach (`wavemaket`)
        - Scenarios where the interpolation table is not sufficiently dense
        - Pixels near intrinsic_waveform nulls or with large parameter slopes

    See Also
    --------
    wavemaket : Table-based computation of wavelet coefficients using precomputed interpolation.
    """
    n_set_old: NDArray[np.integer] = wavelet_waveform.n_set.copy()

    nc_waveform: int = wavelet_waveform.wave_value.shape[0]

    wavelet_norm = taylor_table.wavelet_norm

    for itrc in range(nc_waveform):
        mm: int = 0
        nf_min: int = 0
        nf_max: int = wc.Nf - 1

        for j in range(nt_lim_waveform.nx_min, nt_lim_waveform.nx_max):
            j_ind: int = j  # keep j_ind separate in case we want to apply a time offset in the future

            c: float = waveform.AT[itrc, j] * np.cos(waveform.PT[itrc, j])
            s: float = waveform.AT[itrc, j] * np.sin(waveform.PT[itrc, j])

            fa: float = waveform.FT[itrc, j]

            y0: float = waveform.FTd[itrc, j] / wc.dfd
            ny = int(np.floor(y0))

            Nfsam1_loc: int = int(wc.Nsf + 2 / 3 * np.abs(ny) * wc.dfdot * wc.Nsf)
            Nfsam2_loc: int = int(wc.Nsf + 2 / 3 * np.abs(ny + 1) * wc.dfdot * wc.Nsf)
            # not sure the - 1 is strictly necessary
            half_bandwidth: float = (min(Nfsam1_loc, Nfsam2_loc) - 1) * wc.df_bw / 2

            # lowest frequency layer
            kmin: int = int(np.ceil((fa - half_bandwidth) / wc.DF))
            kmin = max(nf_min, kmin)
            kmin = min(nf_max, kmin)

            # highest frequency layer
            kmax: int = min(nf_max, int(np.floor((fa + half_bandwidth) / wc.DF)))
            kmax = max(nf_min, kmax)
            kmax = max(kmin, kmax)

            for k in range(kmin, kmax + 1):
                wavelet_waveform.pixel_index[itrc, mm] = j_ind * wc.Nf + k

                y, z = get_taylor_time_pixel_direct(fa, waveform.FTd[itrc, j], k, wavelet_norm, wc)

                if (j_ind + k) % 2:
                    wavelet_waveform.wave_value[itrc, mm] = -(c * z + s * y)
                else:
                    wavelet_waveform.wave_value[itrc, mm] = c * y - s * z

                mm += 1
                # end loop over frequency layers

        wavelet_waveform.n_set[itrc] = mm

    # clean up any pixels that were set in the old intrinsic_waveform but aren't anymore
    for itrc in range(nc_waveform):
        for itrm in range(int(wavelet_waveform.n_set[itrc]), int(n_set_old[itrc])):
            wavelet_waveform.pixel_index[itrc, itrm] = -1
            wavelet_waveform.wave_value[itrc, itrm] = 0.0
