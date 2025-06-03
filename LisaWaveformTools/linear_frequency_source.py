"""A source with linearly increasing frequency and constant amplitude."""

from collections import namedtuple

import numpy as np
from numba import njit
from numpy.typing import NDArray

from LisaWaveformTools.lisa_config import LISAConstants
from LisaWaveformTools.ra_waveform_freq import AntennaResponseChannels, get_spacecraft_vec, get_tensor_basis, get_wavefront_time, rigid_adiabatic_antenna
from LisaWaveformTools.ra_waveform_time import StationaryWaveformTime, get_time_tdi_amp_phase
from LisaWaveformTools.stationary_source_waveform import StationarySourceWaveform
from WaveletWaveforms.sparse_waveform_functions import PixelTimeRange
from WaveletWaveforms.wdm_config import WDMWaveletConstants

LinearFrequencyParams = namedtuple('LinearFrequencyParams', ['intrinsic', 'extrinsic'])

LinearFrequencyParams.__doc__ = """
Store the parameters for a source with linearly increasing frequency.
Galactic binaries are the prototypical example of such a source.
Parameters
----------
intrinsic: LinearFrequencyIntrinsicParams
    The intrinsic parameters for the source, see LinearFrequencyIntrinsicParams
extrinsic: ExtrinsicParams
    The extrinsic parameters for the source, see ExtrinsicParams
"""

LinearFrequencyIntrinsicParams = namedtuple('LinearFrequencyIntrinsicParams', ['amp0_t', 'phi0', 'F0', 'FTd0'])

LinearFrequencyIntrinsicParams.__doc__ = """
Store the intrinsic parameters for a source with linearly increasing frequency.

Galactic binaries are the prototypical example of such a source.
The amplitude is assumed constant in the time domain.
The time domain waveform is given
h(t) = amp_t*cos(-phi0 + 2*pi*F0*t + pi*FTd0*t^2)

Parameters
----------
amp0_t: float
    The constant time domain waveform amplitude
phi0: float
    The time domain phase at t=0
F0: float
    The frequency at t=0
FTd0: float
    The frequency derivative with respect to time dF/dt at t=0
"""


# TODO check factor of 2pi
@njit()
def linear_frequency_intrinsic(waveform: StationaryWaveformTime, intrinsic_params: LinearFrequencyIntrinsicParams, t_in: NDArray[np.float64]) -> None:
    """
    Get time domain waveform for a linearly increasing frequency source with constant amplitude.

    For use in the stationary phase approximation. The actual waveform would be given:
    h = amp_t*np.cos(-phi0 + 2*pi*F0*t + pi*FTd0*t**2)
    Note that t_in is not the same as the time coordinate in the input waveform object;
    it may be a different time coordinate, due to the need to convert from the solar system barycenter
    frame to the constellation guiding center frame (or the frames of the individual spacecraft).

    Parameters
    ----------
    waveform : StationaryWaveformTime
        The waveform object to be updated in place with the time domain waveform.
    intrinsic_params : LinearFrequencyIntrinsicParams
        Dictionary or namespace containing intrinsic parameters of the source.
    t_in : np.ndarray
        Array of arrival times (or evaluation times) for the signal.

    Returns
    -------
    None
        The function updates the input `waveform` object in place.
    """
    AT = waveform.AT
    PT = waveform.PT
    FT = waveform.FT
    FTd = waveform.FTd

    nt_loc = t_in.size
    #  compute the intrinsic frequency, phase and amplitude
    for n in range(nt_loc):
        t = t_in[n]
        FT[n] = intrinsic_params.F0 + intrinsic_params.FTd0 * t
        FTd[n] = intrinsic_params.FTd0
        PT[n] = -intrinsic_params.phi0 + 2 * np.pi * intrinsic_params.F0 * t + np.pi * intrinsic_params.FTd0 * t ** 2
        AT[n] = intrinsic_params.amp0_t


# TODO do consistency checks
class LinearFrequencyWaveformTime(StationarySourceWaveform):
    """Store a binary waveform with linearly increasing frequency and constant amplitude in the time domain.
    """

    def __init__(self, params: LinearFrequencyParams, nt_lim_waveform: PixelTimeRange, lc: LISAConstants, wc: WDMWaveletConstants) -> None:
        """Initalize the object"""
        self.params: LinearFrequencyParams = params
        self.nt_lim_waveform: PixelTimeRange = nt_lim_waveform
        self.nt_range: int = self.nt_lim_waveform.nt_max - self.nt_lim_waveform.nt_min
        self.lc: LISAConstants = lc
        self.wc: WDMWaveletConstants = wc
        self.nc_waveform: int = self.lc.nc_waveform

        self.TTs: NDArray[np.float64] = self.wc.DT * np.arange(self.nt_lim_waveform.nt_min, self.nt_lim_waveform.nt_max)

        AmpTs = np.zeros(self.nt_range)
        PPTs = np.zeros(self.nt_range)
        FTs = np.zeros(self.nt_range)
        FTds = np.zeros(self.nt_range)

        self.waveform = StationaryWaveformTime(self.TTs, PPTs, FTs, FTds, AmpTs)

        del AmpTs
        del PPTs
        del FTs
        del FTds

        RRs = np.zeros((self.nc_waveform, self.nt_range))
        IIs = np.zeros((self.nc_waveform, self.nt_range))
        dRRs = np.zeros((self.nc_waveform, self.nt_range))
        dIIs = np.zeros((self.nc_waveform, self.nt_range))

        self.spacecraft_channels = AntennaResponseChannels(self.TTs, RRs, IIs, dRRs, dIIs)

        del RRs
        del IIs
        del dRRs
        del dIIs

        self.spacecraft_orbits = get_spacecraft_vec(self.TTs, self.lc)

        self.wavefront_time = np.zeros(self.nt_range)
        self.kdotx = np.zeros(self.nt_range)

        AET_AmpTs = np.zeros((self.nc_waveform, self.nt_range))
        AET_PPTs = np.zeros((self.nc_waveform, self.nt_range))
        AET_FTs = np.zeros((self.nc_waveform, self.nt_range))
        AET_FTds = np.zeros((self.nc_waveform, self.nt_range))

        self.AET_waveform = StationaryWaveformTime(self.TTs, AET_PPTs, AET_FTs, AET_FTds, AET_AmpTs)

        del AET_AmpTs
        del AET_PPTs
        del AET_FTs
        del AET_FTds

        self.update_params(params)

    def _update_intrinsic(self) -> None:
        """Update the waveform with respect to the intrinsic parameters."""
        tb = get_tensor_basis(self.params.extrinsic)
        get_wavefront_time(self.lc, tb, self.TTs, self.spacecraft_orbits, self.wavefront_time)

        linear_frequency_intrinsic(self.waveform, self.params.intrinsic, self.wavefront_time)

    def _update_extrinsic(self) -> None:
        """Update the waveform with respect to the extrinsic parameters."""
        # TODO fix F_min and nf_range
        rigid_adiabatic_antenna(
            self.spacecraft_channels,
            self.params.extrinsic,
            self.TTs,
            self.waveform.FT,
            0,
            self.nt_range,
            self.kdotx,
            self.lc,
        )
        get_time_tdi_amp_phase(
            self.spacecraft_channels, self.AET_waveform, self.waveform, self.lc, self.wc.DT,
        )

    def update_params(self, params: LinearFrequencyParams) -> None:
        """Update the waveform to match the given input parameters."""
        self.params = params
        self._update_intrinsic()
        self._update_extrinsic()
