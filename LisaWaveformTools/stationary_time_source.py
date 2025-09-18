"""A source with linearly increasing frequency and constant amplitude."""

from typing import override

import numpy as np
from numpy.typing import NDArray

from LisaWaveformTools.lisa_config import LISAConstants
from LisaWaveformTools.ra_waveform_freq import AntennaResponseChannels, SpacecraftOrbits, get_spacecraft_vec, get_tensor_basis, get_wavefront_time, rigid_adiabatic_antenna
from LisaWaveformTools.ra_waveform_time import get_time_tdi_amp_phase
from LisaWaveformTools.stationary_source_waveform import ExtrinsicParams, SourceParams, StationarySourceWaveform, StationaryWaveformTime
from WaveletWaveforms.sparse_waveform_functions import PixelTimeRange


class StationarySourceWaveformTime(StationarySourceWaveform[StationaryWaveformTime]):
    """Store a binary intrinsic_waveform with linearly increasing frequency and constant amplitude in the time domain.
    """

    def __init__(self, params: SourceParams, nt_lim_waveform: PixelTimeRange, lc: LISAConstants) -> None:
        """Initalize the object"""
        self._nt_lim_waveform: PixelTimeRange = nt_lim_waveform
        self._nt_range: int = self._nt_lim_waveform.nt_max - self._nt_lim_waveform.nt_min
        self._lc: LISAConstants = lc
        self._nc_waveform: int = self._lc.nc_waveform
        self._consistent_extrinsic: bool = False

        self._TTs: NDArray[np.float64] = self._nt_lim_waveform.dt * np.arange(self._nt_lim_waveform.nt_min, self._nt_lim_waveform.nt_max)

        AmpTs = np.zeros(self._nt_range)
        PPTs = np.zeros(self._nt_range)
        FTs = np.zeros(self._nt_range)
        FTds = np.zeros(self._nt_range)

        intrinsic_waveform: StationaryWaveformTime = StationaryWaveformTime(self._TTs, PPTs, FTs, FTds, AmpTs)

        del AmpTs
        del PPTs
        del FTs
        del FTds

        RRs = np.zeros((self._nc_waveform, self._nt_range))
        IIs = np.zeros((self._nc_waveform, self._nt_range))
        dRRs = np.zeros((self._nc_waveform, self._nt_range))
        dIIs = np.zeros((self._nc_waveform, self._nt_range))

        self._spacecraft_channels: AntennaResponseChannels = AntennaResponseChannels(self._TTs, RRs, IIs, dRRs, dIIs)

        del RRs
        del IIs
        del dRRs
        del dIIs

        self._spacecraft_orbits: SpacecraftOrbits = get_spacecraft_vec(self._TTs, self._lc)

        self._wavefront_time: NDArray[np.float64] = np.zeros(self._nt_range)
        self._kdotx: NDArray[np.float64] = np.zeros(self._nt_range)

        AET_AmpTs = np.zeros((self._nc_waveform, self._nt_range))
        AET_PPTs = np.zeros((self._nc_waveform, self._nt_range))
        AET_FTs = np.zeros((self._nc_waveform, self._nt_range))
        AET_FTds = np.zeros((self._nc_waveform, self._nt_range))

        tdi_waveform = StationaryWaveformTime(self._TTs, AET_PPTs, AET_FTs, AET_FTds, AET_AmpTs)

        del AET_AmpTs
        del AET_PPTs
        del AET_FTs
        del AET_FTds

        self._tdi_waveform: StationaryWaveformTime = tdi_waveform
        self._intrinsic_waveform: StationaryWaveformTime = intrinsic_waveform

        super().__init__(params, intrinsic_waveform, tdi_waveform)

    @override
    def _update_extrinsic(self) -> None:
        """Update the intrinsic_waveform with respect to the extrinsic parameters."""
        if not isinstance(self.params.extrinsic, ExtrinsicParams):
            msg = 'Extrinsic parameters must be of type ExtrinsicParams.'
            raise TypeError(msg)

        # TODO fix F_min and nf_range
        rigid_adiabatic_antenna(
            self._spacecraft_channels,
            self.params.extrinsic,
            self._TTs,
            self.intrinsic_waveform.FT,
            0,
            self._nt_range,
            self._kdotx,
            self._lc,
        )
        get_time_tdi_amp_phase(
            self._spacecraft_channels, self._tdi_waveform, self.intrinsic_waveform, self._lc, self._nt_lim_waveform.dt,
        )
        self._consistent_extrinsic = True

    @property
    @override
    def nc_waveform(self) -> int:
        """Return the number of channels in the waveform."""
        return self._nc_waveform

    @property
    def wavefront_time(self) -> NDArray[np.float64]:
        """Get the wavefront arrival times at the guiding center."""
        if not self._consistent_intrinsic:
            tb = get_tensor_basis(self.params.extrinsic)
            get_wavefront_time(self._lc, tb, self._TTs, self._spacecraft_orbits, self._wavefront_time)
        return self._wavefront_time
