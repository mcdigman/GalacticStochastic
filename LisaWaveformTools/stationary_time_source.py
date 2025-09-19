"""A source with linearly increasing frequency and constant amplitude."""

from typing import override

import numpy as np
from numpy.typing import NDArray

from LisaWaveformTools.lisa_config import LISAConstants
from LisaWaveformTools.ra_waveform_freq import AntennaResponseChannels, SpacecraftOrbits, get_spacecraft_vec, get_tensor_basis, get_wavefront_time, rigid_adiabatic_antenna
from LisaWaveformTools.ra_waveform_time import get_time_tdi_amp_phase
from LisaWaveformTools.spacecraft_objects import EdgeRiseModel
from LisaWaveformTools.stationary_source_waveform import ExtrinsicParams, SourceParams, StationarySourceWaveform, StationaryWaveformTime
from WaveletWaveforms.sparse_waveform_functions import PixelGenericRange


class StationarySourceWaveformTime(StationarySourceWaveform[StationaryWaveformTime]):
    """Store a binary intrinsic_waveform with linearly increasing frequency and constant amplitude in the time domain.
    """

    def __init__(self, params: SourceParams, nt_lim_waveform: PixelGenericRange, lc: LISAConstants, *, response_mode: int = 0) -> None:
        """Initalize the object"""
        self._nt_lim_waveform: PixelGenericRange = nt_lim_waveform
        self._nt_range: int = self._nt_lim_waveform.nx_max - self._nt_lim_waveform.nx_min
        self._lc: LISAConstants = lc
        self._nc_waveform: int = self._lc.nc_waveform
        self._consistent_extrinsic: bool = False

        self._TTs: NDArray[np.float64] = self._nt_lim_waveform.dx * np.arange(self._nt_lim_waveform.nx_min, self._nt_lim_waveform.nx_max)
        self._spacecraft_orbits: SpacecraftOrbits = get_spacecraft_vec(self._TTs, self._lc)

        self._response_mode: int = -1
        if lc.rise_mode == 3:
            self._er = EdgeRiseModel(-np.inf, np.inf)
        else:
            msg = 'Only rise_mode 3 (no edge) is implemented.'
            raise NotImplementedError(msg)
        self.response_mode = response_mode

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
        nt_lim = PixelGenericRange(0, self._nt_range, self._nt_lim_waveform.dx, self._lc.t0)
        if self.response_mode in (0, 1):
            rigid_adiabatic_antenna(
                self._spacecraft_channels,
                self.params.extrinsic,
                self._TTs,
                self.intrinsic_waveform.FT,
                nt_lim,
                self._kdotx,
                self._lc,
            )
            get_time_tdi_amp_phase(
                self._spacecraft_channels, self._tdi_waveform, self.intrinsic_waveform, self._lc, self._er,
                nt_lim,
            )
        elif self.response_mode == 2:
            # intrinsic only, no rigid adiabatic response
            self._spacecraft_channels.RR[:] = 1.
            self._spacecraft_channels.II[:] = 0.
            self._spacecraft_channels.dRR[:] = 0.
            self._spacecraft_channels.dII[:] = 0.
            self._kdotx[:] = 0.
            self._tdi_waveform.AT[:] = self.intrinsic_waveform.AT
            self._tdi_waveform.PT[:] = self.intrinsic_waveform.PT
            self._tdi_waveform.FT[:] = self.intrinsic_waveform.FT
            self._tdi_waveform.FTd[:] = self.intrinsic_waveform.FTd

        self._consistent_extrinsic = True

    @property
    @override
    def nc_waveform(self) -> int:
        """Return the number of channels in the waveform."""
        return self._nc_waveform

    @property
    def wavefront_time(self) -> NDArray[np.float64]:
        """Get the wavefront arrival times at the guiding center."""
        if self.response_mode == 2:
            return self._TTs
        if not self._consistent_intrinsic or not self._consistent_extrinsic:
            tb = get_tensor_basis(self.params.extrinsic)
            get_wavefront_time(self._lc, tb, self._TTs, self._spacecraft_orbits, self._wavefront_time)
            self._consistent_extrinsic = False
        return self._wavefront_time

    @property
    def response_mode(self) -> int:
        """Get the current response mode."""
        return self._response_mode

    @response_mode.setter
    def response_mode(self, mode: int) -> None:
        if mode not in (0, 1, 2):
            msg = 'Response mode must be 0 (doppler + antenna), 1 (rotation no doppler), or 2 (intrinsic only).'
            raise NotImplementedError(msg)
        if self._response_mode != mode:
            if self._response_mode == -1:  # not initialized yet
                self._response_mode = mode
                self._consistent_extrinsic = False
                self._consistent_intrinsic = False
                return

            spacecraft_orbits_loc: SpacecraftOrbits = get_spacecraft_vec(self._TTs, self._lc)
            if mode == 0:
                pass
            elif mode == 1:
                # move guiding center of constellation to solar system barycenter
                spacecraft_orbits_loc.xs[:] -= spacecraft_orbits_loc.xas
                spacecraft_orbits_loc.ys[:] -= spacecraft_orbits_loc.yas
                spacecraft_orbits_loc.zs[:] -= spacecraft_orbits_loc.zas
                spacecraft_orbits_loc.xas[:] = 0.
                spacecraft_orbits_loc.yas[:] = 0.
                spacecraft_orbits_loc.zas[:] = 0.
            elif mode == 2:
                # no spacecraft motion
                spacecraft_orbits_loc.xs[:] = 0.
                spacecraft_orbits_loc.ys[:] = 0.
                spacecraft_orbits_loc.zs[:] = 0.
                spacecraft_orbits_loc.xas[:] = 0.
                spacecraft_orbits_loc.yas[:] = 0.
                spacecraft_orbits_loc.zas[:] = 0.
            else:
                msg = 'Response mode must be 0 (doppler + antenna), 1 (rotation no doppler), or 2 (intrinsic only).'
                raise NotImplementedError(msg)

            self._spacecraft_orbits = spacecraft_orbits_loc

            self._response_mode = mode
            self._consistent_extrinsic = False
            self._consistent_intrinsic = False
            self.update_params(self.params)
