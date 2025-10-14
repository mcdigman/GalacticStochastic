"""Stationary source waveform in frequency domain."""

from abc import ABC
from typing import TYPE_CHECKING, Generic, override

import h5py
import numpy as np

from LisaWaveformTools.lisa_config import LISAConstants
from LisaWaveformTools.ra_waveform_freq import get_freq_tdi_amp_phase, rigid_adiabatic_antenna
from LisaWaveformTools.source_params import ExtrinsicParamsType, IntrinsicParamsType, SourceParams
from LisaWaveformTools.spacecraft_objects import AntennaResponseChannels, EdgeRiseModel
from LisaWaveformTools.stationary_source_waveform import StationarySourceWaveform, StationaryWaveformFreq
from WaveletWaveforms.sparse_waveform_functions import PixelGenericRange

if TYPE_CHECKING:
    from numpy.typing import NDArray


class StationarySourceWaveformFreq(Generic[IntrinsicParamsType, ExtrinsicParamsType], StationarySourceWaveform[StationaryWaveformFreq, IntrinsicParamsType, ExtrinsicParamsType], ABC):
    """Class to store a binary waveform in frequency domain and update for search."""

    def __init__(self, params: SourceParams, lc: LISAConstants, nf_lim_absolute: PixelGenericRange, freeze_limits: int, t_obs: float, n_pad_F: int = 10, response_mode: int = 0) -> None:
        """Construct a binary wavelet object."""
        self._lc: LISAConstants = lc
        self._nc_waveform: int = self._lc.nc_waveform
        self._consistent_extrinsic: bool = False
        self._n_pad_F: int = n_pad_F

        if lc.rise_mode == 3:
            self._er: EdgeRiseModel = EdgeRiseModel(-np.inf, np.inf)
        else:
            msg = 'Only rise_mode 3 (no edge) is implemented.'
            raise NotImplementedError(msg)

        self.nf_lim_absolute: PixelGenericRange = nf_lim_absolute
        self.freeze_limits: int = 0

        # TODO FFs can possibly be eliminated
        self.FFs: NDArray[np.floating] = self.nf_lim_absolute.dx * np.arange(self.nf_lim_absolute.nx_min, self.nf_lim_absolute.nx_max)  # center of the pixels
        self.NF: int = self.FFs.size

        if self.nf_lim_absolute.nx_min == 0:
            self.FFs[0] = 1.e-5 * self.nf_lim_absolute.dx

        F_min = float(self.nf_lim_absolute.dx * self.nf_lim_absolute.nx_min)
        self.nf_lim: PixelGenericRange = PixelGenericRange(0, self.NF, self.nf_lim_absolute.dx, F_min)
        self.itrFCut: int = self.nf_lim.nx_max - self.nf_lim.nx_min

        AmpFs = np.zeros(self.NF)
        PPFs = np.zeros(self.NF)
        TFs = np.zeros(self.NF)
        TFps = np.zeros(self.NF)
        intrinsic_waveform = StationaryWaveformFreq(self.FFs, AmpFs, PPFs, TFs, TFps)

        del AmpFs
        del PPFs
        del TFs
        del TFps

        RRs = np.zeros((self.nc_waveform, self.NF))
        IIs = np.zeros((self.nc_waveform, self.NF))
        dRRs = np.zeros((self.nc_waveform, self.NF))
        dIIs = np.zeros((self.nc_waveform, self.NF))

        self._spacecraft_channels: AntennaResponseChannels = AntennaResponseChannels(self.FFs, RRs, IIs, dRRs, dIIs)

        del RRs
        del IIs
        del dRRs
        del dIIs

        self.kdotx: NDArray[np.floating] = np.zeros(self.NF)

        AET_AmpFs = np.zeros((self.nc_waveform, self.NF))
        AET_PPFs = np.zeros((self.nc_waveform, self.NF))
        AET_TFs = np.zeros((self.nc_waveform, self.NF))
        AET_TFps = np.zeros((self.nc_waveform, self.NF))

        tdi_waveform = StationaryWaveformFreq(self.FFs, AET_AmpFs, AET_PPFs, AET_TFs, AET_TFps)
        self._t_gen: float = self._lc.t0 + t_obs + self._lc.t_rise
        del AET_AmpFs
        del AET_PPFs
        del AET_TFs
        del AET_TFps

        self.TTRef: float = 0.

        self._tdi_waveform: StationaryWaveformFreq = tdi_waveform
        self._intrinsic_waveform: StationaryWaveformFreq = intrinsic_waveform

        super().__init__(params, intrinsic_waveform, tdi_waveform, response_mode=response_mode)

        self.freeze_limits = freeze_limits

    @override
    def store_hdf5(self, hf_in: h5py.Group, *, group_name: str = 'source_waveform', group_mode: int = 0) -> h5py.Group:
        hf_source = super().store_hdf5(hf_in, group_name=group_name, group_mode=group_mode)

        hf_source.attrs['channels_name'] = self._spacecraft_channels.__class__.__name__
        hf_source.attrs['response_mode'] = self._response_mode
        hf_source.attrs['nc_waveform'] = self._nc_waveform
        hf_source.attrs['itrfCut'] = self.itrFCut
        hf_source.attrs['TTRef'] = self.TTRef
        hf_source.attrs['freeze_limits'] = self.freeze_limits
        hf_source.attrs['n_pad_F'] = self._n_pad_F
        hf_source.attrs['NF'] = self.NF
        hf_source.attrs['t_gen'] = self._t_gen

        _ = hf_source.create_dataset('T', data=self.FFs, compression='gzip')

        _ = hf_source.create_dataset('spacecraft_channel_t', data=self._spacecraft_channels.x, compression='gzip')
        _ = hf_source.create_dataset('RR', data=self._spacecraft_channels.RR, compression='gzip')
        _ = hf_source.create_dataset('II', data=self._spacecraft_channels.II, compression='gzip')
        _ = hf_source.create_dataset('dRR', data=self._spacecraft_channels.dRR, compression='gzip')
        _ = hf_source.create_dataset('dII', data=self._spacecraft_channels.dII, compression='gzip')

        _ = hf_source.create_dataset('kdotx', data=self.kdotx, compression='gzip')

        _ = hf_source.create_dataset('tdi_F', data=self._tdi_waveform.F, compression='gzip')
        _ = hf_source.create_dataset('tdi_AF', data=self._tdi_waveform.AF, compression='gzip')
        _ = hf_source.create_dataset('tdi_TF', data=self._tdi_waveform.TF, compression='gzip')
        _ = hf_source.create_dataset('tdi_TFp', data=self._tdi_waveform.TFp, compression='gzip')

        _ = hf_source.create_dataset('intrinsic_F', data=self._intrinsic_waveform.F, compression='gzip')
        _ = hf_source.create_dataset('intrinsic_AF', data=self._intrinsic_waveform.AF, compression='gzip')
        _ = hf_source.create_dataset('intrinsic_TF', data=self._intrinsic_waveform.TF, compression='gzip')
        _ = hf_source.create_dataset('intrinsic_TFp', data=self._intrinsic_waveform.TFp, compression='gzip')

        hf_lc = hf_source.create_group('lc')
        hf_source.attrs['lc_name'] = self._lc.__class__.__name__
        for key in self._lc._fields:
            hf_lc.attrs[key] = getattr(self._lc, key)

        hf_er = hf_source.create_group('er')
        hf_source.attrs['er_name'] = self._er.__class__.__name__
        for key in self._er._fields:
            hf_er.attrs[key] = getattr(self._er, key)

        hf_source.attrs['nf_lim_name'] = self.nf_lim.__class__.__name__
        hf_nf = hf_source.create_group('nf_lim_waveform')
        for key in self.nf_lim._fields:
            hf_nf.attrs[key] = getattr(self.nf_lim, key)

        hf_source.attrs['nf_lim_absolute_name'] = self.nf_lim_absolute.__class__.__name__
        hf_nfa = hf_source.create_group('nf_lim_absolute')
        for key in self.nf_lim_absolute._fields:
            hf_nfa.attrs[key] = getattr(self.nf_lim_absolute, key)

        return hf_source

    def _update_bounds(self) -> None:
        """Update the boundaries to calculate extrinsic parameters at."""
        # TODO something is malfunctioning here, trap the case where itrFCutOld would segfault?
        # TODO should handle nonmonotonic time from searchsorted
        # TODO need to handle bounds edge correctly
        nf_lim_old = self.nf_lim
        itrFCutOld = min(self.itrFCut + nf_lim_old.nx_min, self.NF)

        nf_low = int(np.searchsorted(self.intrinsic_waveform.TF[:itrFCutOld], self._lc.t0 - self._lc.t_rise, side='right') - self._n_pad_F)
        nf_low = max(0, nf_low)
        # TODO need to recalculate subtraction if Nf old breaks
        # TODO is this the right trap??? Why is the subtraction even needed?
        if self.itrFCut + nf_low < self.NF:
            self.itrFCut = min(max(0, self.itrFCut + nf_lim_old.nx_min - nf_low), self.NF)
        else:
            self.itrFCut = self.NF  # itrFCut

        nf_high = self.NF
        # clip to avoid wasting time computing values that are outside the observation
        if self.intrinsic_waveform.TF[itrFCutOld - 1] > self._t_gen:
            # TODO check this searchsorted correctly handles TFs dropping to 0 at end
            nf_high = min(nf_high, int(np.searchsorted(self.intrinsic_waveform.TF[:itrFCutOld], self._t_gen, side='right')) + self._n_pad_F)

        # make sure order is correct
        nf_high = min(nf_high, self.itrFCut)
        nf_high = max(nf_low, nf_high)
        nf_high = min(nf_high, self.NF)

        # enforce cleanup of values that will not be reset
        if nf_lim_old.nx_max > nf_high:
            # TODO this manipulation of kdotx should only happen if it is a private variable
            self.kdotx[nf_high:nf_lim_old.nx_max] = 0.
            self._tdi_waveform.AF[:, nf_high:nf_lim_old.nx_max] = 0.
            self._tdi_waveform.PF[:, nf_high:nf_lim_old.nx_max] = 0.
            self._tdi_waveform.TF[:, nf_high:nf_lim_old.nx_max] = 0.
            self._tdi_waveform.TFp[:, nf_high:nf_lim_old.nx_max] = 0.
            self._spacecraft_channels.RR[:, nf_high:nf_lim_old.nx_max] = 0.
            self._spacecraft_channels.II[:, nf_high:nf_lim_old.nx_max] = 0.
            self._spacecraft_channels.dRR[:, nf_high:nf_lim_old.nx_max] = 0.
            self._spacecraft_channels.dII[:, nf_high:nf_lim_old.nx_max] = 0.

        if nf_lim_old.nx_min < nf_low:
            # TODO this manipulation of kdotx should only happen if it is a private variable
            self.kdotx[nf_lim_old.nx_min:nf_low] = 0.
            self._tdi_waveform.AF[:, nf_lim_old.nx_min:nf_low] = 0.
            self._tdi_waveform.PF[:, nf_lim_old.nx_min:nf_low] = 0.
            self._tdi_waveform.TF[:, nf_lim_old.nx_min:nf_low] = 0.
            self._tdi_waveform.TFp[:, nf_lim_old.nx_min:nf_low] = 0.
            self._spacecraft_channels.RR[:, nf_lim_old.nx_min:nf_low] = 0.
            self._spacecraft_channels.II[:, nf_lim_old.nx_min:nf_low] = 0.
            self._spacecraft_channels.dRR[:, nf_lim_old.nx_min:nf_low] = 0.
            self._spacecraft_channels.dII[:, nf_lim_old.nx_min:nf_low] = 0.

        self.nf_lim = PixelGenericRange(nf_low, nf_high, nf_lim_old.dx, nf_lim_old.x_min)

    @override
    def _update_extrinsic(self) -> None:
        """
        Update waveform to match the extrinsic parameters of spacecraft response.

        If abbreviated, don't get AET_TFs or AET_TFps, and don't track modulus of AET_PPFs
        """
        rigid_adiabatic_antenna(self._spacecraft_channels, self.params.extrinsic, self.intrinsic_waveform.TF, self.FFs, self.nf_lim, self.kdotx, self._lc)

        get_freq_tdi_amp_phase(self._tdi_waveform, self.intrinsic_waveform, self._spacecraft_channels, self._lc, self.nf_lim, self.kdotx, self._er)
        self._consistent_extrinsic = True

    @override
    def update_params(self, params: SourceParams) -> None:
        """
        Recompute the waveform with updated parameters.

        Ff abbreviated skip getting AET_TFs and AET_TFps
        """
        self.params: SourceParams = params
        self._update_intrinsic()
        if self.freeze_limits == 0:
            self._update_bounds()
        self._update_extrinsic()
        self._consistent: bool = True

    @property
    @override
    def nc_waveform(self) -> int:
        """Return the number of channels in the waveform.

        Returns
        -------
        int
            Number of channels in the waveform.
        """
        return self._nc_waveform
