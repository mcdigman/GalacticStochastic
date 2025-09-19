"""functions to compute rigid adiabatic response in frequency domain"""

from typing import override

import numpy as np

from LisaWaveformTools.lisa_config import LISAConstants
from LisaWaveformTools.ra_waveform_freq import get_freq_tdi_amp_phase, rigid_adiabatic_antenna
from LisaWaveformTools.spacecraft_objects import AntennaResponseChannels, EdgeRiseModel
from LisaWaveformTools.stationary_source_waveform import SourceParams, StationarySourceWaveform, StationaryWaveformFreq
from WaveletWaveforms.sparse_waveform_functions import PixelFreqRange
from WaveletWaveforms.wdm_config import WDMWaveletConstants


class StationarySourceWaveformFreq(StationarySourceWaveform[StationaryWaveformFreq]):
    """class to store a binary waveform in frequency domain and update for search"""
    def __init__(self, params: SourceParams, lc: LISAConstants, wc: WDMWaveletConstants, NF_min, NF_max, freeze_limits, n_pad_F=10):
        """Construct a binary wavelet object"""
        self._lc: LISAConstants = lc
        # TODO eliminate wc as argument
        self._wc: WDMWaveletConstants = wc
        self._nc_waveform: int = self._lc.nc_waveform
        self._consistent_extrinsic: bool = False
        self._n_pad_F = n_pad_F

        if lc.rise_mode == 3:
            self._er = EdgeRiseModel(-np.inf, np.inf)
        else:
            msg = 'Only rise_mode 3 (no edge) is implemented.'
            raise NotImplementedError(msg)

        self.NF_min = NF_min
        self.NF_max = NF_max
        self.nf_offset = NF_min
        self.freeze_limits = False

        # TODO FFs can possibly be eliminated
        self.FFs = wc.DF * np.arange(self.NF_min, self.NF_max)  # center of the pixels
        self.NF = self.FFs.size

        if self.NF_min == 0:
            self.FFs[0] = 1.e-5 * wc.DF

        self.nf_low = 0
        self.nf_high = self.NF
        self.nf_range = self.nf_high - self.nf_low
        self.itrFCut = self.nf_range

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

        self._spacecraft_channels = AntennaResponseChannels(self.FFs, RRs, IIs, dRRs, dIIs)

        del RRs
        del IIs
        del dRRs
        del dIIs

        self.kdotx = np.zeros(self.NF)

        AET_AmpFs = np.zeros((self.nc_waveform, self.NF))
        AET_PPFs = np.zeros((self.nc_waveform, self.NF))
        AET_TFs = np.zeros((self.nc_waveform, self.NF))
        AET_TFps = np.zeros((self.nc_waveform, self.NF))

        tdi_waveform = StationaryWaveformFreq(self.FFs, AET_AmpFs, AET_PPFs, AET_TFs, AET_TFps)

        del AET_AmpFs
        del AET_PPFs
        del AET_TFs
        del AET_TFps

        self.TTRef = 0.
        self.delta_tm = 0.
        # moved this line from end
        self.Tend = np.inf

        self._tdi_waveform: StationaryWaveformFreq = tdi_waveform
        self._intrinsic_waveform: StationaryWaveformFreq = intrinsic_waveform

        super().__init__(params, intrinsic_waveform, tdi_waveform)

        self.freeze_limits = freeze_limits

    def _update_bounds(self):
        """Update the boundaries to calculate extrinsic parameters at"""
        # TODO something is malfunctioning here, trap the case where itrFCutOld would segfault?
        # TODO should handle nonmonotonic time from searchsorted
        # TODO need to handle bounds edge correctly
        itrFCutOld = min(self.itrFCut + self.nf_low, self.NF)
        # print('4',self.nf_low,self.nf_high,self.itrFCut,itrFCutOld)
        nf_low_old = self.nf_low
        self.nf_low = max(0, np.searchsorted(self.intrinsic_waveform.TF[:itrFCutOld], self._lc.t0 - self._lc.t_rise, side='right') - self._n_pad_F)
        # TODO need to recalculate subtraction if Nf old breaks
        # print(itrFCutOld,nf_low_old,self.nf_low,self.nf_high,self.nf_range,self.itrFCut+nf_low_old-self.nf_low)
        # TODO is this the right trap??? Why is the subtraction even needed?
        if self.itrFCut + self.nf_low < self.NF:
            self.itrFCut = min(max(0, self.itrFCut + nf_low_old - self.nf_low), self.NF)
        else:
            self.itrFCut = self.NF  # itrFCut

        self.nf_offset = self.nf_low + self.NF_min
        self.nf_high = self.NF
        # clip to avoid wasting time computing values that are outside the observation
        # print('5',self.nf_low,self.nf_high,self.itrFCut,itrFCutOld)
        if self.intrinsic_waveform.TF[itrFCutOld - 1] > self._lc.t0 + self._wc.Tobs + self._lc.t_rise:
            # TODO check this searchsorted correctly handles TFs dropping to 0 at end
            self.nf_high = min(self.nf_high, np.searchsorted(self.intrinsic_waveform.TF[:itrFCutOld], self._lc.t0 + self._wc.Tobs + self._lc.t_rise, side='right') + self._n_pad_F)

        # make sure order is correct
        self.nf_high = min(self.nf_high, self.itrFCut)
        self.nf_high = max(self.nf_low, self.nf_high)
        self.nf_high = min(self.nf_high, self.NF)
        nf_range_old = self.nf_range
        nf_high_old = nf_range_old + nf_low_old
        self.nf_range = self.nf_high - self.nf_low

        # enforce cleanup of values that will not be reset
        if nf_high_old > self.nf_high:
            # TODO this manipulation of kdotx should only happen if it is a private variable
            self.kdotx[self.nf_high:nf_high_old] = 0.
            self._tdi_waveform.AF[:, self.nf_high:nf_high_old] = 0.
            self._tdi_waveform.PF[:, self.nf_high:nf_high_old] = 0.
            self._tdi_waveform.TF[:, self.nf_high:nf_high_old] = 0.
            self._tdi_waveform.TFp[:, self.nf_high:nf_high_old] = 0.
            self._spacecraft_channels.RR[:, self.nf_high:nf_high_old] = 0.
            self._spacecraft_channels.II[:, self.nf_high:nf_high_old] = 0.
            self._spacecraft_channels.dRR[:, self.nf_high:nf_high_old] = 0.
            self._spacecraft_channels.dII[:, self.nf_high:nf_high_old] = 0.

        if nf_low_old < self.nf_low:
            # TODO this manipulation of kdotx should only happen if it is a private variable
            self.kdotx[nf_low_old:self.nf_low] = 0.
            self._tdi_waveform.AF[:, nf_low_old:self.nf_low] = 0.
            self._tdi_waveform.PF[:, nf_low_old:self.nf_low] = 0.
            self._tdi_waveform.TF[:, nf_low_old:self.nf_low] = 0.
            self._tdi_waveform.TFp[:, nf_low_old:self.nf_low] = 0.
            self._spacecraft_channels.RR[:, nf_low_old:self.nf_low] = 0.
            self._spacecraft_channels.II[:, nf_low_old:self.nf_low] = 0.
            self._spacecraft_channels.dRR[:, nf_low_old:self.nf_low] = 0.
            self._spacecraft_channels.dII[:, nf_low_old:self.nf_low] = 0.

    @override
    def _update_extrinsic(self):
        """
        Update waveform to match the extrinsic parameters of spacecraft response
        if abbreviated, don't get AET_TFs or AET_TFps, and don't track modulus of AET_PPFs
        """
        F_min = self._wc.DF * (self.nf_offset - self.nf_low)

        rigid_adiabatic_antenna(self._spacecraft_channels, self.params.extrinsic, self.intrinsic_waveform.TF, self.FFs, self.nf_low, self.nf_range, self.kdotx, self._lc)
        nf_lim = PixelFreqRange(self.nf_low, self.nf_low + self.nf_range, self._wc.DF)
        get_freq_tdi_amp_phase(self._tdi_waveform, self.intrinsic_waveform, self._spacecraft_channels, self._lc, nf_lim, F_min, self.kdotx, self._er)
        self._consistent_extrinsic = True

    @override
    def update_params(self, params: SourceParams):
        """Recompute the waveform with updated parameters,
            if abbreviated skip getting AET_TFs and AET_TFps
        """
        self.params = params
        self._update_intrinsic()
        if not self.freeze_limits:
            self._update_bounds()
        self._update_extrinsic()
        self._consistent = True

    @property
    @override
    def nc_waveform(self) -> int:
        """Return the number of channels in the waveform."""
        return self._nc_waveform
