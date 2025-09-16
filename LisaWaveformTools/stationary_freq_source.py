"""functions to compute rigid adiabatic response in frequency domain"""

from typing import override

import numpy as np
from numba import njit

from LisaWaveformTools.lisa_config import LISAConstants
from LisaWaveformTools.ra_waveform_freq import AntennaResponseChannels, rigid_adiabatic_antenna
from LisaWaveformTools.ra_waveform_time import spacecraft_channel_deriv_helper
from LisaWaveformTools.stationary_source_waveform import SourceParams, StationarySourceWaveform, StationaryWaveformFreq
from WaveletWaveforms.wdm_config import WDMWaveletConstants


@njit(fastmath=True)
def Extrinsic_inplace(AET_waveform: StationaryWaveformFreq, waveform: StationaryWaveformFreq, spacecraft_channels: AntennaResponseChannels, lc: LISAConstants, DF: float, NF, nf_low, F_min, kdotx, Tend=np.inf):
    """Helper for getting LISA response in frequency domain"""
    # TODO figure out how to set Tend properly
    # TODO may be good to set 2*pi multiple reproducibly
    AET_Amps = AET_waveform.AF
    AET_Phases = AET_waveform.PF
    AET_TFs = AET_waveform.TF
    AET_TFps = AET_waveform.TFp

    nc_loc = AET_Amps.shape[0]

    AA = waveform.AF
    PF = waveform.PF
    TF = waveform.TF
    TFp = waveform.TFp

    Phase_accums = np.zeros(nc_loc)

    # for the derivative of RR and II absorb 1/(2*DF) into the constant in AET_TFs
    spacecraft_channel_deriv_helper(spacecraft_channels, 1.0)

    # Merger kdotx
    polds = np.zeros(nc_loc)

    n = NF - 1

    for itrc in range(nc_loc):
        RR = spacecraft_channels.RR[itrc, n]
        II = spacecraft_channels.II[itrc, n]
        polds[itrc] = np.arctan2(II, RR)
        if polds[itrc] < 0.0:
            polds[itrc] += 2 * np.pi

    js = np.zeros(nc_loc)

    for n in range(NF - 1, -1, -1):

        # Barycenter time and frequency
        t = TF[n + nf_low]
        f = n * DF + F_min  # FF[n]
        # kdotx = t-xis[n+nf_low]

        x = 1.0
#        if Tstart<t<Tstart+lc.t_rise:
#            x = 0.5*(1.-np.cos(np.pi*(t-Tstart))/lc.t_rise)
#        if (Tend-lc.t_rise)<t<Tend:
#            x = 0.5*(1.0-np.cos(np.pi*(t-Tend)/lc.t_rise))
        if t > Tend:
            x = 0.0
#        if t<Tstart:
#            x = 0.0

        fonfs = f / lc.fstr  # Derivation says this is needed in the phase. Doesn't seem to be.
        Amp = 8 * x * AA[n + nf_low] * (fonfs * np.sin(fonfs))
        # TODO check and make consistent across versions
        Phase = -2 * np.pi * f * kdotx[n] - PF[n + nf_low]  # TODO check pi/4

        for itrc in range(nc_loc):
            RR = spacecraft_channels.RR[itrc, n]
            II = spacecraft_channels.II[itrc, n]
            dRR = spacecraft_channels.dRR[itrc, n]
            dII = spacecraft_channels.dII[itrc, n]

            # TODO find better hack for being exactly 0 to avoid dividing by 0
            if RR == 0. and II == 0.:
                AET_TFs[itrc, n] = t
                p = 0.
            else:

                AET_TFs[itrc, n] = t + (II * dRR - RR * dII) / (RR**2 + II**2) * 1 / (4 * np.pi * DF)
                # get phase
                p = np.arctan2(II, RR)
                # shift phase to 0 to 2pi range
                if p < 0.0:
                    p += 2 * np.pi

            if n == NF - 1:
                Phase_accums[itrc] = -p - js[itrc]
            else:
                Phase_accums[itrc] -= np.pi * DF * (AET_TFs[itrc, n] - t + AET_TFs[itrc, n + 1] - TF[n + 1 + nf_low])

            if (Phase_accums[itrc] + p + js[itrc]) > np.pi:
                js[itrc] -= 2 * np.pi
            if (-p - js[itrc] - Phase_accums[itrc]) > np.pi:
                js[itrc] += 2 * np.pi
            polds[itrc] = p

            # TODO check h22fac absence
            AET_Amps[itrc, n] = Amp * np.sqrt(RR**2 + II**2)
            AET_Phases[itrc, n] = -Phase - p - js[itrc]

    # TODO check this phasing relative to taylorT3

    # stabilized_gradient_uniform_inplace(waveform.TF, waveform.TFp, tdi_waveform.TF, tdi_waveform.TFp, DF)

    # compute AET_TFps as perturbation on TFps
    # compute the gradient dy/dx using a second order accurate central finite difference assuming constant x grid along second axis, forward/backward first order accurate at boundaries, and apply a TFps base
    for itrc in range(nc_loc):
        AET_TFps[itrc, 0] = (AET_TFs[itrc, 1] - AET_TFs[itrc, 0] - TF[nf_low + 1] + TF[nf_low]) / DF + TFp[nf_low]
        AET_TFps[itrc, NF - 1] = (AET_TFs[itrc, NF - 1] - AET_TFs[itrc, NF - 2] - TF[nf_low + NF - 1] + TF[nf_low + NF - 2]) / DF + TFp[nf_low + NF - 1]

    for n in range(1, NF - 1):
        TF_shift = -TF[nf_low + n + 1] + TF[nf_low + n - 1]
        TFp_shift = TFp[nf_low + n]
        for itrc in range(nc_loc):
            AET_TFps[itrc, n] = (AET_TFs[itrc, n + 1] - AET_TFs[itrc, n - 1] + TF_shift) / (2 * DF) + TFp_shift
    return


class StationarySourceWaveformFreq(StationarySourceWaveform[StationaryWaveformFreq]):
    """class to store a binary waveform in frequency domain and update for search"""
    def __init__(self, params: SourceParams, lc: LISAConstants, wc: WDMWaveletConstants, NF_min, NF_max, freeze_limits, n_pad_F=10):
        """Construct a binary wavelet object"""
        self._lc: LISAConstants = lc
        self._wc: WDMWaveletConstants = wc
        self._nc_waveform: int = self._lc.nc_waveform
        self._consistent_extrinsic: bool = False
        self._n_pad_F = n_pad_F

        self.NF_min = NF_min
        self.NF_max = NF_max
        self.nf_offset = NF_min
        self.freeze_limits = False
        self.freeze_extrinsic = False

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
        self.nf_range = self.nf_high - self.nf_low

        # enforce cleanup of values that will not be reset
        if nf_range_old > self.nf_range:
            self._tdi_waveform.AF[:, self.nf_range:nf_range_old] = 0.
            self._tdi_waveform.PF[:, self.nf_range:nf_range_old] = 0.
            self._tdi_waveform.TF[:, self.nf_range:nf_range_old] = 0.
            self._tdi_waveform.TFp[:, self.nf_range:nf_range_old] = 0.
            self._spacecraft_channels.RR[:, self.nf_range:nf_range_old] = 0.
            self._spacecraft_channels.II[:, self.nf_range:nf_range_old] = 0.
            self._spacecraft_channels.dRR[:, self.nf_range:nf_range_old] = 0.
            self._spacecraft_channels.dII[:, self.nf_range:nf_range_old] = 0.

    @override
    def _update_extrinsic(self):
        """
        Update waveform to match the extrinsic parameters of spacecraft response
        if abbreviated, don't get AET_TFs or AET_TFps, and don't track modulus of AET_PPFs
        """
        F_min = self._wc.DF * self.nf_offset

        rigid_adiabatic_antenna(self._spacecraft_channels, self.params.extrinsic, self.intrinsic_waveform.TF, self.FFs, self.nf_low, self.nf_range, self.kdotx, self._lc)
        Extrinsic_inplace(self._tdi_waveform, self.intrinsic_waveform, self._spacecraft_channels, self._lc, self._wc.DF, self.nf_range, self.nf_low, F_min, self.kdotx, Tend=self.Tend)
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
        if not self.freeze_extrinsic:
            self._update_extrinsic()
            self._consistent = True

    @property
    @override
    def nc_waveform(self) -> int:
        """Return the number of channels in the waveform."""
        return self._nc_waveform
