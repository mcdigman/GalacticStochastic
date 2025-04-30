"""subroutines for running lisa binary monte carlo search"""
import numpy as np

from coefficientsWDM_time_helpers import get_evTs
from taylor_wdm_funcs import wavemaket_multi_inplace

from ra_waveform_time import BinaryTimeWaveformAmpFreqD


class BinaryWaveletAmpFreqDT():
    """class to store a sparse binary wavelet and update for search"""
    def __init__(self, params, wc, lc, NT_min=0, NT_max=-1, NMT_use=-1, n_pad_T=0, freeze_limits=False):
        """construct a binary wavelet object, if NMF_use is not -1 it overrides the default"""
        self.wc = wc
        self.lc = lc
        if NT_max == -1:
            self.NT_max = self.wc.Nt
        else:
            self.NT_max = NT_max
        self.NT_min = NT_min
        self.freeze_limits = freeze_limits

        self.params = params

        # interpolation for wavelet taylor expansion
        self.NfsamT, self.evcTs, self.evsTs = get_evTs(self.wc, check_cache=False, hf_out=False)

        # maximum number of wavelet coefficients needed by taylor expansion
        self.fds = self.wc.dfd*np.arange(-self.wc.Nfd_negative, self.wc.Nfd-self.wc.Nfd_negative)
        self.NMT_max = np.int64(np.ceil((self.wc.BW+self.fds[self.wc.Nfd-1]*self.wc.Tw)/self.wc.DF))*self.wc.Nt

        #number of frequencies to tack on for padding, e.g. for interpolating and splines
        self.n_pad_T = n_pad_T

        #get the waveform in frequency space
        self.fwt = BinaryTimeWaveformAmpFreqD(self.params, self.NT_min, self.NT_max, self.lc, self.wc, freeze_limits, self.n_pad_T)

        if NMT_use==-1:
            self.NMT_use = self.NMT_max
        else:
            self.NMT_use = min(self.NMT_max, NMT_use)

        self.Tlists = np.full((self.wc.NC, self.NMT_use), -1, dtype=np.int64)
        self.waveT = np.zeros((self.wc.NC, self.NMT_use))
        self.NUTs = np.zeros(self.wc.NC, dtype=np.int64)
        self.consistent = False

        #initialize to input parameters
        self.update_params(params)

    def update_params(self, params_in):
        """update the internal wavelet representation to match the input parameters"""
        self.consistent = False
        self.params = params_in

        self.fwt.update_params(params_in)

        #TODO should trap exception if NMF_use is too small
        if self.NMT_use>0:
            AET_AmpTs = self.fwt.AET_waveform.AT
            AET_PPTs = self.fwt.AET_waveform.PT
            AET_FTs = self.fwt.AET_waveform.FT
            AET_FTds = self.fwt.AET_waveform.FTd
            NUTs_new = wavemaket_multi_inplace(self.waveT, self.Tlists, AET_PPTs[:, self.n_pad_T:self.n_pad_T+self.wc.Nt], AET_FTs[:, self.n_pad_T:self.n_pad_T+self.wc.Nt], AET_FTds[:, self.n_pad_T:self.n_pad_T+self.wc.Nt], AET_AmpTs[:, self.n_pad_T:self.n_pad_T+self.wc.Nt], self.wc.NC, self.wc.Nt, self.wc, self.evcTs, self.evsTs, self.NfsamT, force_nulls=False)
            for itrc in range(0, self.wc.NC):
                if NUTs_new[itrc]<self.NUTs[itrc]:
                    self.waveT[itrc, NUTs_new[itrc]:self.NUTs[itrc]] = 0.
                    self.Tlists[itrc, NUTs_new[itrc]:self.NUTs[itrc]] = -1
            self.NUTs = NUTs_new
        self.consistent = True

    def get_unsorted_coeffs(self):
        """get coefficients in the order they are generated"""
        if not self.consistent:
            self.update_params(self.params)
        return self.Tlists, self.waveT, self.NUTs
