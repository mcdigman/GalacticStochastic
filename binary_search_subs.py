"""subroutines for running lisa binary monte carlo search"""
import numpy as np


from taylor_wdm_funcs import wavemaket_multi_inplace
from ra_waveform_time import BinaryTimeWaveformAmpFreqD

from wdm_const import wdm_const as wc
import mcmc_params as mcp

class BinaryWaveletAmpFreqDT():
    """class to store a sparse binary wavelet and update for search"""
    def __init__(self,params,NT_min=0,NT_max=wc.Nt,NMT_use=-1,freeze_limits=False):
        """construct a binary wavelet object, if NMF_use is not -1 it overrides the default"""
        self.NT_min = NT_min
        self.NT_max = NT_max
        self.freeze_limits = freeze_limits

        self.params = params

        #get the waveform in frequency space
        #self.fwt = BinaryTimeWaveformTaylorT3T(self.params,self.NT_min,self.NT_max,freeze_limits)
        self.fwt = BinaryTimeWaveformAmpFreqD(self.params,self.NT_min,self.NT_max,freeze_limits)

        if NMT_use==-1:
            self.NMT_use = mcp.NMT_max
        else:
            self.NMT_use = min(mcp.NMT_max,NMT_use)

        self.Tlists = np.full((wc.NC,self.NMT_use),-1,dtype=np.int64)
        self.waveT = np.zeros((wc.NC,self.NMT_use))
        self.NUTs = np.zeros(wc.NC,dtype=np.int64)
        self.consistent = False

        #initialize to input parameters
        self.update_params(params)

    def update_params(self,params_in):
        """update the internal wavelet representation to match the input parameters"""
        self.consistent = False
        self.params = params_in

        self.fwt.update_params(params_in)

        #TODO should trap exception if NMF_use is too small
        if self.NMT_use>0:
            NUTs_new = wavemaket_multi_inplace(self.waveT,self.Tlists,self.fwt.AET_PPTs[:,mcp.n_pad_T:mcp.n_pad_T+wc.Nt],self.fwt.AET_FTs[:,mcp.n_pad_T:mcp.n_pad_T+wc.Nt],self.fwt.AET_FTds[:,mcp.n_pad_T:mcp.n_pad_T+wc.Nt],self.fwt.AET_AmpTs[:,mcp.n_pad_T:mcp.n_pad_T+wc.Nt],wc.NC,wc.Nt,force_nulls=False)
            for itrc in range(0,wc.NC):
                if NUTs_new[itrc]<self.NUTs[itrc]:
                    self.waveT[itrc,NUTs_new[itrc]:self.NUTs[itrc]] = 0.
                    self.Tlists[itrc,NUTs_new[itrc]:self.NUTs[itrc]] = -1
            self.NUTs = NUTs_new
        self.consistent = True

    def get_unsorted_coeffs(self):
        """get coefficients in the order they are generated"""
        if not self.consistent:
            self.update_params(self.params)
        return self.Tlists,self.waveT,self.NUTs
