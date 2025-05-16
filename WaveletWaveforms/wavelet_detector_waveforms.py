"""subroutines for running lisa binary monte carlo search"""

from LisaWaveformTools.ra_waveform_time import BinaryTimeWaveformAmpFreqD
from WaveletWaveforms.coefficientsWDM_time_helpers import get_empty_sparse_taylor_time_waveform, get_evTs
from WaveletWaveforms.taylor_wdm_funcs import wavemaket_multi_inplace


class BinaryWaveletAmpFreqDT:
    """class to store a sparse binary wavelet and update for search"""

    def __init__(self, params, wc, lc, nt_min=0, nt_max=-1):
        """Construct a binary wavelet object, if NMF_use is not -1 it overrides the default"""
        self.wc = wc
        self.lc = lc
        if nt_max == -1:
            self.nt_max = self.wc.Nt
        else:
            self.nt_max = nt_max
        self.nt_min = nt_min

        self.params = params

        # interpolation for wavelet taylor expansion
        self.taylor_time_table = get_evTs(self.wc, check_cache=False, hf_out=False)

        # get the waveform in frequency space
        self.fwt = BinaryTimeWaveformAmpFreqD(self.params, self.nt_min, self.nt_max, self.lc, self.wc)

        # get a blank waveform in the sparse wavelet domain
        # when consistent is set it will be correct
        self.wavelet_waveform = get_empty_sparse_taylor_time_waveform(self.wc.NC, wc)

        self.consistent = False

        # initialize to input parameters
        self.update_params(params)

    def update_params(self, params_in):
        """Update the internal wavelet representation to match the input parameters"""
        self.consistent = False
        self.params = params_in

        self.fwt.update_params(params_in)

        self.wavelet_waveform = wavemaket_multi_inplace(self.wavelet_waveform, self.fwt.AET_waveform, self.nt_min, self.nt_max, self.wc, self.taylor_time_table, force_nulls=False)
        self.consistent = True

    def get_unsorted_coeffs(self):
        """Get coefficients in the order they are generated"""
        if not self.consistent:
            self.update_params(self.params)
        return self.wavelet_waveform
