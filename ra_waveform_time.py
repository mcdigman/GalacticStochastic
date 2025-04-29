"""subroutines for running lisa code"""
import numpy as np
from numba import njit

from ra_waveform_freq import RAantenna_inplace,spacecraft_vec,get_xis_inplace,get_tensor_basis

from algebra_tools import gradient_homog_2d_inplace
import mcmc_params as mcp

from wdm_const import wdm_const as wc
from wdm_const import lisa_const as lc
import global_const as gc

#TODO do consistency checks
class BinaryTimeWaveformAmpFreqD():
    """class to store a binary waveform in time domain and update for search
        assuming input binary format based on amplitude, frequency, and frequency derivative"""
    def __init__(self,params,NT_min,NT_max,freeze_limits=False):
        """initalize the object"""
        self.params = params
        self.NT_min = NT_min-mcp.n_pad_T
        self.NT_max = NT_max+mcp.n_pad_T
        self.NT = self.NT_max-self.NT_min
        self.freeze_limits = freeze_limits

        #TODO ensure this handles padding self consistently
        self.nt_low = self.NT_min#-mcp.n_pad_T
        self.nt_high = self.NT_max#self.NT+mcp.n_pad_T

        self.nt_range = self.nt_high-self.nt_low

        self.TTs = wc.DT*np.arange(self.nt_low,self.nt_high)

        self.AmpTs = np.zeros(self.NT)
        self.PPTs = np.zeros(self.NT)
        self.FTs = np.zeros(self.NT)
        self.FTds = np.zeros(self.NT)

        self.dRRs = np.zeros((wc.NC,self.NT))
        self.dIIs = np.zeros((wc.NC,self.NT))
        self.RRs = np.zeros((wc.NC,self.NT))
        self.IIs = np.zeros((wc.NC,self.NT))

        #self.xs = np.zeros((wc.NC,self.NT))
        #self.ys = np.zeros((wc.NC,self.NT))
        #self.zs = np.zeros((wc.NC,self.NT))
        self.xas = np.zeros(self.NT)
        self.yas = np.zeros(self.NT)
        self.zas = np.zeros(self.NT)
        self.xis = np.zeros(self.NT)
        self.kdotx = np.zeros(self.NT)

        _, _, _, self.xas[:], self.yas[:], self.zas[:] = spacecraft_vec(self.TTs)
        #spacecraft_vec_inplace(self.TTs,self.xs,self.ys,self.zs,0,self.nt_range)

        #self.kdotx = np.zeros(self.NT)

        self.AET_AmpTs = np.zeros((wc.NC,self.NT))
        self.AET_PPTs = np.zeros((wc.NC,self.NT))
        self.AET_FTs = np.zeros((wc.NC,self.NT))
        self.AET_FTds = np.zeros((wc.NC,self.NT))

        #self.TTRef = 0.

        #spacecraft_vec_inplace(self.TTs,self.xs,self.ys,self.zs,0,self.NT) #TODO fix nf_low and nf_range

        self.update_params(params)

    def update_params(self,params):
        self.params = params
        self.update_intrinsic()
        #TODO add update_bounds if appropriate
        self.update_extrinsic()

    def update_intrinsic(self):
        """get amplitude and phase for taylorT3"""
        amp = self.params[0]
        costh = np.cos(np.pi/2-self.params[1] ) #TODO check
        phi = self.params[2]
        freq0 = self.params[3]
        freqD = self.params[4]
        phi0 = self.params[6]+np.pi

        kv, _, _ = get_tensor_basis(phi,costh) #TODO check intrinsic extrinsic separation here
        get_xis_inplace(kv,self.TTs,self.xas,self.yas,self.zas,self.xis)

        AmpFreqDeriv_inplace(self.AmpTs,self.PPTs,self.FTs,self.FTds,amp,phi0,freq0,freqD,self.xis,self.TTs.size)

    def update_extrinsic(self):
        #Calculate cos and sin of sky position, inclination, polarization
        costh = np.cos(np.pi/2-self.params[1])
        phi = self.params[2]
        cosi = np.cos(self.params[5])
        psi = self.params[7]

        RAantenna_inplace(self.RRs,self.IIs,cosi,psi,phi,costh,self.TTs,self.FTs,0,self.NT,self.kdotx) #TODO fix F_min and nf_range
        ExtractAmpPhase_inplace(self.AET_AmpTs,self.AET_PPTs,self.AET_FTs,self.AET_FTds,self.AmpTs,self.PPTs,self.FTs,self.FTds,self.RRs,self.IIs,self.dRRs,self.dIIs,self.NT)

@njit(fastmath=True)
def ExtractAmpPhase_inplace(AET_Amps,AET_Phases,AET_FTs,AET_FTds,AA,PP,FT,FTd,RRs,IIs,dRRs,dIIs,NT):
    """get the amplitude and phase for LISA"""
    #TODO check absolute phase aligns with Extrinsic_inplace
    polds = np.zeros(wc.NC)
    js = np.zeros(wc.NC)

    gradient_homog_2d_inplace(RRs,wc.DT,NT,3,dRRs)
    gradient_homog_2d_inplace(IIs,wc.DT,NT,3,dIIs)

    n = 0
    for itrc in range(0,wc.NC):
        polds[itrc] = np.arctan2(IIs[itrc,n],RRs[itrc,n])
        if polds[itrc]<0.:
            polds[itrc] += 2*np.pi
    for n in range(0,NT):
        fonfs = FT[n]/lc.fstr

        # including TDI + fractional frequency modifiers
        Ampx = AA[n]*(8*fonfs*np.sin(fonfs))
        Phase = PP[n]
        for itrc in range(0,wc.NC):
            RR = RRs[itrc,n]
            II = IIs[itrc,n]

            if RR==0. and II==0.:
                #TODO is this the correct way to handle both ps being 0?
                p = 0.
                AET_FTs[itrc,n] = FT[n]
            else:
                p = np.arctan2(II,RR)
                AET_FTs[itrc,n] = FT[n]-(II*dRRs[itrc,n]-RR*dIIs[itrc,n])/(RR**2+II**2)/(2*np.pi)


            if p<0.:
                p += 2*np.pi

            #TODO implement integral tracking of js
            if p-polds[itrc] > 6.:
                js[itrc] -= 2*np.pi
            if polds[itrc]-p > 6.:
                js[itrc] += 2*np.pi
            polds[itrc] = p

            AET_Amps[itrc,n] = Ampx*np.sqrt(RR**2+II**2)
            AET_Phases[itrc,n] = Phase+p+js[itrc]#+2*np.pi*kdotx[n]*FT[n]

    for itrc in range(0,wc.NC):
        AET_FTds[itrc,0] = (AET_FTs[itrc,1]-AET_FTs[itrc,0]-FT[1]+FT[0])/wc.DT+FTd[0]
        AET_FTds[itrc,NT-1] = (AET_FTs[itrc,NT-1]-AET_FTs[itrc,NT-2]-FT[NT-1]+FT[NT-2])/wc.DT+FTd[NT-1]

    for n in range(1,NT-1):
        FT_shift = -FT[n+1]+FT[n-1]
        FTd_shift = FTd[n]
        for itrc in range(0,wc.NC):
            AET_FTds[itrc,n] = (AET_FTs[itrc,n+1]-AET_FTs[itrc,n-1]+FT_shift)/(2*wc.DT)+FTd_shift

#TODO check factor of 2pi
@njit()
def AmpFreqDeriv_inplace(AS,PS,FS,FDS,Amp,phi0,FI,FD0,TS,NT):
    """Get time domain waveform to lowest order, simple constant fdot"""
    #Amp = 4*np.pi**(2/3)*pow(Mc,5/3)/np.exp(logD)

#  compute the intrinsic frequency, phase and amplitude
    for n in range(0,NT):
        t = TS[n]
        FS[n] = FI+FD0*t
        FDS[n] = FD0
        PS[n] = -phi0+2*np.pi*FI*t+np.pi*FD0*t**2
        AS[n] = Amp#/FI**(2/3)*FS[n]**(2/3) #TODO is this amplitude right?
