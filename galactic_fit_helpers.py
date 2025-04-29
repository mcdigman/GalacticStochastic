"""scratch to test processing of galactic background"""
from numba import njit,prange
import numpy as np
from time import perf_counter
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import least_squares,dual_annealing,differential_evolution
import scipy.ndimage

import WDMWaveletTransforms.fft_funcs as fft

from wdm_const import wdm_const as wc
import global_const as gc

TobsYEAR = wc.Tobs/gc.SECSYEAR

def SAE_gal_model(f,log10A,log10f2,log10f1,log10fknee,alpha):
    """model from arXiv:2103.14598 for galactic binary confusion noise amplitude"""
    return 10**log10A/2*f**(5/3)*np.exp(-(f/10**log10f1)**alpha)*(1+np.tanh((10**log10fknee-f)/10**log10f2))

def SAE_gal_model_alt(f,A,alpha,beta,kappa,gamma,fknee):
    """model from arXiv:1703.09858 for galactic binary confusion noise amplitude"""
    return A*f**(7/3)*np.exp(-f**alpha+beta*f*np.sin(kappa*f))*(1+np.tanh(gamma*(fknee-f)))


def log10_SAE_gal_model(f,A,f2,a1,b1,ak,bk,alpha):
    """model from arXiv:2103.14598 for galactic binary confusion noise amplitude"""
    f1 = 10**(a1*np.log10(TobsYEAR)+b1)
    fknee = 10**(ak*np.log10(TobsYEAR)+bk)
    return np.log10(A)-np.log10(2)+-7/3*np.log10(f)-(f/f1)**alpha/np.log(10)+np.log10((1+np.tanh((fknee-f)/f2)))

def gen_wave_sum(noise_AET_dense,waveT_ini,n_bin_use,params_gb,force_suppress,nt_min,nt_max,snr_thresh,get_snrs=True):
    #do the finishing step for itrn=0 to set everything at the end of the loop as it should be
    snrs_tot = np.zeros(n_bin_use)
    snrs = np.zeros((n_bin_use,wc.NC))
    var_suppress = np.zeros(n_bin_use,dtype=np.bool_)
    galactic_bg = np.zeros((wc.Nt*wc.Nf,wc.NC))

    for itrb in range(0,n_bin_use):
        if not force_suppress[itrb]:
            waveT_ini.update_params(params_gb[itrb].copy())
            listT_temp,waveT_temp,NUTs_temp = waveT_ini.get_unsorted_coeffs()

            if get_snrs:
                snrs[itrb] = noise_AET_dense.get_sparse_snrs(NUTs_temp,listT_temp,waveT_temp,nt_min,nt_max)
                snrs_tot[itrb] = np.linalg.norm(snrs[itrb])

                if snrs_tot[itrb]>=snr_thresh:
                    var_suppress[itrb] = True
                elif np.isnan(snrs_tot[itrb]):#itrn>=n_itr_const_build:
                    print('nan detected in snr at '+str(itrb))

            if not var_suppress[itrb]:
                for itrc in range(0,2):
                    galactic_bg[listT_temp[itrc,:NUTs_temp[itrc]],itrc] += waveT_temp[itrc,:NUTs_temp[itrc]]

    return galactic_bg,snrs,snrs_tot,var_suppress

def snr_suppress_consistency_check(noise_AET_dense,waveT_ini,n_bin_use,params_gb,force_suppress,nt_min,nt_max,snr_thresh):
    #do the finishing step for itrn=0 to set everything at the end of the loop as it should be
    snrs_tot = np.zeros(n_bin_use)
    snrs = np.zeros((n_bin_use,wc.NC))
    var_suppress = np.zeros(n_bin_use,dtype=np.bool_)

    for itrb in range(0,n_bin_use):
        if not force_suppress[itrb]:
            waveT_ini.update_params(params_gb[itrb].copy())
            listT_temp,waveT_temp,NUTs_temp = waveT_ini.get_unsorted_coeffs()

            snrs[itrb] = noise_AET_dense.get_sparse_snrs(NUTs_temp,listT_temp,waveT_temp,nt_min,nt_max)
            snrs_tot[itrb] = np.linalg.norm(snrs[itrb])

            if snrs_tot[itrb]>=snr_thresh:
                var_suppress[itrb] = True
            elif np.isnan(snrs_tot[itrb]):#itrn>=n_itr_const_build:
                print('nan detected in snr at '+str(itrb))
    return snrs,snrs_tot,var_suppress


def filter_periods_fft(r_got1,Nt_loc,period_list):
    ts = np.arange(0,Nt_loc)*wc.DT
    wts = 2*np.pi/gc.SECSYEAR*ts
    r_fft1 = np.zeros((wc.Nt,wc.NC))
    for itrc in range(0,2):
        res_fft = fft.rfft(r_got1[:,itrc]-1.)*2/Nt_loc
        abs_fft = np.abs(res_fft)
        angle_fft = -np.angle(res_fft)
        rec = 1.+abs_fft[0]/2+np.zeros(Nt_loc)
        for k in period_list:
            idx = np.int64(wc.Tobs/gc.SECSYEAR)*k
            rec += abs_fft[idx]*np.cos(k*wts-angle_fft[idx])
        angle_fftm = angle_fft%(2*np.pi)
        print("%5.3f & %5.2f & %5.3f & %5.2f & %5.3f & %5.2f & %5.3f & %5.2f & %5.3f & %5.2f"%(abs_fft[1*8],angle_fftm[1*8],abs_fft[2*8],angle_fftm[2*8],abs_fft[3*8],angle_fftm[3*8],abs_fft[4*8],angle_fftm[4*8],abs_fft[5*8],angle_fftm[5*8]))
        r_fft1[:,itrc] = rec
    return r_fft1


def get_SAET_cyclostationary_mean(galactic_bg,SAET_m,smooth_lengthf=4,filter_periods=False,period_list=None,Nt_loc = wc.Nt):
    SAET_pure_in = (galactic_bg.reshape((wc.Nt,wc.Nf,wc.NC)))**2
    SAET_pure_mean = np.mean(SAET_pure_in,axis=0)
    if not filter_periods:
        rec_use = np.zeros((wc.Nt,wc.NC))+1.
    else:
        r_got1 = np.zeros((wc.Nt,wc.NC))
        SAET_pure_white = SAET_pure_mean/SAET_m

        for itrc in range(0,2):
            r_eval_mask = SAET_pure_white[:,itrc]>(0.1*np.max(SAET_pure_white[:,itrc]))
            r_got1[:,itrc] = np.mean(SAET_pure_in[:,r_eval_mask,itrc]/(SAET_pure_mean[r_eval_mask,itrc]+1.e-13*np.max(SAET_pure_mean[r_eval_mask,itrc])),axis=1)

        if period_list is None:
            period_list = np.arange(1,np.int64(gc.SECSYEAR//wc.DT)//2+1)


        r_fft1 = filter_periods_fft(r_got1,Nt_loc,period_list)

        rec_use = r_fft1

    SAET_pure_mod = SAET_pure_in.copy()
    for itrc in range(0,2):
        SAET_pure_mod[:,:,itrc] = (SAET_pure_mod[:,:,itrc].T/rec_use[:,itrc]).T


    SAET_pures = np.mean(SAET_pure_mod,axis=0)


    SAET_pures_smooth2 = np.zeros((wc.Nf,3))
    SAET_pures_smooth2[0,:] = SAET_pures[0,:]

    log_fs = np.log10(np.arange(1,wc.Nf)*wc.DF)
    n_f_interp = 10*wc.Nf
    log_fs_interp = np.linspace(np.log10(wc.DF),np.log10(wc.DF*(wc.Nf-1)),n_f_interp)

    for itrc in range(0,3):
        log_SAE_pure_loc_smooth1 = np.log10(SAET_pures[1:,itrc]+1.e-50)
        log_SAE_interp_loc = InterpolatedUnivariateSpline(log_fs,log_SAE_pure_loc_smooth1,k=3,ext=2)(log_fs_interp)
        log_SAE_interp_loc_smooth = scipy.ndimage.gaussian_filter(log_SAE_interp_loc,smooth_lengthf*10.)
        SAET_pures_smooth2[1:,itrc] = 10**InterpolatedUnivariateSpline(log_fs_interp,log_SAE_interp_loc_smooth,k=3,ext=2)(log_fs)-1.e-50

    SAET_res = np.zeros((wc.Nt,wc.Nf,wc.NC))+SAET_m
    for itrc in range(0,2):
        SAET_res[:,:,itrc] += np.outer(rec_use[:,itrc],SAET_pures_smooth2[:,itrc])

    assert np.all(np.isfinite(SAET_res))

    return SAET_res,rec_use,SAET_pures_smooth2

def fit_gb_spectrum_evolve(SAET_goals,fs,fs_report,nt_ranges,offset):
    a1 = -0.25#-0.15
    b1 = -2.70#-0.37
    ak = -0.27#-2.72
    bk = -2.47#-2.49
    log10A = np.log10(7.e-39)
    log10f2 = np.log10(0.00051)#1.0292637e-4#0.00067
    alpha = 1.6#1.56

    TobsYEAR_locs = nt_ranges*wc.DT/gc.SECSYEAR
    n_spect = SAET_goals.shape[0]

    log_SAE_goals = np.log10(SAET_goals[:,:,0:2])


    def SAE_func_temp(tpl):
        resid = 0.
        a1 = tpl[0]
        ak = tpl[1]
        b1 = tpl[2]
        bk = tpl[3]
        log10A = tpl[4]
        log10f2 = tpl[5]
        alpha = tpl[6]
        for itry in range(n_spect):
            log10f1 = (a1*np.log10(TobsYEAR_locs[itry])+b1)
            log10fknee = (ak*np.log10(TobsYEAR_locs[itry])+bk)
            resid += np.sum((np.log10(np.abs(SAE_gal_model(fs,log10A,log10f2,log10f1,log10fknee,alpha))+offset)-log_SAE_goals[itry,:,:].T).flatten()**2)
        return resid

    bounds = np.zeros((7,2))
    bounds[0,0] = a1-0.2#-0.35
    bounds[0,1] = a1+0.2#-0.05
    bounds[1,0] = ak-0.2
    bounds[1,1] = ak+0.2
    bounds[2,0] = b1-0.4
    bounds[2,1] = b1+0.4
    bounds[3,0] = bk-0.4
    bounds[3,1] = bk+0.4
    bounds[4,0] = log10A-0.5
    bounds[4,1] = log10A+0.5
    bounds[5,0] = log10f2-1.5
    bounds[5,1] = log10f2+1.5
    bounds[6,0] = 1.35
    bounds[6,1] = 2.25

    res_found = dual_annealing(SAE_func_temp,bounds,maxiter=2000)


    res = res_found['x']
    print(res_found)

    a1 = res[0]
    ak = res[1]
    b1 = res[2]
    bk = res[3]
    log10A = res[4]
    log10f2 = res[5]
    alpha = res[6]

    SAE_base_res = np.zeros((n_spect,fs_report.size))
    for itry in range(n_spect):
        log10f1 = (a1*np.log10(TobsYEAR_locs[itry])+b1)
        log10fknee = (ak*np.log10(TobsYEAR_locs[itry])+bk)
        SAE_base_res[itry,:] = SAE_gal_model(fs_report,log10A,log10f2,log10f1,log10fknee,alpha)

    return SAE_base_res,res

def fit_gb_spectrum_pure(SAET_goal,fs,fs_report,nt_range=wc.Nt,same_spectra=False):
    a1 = -2.0873951428e-1#-0.15
    b1 = -2.6083913865e0#-0.37
    ak = -2.7979471707e-1#-2.72
    bk = -2.2429123040e0#-2.49
    log10A = np.log10(1.e-38)
    log10f2 = np.log10(1.029e-3)#1.0292637e-4#0.00067
    alpha = 2.#1.56

    TobsYEAR_loc = nt_range*wc.DT/gc.SECSYEAR

    log10f1 = (a1*np.log10(TobsYEAR_loc)+b1)
    log10fknee = (ak*np.log10(TobsYEAR_loc)+bk)
    print('f1, knee',log10f1,log10fknee)

    nfr = 10000
    fs_resample = 10**np.linspace(np.log10(np.min(fs)),np.log10(np.max(fs)),nfr)

    SAET_goal_resample = np.zeros((nfr,2))
    SAE_base_res = np.zeros((fs_report.size,2))
    res_param = np.zeros((5,2))

    for itrc in range(0,2):
        SAET_goal_resample[:,itrc] = 10**InterpolatedUnivariateSpline(np.log10(fs),np.log10(SAET_goal[:,itrc]))(np.log10(fs_resample))

    if not same_spectra:
        for itrc in range(0,2):
            def SAE_func_temp(tpl):
                return np.log10(SAE_gal_model(fs_resample,tpl[0],tpl[1],tpl[2],tpl[3],tpl[4])+1.e-44)-np.log10(SAET_goal_resample[:,itrc])

            res = least_squares(SAE_func_temp,np.array([log10A,log10f2,log10f1,log10fknee,alpha]))['x']
            res_param[:,itrc] = res
            SAE_base_res[:,itrc] = SAE_gal_model(fs_report,res[0],res[1],res[2],res[3],res[4])
            print('res params',res)
    else:
        def SAE_func_temp(tpl):
            return np.sum((np.log10(np.abs(SAE_gal_model(fs,tpl[0],tpl[1],tpl[2],tpl[3],tpl[4]))+1.e-44)-np.log10(SAET_goal[:,0:2].T)).flatten()**2)

        res_found = least_squares(SAE_func_temp,np.array([log10A,log10f2,log10f1,log10fknee,alpha]))
        bounds = np.zeros((5,2))
        bounds[0,0] = np.log10(2.5e-39)
        bounds[0,1] = np.log10(3.5e-37)
        bounds[1,0] = np.log10(5.e-5)
        bounds[1,1] = np.log10(2.e-3)
        bounds[2,0] = log10f1-0.1#f1-f1*0.1
        bounds[2,1] = log10f1+0.1#f1+f1*0.1
        bounds[3,0] = log10fknee-0.1#fknee*0.1
        bounds[3,1] = log10fknee+0.1
        bounds[4,0] = 1.50
        bounds[4,1] = 2.32
        res_found = differential_evolution(SAE_func_temp,bounds,maxiter=1000)
        res = res_found['x']
        print(res_found)

        for itrc in range(0,2):
            res_param[:,itrc] = res
            SAE_base_res[:,itrc] = SAE_gal_model(fs_report,res[0],res[1],res[2],res[3],res[4])

    print(res_param[:,0])
    return SAE_base_res,res_param


@njit()
def get_SAET_smooth(galactic_bg_res,SAET_m,noise_realization,smooth_lengthf,smooth_lengtht,Nt_loc=wc.Nt):
    #skipping actually smoothing T right now because we don't use it
    galactic_bg_full = galactic_bg_res.reshape(Nt_loc,wc.Nf,wc.NC)#+noise_realization

    signal_full = (galactic_bg_full+noise_realization)#/np.sqrt((wc.Tobs/(8*wc.Nt*wc.Nf)))

    SAET_galactic_bg_smooth = np.zeros((Nt_loc,wc.Nf,wc.NC))

    for itrc in range(0,2):
        SAET_galactic_bg_white = signal_full[:,:,itrc]**2/SAET_m[:,itrc]
        SAET_galactic_bg_smoothf_white = np.zeros((Nt_loc,wc.Nf))
        SAET_galactic_bg_smoothft_white = np.zeros((Nt_loc,wc.Nf))

        for itrt in prange(0,Nt_loc):
            sum_loc = 0.
            n_inc = 0
            for itrn in range(0,smooth_lengthf+1):
                sum_loc += SAET_galactic_bg_white[itrt,itrn]
                n_inc += 1
            for itrf in range(0,smooth_lengthf):
                SAET_galactic_bg_smoothf_white[itrt,itrf] = SAET_galactic_bg_white[itrt,itrn]#sum_loc/n_inc
                sum_loc += SAET_galactic_bg_white[itrt,itrf+smooth_lengthf+1]
                n_inc += 1
            for itrf in range(smooth_lengthf,wc.Nf-smooth_lengthf-1):
                SAET_galactic_bg_smoothf_white[itrt,itrf] = sum_loc/n_inc
                sum_loc += SAET_galactic_bg_white[itrt,itrf+smooth_lengthf+1]
                sum_loc -= SAET_galactic_bg_white[itrt,itrf-smooth_lengthf]
            for itrf in range(wc.Nf-smooth_lengthf-1,wc.Nf):
                SAET_galactic_bg_smoothf_white[itrt,itrf] = SAET_galactic_bg_white[itrt,itrn]#sum_loc/n_inc
                sum_loc -= SAET_galactic_bg_white[itrt,itrf-smooth_lengthf]
                n_inc -= 1

        for itrf in prange(0,wc.Nf):
            sum_loc = 0.
            n_inc = 0
            for itrn in range(0,smooth_lengtht+1):
                sum_loc += SAET_galactic_bg_smoothf_white[itrn,itrf]
                n_inc += 1
            for itrt in range(0,smooth_lengtht):
                SAET_galactic_bg_smoothft_white[itrt,itrf] = sum_loc/n_inc
                sum_loc += SAET_galactic_bg_smoothf_white[itrt+smooth_lengtht+1,itrf]
                n_inc += 1
            for itrt in range(smooth_lengtht,Nt_loc-smooth_lengtht-1):
                SAET_galactic_bg_smoothft_white[itrt,itrf] = sum_loc/n_inc
                sum_loc += SAET_galactic_bg_smoothf_white[itrt+smooth_lengtht+1,itrf]
                sum_loc -= SAET_galactic_bg_smoothf_white[itrt-smooth_lengtht,itrf]
            for itrt in range(Nt_loc-smooth_lengtht-1,Nt_loc):
                SAET_galactic_bg_smoothft_white[itrt,itrf] = sum_loc/n_inc
                sum_loc -= SAET_galactic_bg_smoothf_white[itrt-smooth_lengtht,itrf]
                n_inc -= 1
        SAET_galactic_bg_smooth[:,:,itrc] = SAET_galactic_bg_smoothft_white[:,:]*SAET_m[:,itrc]

    for itrt in range(0,Nt_loc):
        SAET_galactic_bg_smooth[itrt,:,2] = SAET_m[:,2]

    return SAET_galactic_bg_smooth
