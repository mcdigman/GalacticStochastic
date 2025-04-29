"""index for loading the current versions of files"""
import numpy as np
from wdm_const import wdm_const as wc
import h5py
from instrument_noise import instrument_noise_AET_wdm_m

global_max_Nt = 4096

full_galactic_params_filename = 'LDC/LDC2_sangria_training_v2.h5'
master_gb_filename = "Galaxy/gb8_full_abbrev_snr="+str(7)+"_Nf="+str(wc.Nf)+"_Nt="+str(global_max_Nt)+"_dt="+str(wc.dt)+".hdf5"

n_par_gb = 8
labels_gb = ['Amplitude','EclipticLatitude','EclipticLongitude','Frequency','FrequencyDerivative','Inclination','InitialPhase','Polarization']

def get_common_noise_filename(snr_thresh,Nf=wc.Nf,Nt=global_max_Nt,dt=wc.dt):
    return 'Galaxy/gb8_full_abbrev_snr='+str(snr_thresh)+'_Nf='+str(wc.Nf)+'_Nt='+str(global_max_Nt)+'_dt='+str(wc.dt)+'.hdf5'

def get_init_filename(snr_thresh,Nf=wc.Nf,Nt=wc.Nt,dt=wc.dt):
    return 'Galaxy/gb8_full_abbrev_snr='+str(snr_thresh)+'_Nf='+str(wc.Nf)+'_Nt='+str(wc.Nt)+'_dt='+str(wc.dt)+'.hdf5'

def get_processed_gb_filename(const_only,snr_thresh,nt_min=0,nt_max=wc.Nt,Nf=wc.Nf,Nt=wc.Nt,smooth_lengtht=0,smooth_lengthf=6,dt=wc.dt):
    return "Galaxy/gb8_processed_smoothf="+str(smooth_lengthf)+'smootht='+str(smooth_lengtht)+'snr'+str(snr_thresh)+"_Nf="+str(Nf)+"_Nt="+str(Nt)+"_dt="+str(dt)+"const="+str(const_only)+"nt_min="+str(nt_min)+"nt_max="+str(nt_max)+".hdf5"

def get_noise_common(snr_thresh,Nf=wc.Nf,Nt=global_max_Nt,dt=wc.dt):
    filename_gb_common = get_common_noise_filename(snr_thresh,Nf=Nf,Nt=Nt,dt=dt)
    hf_in = h5py.File(filename_gb_common,'r')
    noise_realization_common = np.asarray(hf_in['SAET']['noise_realization'])
    hf_in.close()
    return filename_gb_common,noise_realization_common

def get_full_galactic_params(fmin=0.00001,fmax=0.1,use_dgb=True,use_igb=True,use_vgb=True):
    """get the galaxy dataset binaries"""
    filename = full_galactic_params_filename

    hf_in = h5py.File(filename,'r')
    #dgb is detached galactic binaries, igb is interacting galactic binaries, vgb is verification
    if use_dgb:
        freqs_dgb = hf_in['sky']['dgb']['cat']['Frequency']
        mask_dgb = (freqs_dgb>fmin)&(freqs_dgb<fmax)
        n_dgb = np.sum(mask_dgb)
    else:
        n_dgb = 0

    if use_igb:
        freqs_igb = hf_in['sky']['igb']['cat']['Frequency']
        mask_igb = (freqs_igb>fmin)&(freqs_igb<fmax)
        n_igb = np.sum(mask_igb)
    else:
        n_igb = 0

    if use_vgb:
        freqs_vgb = hf_in['sky']['vgb']['cat']['Frequency']
        mask_vgb = (freqs_vgb>fmin)&(freqs_vgb<fmax)
        n_vgb = np.sum(mask_vgb)
    else:
        n_vgb = 0

    n_tot = n_dgb+n_igb+n_vgb
    #n_tot = n_vgb
    print('detached',n_dgb)
    print('interact',n_igb)
    print('verify',n_vgb)
    print('totals  ',n_tot)
    params_gb = np.zeros((n_tot,n_par_gb))
    for itrl in range(0,n_par_gb):
        if use_dgb:
            params_gb[:n_dgb,itrl] = hf_in['sky']['dgb']['cat'][labels_gb[itrl]][mask_dgb]
        if use_igb:
            params_gb[n_dgb:n_dgb+n_igb,itrl] = hf_in['sky']['igb']['cat'][labels_gb[itrl]][mask_igb]
        if use_vgb:
            params_gb[n_dgb+n_igb:,itrl] = hf_in['sky']['vgb']['cat'][labels_gb[itrl]][mask_vgb]
        #params_gb[:,itrl] = hf_in['sky']['vgb']['cat'][labels_gb[itrl]][use_vgb]

    hf_in.close()
    return params_gb,n_dgb,n_igb,n_vgb,n_tot

def load_master_galactic_file():
    hf_in = h5py.File(master_gb_filename,'r')
    gb_file_source = hf_in['SAET']['source_gb_file'][()].decode()
    print(gb_file_source)
    print(full_galactic_params_filename)
    assert gb_file_source==full_galactic_params_filename
    galactic_bg_const_in = np.asarray(hf_in['SAET']['galactic_bg_const'])
    noise_realization_common = np.asarray(hf_in['SAET']['noise_realization'])
    snrs_tot_in = np.asarray(hf_in['SAET']['snrs_tot'])
    hf_in.close()
    return galactic_bg_const_in,noise_realization_common,snrs_tot_in

def load_init_galactic_file(snr_thresh,Nf=wc.Nf,Nt=wc.Nt,dt=wc.dt):
    filename_gb_init = get_init_filename(snr_thresh,Nf=wc.Nf,Nt=wc.Nt,dt=wc.dt)
    hf_in = h5py.File(filename_gb_init,'r')

    #check given parameters match expectations
    Nt_got = hf_in['SAET']['Nt'][()]
    Nf_got = hf_in['SAET']['Nf'][()]
    dt_got = hf_in['SAET']['dt'][()]
    snr_thresh_got = hf_in['SAET']['snr_thresh'][()]
    snr_min_got = hf_in['SAET']['snr_min'][()]

    assert Nt_got == Nt
    assert Nf_got == Nf
    assert dt_got == dt
    assert snr_thresh_got == snr_thresh
    #snr_min_got = snr_min

    #galactic_bg_const_in = np.sqrt(wc.Tobs/(8*wc.Nt*wc.Nf))*np.asarray(hf_in['SAET']['galactic_bg_const'])
    galactic_bg_const_in = np.asarray(hf_in['SAET']['galactic_bg_const'])
    noise_realization_got = np.asarray(hf_in['SAET']['noise_realization'])
    smooth_lengthf_got = hf_in['SAET']['smooth_lengthf'][()]
    smooth_lengtht_got = hf_in['SAET']['smooth_lengtht'][()]
    n_iterations_got = hf_in['SAET']['n_iterations'][()]

    snr_tots_in = np.asarray(hf_in['SAET']['snrs_tot'])
    SAET_m = np.asarray(hf_in['SAET']['SAET_m'])

    #check input SAET makes sense, first value not checked as it may not be consistent
    SAET_m_alt = instrument_noise_AET_wdm_m()
    assert np.allclose(SAET_m[1:],SAET_m_alt[1:],atol=1.e-80,rtol=1.e-13)

    hf_in.close()

    return filename_gb_init,snr_min_got,galactic_bg_const_in,noise_realization_got,smooth_lengthf_got,smooth_lengtht_got,n_iterations_got,snr_tots_in,SAET_m
