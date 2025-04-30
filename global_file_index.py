"""index for loading the current versions of files"""
import numpy as np
import h5py
from instrument_noise import instrument_noise_AET_wdm_m

import wdm_const

#full_galactic_params_filename = 'LDC/LDC2_sangria_training_v2.h5'

n_par_gb = 8
labels_gb = ['Amplitude', 'EclipticLatitude', 'EclipticLongitude', 'Frequency', 'FrequencyDerivative', 'Inclination', 'InitialPhase', 'Polarization']


def get_common_noise_filename(galaxy_dir, snr_thresh, wc):
    return galaxy_dir + 'gb8_full_abbrev_snr='+str(snr_thresh)+'_Nf='+str(wc.Nf)+'_Nt='+str(wc.Nt)+'_dt=%.2f.hdf5' % wc.dt


def get_init_filename(galaxy_dir, snr_thresh, Nf, Nt, dt):
    return galaxy_dir + 'gb8_full_abbrev_snr='+str(snr_thresh)+'_Nf='+str(Nf)+'_Nt='+str(Nt)+'_dt=%.2f.hdf5' % dt


def get_preliminary_filename(galaxy_dir, snr_thresh, Nf, Nt, dt):
    return galaxy_dir + "gb8_full_abbrev_snr="+str(snr_thresh)+"_Nf="+str(Nf)+"_Nt="+str(Nt)+"_dt=%.2f.hdf5" % dt


def get_galaxy_filename(galaxy_file, galaxy_dir):
    return galaxy_dir + galaxy_file


def get_processed_gb_filename(const_only, snr_thresh, wc, nt_min=0, nt_max=-1, smooth_lengtht=0, smooth_lengthf=6):
    if nt_max == -1:
        nt_max = wc.Nt
    return "Galaxies/Galaxy1/gb8_processed_smoothf="+str(smooth_lengthf)+'smootht='+str(smooth_lengtht)+'snr'+str(snr_thresh)+"_Nf="+str(wc.Nf)+"_Nt="+str(wc.Nt)+"_dt="+str(wc.dt)+"const="+str(const_only)+"nt_min="+str(nt_min)+"nt_max="+str(nt_max)+".hdf5"


def get_noise_common(galaxy_dir, snr_thresh, wc, lc):
    filename_gb_common = get_common_noise_filename(galaxy_dir, snr_thresh, wc)
    hf_in = h5py.File(filename_gb_common, 'r')
    noise_realization_common = np.asarray(hf_in['SAET']['noise_realization'])

    wc2 = wdm_const.WDMWaveletConstants(**{key:hf_in['wc'][key][()] for key in hf_in['wc'].keys()})
    lc2 = wdm_const.LISAConstants(**{key:hf_in['lc'][key][()] for key in hf_in['lc'].keys()})

    assert wc == wc2
    assert lc == lc2

    hf_in.close()
    return filename_gb_common, noise_realization_common


def get_full_galactic_params(galaxy_file, galaxy_dir, fmin=0.00001, fmax=0.1, use_dgb=True, use_igb=True, use_vgb=True):
    """get the galaxy dataset binaries"""
    full_galactic_params_filename = get_galaxy_filename(galaxy_file, galaxy_dir)
    filename = full_galactic_params_filename

    hf_in = h5py.File(filename, 'r')
    #dgb is detached galactic binaries, igb is interacting galactic binaries, vgb is verification
    if use_dgb:
        freqs_dgb = np.asarray(hf_in['sky']['dgb']['cat']['Frequency'])
        mask_dgb = (freqs_dgb>fmin)&(freqs_dgb<fmax)
        n_dgb = np.sum(mask_dgb)
    else:
        n_dgb = 0

    if use_igb:
        freqs_igb = np.asarray(hf_in['sky']['igb']['cat']['Frequency'])
        mask_igb = (freqs_igb>fmin)&(freqs_igb<fmax)
        n_igb = np.sum(mask_igb)
    else:
        n_igb = 0

    if use_vgb:
        freqs_vgb = np.asarray(hf_in['sky']['vgb']['cat']['Frequency'])
        mask_vgb = (freqs_vgb>fmin)&(freqs_vgb<fmax)
        n_vgb = np.sum(mask_vgb)
    else:
        n_vgb = 0

    n_tot = n_dgb+n_igb+n_vgb
    #n_tot = n_vgb
    print('detached', n_dgb)
    print('interact', n_igb)
    print('verify', n_vgb)
    print('totals  ', n_tot)
    params_gb = np.zeros((n_tot, n_par_gb))
    for itrl in range(0, n_par_gb):
        if use_dgb:
            params_gb[:n_dgb, itrl] = hf_in['sky']['dgb']['cat'][labels_gb[itrl]][mask_dgb]
        if use_igb:
            params_gb[n_dgb:n_dgb+n_igb, itrl] = hf_in['sky']['igb']['cat'][labels_gb[itrl]][mask_igb]
        if use_vgb:
            params_gb[n_dgb+n_igb:, itrl] = hf_in['sky']['vgb']['cat'][labels_gb[itrl]][mask_vgb]
        #params_gb[:, itrl] = hf_in['sky']['vgb']['cat'][labels_gb[itrl]][use_vgb]

    hf_in.close()
    return params_gb, n_dgb, n_igb, n_vgb, n_tot


def load_preliminary_galactic_file(galaxy_file, galaxy_dir, snr_thresh, Nf, Nt, dt):
    preliminary_gb_filename = get_preliminary_filename(galaxy_dir, snr_thresh, Nf, Nt, dt)

    hf_in = h5py.File(preliminary_gb_filename, 'r')

    wc = wdm_const.WDMWaveletConstants(**{key:hf_in['wc'][key][()] for key in hf_in['wc'].keys()})
    lc = wdm_const.LISAConstants(**{key:hf_in['lc'][key][()] for key in hf_in['lc'].keys()})

    gb_file_source = hf_in['SAET']['source_gb_file'][()].decode()
    full_galactic_params_filename = get_galaxy_filename(galaxy_file, galaxy_dir)
    assert gb_file_source==full_galactic_params_filename
    galactic_bg_const_in = np.asarray(hf_in['SAET']['galactic_bg_const'])
    noise_realization_common = np.asarray(hf_in['SAET']['noise_realization'])
    snrs_tot_in = np.asarray(hf_in['SAET']['snrs_tot'])
    hf_in.close()
    return galactic_bg_const_in, noise_realization_common, snrs_tot_in, wc, lc


def load_init_galactic_file(galaxy_dir, snr_thresh, Nf, Nt, dt):
    filename_gb_init = get_init_filename(galaxy_dir, snr_thresh, Nf, Nt, dt)
    hf_in = h5py.File(filename_gb_init, 'r')

    #check given parameters match expectations
    Nt_got = hf_in['SAET']['Nt'][()]
    Nf_got = hf_in['SAET']['Nf'][()]
    dt_got = hf_in['SAET']['dt'][()]

    wc = wdm_const.WDMWaveletConstants(**{key:hf_in['wc'][key][()] for key in hf_in['wc'].keys()})
    lc = wdm_const.LISAConstants(**{key:hf_in['lc'][key][()] for key in hf_in['lc'].keys()})

    snr_thresh_got = hf_in['SAET']['snr_thresh'][()]
    snr_min_got = hf_in['SAET']['snr_min'][()]

    assert Nt_got == Nt
    assert Nf_got == wc.Nf
    assert dt_got == wc.dt
    assert snr_thresh_got == snr_thresh
    #snr_min_got = snr_min

    galactic_bg_const_in = np.asarray(hf_in['SAET']['galactic_bg_const'])
    noise_realization_got = np.asarray(hf_in['SAET']['noise_realization'])
    smooth_lengthf_got = hf_in['SAET']['smooth_lengthf'][()]
    smooth_lengtht_got = hf_in['SAET']['smooth_lengtht'][()]
    n_iterations_got = hf_in['SAET']['n_iterations'][()]

    snr_tots_in = np.asarray(hf_in['SAET']['snrs_tot'])
    SAET_m = np.asarray(hf_in['SAET']['SAET_m'])

    #check input SAET makes sense, first value not checked as it may not be consistent
    SAET_m_alt = instrument_noise_AET_wdm_m(lc, wc)
    assert np.allclose(SAET_m[1:], SAET_m_alt[1:], atol=1.e-80, rtol=1.e-13)

    hf_in.close()

    return filename_gb_init, snr_min_got, galactic_bg_const_in, noise_realization_got, smooth_lengthf_got, smooth_lengtht_got, n_iterations_got, snr_tots_in, SAET_m, wc, lc
