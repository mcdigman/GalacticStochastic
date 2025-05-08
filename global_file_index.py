"""index for loading the current versions of files"""
import numpy as np
import h5py

from instrument_noise import instrument_noise_AET_wdm_m

import iterative_fit_helpers as ifh

import wdm_config

import lisa_config

n_par_gb = 8
labels_gb = ['Amplitude', 'EclipticLatitude', 'EclipticLongitude', 'Frequency', 'FrequencyDerivative', 'Inclination', 'InitialPhase', 'Polarization']


def get_common_noise_filename(galaxy_dir, snr_thresh, wc):
    return galaxy_dir + ('gb8_full_abbrev_snr=%.2f' % snr_thresh)+'_Nf='+str(wc.Nf)+'_Nt='+str(wc.Nt)+'_dt=%.2f.hdf5' % wc.dt


def get_init_filename(galaxy_dir, snr_thresh, Nf, Nt, dt):
    return galaxy_dir + ('gb8_full_abbrev_snr=%.2f' % snr_thresh) +'_Nf='+str(Nf)+'_Nt='+str(Nt)+'_dt=%.2f.hdf5' % dt


def get_preliminary_filename(galaxy_dir, snr_thresh, Nf, Nt, dt):
    return galaxy_dir + ('gb8_full_abbrev_snr=%.2f' % snr_thresh) +'_Nf='+str(Nf)+'_Nt='+str(Nt)+'_dt=%.2f.hdf5' % dt


def get_galaxy_filename(galaxy_file, galaxy_dir):
    return galaxy_dir + galaxy_file


def get_processed_gb_filename(galaxy_dir, const_only, snr_thresh, wc, nt_min, nt_max):
    return galaxy_dir + ('gb8_processed_snr=%.2f' % snr_thresh) +'_Nf='+str(wc.Nf)+'_Nt='+str(wc.Nt) + ('_dt=%.2f' % (wc.dt)) + '_const='+str(const_only)+'_nt_min='+str(nt_min)+'_nt_max='+str(nt_max)+'.hdf5'


def get_noise_common(galaxy_dir, snr_thresh, wc, lc):
    filename_gb_common = get_common_noise_filename(galaxy_dir, snr_thresh, wc)
    hf_in = h5py.File(filename_gb_common, 'r')
    noise_realization_common = np.asarray(hf_in['SAET']['noise_realization'])

    wc2 = wdm_config.WDMWaveletConstants(**{key:hf_in['wc'][key][()] for key in hf_in['wc'].keys()})
    lc2 = lisa_config.LISAConstants(**{key:hf_in['lc'][key][()] for key in hf_in['lc'].keys()})

    assert wc == wc2
    assert lc == lc2

    hf_in.close()
    return noise_realization_common


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

    wc = wdm_config.WDMWaveletConstants(**{key:hf_in['wc'][key][()] for key in hf_in['wc'].keys()})
    lc = lisa_config.LISAConstants(**{key:hf_in['lc'][key][()] for key in hf_in['lc'].keys()})

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

    wc = wdm_config.WDMWaveletConstants(**{key:hf_in['wc'][key][()] for key in hf_in['wc'].keys()})
    lc = lisa_config.LISAConstants(**{key:hf_in['lc'][key][()] for key in hf_in['lc'].keys()})
    preliminary_ic = ifh.IterationConfig(**{key:hf_in['preliminary_ic'][key][()] for key in hf_in['preliminary_ic'].keys()})

    # TODO add check for wc and lc match expectations
    galactic_bg_const_in = np.asarray(hf_in['SAET']['galactic_bg_const'])
    noise_realization_got = np.asarray(hf_in['SAET']['noise_realization'])

    snr_tots_in = np.asarray(hf_in['SAET']['snrs_tot'])
    SAET_m = np.asarray(hf_in['SAET']['SAET_m'])

    #check input SAET makes sense, first value not checked as it may not be consistent
    SAET_m_alt = instrument_noise_AET_wdm_m(lc, wc)
    assert np.allclose(SAET_m[1:], SAET_m_alt[1:], atol=1.e-80, rtol=1.e-13)

    hf_in.close()

    return galactic_bg_const_in, noise_realization_got, snr_tots_in, SAET_m, wc, lc, preliminary_ic

def load_processed_gb_file(galaxy_dir, snr_thresh, wc, lc, nt_min, nt_max, const_only):
    filename_in = get_processed_gb_filename(galaxy_dir, const_only, snr_thresh, wc, nt_min, nt_max)
    hf_in = h5py.File(filename_in,'r')

    # check parameters in file match current parameters
    wc2 = wdm_config.WDMWaveletConstants(**{key:hf_in['wc'][key][()] for key in hf_in['wc'].keys()})
    lc2 = lisa_config.LISAConstants(**{key:hf_in['lc'][key][()] for key in hf_in['lc'].keys()})

    assert wc2 == wc
    assert lc2 == lc

    galactic_bg_const = np.asarray(hf_in['SAET']['galactic_bg_const'])
    galactic_bg = np.asarray(hf_in['SAET']['galactic_bg'])

    SAET_m = np.asarray(hf_in['SAET']['SAET_m'])

    SAETf_got = np.zeros((wc.Nt,wc.Nf,wc.NC))
    SAET1_got = np.zeros((wc.Nt,wc.Nf,wc.NC))

    SAETf_got[:,:,:2] = np.asarray(hf_in['SAET']['SAEf'])

    SAETf_got[:,:,2] = SAET_m[:,2]
    SAET1_got[:,:,2] = SAET_m[:,2]

    argbinmap = np.asarray(hf_in['SAET']['argbinmap'])

    hf_in.close()

    return argbinmap, (galactic_bg_const+galactic_bg).reshape((wc.Nt,wc.Nf,wc.NC))


def store_preliminary_gb_file(galaxy_dir, galaxy_file, wc, lc, ic, galactic_bg_const, noise_realization, n_bin_use, SAET_m, snrs_tot):
    Nf = wc.Nf
    Nt = wc.Nt
    dt = wc.dt
    filename_out = get_preliminary_filename(galaxy_dir, ic.snr_thresh, Nf, Nt, dt)
    hf_out = h5py.File(filename_out, 'w')
    hf_out.create_group('SAET')
    hf_out['SAET'].create_dataset('galactic_bg_const', data=galactic_bg_const, compression='gzip')
    hf_out['SAET'].create_dataset('noise_realization', data=noise_realization, compression='gzip')
    hf_out['SAET'].create_dataset('smooth_lengthf', data=ic.smooth_lengthf)
    hf_out['SAET'].create_dataset('smooth_lengtht', data=ic.smooth_lengtht)
    hf_out['SAET'].create_dataset('snr_thresh', data=ic.snr_thresh)
    hf_out['SAET'].create_dataset('snr_min', data=ic.snr_min)
    hf_out['SAET'].create_dataset('Nt', data=wc.Nt)
    hf_out['SAET'].create_dataset('Nf', data=wc.Nf)
    hf_out['SAET'].create_dataset('dt', data=wc.dt)
    hf_out['SAET'].create_dataset('n_iterations', data=ic.n_iterations)
    hf_out['SAET'].create_dataset('n_bin_use', data=n_bin_use)
    hf_out['SAET'].create_dataset('SAET_m', data=SAET_m)
    hf_out['SAET'].create_dataset('snrs_tot', data=snrs_tot[0], compression='gzip')
    hf_out['SAET'].create_dataset('source_gb_file', data=get_galaxy_filename(galaxy_file, galaxy_dir))
    # TODO I think the stored preliminary filename needs to handle second calls to this differently
    hf_out['SAET'].create_dataset('preliminary_gb_file', data=get_preliminary_filename(galaxy_dir, ic.snr_thresh, wc.Nf, wc.Nt, wc.dt))

    hf_out.create_group('preliminary_ic')
    for key in ic._fields:
        hf_out['preliminary_ic'].create_dataset(key, data=getattr(ic, key))

    hf_out.create_group('wc')
    for key in wc._fields:
        hf_out['wc'].create_dataset(key, data=getattr(wc, key))

    hf_out.create_group('lc')
    for key in lc._fields:
        hf_out['lc'].create_dataset(key, data=getattr(lc, key))

    hf_out.close()

def store_processed_gb_file(galaxy_dir, galaxy_file, wc, lc, ic, nt_min, nt_max, bgd, period_list1, n_bin_use, SAET_m, SAE_fin, const_only, snrs_tot, n_full_converged, argbinmap, const_suppress, const_suppress2, var_suppress, ic_preliminary):
    filename_gb_init = get_preliminary_filename(galaxy_dir, ic_preliminary.snr_thresh, wc.Nf, wc.Nt, wc.dt)
    filename_gb_common = get_common_noise_filename(galaxy_dir, ic_preliminary.snr_thresh, wc)
    filename_out = get_processed_gb_filename(galaxy_dir, const_only, ic.snr_thresh, wc, nt_min, nt_max)

    hf_out = h5py.File(filename_out, 'w')
    hf_out.create_group('SAET')
    hf_out['SAET'].create_dataset('galactic_bg_const', data=bgd.galactic_bg_const + bgd.galactic_bg_const_base, compression='gzip')
    hf_out['SAET'].create_dataset('galactic_bg_suppress', data=bgd.galactic_bg_suppress, compression='gzip')
    hf_out['SAET'].create_dataset('galactic_bg', data=bgd.galactic_bg, compression='gzip')
    hf_out['SAET'].create_dataset('period_list', data=period_list1)

    hf_out['SAET'].create_dataset('n_bin_use', data=n_bin_use)
    hf_out['SAET'].create_dataset('SAET_m', data=SAET_m)
    hf_out['SAET'].create_dataset('snrs_tot', data=snrs_tot[n_full_converged], compression='gzip')
    hf_out['SAET'].create_dataset('argbinmap', data=argbinmap, compression='gzip')

    hf_out['SAET'].create_dataset('const_suppress', data=const_suppress, compression='gzip')
    hf_out['SAET'].create_dataset('const_suppress2', data=const_suppress2[n_full_converged], compression='gzip')

    hf_out['SAET'].create_dataset('var_suppress', data=var_suppress[n_full_converged], compression='gzip')
    hf_out['SAET'].create_dataset('SAEf', data=SAE_fin, compression='gzip')

    hf_out['SAET'].create_dataset('source_gb_file', data=get_galaxy_filename(galaxy_file, galaxy_dir))
    hf_out['SAET'].create_dataset('preliminary_gb_file', data=filename_gb_init) # TODO these are redundant as constructed
    hf_out['SAET'].create_dataset('init_gb_file', data=filename_gb_init)
    hf_out['SAET'].create_dataset('common_gb_noise_file', data=filename_gb_common)

    hf_out.create_group('wc')
    for key in wc._fields:
        hf_out['wc'].create_dataset(key, data=getattr(wc, key))

    hf_out.create_group('lc')
    for key in lc._fields:
        hf_out['lc'].create_dataset(key, data=getattr(lc, key))

    hf_out.create_group('ic')
    for key in ic._fields:
        hf_out['ic'].create_dataset(key, data=getattr(ic, key))

    hf_out.create_group('ic_preliminary')
    for key in ic_preliminary._fields:
        hf_out['ic_preliminary'].create_dataset(key, data=getattr(ic_preliminary, key))

    hf_out.close()
