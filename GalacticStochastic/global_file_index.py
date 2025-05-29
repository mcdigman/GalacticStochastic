"""index for loading the current versions of files"""

from pathlib import Path

import h5py
import numpy as np

from GalacticStochastic.background_decomposition import BGDecomposition
from GalacticStochastic.iteration_config import IterationConfig
from LisaWaveformTools import lisa_config
from LisaWaveformTools.instrument_noise import instrument_noise_AET_wdm_m
from LisaWaveformTools.lisa_config import LISAConstants
from WaveletWaveforms import wdm_config
from WaveletWaveforms.sparse_waveform_functions import PixelTimeRange
from WaveletWaveforms.wdm_config import WDMWaveletConstants

n_par_gb = 8
labels_gb = [
    'Amplitude',
    'EclipticLatitude',
    'EclipticLongitude',
    'Frequency',
    'FrequencyDerivative',
    'Inclination',
    'InitialPhase',
    'Polarization',
]


def get_common_noise_filename(galaxy_dir, snr_thresh, wc: WDMWaveletConstants):
    return (
        galaxy_dir
        + ('preprocessed_background=%.2f' % snr_thresh)
        + '_Nf='
        + str(wc.Nf)
        + '_Nt='
        + str(wc.Nt)
        + '_dt=%.2f.hdf5' % wc.dt
    )


def get_init_filename(galaxy_dir, snr_thresh, Nf, Nt, dt):
    return (
        galaxy_dir
        + ('preprocessed_background=%.2f' % snr_thresh)
        + '_Nf='
        + str(Nf)
        + '_Nt='
        + str(Nt)
        + '_dt=%.2f.hdf5' % dt
    )


def get_preliminary_filename(galaxy_dir, snr_thresh, Nf, Nt, dt):
    return (
        galaxy_dir
        + ('preprocessed_background=%.2f' % snr_thresh)
        + '_Nf='
        + str(Nf)
        + '_Nt='
        + str(Nt)
        + '_dt=%.2f.hdf5' % dt
    )


def get_galaxy_filename(galaxy_file, galaxy_dir):
    return galaxy_dir + galaxy_file


def get_processed_gb_filename(galaxy_dir, stat_only, snr_thresh, wc: WDMWaveletConstants, nt_lim_snr: PixelTimeRange):
    return (
        galaxy_dir
        + ('gb8_processed_snr=%.2f' % snr_thresh)
        + '_Nf='
        + str(wc.Nf)
        + '_Nt='
        + str(wc.Nt)
        + ('_dt=%.2f' % (wc.dt))
        + '_const='
        + str(stat_only)
        + '_nt_min='
        + str(nt_lim_snr.nt_min)
        + '_nt_max='
        + str(nt_lim_snr.nt_max)
        + '.hdf5'
    )


def get_noise_common(galaxy_dir, snr_thresh, wc: WDMWaveletConstants):
    filename_gb_common = get_common_noise_filename(galaxy_dir, snr_thresh, wc)
    hf_in = h5py.File(filename_gb_common, 'r')
    noise_realization_common = np.asarray(hf_in['S']['noise_realization'])

    # wc2 = wdm_config.WDMWaveletConstants(**{key: hf_in['wc'][key][()] for key in wc._fields})
    # lc2 = lisa_config.LISAConstants(**{key: hf_in['lc'][key][()] for key in lc._fields})

    # assert wc == wc2
    # assert lc == lc2

    hf_in.close()
    return noise_realization_common


def get_full_galactic_params(
    galaxy_file, galaxy_dir, *, fmin=0.00001, fmax=0.1, use_dgb=True, use_igb=True, use_vgb=True
):
    """Get the galaxy dataset binaries"""
    full_galactic_params_filename = get_galaxy_filename(galaxy_file, galaxy_dir)
    filename = full_galactic_params_filename

    hf_in = h5py.File(filename, 'r')
    # dgb is detached galactic binaries, igb is interacting galactic binaries, vgb is verification
    if use_dgb:
        freqs_dgb = np.asarray(hf_in['sky']['dgb']['cat']['Frequency'])
        mask_dgb = (freqs_dgb > fmin) & (freqs_dgb < fmax)
        n_dgb = np.sum(mask_dgb)
    else:
        n_dgb = 0

    if use_igb:
        freqs_igb = np.asarray(hf_in['sky']['igb']['cat']['Frequency'])
        mask_igb = (freqs_igb > fmin) & (freqs_igb < fmax)
        n_igb = np.sum(mask_igb)
    else:
        n_igb = 0

    if use_vgb:
        freqs_vgb = np.asarray(hf_in['sky']['vgb']['cat']['Frequency'])
        mask_vgb = (freqs_vgb > fmin) & (freqs_vgb < fmax)
        n_vgb = np.sum(mask_vgb)
    else:
        n_vgb = 0

    n_tot = n_dgb + n_igb + n_vgb
    print('detached', n_dgb)
    print('interact', n_igb)
    print('verify', n_vgb)
    print('totals  ', n_tot)
    params_gb = np.zeros((n_tot, n_par_gb))
    for itrl in range(n_par_gb):
        if use_dgb:
            params_gb[:n_dgb, itrl] = hf_in['sky']['dgb']['cat'][labels_gb[itrl]][mask_dgb]
        if use_igb:
            params_gb[n_dgb:n_dgb + n_igb, itrl] = hf_in['sky']['igb']['cat'][labels_gb[itrl]][mask_igb]
        if use_vgb:
            params_gb[n_dgb + n_igb:, itrl] = hf_in['sky']['vgb']['cat'][labels_gb[itrl]][mask_vgb]

    hf_in.close()
    return params_gb, n_dgb, n_igb, n_vgb, n_tot


def load_preliminary_galactic_file(galaxy_file, galaxy_dir, snr_thresh, wc: WDMWaveletConstants, lc: LISAConstants):
    preliminary_gb_filename = get_preliminary_filename(galaxy_dir, snr_thresh, wc.Nf, wc.Nt, wc.dt)

    hf_in = h5py.File(preliminary_gb_filename, 'r')

    # check the stored file matches necessary current parameters

    # load the wavelet constants
    # wc_in = wdm_config.WDMWaveletConstants(**{key: hf_in['configuration']['wc'][key][()] for key in wc._fields})
    # assert wc_in == wc

    # load the lisa constants
    # lc_in = lisa_config.LISAConstants(**{key: hf_in['configuration']['lc'][key][()] for key in lc._fields})
    lc_in = lc
    assert lc_in == lc

    # check the galaxy filename matches
    gb_file_source = hf_in['galaxy']['binaries']['galaxy_file'][()].decode()
    full_galactic_params_filename = get_galaxy_filename(galaxy_file, galaxy_dir)
    assert gb_file_source == full_galactic_params_filename

    galactic_below_in = np.asarray(hf_in['galaxy']['signal']['galactic_below'])

    snrs_tot_upper_in = np.asarray(hf_in['galaxy']['binaries']['snrs_tot_upper'])

    S_inst_m = np.asarray(hf_in['noise_model']['S_instrument'])

    # check input S makes sense, first value not checked as it may not be consistent
    S_inst_m_alt = instrument_noise_AET_wdm_m(lc, wc)
    assert np.allclose(S_inst_m[1:], S_inst_m_alt[1:], atol=1.0e-80, rtol=1.0e-13)

    hf_in.close()

    return galactic_below_in, snrs_tot_upper_in, S_inst_m, wc, lc


def load_preliminary_galactic_file_old(galaxy_file, galaxy_dir, snr_thresh, Nf, Nt, dt):
    preliminary_gb_filename = get_preliminary_filename(galaxy_dir, snr_thresh, Nf, Nt, dt)

    hf_in = h5py.File(preliminary_gb_filename, 'r')

    wc = wdm_config.WDMWaveletConstants(**{key: hf_in['wc'][key][()] for key in hf_in['wc']})
    lc = lisa_config.LISAConstants(**{key: hf_in['lc'][key][()] for key in hf_in['lc']})

    gb_file_source = hf_in['S']['source_gb_file'][()].decode()
    full_galactic_params_filename = get_galaxy_filename(galaxy_file, galaxy_dir)
    assert gb_file_source == full_galactic_params_filename
    # preserving ability to read some files some legacy key names
    try:
        galactic_below_in = np.asarray(hf_in['S']['galactic_below'])
    except KeyError:
        galactic_below_in = np.asarray(hf_in['S']['galactic_bg_const'])

    try:
        snrs_tot_upper_in = np.asarray(hf_in['S']['snrs_tot_upper'])
    except KeyError:
        snrs_tot_upper_in = np.asarray(hf_in['S']['snrs_tot'])

    hf_in.close()
    return galactic_below_in, snrs_tot_upper_in, wc, lc


def load_init_galactic_file(galaxy_dir, snr_thresh, Nf, Nt, dt):
    filename_gb_init = get_init_filename(galaxy_dir, snr_thresh, Nf, Nt, dt)
    hf_in = h5py.File(filename_gb_init, 'r')

    # check given parameters match expectations

    wc = wdm_config.WDMWaveletConstants(**{key: hf_in['wc'][key][()] for key in hf_in['wc']})
    lc = lisa_config.LISAConstants(**{key: hf_in['lc'][key][()] for key in hf_in['lc']})
    snr_min = hf_in['preliminary_ic']['snr_min'][()]

    # TODO add check for wc and lc match expectations
    try:
        galactic_below_in = np.asarray(hf_in['S']['galactic_below'])
    except KeyError:
        galactic_below_in = np.asarray(hf_in['S']['galactic_bg_const'])

    try:
        snr_tots_in = np.asarray(hf_in['S']['snrs_tot_upper'])
    except KeyError:
        snr_tots_in = np.asarray(hf_in['S']['snrs_tot'])

    S_inst_m = np.asarray(hf_in['S']['S_stat_m'])

    # check input S makes sense, first value not checked as it may not be consistent
    S_inst_m_alt = instrument_noise_AET_wdm_m(lc, wc)
    assert np.allclose(S_inst_m[1:], S_inst_m_alt[1:], atol=1.0e-80, rtol=1.0e-13)

    hf_in.close()

    return galactic_below_in, snr_tots_in, S_inst_m, wc, lc, snr_min


def load_processed_gb_file(
    galaxy_dir, snr_thresh, wc: WDMWaveletConstants, lc: LISAConstants, nt_lim_snr: PixelTimeRange, *, stat_only
):
    # TODO loading should produce a galactic background decomposition object
    filename_in = get_processed_gb_filename(galaxy_dir, stat_only, snr_thresh, wc, nt_lim_snr)
    hf_in = h5py.File(filename_in, 'r')

    # check parameters in file match current parameters
    # wc2 = wdm_config.WDMWaveletConstants(**{key: hf_in['wc'][key][()] for key in wc._fields})
    # lc2 = lisa_config.LISAConstants(**{key: hf_in['lc'][key][()] for key in lc._fields})
    lc2 = lc
    # assert wc2 == wc
    assert lc2 == lc

    galactic_below = np.asarray(hf_in['S']['galactic_below'])
    try:
        galactic_undecided = np.asarray(hf_in['S']['galactic_undecided'])
    except ImportError:
        galactic_undecided = np.asarray(hf_in['S']['galactic_bg'])

    argbinmap = np.asarray(hf_in['S']['argbinmap'])

    hf_in.close()

    return argbinmap, (galactic_below + galactic_undecided).reshape((wc.Nt, wc.Nf, galactic_below.shape[-1]))


def store_preliminary_gb_file(
    config_filename,
    galaxy_dir,
    galaxy_file,
    wc: WDMWaveletConstants,
    lc: LISAConstants,
    ic: IterationConfig,
    galactic_below,
    S_inst_m,
    snrs_tot_upper,
) -> None:
    filename_out = get_preliminary_filename(galaxy_dir, ic.snr_thresh, wc.Nf, wc.Nt, wc.dt)
    hf_out = h5py.File(filename_out, 'w')

    # store results related to the input galaxy
    hf_out.create_group('galaxy')

    hf_out['galaxy'].create_group('signal')
    hf_out['galaxy'].create_group('binaries')

    # the faint part of the galactic signal
    hf_out['galaxy']['signal'].create_dataset('galactic_below', data=galactic_below, compression='gzip')

    # the filename of the file with the galaxy in it
    hf_out['galaxy']['binaries'].create_dataset('galaxy_file', data=get_galaxy_filename(galaxy_file, galaxy_dir))

    # the initial computed snr
    hf_out['galaxy']['binaries'].create_dataset('snrs_tot_upper', data=snrs_tot_upper[0], compression='gzip')

    # store parameters related to the noise model
    hf_out.create_group('noise_model')

    hf_out['noise_model'].create_dataset('S_instrument', data=S_inst_m)

    # store configuration parameters
    hf_out.create_group('configuration')
    hf_out['configuration'].create_group('wc')
    hf_out['configuration'].create_group('ic')
    hf_out['configuration'].create_group('lc')
    hf_out['configuration'].create_group('config_text')

    # store all the configuration objects to the file

    # the wavelet constants
    for key in wc._fields:
        hf_out['configuration']['wc'].create_dataset(key, data=getattr(wc, key))

    # lisa related constants
    for key in lc._fields:
        hf_out['configuration']['lc'].create_dataset(key, data=getattr(lc, key))

    # iterative fit related constants
    for key in ic._fields:
        hf_out['configuration']['ic'].create_dataset(key, data=getattr(ic, key))

    # archive the entire raw text of the configuration file to the hdf5 file as well
    with Path.open(config_filename) as file:
        file_content = file.read()

    hf_out['configuration']['config_text'].create_dataset(config_filename, data=file_content)

    hf_out.close()


def store_processed_gb_file(
    galaxy_dir,
    galaxy_file,
    wc: WDMWaveletConstants,
    lc: LISAConstants,
    ic: IterationConfig,
    nt_lim_snr: PixelTimeRange,
    bgd: BGDecomposition,
    period_list,
    n_bin_use,
    S_inst_m,
    S_final,
    stat_only,
    snrs_tot_upper,
    n_full_converged,
    argbinmap,
    faints_old,
    faints_cur,
    brights,
) -> None:
    filename_gb_init = get_preliminary_filename(galaxy_dir, ic.snr_thresh, wc.Nf, wc.Nt, wc.dt)
    filename_gb_common = get_common_noise_filename(galaxy_dir, ic.snr_thresh, wc)
    filename_out = get_processed_gb_filename(galaxy_dir, stat_only, ic.snr_thresh, wc, nt_lim_snr)

    hf_out = h5py.File(filename_out, 'w')
    hf_out.create_group('S')
    hf_out['S'].create_dataset('galactic_below', data=bgd.get_galactic_below_low(), compression='gzip')
    hf_out['S'].create_dataset('galactic_above', data=bgd.get_galactic_coadd_resolvable(), compression='gzip')
    hf_out['S'].create_dataset('galactic_undecided', data=bgd.get_galactic_coadd_undecided(), compression='gzip')
    hf_out['S'].create_dataset('period_list', data=period_list)

    hf_out['S'].create_dataset('n_bin_use', data=n_bin_use)
    hf_out['S'].create_dataset('S_stat_m', data=S_inst_m)
    hf_out['S'].create_dataset('snrs_tot_upper', data=snrs_tot_upper[n_full_converged], compression='gzip')
    hf_out['S'].create_dataset('argbinmap', data=argbinmap, compression='gzip')

    hf_out['S'].create_dataset('faints_old', data=faints_old, compression='gzip')
    hf_out['S'].create_dataset('faints_cur', data=faints_cur[n_full_converged], compression='gzip')

    hf_out['S'].create_dataset('brights', data=brights[n_full_converged], compression='gzip')
    hf_out['S'].create_dataset('S_final', data=S_final, compression='gzip')

    hf_out['S'].create_dataset('source_gb_file', data=get_galaxy_filename(galaxy_file, galaxy_dir))
    hf_out['S'].create_dataset('preliminary_gb_file', data=filename_gb_init)  # TODO these are redundant as constructed
    hf_out['S'].create_dataset('init_gb_file', data=filename_gb_init)
    hf_out['S'].create_dataset('common_gb_noise_file', data=filename_gb_common)

    hf_out.create_group('wc')
    for key in wc._fields:
        hf_out['wc'].create_dataset(key, data=getattr(wc, key))

    hf_out.create_group('lc')
    for key in lc._fields:
        hf_out['lc'].create_dataset(key, data=getattr(lc, key))

    hf_out.create_group('ic')
    for key in ic._fields:
        hf_out['ic'].create_dataset(key, data=getattr(ic, key))

    hf_out.close()
