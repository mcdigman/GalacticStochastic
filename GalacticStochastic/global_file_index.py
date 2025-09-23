"""index for loading the current versions of files"""

from pathlib import Path

import h5py
import numpy as np

from GalacticStochastic.background_decomposition import BGDecomposition
from GalacticStochastic.inclusion_state_manager import BinaryInclusionState
from GalacticStochastic.iteration_config import IterationConfig
from GalacticStochastic.noise_manager import NoiseModelManager
from LisaWaveformTools.instrument_noise import instrument_noise_AET_wdm_m
from LisaWaveformTools.lisa_config import LISAConstants
from WaveletWaveforms.sparse_waveform_functions import PixelGenericRange
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


def get_common_noise_filename(config: dict, snr_thresh, wc: WDMWaveletConstants) -> str:
    galaxy_dir = config['files']['galaxy_dir']
    return (
        galaxy_dir
        + ('preprocessed_background=%.2f' % snr_thresh)
        + '_Nf='
        + str(wc.Nf)
        + '_Nt='
        + str(wc.Nt)
        + '_dt=%.2f.hdf5' % wc.dt
    )


def get_preliminary_filename(config: dict, snr_thresh: float, Nf: int, Nt: int, dt: float) -> str:
    galaxy_dir = config['files']['galaxy_dir']
    return (
        galaxy_dir
        + ('preprocessed_background=%.2f' % snr_thresh)
        + '_Nf='
        + str(Nf)
        + '_Nt='
        + str(Nt)
        + '_dt=%.2f.hdf5' % dt
    )


def get_galaxy_filename(config: dict) -> str:
    return config['files']['galaxy_dir'] + config['files']['galaxy_file']


def get_processed_gb_filename(config: dict, stat_only, snr_thresh, wc: WDMWaveletConstants, nt_lim_snr: PixelGenericRange) -> str:
    galaxy_dir = config['files']['galaxy_dir']
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
        + str(nt_lim_snr.nx_min)
        + '_nt_max='
        + str(nt_lim_snr.nx_max)
        + '.hdf5'
    )


def get_noise_common(config, snr_thresh, wc: WDMWaveletConstants):
    galaxy_dir = config['files']['galaxy_dir']
    filename_gb_common = get_common_noise_filename(galaxy_dir, snr_thresh, wc)
    hf_in = h5py.File(filename_gb_common, 'r')
    noise_realization_common = np.asarray(hf_in['S']['noise_realization'])

    hf_in.close()
    return noise_realization_common


def get_full_galactic_params(
    config: dict, *, fmin: float = 0.00001, fmax: float = 0.1, use_dgb: bool = True, use_igb: bool = True, use_vgb: bool = True,
):
    """Get the galaxy dataset binaries"""
    full_galactic_params_filename = get_galaxy_filename(config)
    filename = full_galactic_params_filename

    hf_in = h5py.File(filename, 'r')
    # dgb is detached galactic binaries, igb is interacting galactic binaries, vgb is verification
    freqs_dgb = np.asarray(hf_in['sky']['dgb']['cat']['Frequency'])
    if use_dgb:
        mask_dgb = (freqs_dgb > fmin) & (freqs_dgb < fmax)
    else:
        mask_dgb = np.zeros(freqs_dgb.shape, dtype=bool)

    n_dgb: int = int(np.sum(1 * mask_dgb))

    freqs_igb = np.asarray(hf_in['sky']['igb']['cat']['Frequency'])
    if use_igb:
        mask_igb = (freqs_igb > fmin) & (freqs_igb < fmax)
    else:
        mask_igb = np.zeros(freqs_igb.shape, dtype=bool)

    n_igb: int = int(np.sum(1 * mask_igb))

    freqs_vgb = np.asarray(hf_in['sky']['vgb']['cat']['Frequency'])
    if use_vgb:
        mask_vgb = (freqs_vgb > fmin) & (freqs_vgb < fmax)
    else:
        mask_vgb = np.zeros(freqs_vgb.shape, dtype=bool)

    n_vgb: int = int(np.sum(1 * mask_vgb))

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


def load_preliminary_galactic_file(config: dict, ic: IterationConfig, wc: WDMWaveletConstants, lc: LISAConstants):
    snr_thresh = ic.snr_thresh
    preliminary_gb_filename = get_preliminary_filename(config, snr_thresh, wc.Nf, wc.Nt, wc.dt)

    hf_in = h5py.File(preliminary_gb_filename, 'r')

    # check the galaxy filename matches
    gb_file_source = hf_in['galaxy']['binaries']['galaxy_file'][()].decode()
    full_galactic_params_filename = get_galaxy_filename(config)
    assert gb_file_source == full_galactic_params_filename

    galactic_below_in = np.asarray(hf_in['galaxy']['signal']['galactic_below'])

    snrs_tot_upper_in = np.asarray(hf_in['galaxy']['binaries']['snrs_tot_upper'])

    S_inst_m = np.asarray(hf_in['noise_model']['S_instrument'])

    # check input S makes sense, first value not checked as it may not be consistent
    S_inst_m_alt = instrument_noise_AET_wdm_m(lc, wc)
    assert np.allclose(S_inst_m[1:], S_inst_m_alt[1:], atol=1.0e-80, rtol=1.0e-13)

    hf_in.close()

    return galactic_below_in, snrs_tot_upper_in, S_inst_m, wc, lc


def load_processed_gb_file(
    config: dict, snr_thresh, wc: WDMWaveletConstants, nt_lim_snr: PixelGenericRange, *, stat_only,
):
    # TODO loading should produce a galactic background decomposition object
    filename_in = get_processed_gb_filename(config, stat_only, snr_thresh, wc, nt_lim_snr)
    hf_in = h5py.File(filename_in, 'r')

    galactic_below = np.asarray(hf_in['S']['galactic_below'])
    try:
        galactic_undecided = np.asarray(hf_in['S']['galactic_undecided'])
    except ImportError:
        galactic_undecided = np.asarray(hf_in['S']['galactic_bg'])

    argbinmap = np.asarray(hf_in['S']['argbinmap'])

    hf_in.close()

    return argbinmap, (galactic_below + galactic_undecided).reshape((wc.Nt, wc.Nf, galactic_below.shape[-1]))


def store_preliminary_gb_file(
    config_filename: str,
    config: dict,
    wc: WDMWaveletConstants,
    lc: LISAConstants,
    ic: IterationConfig,
    galactic_below,
    S_inst_m,
    snrs_tot_upper,
) -> None:
    filename_out = get_preliminary_filename(config, ic.snr_thresh, wc.Nf, wc.Nt, wc.dt)
    hf_out = h5py.File(filename_out, 'w')

    # store results related to the input galaxy
    hf_out.create_group('galaxy')

    hf_out['galaxy'].create_group('signal')
    hf_out['galaxy'].create_group('binaries')

    # the faint part of the galactic signal
    hf_out['galaxy']['signal'].create_dataset('galactic_below', data=galactic_below, compression='gzip')

    # the filename of the file with the galaxy in it
    hf_out['galaxy']['binaries'].create_dataset('galaxy_file', data=get_galaxy_filename(config))

    # the initial computed snr
    hf_out['galaxy']['binaries'].create_dataset('snrs_tot_upper', data=snrs_tot_upper[0], compression='gzip')

    # store parameters related to the noise model
    hf_out.create_group('noise_model')

    hf_out['noise_model'].create_dataset('S_instrument', data=S_inst_m)

    # store configuration parameters
    hf_out.create_group('configuration')
    hf_out['configuration'].create_group('_wc')
    hf_out['configuration'].create_group('ic')
    hf_out['configuration'].create_group('_lc')
    hf_out['configuration'].create_group('config_text')

    # store all the configuration objects to the file

    # the wavelet constants
    for key in wc._fields:
        hf_out['configuration']['_wc'].create_dataset(key, data=getattr(wc, key))

    # lisa related constants
    for key in lc._fields:
        hf_out['configuration']['_lc'].create_dataset(key, data=getattr(lc, key))

    # iterative fit related constants
    for key in ic._fields:
        hf_out['configuration']['ic'].create_dataset(key, data=getattr(ic, key))

    # archive the entire raw text of the configuration file to the hdf5 file as well
    with Path(config_filename).open('rb') as file:
        file_content = file.read()

    hf_out['configuration']['config_text'].create_dataset(config_filename, data=file_content)

    hf_out.close()


def store_processed_gb_file(
    config: dict,
    wc: WDMWaveletConstants,
    lc: LISAConstants,
    ic: IterationConfig,
    noise_manager: NoiseModelManager,
    bgd: BGDecomposition,
    n_full_converged,
    bis: BinaryInclusionState,
) -> None:

    nt_lim_snr = noise_manager.nt_lim_snr
    S_inst_m = noise_manager.S_inst_m
    S_final = noise_manager.S_final
    stat_only = noise_manager.stat_only

    filename_gb_init = get_preliminary_filename(config, ic.snr_thresh, wc.Nf, wc.Nt, wc.dt)
    filename_gb_common = get_common_noise_filename(config, ic.snr_thresh, wc)
    filename_out = get_processed_gb_filename(config, stat_only, ic.snr_thresh, wc, nt_lim_snr)

    period_list = ic.period_list

    n_bin_use = bis.n_bin_use
    snrs_tot_upper = bis.snrs_tot_upper
    argbinmap = bis.argbinmap
    faints_old = bis.faints_old
    faints_cur = bis.faints_cur
    brights = bis.brights

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

    hf_out['S'].create_dataset('source_gb_file', data=get_galaxy_filename(config))
    hf_out['S'].create_dataset('preliminary_gb_file', data=filename_gb_init)  # TODO these are redundant as constructed
    hf_out['S'].create_dataset('init_gb_file', data=filename_gb_init)
    hf_out['S'].create_dataset('common_gb_noise_file', data=filename_gb_common)

    hf_out.create_group('_wc')
    for key in wc._fields:
        hf_out['_wc'].create_dataset(key, data=getattr(wc, key))

    hf_out.create_group('_lc')
    for key in lc._fields:
        hf_out['_lc'].create_dataset(key, data=getattr(lc, key))

    hf_out.create_group('ic')
    for key in ic._fields:
        hf_out['ic'].create_dataset(key, data=getattr(ic, key))

    hf_out.close()
