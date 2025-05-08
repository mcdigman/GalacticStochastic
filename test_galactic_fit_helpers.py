"""unit tests for code in galactic_fit_helpers.py"""

import configparser

import pytest 

import numpy as np

import scipy.ndimage
from scipy.interpolate import InterpolatedUnivariateSpline


from galactic_fit_helpers import get_SAET_cyclostationary_mean

from wdm_config import get_wavelet_model

import global_const as gc

# we  can use the same baise noise for most things and modulate it as necessary
config = configparser.ConfigParser()
config.read('tests/galactic_fit_test_config1.ini')

wc = get_wavelet_model(config)

bg_base = np.random.normal(0.,1.,(wc.Nt, wc.Nf, wc.NC))

def test_stationary_mean_scramble_invariance():
    """SAET for stationary mean should be independent of time order of the samples; check this is true""" 

    config = configparser.ConfigParser()
    config.read('tests/galactic_fit_test_config1.ini')

    wc = get_wavelet_model(config)

    # get the background
    bg_here1 = 1000*bg_base.copy()

    # get scrambled time indices
    idx_sel2 = np.arange(0,wc.Nt)
    np.random.shuffle(idx_sel2)
    
    # get the same background with time indices scrambled
    bg_here2 = bg_here1[idx_sel2].copy()

    SAET_m = np.full((wc.Nf, wc.NC), 1.)

    # get both SAETs
    SAET_got1, _, _, _, _ = get_SAET_cyclostationary_mean(bg_here1, SAET_m, wc, smooth_lengthf=1., filter_periods=False, period_list=np.array([])) 
    SAET_got2, _, _, _, _ = get_SAET_cyclostationary_mean(bg_here2, SAET_m, wc, smooth_lengthf=1., filter_periods=False, period_list=np.array([])) 

    # check for expected invariance
    assert np.allclose(SAET_got1, SAET_got2, atol=1.e-14, rtol=1.e-13)


def get_noise_model_helper(model_name):
    """helper to get some useful noise model multipliers given a name"""
    if model_name == 'powerlaw1':
        f_mult = ((np.arange(0, wc.Nf)+1)/wc.Nf)**2
    elif model_name == 'powerlaw2':
        f_mult = ((np.arange(0, wc.Nf)+1)/wc.Nf)**-2
    elif model_name == 'white_faint':
        f_mult = np.full(wc.Nf, 1.e-3)
    elif model_name == 'white_equal':
        f_mult = np.full(wc.Nf, 1.)
    elif model_name == 'white_bright':
        f_mult = np.full(wc.Nf, 1000.)
    elif model_name == 'sin1':
        f_mult = 1.+0.5*np.sin(2*np.pi*np.arange(0, wc.Nf)/10.)
    elif model_name == 'sin2':
        f_mult = 1.+0.5*np.sin(2*np.pi*np.arange(0, wc.Nf)/100.)
    elif model_name == 'sin3':
        f_mult = 1.+0.5*np.sin(2*np.pi*np.arange(0, wc.Nf)/2.)
    elif model_name == 'dirac1':
        f_mult = np.full(wc.Nf, 1.e-3)
        f_mult[0] = 1.e4
    elif model_name == 'dirac2':
        f_mult = np.full(wc.Nf, 1.e-3)
        f_mult[wc.Nf//2] = 1.e4
    else:
        raise ValueError('unrecognized option for bg model')
    return f_mult

def stationary_mean_smooth_helper(bg_models, noise_models, smooth_lengthf, filter_periods):
    """helper to test stationary mean with several lengths of spectral smoothing can reproduce injected input spectrum""" 

    f_mult = np.full((wc.Nf, wc.NC), 0.)
    for itrc in range(0, wc.NC):
        f_mult[:, itrc] = get_noise_model_helper(bg_models[itrc])

    bg_here = bg_base.copy()

    for itrf in range(0, wc.Nf):
        bg_here[:, itrf, :] *= f_mult[itrf]

    SAET_m = np.full((wc.Nf, wc.NC), 0.)
    for itrc in range(0, wc.NC):
        SAET_m[:,itrc] = get_noise_model_helper(noise_models[itrc])

    SAET_got, _, _, _, _ = get_SAET_cyclostationary_mean(bg_here, SAET_m, wc, smooth_lengthf=smooth_lengthf, filter_periods=filter_periods, period_list=np.array([])) 

    # replicate expected smoothed multiplier
    f_mult_smooth = np.zeros_like(f_mult)
    interp_mult = 10
    n_f_interp = interp_mult*wc.Nf
    log_fs_interp = np.linspace(np.log10(wc.DF), np.log10(wc.DF*(wc.Nf-1)), n_f_interp)
    log_fs = np.log10(np.arange(1, wc.Nf)*wc.DF)
    for itrc in range(0, wc.NC):
        log_f_mult_interp = InterpolatedUnivariateSpline(log_fs, np.log10(f_mult[1:, itrc]**2+1.e-50), k=3, ext=2)(log_fs_interp)

        log_f_mult_smooth_interp = scipy.ndimage.gaussian_filter(log_f_mult_interp, smooth_lengthf*interp_mult)
        f_mult_smooth[:, itrc] = np.hstack([f_mult[0, itrc],np.sqrt(10**InterpolatedUnivariateSpline(log_fs_interp, log_f_mult_smooth_interp, k=3, ext=2)(log_fs)-1.e-50)])
    
    for itrc in range(0, wc.NC):
        # check that in constant model the resulting spectrum is indeed constant
        assert np.all(SAET_got[:, :, itrc] == SAET_got[0, :, itrc])

    #import matplotlib.pyplot as plt
    #plt.plot( 1.+f_mult_smooth[:]**2)
    #plt.semilogy(SAET_got[0, :, 0])
    #plt.show()

    # check that adding known noise produces known spectrum
    for itrc in range(0, 2):
        # check no rows outside ~5 sigma of being consistent with expected result
        for itrf in range(0, wc.Nf):
            got_loc = SAET_got[0, itrf, itrc]
            pred_loc = SAET_m[itrf, itrc]+f_mult_smooth[itrf, itrc]**2
            print(itrf, got_loc, SAET_m[itrf, itrc], f_mult_smooth[itrf, itrc]**2, pred_loc, (got_loc-pred_loc)/((f_mult_smooth[itrf, itrc]**2)/np.sqrt(wc.Nt)))
            assert np.allclose(got_loc, pred_loc, atol=5*(f_mult_smooth[itrf, itrc]**2)/np.sqrt(wc.Nt), rtol=5*(f_mult_smooth[itrf, itrc]**2)/np.sqrt(wc.Nt))

    # NOTE this is verifying the current behavior, although it probably isn't actually good behavior
    assert np.all(SAET_got[:,:,2] == SAET_m[:,2])

def nonstationary_mean_smooth_helper(bg_models, noise_models, smooth_lengthf, filter_periods, period_list, amp_list, phase_list):
    """helper to test stationary mean with several lengths of spectral smoothing can reproduce injected input spectrum""" 

    f_mult = np.full((wc.Nf, wc.NC), 0.)
    for itrc in range(0, wc.NC):
        f_mult[:, itrc] = get_noise_model_helper(bg_models[itrc])

    ts = np.arange(0, wc.Nt)*wc.DT
    t_mult = np.full((wc.Nt, wc.NC), 1.)
    for itrc in range(0, wc.NC):
        for itrp in range(0, period_list.size):
            t_mult[:, itrc] += amp_list[itrp]*np.cos(2*np.pi/gc.SECSYEAR*ts*period_list[itrp] - phase_list[itrp])


    bg_here = bg_base.copy()

    for itrf in range(0, wc.Nf):
        bg_here[:, itrf, :] *= f_mult[itrf]

    for itrt in range(0, wc.Nt):
        bg_here[itrt, :, :] *= t_mult[itrt,:]

    SAET_m = np.full((wc.Nf, wc.NC), 0.)
    for itrc in range(0, wc.NC):
        SAET_m[:,itrc] = get_noise_model_helper(noise_models[itrc])

    SAET_got, _, _, amp_got, angle_got = get_SAET_cyclostationary_mean(bg_here, SAET_m, wc, smooth_lengthf=smooth_lengthf, filter_periods=filter_periods, period_list=period_list) 
    print('amp 0',amp_got)
    # TODO why is this factor of 2 appearing?
    print('amp 1',2*amp_list)


    print('angle 0',angle_got)
    print('angle 1',phase_list)

    print((angle_got - phase_list + np.pi) % (2*np.pi) + phase_list - np.pi,)
    print(phase_list)

    for itrp in range(period_list.size):
        assert np.isclose(amp_got[itrp], 2*amp_list[itrp], atol=1.e-2, rtol=1.e-1)
        assert np.isclose((angle_got[itrp] - phase_list[itrp] + np.pi) % (2*np.pi) + phase_list[itrp] - np.pi, phase_list[itrp], atol=1.e-2/(amp_got[itrp]+0.001), rtol=1.e-1)


    # replicate expected smoothed multiplier
    f_mult_smooth = np.zeros_like(f_mult)
    interp_mult = 10
    n_f_interp = interp_mult*wc.Nf
    log_fs_interp = np.linspace(np.log10(wc.DF), np.log10(wc.DF*(wc.Nf-1)), n_f_interp)
    log_fs = np.log10(np.arange(1, wc.Nf)*wc.DF)
    for itrc in range(0, wc.NC):
        log_f_mult_interp = InterpolatedUnivariateSpline(log_fs, np.log10(f_mult[1:, itrc]**2+1.e-50), k=3, ext=2)(log_fs_interp)

        log_f_mult_smooth_interp = scipy.ndimage.gaussian_filter(log_f_mult_interp, smooth_lengthf*interp_mult)
        f_mult_smooth[:, itrc] = np.hstack([f_mult[0, itrc],np.sqrt(10**InterpolatedUnivariateSpline(log_fs_interp, log_f_mult_smooth_interp, k=3, ext=2)(log_fs)-1.e-50)])
    
    #import matplotlib.pyplot as plt
    #plt.plot( 1.+f_mult_smooth[:]**2)
    #plt.semilogy(SAET_got[0, :, 0])
    #plt.show()

    # check that adding known noise produces known spectrum
    for itrc in range(0, 2):
        # check no rows outside ~5 sigma of being consistent with expected result
        for itrf in range(0, wc.Nf):
            got_loc = SAET_got[:, itrf, itrc]
            bg_loc = f_mult_smooth[itrf, itrc]**2*t_mult[:,itrc]**2
            pred_loc = SAET_m[itrf, itrc] + bg_loc
            print(itrf, got_loc, SAET_m[itrf, itrc], f_mult_smooth[itrf, itrc]**2, pred_loc, (got_loc-pred_loc)/(bg_loc/np.sqrt(wc.Nt)))
            if not np.allclose(got_loc, pred_loc, atol=5*(f_mult_smooth[itrf, itrc]**2)/np.sqrt(wc.Nt), rtol=5*(bg_loc)/np.sqrt(wc.Nt)):
                import matplotlib.pyplot as plt
                plt.plot(got_loc)
                plt.plot(pred_loc)
                plt.show()
            assert np.allclose(got_loc, pred_loc, atol=5*(f_mult_smooth[itrf, itrc]**2)/np.sqrt(wc.Nt), rtol=5*(bg_loc)/np.sqrt(wc.Nt))

    # NOTE this is verifying the current behavior, although it probably isn't actually good behavior
    assert np.all(SAET_got[:,:,2] == SAET_m[:,2])

@pytest.mark.parametrize('bg_model', ['white_bright'])
@pytest.mark.parametrize('noise_model', ['white_equal'])
@pytest.mark.parametrize('itrk', [2])
@pytest.mark.parametrize('phase', [0.2])
@pytest.mark.parametrize('amp', [0., 0.1, 0.2, 0.4, 0.5, 0.8, 0.999])
def test_nonstationary_bg_power_bg_amp(bg_model, noise_model, itrk, phase, amp):
    """ test that smoothed time varying spectrum with different modulation amplitudes constant noise model produces expected results"""
    nonstationary_mean_smooth_helper([bg_model, bg_model, bg_model], [noise_model, noise_model, noise_model], 1., True, np.array([itrk]), np.array([amp]), np.array([phase]))

@pytest.mark.parametrize('bg_model', ['white_bright'])
@pytest.mark.parametrize('noise_model', ['white_equal'])
@pytest.mark.parametrize('itrk', [1, 2, 3, 4, 5, 16, 32, 64, 127, 128, 129])
@pytest.mark.parametrize('phase', [0.7])
def test_nonstationary_bg_power_harmonic(bg_model, noise_model, itrk, phase):
    """ test that smoothed time varying spectrum with constant noise model produces expected results with different known injected time variation"""
    nonstationary_mean_smooth_helper([bg_model, bg_model, bg_model], [noise_model, noise_model, noise_model], 1., True, np.array([itrk]), np.array([0.2]), np.array([phase]))

@pytest.mark.parametrize('bg_model', ['white_faint', 'white_equal','white_bright'])
@pytest.mark.parametrize('noise_model', ['white_equal'])
@pytest.mark.parametrize('itrk', [16])
@pytest.mark.parametrize('phase', [0.2])
def test_nonstationary_bg_power_bg_brightness(bg_model, noise_model, itrk, phase):
    """ test that smoothed time varying spectrum with different brightnesses constant noise model produces expected results"""
    nonstationary_mean_smooth_helper([bg_model, bg_model, bg_model], [noise_model, noise_model, noise_model], 1., True, np.array([itrk]), np.array([0.2]), np.array([phase]))


@pytest.mark.parametrize('bg_model', ['white_equal'])
@pytest.mark.parametrize('noise_model', ['white_faint', 'white_equal', 'white_bright'])
@pytest.mark.parametrize('itrk', [16])
@pytest.mark.parametrize('phase', [0.2])
def test_nonstationary_bg_power_noise_brightness(bg_model, noise_model, itrk, phase):
    """ test that smoothed time varying spectrum with different noises brightnesses constant background brightness produces expected results"""
    nonstationary_mean_smooth_helper([bg_model, bg_model, bg_model], [noise_model, noise_model, noise_model], 1., True, np.array([itrk]), np.array([0.2]), np.array([phase]))

@pytest.mark.parametrize('bg_model', ['white_bright'])
@pytest.mark.parametrize('noise_model', ['white_equal'])
@pytest.mark.parametrize('itrk', [1,3,16])
@pytest.mark.parametrize('phase', [0.,0.2,0.3,np.pi/2-0.01,np.pi/2., np.pi/2+0.01, np.pi-0.01,np.pi, np.pi+0.01, 3*np.pi/2-0.01,3*np.pi/2., 3*np.pi/2+0.01, 2*np.pi-0.01,2*np.pi, 2*np.pi+0.01])
def test_nonstationary_bg_power_phase(bg_model, noise_model, itrk, phase):
    """ test that smoothed time varying spectrum with constant noise model produces expected results with different phases"""
    nonstationary_mean_smooth_helper([bg_model, bg_model, bg_model], [noise_model, noise_model, noise_model], 1., True, np.array([itrk]), np.array([0.2]), np.array([phase]))

def test_different_bg_spectra():
    """ test that smoothed time invariant spectrum produce expected results if background spectrum differs between channels"""
    stationary_mean_smooth_helper(['powerlaw1', 'white_bright', 'powerlaw2'], ['white_equal', 'white_equal', 'white_equal'], 1., False)

def test_different_noise_spectra():
    """ test that smoothed time invariant spectrum produce expected results if noise spectrum differs between channels"""
    stationary_mean_smooth_helper(['white_equal', 'white_equal', 'white_equal'], ['powerlaw1', 'white_equal', 'powerlaw2'], 1., False)

def test_different_noise_bg_spectra():
    """ test that smoothed time invariant spectrum produce expected results if noise and background spectra differs between channels"""
    stationary_mean_smooth_helper(['powerlaw1', 'sin1', 'powerlaw2'], ['sin1', 'sin2', 'sin3'], 1., False)


@pytest.mark.parametrize('bg_model', ['white_faint', 'white_equal', 'white_bright'])
@pytest.mark.parametrize('noise_model', ['white_equal'])
def test_stationary_bg_power(bg_model, noise_model):
    """ test that smoothed time invariant spectrum with constant noise model produces expected results with different background noise powers"""
    stationary_mean_smooth_helper([bg_model, bg_model, bg_model], [noise_model, noise_model, noise_model], 1., False)

@pytest.mark.parametrize('bg_model', ['white_equal'])
@pytest.mark.parametrize('noise_model', ['white_faint', 'white_equal', 'white_bright'])
def test_stationary_noise_power(bg_model, noise_model):
    """ test that smoothed time invariant spectrum with constant noise model produces expected results with different instrument noise powers"""
    stationary_mean_smooth_helper([bg_model, bg_model, bg_model], [noise_model, noise_model, noise_model], 1., False)

@pytest.mark.parametrize('bg_model', ['sin1','sin2','sin3','powerlaw1', 'powerlaw2', 'white_equal', 'dirac1', 'dirac2'], False)
@pytest.mark.parametrize('noise_model', ['white_equal'])
@pytest.mark.parametrize('smooth_lengthf', [1.])
def test_stationary_filter_bg_(bg_model, noise_model, smooth_lengthf):
    """ test nothing unexpected happens if filter_periods is true but the noise model has no harmonics"""
    stationary_mean_smooth_helper([bg_model, bg_model, bg_model], [noise_model, noise_model, noise_model], smooth_lengthf, True)

@pytest.mark.parametrize('bg_model', ['sin1','sin2','sin3','powerlaw1', 'powerlaw2', 'white_equal', 'dirac1', 'dirac2'], False)
@pytest.mark.parametrize('noise_model', ['white_equal'])
@pytest.mark.parametrize('smooth_lengthf', [0.1, 1., 10., 100.])
def test_stationary_mean_bg_smooth(bg_model, noise_model, smooth_lengthf):
    """ test that smoothed time invariant spectrum with constant instrument noise model produces expected results with different smoothing lengths"""
    stationary_mean_smooth_helper([bg_model, bg_model, bg_model], [noise_model, noise_model, noise_model], smooth_lengthf, False)

@pytest.mark.parametrize('bg_model', ['sin2', 'sin3', 'powerlaw1', 'powerlaw2', 'white_equal', 'dirac1', 'dirac2'])
@pytest.mark.parametrize('noise_model', ['sin1','sin2','sin3','powerlaw1', 'powerlaw2', 'white_equal', 'dirac1', 'dirac2'])
@pytest.mark.parametrize('smooth_lengthf', [1.,100.])
def test_stationary_mean__instrument_smooth(bg_model, noise_model, smooth_lengthf):
    """ test that smoothed time invariant spectrum with different pairs of instrument and bg noise model produces expected results"""
    stationary_mean_smooth_helper([bg_model, bg_model, bg_model], [noise_model, noise_model, noise_model], smooth_lengthf, False)
