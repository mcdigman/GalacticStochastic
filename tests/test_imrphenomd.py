"""test the IMRPhenomD module C 2021 Matthew Digman"""


import numpy as np
import PyIMRPhenomD.IMRPhenomD_const as imrc
import pytest
from numpy import gradient
from numpy.testing import assert_allclose
from numpy.typing import NDArray
from PyIMRPhenomD.IMRPhenomD import IMRPhenomDGenerateh22FDAmpPhase
from PyIMRPhenomD.IMRPhenomD_fring_helper import QNMData_a, QNMData_fdamp, QNMData_fring, fdamp_interp, fring_interp
from PyIMRPhenomD.IMRPhenomD_internals import AmpInsAnsatz, AmpIntAnsatz, AmpMRDAnsatz, AmpPhaseFDWaveform, DAmpInsAnsatz, DAmpIntAnsatz, DAmpMRDAnsatz, DDPhiInsAnsatzInt, DDPhiIntAnsatz, DDPhiMRD, DPhiInsAnsatzInt, DPhiIntAnsatz, DPhiMRD, FinalSpin0815, IMRPhenDAmplitude, IMRPhenDDAmplitude, IMRPhenDPhase, PhiInsAnsatzInt, PhiIntAnsatz, PhiMRDAnsatzInt, amp0Func, chiPN, fmaxCalc, fringdown
from scipy.interpolate import InterpolatedUnivariateSpline

from LisaWaveformTools.algebra_tools import stabilized_gradient_uniform_inplace
from LisaWaveformTools.binary_params_manager import BinaryIntrinsicParams, BinaryIntrinsicParamsManager
from LisaWaveformTools.imrphenomd_waveform import AmpInsAnsatzInplace, AmpIntAnsatzInplace, AmpMRDAnsatzInplace, AmpPhaseSeriesInsAnsatz, IMRPhenDAmplitudeFI, IMRPhenDAmpPhase_tc, IMRPhenDAmpPhaseFI, IMRPhenDPhaseFI, IMRPhenomDParams, PhiSeriesInsAnsatz, PhiSeriesIntAnsatz, PhiSeriesMRDAnsatz, get_imr_phenomd_params
from LisaWaveformTools.stationary_source_waveform import StationaryWaveformFreq
from LisaWaveformTools.taylorf2_helpers import TaylorF2_aligned_inplace
from WaveletWaveforms.sparse_waveform_functions import PixelGenericRange

# TODO still need tests for correct anchoring of reference frequency, phase, and time
# TODO test amplitude normalization


def setup_test_helper(m1_solar: float, m2_solar: float, chi1: float = 0.7534821857057837, chi2: float = 0.6215875279643664) -> tuple[BinaryIntrinsicParams, float]:

    distance: float = 56.00578366287752 * 1.0e9 * imrc.PC_SI
    m1_sec: float = m1_solar * imrc.MTSUN_SI
    m2_sec: float = m2_solar * imrc.MTSUN_SI
    Mt_sec: float = m1_sec + m2_sec

    tc: float = 2.496000e+07

    eta: float = m1_sec * m2_sec / Mt_sec**2
    Mc: float = eta**(3 / 5) * Mt_sec
    assert_allclose(eta, (Mc / Mt_sec)**(5 / 3))

    chis: float = (chi1 + chi2) / 2
    chia: float = (chi1 - chi2) / 2
    phic: float = 2.848705 / 2
    FI: float = 3.4956509169372e-05
    MfRef_in: float = FI * Mt_sec
    chi: float = chiPN(eta, chis, chia)
    chi_postnewtonian_norm: float = chi / (1. - 76. / 113. * eta)

    intrinsic_params_packed: NDArray[np.floating] = np.array([
        np.log(distance),  # Log luminosity distance in meters
        Mt_sec,  # Total mass in seconds
        Mc,  # Chirp mass in seconds
        FI,              # Initial frequency in Hz
        tc,             # Coalescence time in seconds
        phic,               # Phase at coalescence
        chi_postnewtonian_norm,               # Normalized postnewtonian spin parameter
        chia,               # Antisymmetric component of aligned spin
        0.0,               # Precessing spin
        0.0,               # Initial eccentricity
    ])

    intrinsic_params_manager: BinaryIntrinsicParamsManager = BinaryIntrinsicParamsManager(intrinsic_params_packed)
    intrinsic: BinaryIntrinsicParams = intrinsic_params_manager.params

    return intrinsic, MfRef_in


def get_waveform(waveform_method: str, intrinsic: BinaryIntrinsicParams, freq: NDArray[np.floating], Mfs: NDArray[np.floating], nf_lim: PixelGenericRange, MfRef_in: float) -> tuple[int, StationaryWaveformFreq]:
    amp0: float = 2. * np.sqrt(5. / (64. * np.pi)) * intrinsic.mass_total_detector_sec**2 / intrinsic.luminosity_distance_m  # * imrc.CLIGHT  # *imrc.MTSUN_SI**2#*imrc.MTSUN_SI**2*wc.CLIGHT
    finspin: float = FinalSpin0815(intrinsic.symmetric_mass_ratio, intrinsic.chi_s, intrinsic.chi_a)  # FinalSpin0815 - 0815 is like a version number
    Mf_ringdown, Mf_damp = fringdown(intrinsic.symmetric_mass_ratio, intrinsic.chi_s, intrinsic.chi_a, finspin)
    amp0_use: float = amp0 * amp0Func(intrinsic.symmetric_mass_ratio)

    ins_mode = 0
    dm = intrinsic.mass_total_detector_sec / (2 * np.pi)
    NF = nf_lim.nx_max - nf_lim.nx_min
    waveform = StationaryWaveformFreq(freq, np.zeros(NF), np.zeros(NF), np.zeros(NF), np.zeros(NF))
    if waveform_method == 'ampphasefull0':
        ins_mode = 1
        _itrFCut, _imr_params = IMRPhenDAmpPhaseFI(waveform, intrinsic, nf_lim, MfRef_in=MfRef_in, phi0=intrinsic.phase_c, amp_mult=amp0, imr_default_t=0)
    elif waveform_method == 'ampphasefull1':
        ins_mode = 1
        _itrFCut, _imr_params = IMRPhenDAmpPhase_tc(waveform, intrinsic, nf_lim, MfRef_in=MfRef_in, phi0=intrinsic.phase_c, amp_mult=amp0, imr_default_t=1, TTRef_in=intrinsic.time_c_sec)
    elif waveform_method == 'ampphasetc0':
        ins_mode = 1
        _itrFCut, _imr_params = IMRPhenDAmpPhase_tc(waveform, intrinsic, nf_lim, MfRef_in=MfRef_in, phi0=intrinsic.phase_c, amp_mult=amp0, imr_default_t=0, TTRef_in=intrinsic.time_c_sec)
    elif waveform_method == 'ampphasetc1':
        ins_mode = 1
        _itrFCut, _imr_params = IMRPhenDAmpPhaseFI(waveform, intrinsic, nf_lim, MfRef_in=MfRef_in, phi0=intrinsic.phase_c, amp_mult=amp0, imr_default_t=1)
    elif waveform_method == 'phasefull0':
        ins_mode = 1
        _itrFCut, _imr_params = IMRPhenDPhaseFI(waveform, intrinsic, nf_lim, MfRef_in=MfRef_in, phi0=intrinsic.phase_c, amp_mult=amp0, imr_default_t=0)
        IMRPhenDAmplitudeFI(waveform, intrinsic, nf_lim, MfRef_in=MfRef_in, phi0=intrinsic.phase_c, amp_mult=amp0, imr_default_t=0)
    elif waveform_method == 'phasefull1':
        ins_mode = 1
        _itrFCut, _imr_params = IMRPhenDPhaseFI(waveform, intrinsic, nf_lim, MfRef_in=MfRef_in, phi0=intrinsic.phase_c, amp_mult=amp0, imr_default_t=1)
        IMRPhenDAmplitudeFI(waveform, intrinsic, nf_lim, MfRef_in=MfRef_in, phi0=intrinsic.phase_c, amp_mult=amp0, imr_default_t=0)
    elif waveform_method == 'amp_phase_series_ins_ansatz0':
        ins_mode = 1
        imr_params = get_imr_phenomd_params(params=intrinsic, MfRef_in=MfRef_in, MfRef_max=np.nan, imr_default_t=0, TTRef_in=np.nan, phi0=intrinsic.phase_c, amp_mult=amp0, t_offset=0.)
        AmpPhaseSeriesInsAnsatz(waveform, imr_params, nf_lim)
    elif waveform_method == 'amp_phase_series_ins_ansatz1':
        ins_mode = 1
        imr_params = get_imr_phenomd_params(params=intrinsic, MfRef_in=MfRef_in, MfRef_max=np.nan, imr_default_t=1, TTRef_in=np.nan, phi0=intrinsic.phase_c, amp_mult=amp0, t_offset=0.)
        AmpPhaseSeriesInsAnsatz(waveform, imr_params, nf_lim)
    elif waveform_method == 'amp_phase_int_ansatz0':
        ins_mode = 0
        imr_params = get_imr_phenomd_params(params=intrinsic, MfRef_in=MfRef_in, MfRef_max=np.nan, imr_default_t=0, TTRef_in=np.nan, phi0=intrinsic.phase_c, amp_mult=amp0, t_offset=0.)
        PhiSeriesIntAnsatz(waveform, imr_params, nf_lim)
        AmpIntAnsatzInplace(waveform, imr_params, nf_lim)
    elif waveform_method == 'amp_phase_int_ansatz1':
        ins_mode = 0
        imr_params = get_imr_phenomd_params(params=intrinsic, MfRef_in=MfRef_in, MfRef_max=np.nan, imr_default_t=1, TTRef_in=np.nan, phi0=intrinsic.phase_c, amp_mult=amp0, t_offset=0.)
        PhiSeriesIntAnsatz(waveform, imr_params, nf_lim)
        AmpIntAnsatzInplace(waveform, imr_params, nf_lim)
    elif waveform_method == 'amp_phase_ins_ansatz0':
        ins_mode = 1
        imr_params = get_imr_phenomd_params(params=intrinsic, MfRef_in=MfRef_in, MfRef_max=np.nan, imr_default_t=0, TTRef_in=np.nan, phi0=intrinsic.phase_c, amp_mult=amp0, t_offset=0.)
        PhiSeriesInsAnsatz(waveform, imr_params, nf_lim)
        AmpInsAnsatzInplace(waveform, imr_params, nf_lim)
    elif waveform_method == 'amp_phase_ins_ansatz1':
        ins_mode = 1
        imr_params = get_imr_phenomd_params(params=intrinsic, MfRef_in=MfRef_in, MfRef_max=np.nan, imr_default_t=1, TTRef_in=np.nan, phi0=intrinsic.phase_c, amp_mult=amp0, t_offset=0.)
        PhiSeriesInsAnsatz(waveform, imr_params, nf_lim)
        AmpInsAnsatzInplace(waveform, imr_params, nf_lim)
    elif waveform_method == 'amp_phase_mrd_ansatz0':
        ins_mode = 0
        imr_params = get_imr_phenomd_params(params=intrinsic, MfRef_in=MfRef_in, MfRef_max=np.nan, imr_default_t=0, TTRef_in=np.nan, phi0=intrinsic.phase_c, amp_mult=amp0, t_offset=0.)
        PhiSeriesMRDAnsatz(waveform, imr_params, nf_lim)
        AmpMRDAnsatzInplace(waveform, imr_params, nf_lim)
    elif waveform_method == 'amp_phase_mrd_ansatz1':
        ins_mode = 0
        imr_params = get_imr_phenomd_params(params=intrinsic, MfRef_in=MfRef_in, MfRef_max=np.nan, imr_default_t=1, TTRef_in=np.nan, phi0=intrinsic.phase_c, amp_mult=amp0, t_offset=0.)
        PhiSeriesMRDAnsatz(waveform, imr_params, nf_lim)
        AmpMRDAnsatzInplace(waveform, imr_params, nf_lim)
    elif waveform_method == 'old_ins_ansatz':
        ins_mode = 1
        waveform.PF[:] = PhiInsAnsatzInt(Mfs, intrinsic.symmetric_mass_ratio, intrinsic.chi_s, intrinsic.chi_a, intrinsic.chi_postnewtonian)
        waveform.TF[:] = DPhiInsAnsatzInt(Mfs, intrinsic.symmetric_mass_ratio, intrinsic.chi_s, intrinsic.chi_a, intrinsic.chi_postnewtonian) * dm
        waveform.TFp[:] = DDPhiInsAnsatzInt(Mfs, intrinsic.symmetric_mass_ratio, intrinsic.chi_s, intrinsic.chi_a, intrinsic.chi_postnewtonian) * dm * intrinsic.mass_total_detector_sec
        waveform.AF[:] = AmpInsAnsatz(Mfs, intrinsic.symmetric_mass_ratio, intrinsic.chi_s, intrinsic.chi_a, intrinsic.chi_postnewtonian, amp0_use)
    elif waveform_method == 'old_int_ansatz':
        ins_mode = 0
        waveform.PF[:] = PhiIntAnsatz(Mfs, intrinsic.symmetric_mass_ratio, intrinsic.chi_postnewtonian)
        waveform.TF[:] = DPhiIntAnsatz(Mfs, intrinsic.symmetric_mass_ratio, intrinsic.chi_postnewtonian) * dm
        waveform.TFp[:] = DDPhiIntAnsatz(Mfs, intrinsic.symmetric_mass_ratio, intrinsic.chi_postnewtonian) * dm * intrinsic.mass_total_detector_sec
        waveform.AF[:] = AmpIntAnsatz(Mfs, Mf_ringdown, Mf_damp, intrinsic.symmetric_mass_ratio, intrinsic.chi_s, intrinsic.chi_a, intrinsic.chi_postnewtonian, amp0_use)
    elif waveform_method == 'old_mrd_ansatz':
        ins_mode = 0
        waveform.PF[:] = PhiMRDAnsatzInt(Mfs, Mf_ringdown, Mf_damp, intrinsic.symmetric_mass_ratio, intrinsic.chi_postnewtonian)
        waveform.TF[:] = DPhiMRD(Mfs, Mf_ringdown, Mf_damp, intrinsic.symmetric_mass_ratio, intrinsic.chi_postnewtonian) * dm
        waveform.TFp[:] = DDPhiMRD(Mfs, Mf_ringdown, Mf_damp, intrinsic.symmetric_mass_ratio, intrinsic.chi_postnewtonian) * dm * intrinsic.mass_total_detector_sec
        waveform.AF[:] = AmpMRDAnsatz(Mfs, Mf_ringdown, Mf_damp, intrinsic.symmetric_mass_ratio, intrinsic.chi_postnewtonian, amp0_use)
    elif waveform_method == 'old_phase':
        ins_mode = 1
        waveform.PF[:], waveform.TF[:], waveform.TFp[:], _t0, _MfRef, _itrFCut = IMRPhenDPhase(Mfs, intrinsic.mass_total_detector_sec, intrinsic.symmetric_mass_ratio, intrinsic.chi_s, intrinsic.chi_a, NF, MfRef_in, intrinsic.phase_c)
        waveform.AF[:] = IMRPhenDAmplitude(Mfs, intrinsic.symmetric_mass_ratio, intrinsic.chi_s, intrinsic.chi_a, NF, amp0)
    elif waveform_method == 'taylorf2':
        ins_mode = 1
        TaylorF2_aligned_inplace(waveform, intrinsic, nf_lim, amplitude_pn_mode=2, include_pn_ss3=0)
    else:
        msg = f'Unrecogized method {waveform_method}'
        raise ValueError(msg)
    return ins_mode, waveform


@pytest.mark.parametrize('q', [0.4781820536061116])
@pytest.mark.parametrize('waveform_method', ['taylorf2', 'old_phase', 'old_mrd_ansatz', 'old_int_ansatz', 'old_ins_ansatz', 'amp_phase_mrd_ansatz0', 'amp_phase_mrd_ansatz1', 'amp_phase_ins_ansatz0', 'amp_phase_ins_ansatz1', 'amp_phase_int_ansatz0', 'amp_phase_int_ansatz1', 'amp_phase_series_ins_ansatz0', 'amp_phase_series_ins_ansatz1', 'ampphasefull0', 'ampphasefull1', 'phasefull0', 'phasefull1', 'ampphasetc0', 'ampphasetc1'])
def test_imrphenomd_amplitude_derivative_consistency(q: float, waveform_method: str) -> None:
    """Check for internal consistency of amplitude derivatives."""
    m_tot_solar = 1242860.685 + 2599137.035
    m1_solar = m_tot_solar / (1 + q)
    m2_solar = m1_solar * q
    intrinsic, MfRef_in = setup_test_helper(m1_solar, m2_solar)
    amp0: float = 2. * np.sqrt(5. / (64. * np.pi)) * intrinsic.mass_total_detector_sec**2 / intrinsic.luminosity_distance_m  # * imrc.CLIGHT  # *imrc.MTSUN_SI**2#*imrc.MTSUN_SI**2*wc.CLIGHT
    amp0_use: float = amp0 * amp0Func(intrinsic.symmetric_mass_ratio)

    NF: int = 16384 * 10

    DF: float = 0.99 * imrc.f_CUT / intrinsic.mass_total_detector_sec / NF
    freq: NDArray[np.floating] = np.arange(1, NF + 1) * DF
    Mfs: NDArray[np.floating] = freq * intrinsic.mass_total_detector_sec

    nf_lim: PixelGenericRange = PixelGenericRange(0, NF, DF, DF)

    finspin: float = FinalSpin0815(intrinsic.symmetric_mass_ratio, intrinsic.chi_s, intrinsic.chi_a)  # FinalSpin0815 - 0815 is like a version number
    Mf_ringdown, Mf_damp = fringdown(intrinsic.symmetric_mass_ratio, intrinsic.chi_s, intrinsic.chi_a, finspin)

    itrlim_low_amp = 30
    itrlim_high_amp = NF - 1
    itr_anchor = NF - 1

    MfRef_in = Mfs[itr_anchor]

    _, waveform = get_waveform(waveform_method, intrinsic, freq, Mfs, nf_lim, MfRef_in)
    if waveform_method in ('old_phase', 'ampphasefull0', 'ampphasefull1', 'ampphasetc0', 'ampphasetc1', 'phasefull0', 'phasefull1'):
        DAmps = IMRPhenDDAmplitude(Mfs, intrinsic.symmetric_mass_ratio, intrinsic.chi_s, intrinsic.chi_a, NF, amp_mult=amp0)
    elif waveform_method in ('old_mrd_ansatz', 'amp_phase_mrd_ansatz0', 'amp_phase_mrd_ansatz1'):
        DAmps = DAmpMRDAnsatz(Mfs, Mf_ringdown, Mf_damp, intrinsic.symmetric_mass_ratio, intrinsic.chi_postnewtonian, amp0_use)
    elif waveform_method in ('old_int_ansatz', 'amp_phase_int_ansatz0', 'amp_phase_int_ansatz1'):
        DAmps = DAmpIntAnsatz(Mfs, Mf_ringdown, Mf_damp, intrinsic.symmetric_mass_ratio, intrinsic.chi_s, intrinsic.chi_a, intrinsic.chi_postnewtonian, amp0_use)
    elif waveform_method in ('old_ins_ansatz', 'taylorf2', 'amp_phase_series_ins_ansatz0', 'amp_phase_series_ins_ansatz1', 'amp_phase_ins_ansatz0', 'amp_phase_ins_ansatz1'):
        DAmps = DAmpInsAnsatz(Mfs, intrinsic.symmetric_mass_ratio, intrinsic.chi_s, intrinsic.chi_a, intrinsic.chi_postnewtonian, amp_mult=amp0_use)
    else:
        msg = f'Amplitude derivative availability not implemented for {waveform_method}'
        raise NotImplementedError(msg)

    DAmps_alt = np.gradient(waveform.AF * Mfs**(7. / 6.), Mfs)
    assert_allclose(DAmps[itrlim_low_amp:itrlim_high_amp], DAmps_alt[itrlim_low_amp:itrlim_high_amp], atol=8.e-30, rtol=8.e-5)


@pytest.mark.parametrize('q', [0.4781820536061116])
@pytest.mark.parametrize('waveform_method1', ['old_phase', 'ampphasefull0', 'ampphasefull1', 'ampphasetc0', 'ampphasetc1', 'phasefull0', 'phasefull1'])
@pytest.mark.parametrize('waveform_method2', ['amp_phase_int_ansatz1', 'amp_phase_mrd_ansatz1', 'amp_phase_int_ansatz0', 'amp_phase_mrd_ansatz0'])
def test_imrphenomd_full_int_mrd_transition(q: float, waveform_method1: str, waveform_method2: str) -> None:
    """Check for requisite continuity at transition between inspiral and intermediate regime"""
    # if waveform_method1 == waveform_method2:
    #    return
    m_tot_solar = 1242860.685 + 2599137.035
    m1_solar = m_tot_solar / (1 + q)
    m2_solar = m1_solar * q
    intrinsic, MfRef_in = setup_test_helper(m1_solar, m2_solar)

    NF: int = 16384 * 100

    DF: float = 0.99 * imrc.f_CUT / intrinsic.mass_total_detector_sec / NF
    freq: NDArray[np.floating] = np.arange(1, NF + 1) * DF
    Mfs: NDArray[np.floating] = freq * intrinsic.mass_total_detector_sec

    nf_lim: PixelGenericRange = PixelGenericRange(0, NF, DF, DF)

    _, waveform1 = get_waveform(waveform_method1, intrinsic, freq, Mfs, nf_lim, MfRef_in)
    _, waveform2 = get_waveform(waveform_method2, intrinsic, freq, Mfs, nf_lim, MfRef_in)

    finspin: float = FinalSpin0815(intrinsic.symmetric_mass_ratio, intrinsic.chi_s, intrinsic.chi_a)  # FinalSpin0815 - 0815 is like a version number
    Mf_ringdown, Mf_damp = fringdown(intrinsic.symmetric_mass_ratio, intrinsic.chi_s, intrinsic.chi_a, finspin)
    MfMRDJoinPhi: float = Mf_ringdown / 2.0
    MfMRDJoinAmp: float = fmaxCalc(Mf_ringdown, Mf_damp, intrinsic.symmetric_mass_ratio, intrinsic.chi_postnewtonian)

    # only check up to inspiral-merger transition
    itrfMRDPhi = int(np.searchsorted(freq, MfMRDJoinPhi / intrinsic.mass_total_detector_sec))
    itrfMRDAmp = int(np.searchsorted(freq, MfMRDJoinAmp / intrinsic.mass_total_detector_sec))
    itr_anchor = itrfMRDPhi - 1
    itrlim_low = itrfMRDPhi - 5
    itrlim_high = itrfMRDPhi + 5
    itrlim_low_amp = itrfMRDAmp - 5
    itrlim_high_amp = itrfMRDAmp + 5

    phi1_adj = waveform1.PF - 2 * np.pi * waveform1.TF[itr_anchor] * (freq - freq[itr_anchor]) - waveform1.PF[itr_anchor]
    phi2_adj = waveform2.PF - 2 * np.pi * waveform2.TF[itr_anchor] * (freq - freq[itr_anchor]) - waveform2.PF[itr_anchor]

    t1_adj = waveform1.TF - waveform1.TF[itr_anchor]
    t2_adj = waveform2.TF - waveform2.TF[itr_anchor]

    # anchor time and phase offsets, can test those are as expected separately
    assert_allclose(waveform1.TFp[itrlim_low:itrlim_high], waveform2.TFp[itrlim_low:itrlim_high], atol=1.e-100, rtol=1.4e-2)
    assert_allclose(t1_adj[itrlim_low:itrlim_high], t2_adj[itrlim_low:itrlim_high], atol=1.3e-4, rtol=1.e-4)
    assert_allclose(phi1_adj[itrlim_low:itrlim_high], phi2_adj[itrlim_low:itrlim_high], atol=2.e-10, rtol=1.e-16)
    assert_allclose(waveform1.AF[itrlim_low_amp:itrlim_high_amp], waveform2.AF[itrlim_low_amp:itrlim_high_amp], atol=1.e-100, rtol=1.e-9)


@pytest.mark.parametrize('q', [0.4781820536061116])
@pytest.mark.parametrize('waveform_method1', ['old_phase', 'ampphasefull0', 'ampphasefull1', 'ampphasetc0', 'ampphasetc1', 'phasefull0', 'phasefull1'])
@pytest.mark.parametrize('waveform_method2', ['amp_phase_ins_ansatz1', 'amp_phase_int_ansatz1', 'amp_phase_ins_ansatz0', 'amp_phase_int_ansatz0'])
def test_imrphenomd_full_ins_int_transition(q: float, waveform_method1: str, waveform_method2: str) -> None:
    """Check for requisite continuity at transition between inspiral and intermediate regime"""
    # if waveform_method1 == waveform_method2:
    #    return
    m_tot_solar = 1242860.685 + 2599137.035
    m1_solar = m_tot_solar / (1 + q)
    m2_solar = m1_solar * q
    intrinsic, MfRef_in = setup_test_helper(m1_solar, m2_solar)

    NF: int = 16384 * 100

    DF: float = 0.99 * imrc.f_CUT / intrinsic.mass_total_detector_sec / NF
    freq: NDArray[np.floating] = np.arange(1, NF + 1) * DF
    Mfs: NDArray[np.floating] = freq * intrinsic.mass_total_detector_sec

    nf_lim: PixelGenericRange = PixelGenericRange(0, NF, DF, DF)

    _, waveform1 = get_waveform(waveform_method1, intrinsic, freq, Mfs, nf_lim, MfRef_in)
    _, waveform2 = get_waveform(waveform_method2, intrinsic, freq, Mfs, nf_lim, MfRef_in)

    # only check up to inspiral-merger transition
    itrfIntPhi = int(np.searchsorted(freq, imrc.PHI_fJoin_INS / intrinsic.mass_total_detector_sec))
    itrfIntAmp = int(np.searchsorted(freq, imrc.AMP_fJoin_INS / intrinsic.mass_total_detector_sec))
    itr_anchor = itrfIntPhi - 1
    itrlim_low = itrfIntPhi - 5
    itrlim_high = itrfIntPhi + 5
    itrlim_low_amp = itrfIntAmp - 5
    itrlim_high_amp = itrfIntAmp + 5

    phi1_adj = waveform1.PF - 2 * np.pi * waveform1.TF[itr_anchor] * (freq - freq[itr_anchor]) - waveform1.PF[itr_anchor]
    phi2_adj = waveform2.PF - 2 * np.pi * waveform2.TF[itr_anchor] * (freq - freq[itr_anchor]) - waveform2.PF[itr_anchor]

    t1_adj = waveform1.TF - waveform1.TF[itr_anchor]
    t2_adj = waveform2.TF - waveform2.TF[itr_anchor]

    # anchor time and phase offsets, can test those are as expected separately
    assert_allclose(waveform1.TFp[itrlim_low:itrlim_high], waveform2.TFp[itrlim_low:itrlim_high], atol=1.e-100, rtol=1.e-1)
    assert_allclose(t1_adj[itrlim_low:itrlim_high], t2_adj[itrlim_low:itrlim_high], atol=4.e-2, rtol=1.e-4)
    assert_allclose(phi1_adj[itrlim_low:itrlim_high], phi2_adj[itrlim_low:itrlim_high], atol=3.e-9, rtol=1.e-16)
    assert_allclose(waveform1.AF[itrlim_low_amp:itrlim_high_amp], waveform2.AF[itrlim_low_amp:itrlim_high_amp], atol=1.e-100, rtol=1.e-9)


@pytest.mark.parametrize('q', [0.4781820536061116])
@pytest.mark.parametrize('waveform_method1', ['old_phase', 'ampphasefull0', 'ampphasefull1', 'ampphasetc0', 'ampphasetc1', 'phasefull0', 'phasefull1'])
@pytest.mark.parametrize('waveform_method2', ['amp_phase_mrd_ansatz1', 'amp_phase_mrd_ansatz0', 'old_mrd_ansatz'])
def test_imrphenomd_full_cross_mrd(q: float, waveform_method1: str, waveform_method2: str) -> None:
    """Various checks for cross consistency of full regime methods and mrd regime, eliminating phase and time absolute setpoints"""
    # if waveform_method1 == waveform_method2:
    #    return
    m_tot_solar = 1242860.685 + 2599137.035
    m1_solar = m_tot_solar / (1 + q)
    m2_solar = m1_solar * q
    intrinsic, MfRef_in = setup_test_helper(m1_solar, m2_solar)

    NF: int = 16384 * 10

    DF: float = 0.99 * imrc.f_CUT / intrinsic.mass_total_detector_sec / NF
    freq: NDArray[np.floating] = np.arange(1, NF + 1) * DF
    Mfs: NDArray[np.floating] = freq * intrinsic.mass_total_detector_sec

    nf_lim: PixelGenericRange = PixelGenericRange(0, NF, DF, DF)

    _, waveform1 = get_waveform(waveform_method1, intrinsic, freq, Mfs, nf_lim, MfRef_in)
    _, waveform2 = get_waveform(waveform_method2, intrinsic, freq, Mfs, nf_lim, MfRef_in)

    finspin: float = FinalSpin0815(intrinsic.symmetric_mass_ratio, intrinsic.chi_s, intrinsic.chi_a)  # FinalSpin0815 - 0815 is like a version number
    Mf_ringdown, Mf_damp = fringdown(intrinsic.symmetric_mass_ratio, intrinsic.chi_s, intrinsic.chi_a, finspin)
    MfMRDJoinPhi: float = Mf_ringdown / 2.0
    MfMRDJoinAmp: float = fmaxCalc(Mf_ringdown, Mf_damp, intrinsic.symmetric_mass_ratio, intrinsic.chi_postnewtonian)

    # only check up to inspiral-merger transition
    itrfMRDPhi = int(np.searchsorted(freq, MfMRDJoinPhi / intrinsic.mass_total_detector_sec))
    itrfMRDAmp = int(np.searchsorted(freq, MfMRDJoinAmp / intrinsic.mass_total_detector_sec))
    itr_anchor = NF - 1
    itrlim_low = itrfMRDPhi
    itrlim_high = NF - 1
    itrlim_low_amp = itrfMRDAmp
    itrlim_high_amp = NF - 1

    phi1_adj = waveform1.PF - 2 * np.pi * waveform1.TF[itr_anchor] * (freq - freq[itr_anchor]) - waveform1.PF[itr_anchor]
    phi2_adj = waveform2.PF - 2 * np.pi * waveform2.TF[itr_anchor] * (freq - freq[itr_anchor]) - waveform2.PF[itr_anchor]

    t1_adj = waveform1.TF - waveform1.TF[itr_anchor]
    t2_adj = waveform2.TF - waveform2.TF[itr_anchor]

    # plt.plot(phi1_adj[20:])
    # plt.plot(phi2_adj[20:])
    # plt.show()
    # plt.plot(t1_adj[20:itrlim_high])
    # plt.plot(t2_adj[20:itrlim_high])
    # plt.show()
    # plt.plot(t1_adj[20:itrlim_high]-t2_adj[20:itrlim_high])
    # plt.show()
    # plt.plot(waveform1.TFp)
    # plt.plot(waveform2.TFp)
    # plt.show()

    # anchor time and phase offsets, can test those are as expected separately
    assert_allclose(waveform1.TFp[itrlim_low:itrlim_high], waveform2.TFp[itrlim_low:itrlim_high], atol=1.e-100, rtol=1.e-6)
    assert_allclose(t1_adj[itrlim_low:itrlim_high], t2_adj[itrlim_low:itrlim_high], atol=1.e-10, rtol=1.e-6)
    assert_allclose(phi1_adj[itrlim_low:itrlim_high], phi2_adj[itrlim_low:itrlim_high], atol=1.e-9, rtol=1.e-6)
    assert_allclose(waveform1.AF[itrlim_low_amp:itrlim_high_amp], waveform2.AF[itrlim_low_amp:itrlim_high_amp], atol=1.e-100, rtol=1.e-6)


@pytest.mark.parametrize('q', [0.4781820536061116])
@pytest.mark.parametrize('waveform_method1', ['old_phase', 'ampphasefull0', 'ampphasefull1', 'ampphasetc0', 'ampphasetc1', 'phasefull0', 'phasefull1'])
@pytest.mark.parametrize('waveform_method2', ['amp_phase_int_ansatz1', 'amp_phase_int_ansatz0', 'old_int_ansatz'])
def test_imrphenomd_full_cross_int(q: float, waveform_method1: str, waveform_method2: str) -> None:
    """Various checks for cross consistency of full regime methods and intermediate regime, eliminating phase and time absolute setpoints"""
    # if waveform_method1 == waveform_method2:
    #    return
    m_tot_solar = 1242860.685 + 2599137.035
    m1_solar = m_tot_solar / (1 + q)
    m2_solar = m1_solar * q
    intrinsic, MfRef_in = setup_test_helper(m1_solar, m2_solar)

    NF: int = 16384 * 10

    DF: float = 0.99 * imrc.f_CUT / intrinsic.mass_total_detector_sec / NF
    freq: NDArray[np.floating] = np.arange(1, NF + 1) * DF
    Mfs: NDArray[np.floating] = freq * intrinsic.mass_total_detector_sec

    nf_lim: PixelGenericRange = PixelGenericRange(0, NF, DF, DF)

    _, waveform1 = get_waveform(waveform_method1, intrinsic, freq, Mfs, nf_lim, MfRef_in)
    _, waveform2 = get_waveform(waveform_method2, intrinsic, freq, Mfs, nf_lim, MfRef_in)

    finspin: float = FinalSpin0815(intrinsic.symmetric_mass_ratio, intrinsic.chi_s, intrinsic.chi_a)  # FinalSpin0815 - 0815 is like a version number
    Mf_ringdown, Mf_damp = fringdown(intrinsic.symmetric_mass_ratio, intrinsic.chi_s, intrinsic.chi_a, finspin)
    MfMRDJoinPhi: float = Mf_ringdown / 2.0
    MfMRDJoinAmp: float = fmaxCalc(Mf_ringdown, Mf_damp, intrinsic.symmetric_mass_ratio, intrinsic.chi_postnewtonian)

    # only check up to inspiral-merger transition
    itrfIntPhi = int(np.searchsorted(freq, imrc.PHI_fJoin_INS / intrinsic.mass_total_detector_sec))
    itrfIntAmp = int(np.searchsorted(freq, imrc.AMP_fJoin_INS / intrinsic.mass_total_detector_sec))
    itrfMRDPhi = int(np.searchsorted(freq, MfMRDJoinPhi / intrinsic.mass_total_detector_sec))
    itrfMRDAmp = int(np.searchsorted(freq, MfMRDJoinAmp / intrinsic.mass_total_detector_sec))
    itr_anchor = itrfMRDPhi - 1
    itrlim_low = itrfIntPhi
    itrlim_high = itrfMRDPhi
    itrlim_low_amp = itrfIntAmp
    itrlim_high_amp = itrfMRDAmp

    phi1_adj = waveform1.PF - 2 * np.pi * waveform1.TF[itr_anchor] * (freq - freq[itr_anchor]) - waveform1.PF[itr_anchor]
    phi2_adj = waveform2.PF - 2 * np.pi * waveform2.TF[itr_anchor] * (freq - freq[itr_anchor]) - waveform2.PF[itr_anchor]

    t1_adj = waveform1.TF - waveform1.TF[itr_anchor]
    t2_adj = waveform2.TF - waveform2.TF[itr_anchor]

    # plt.plot(phi1_adj[20:])
    # plt.plot(phi2_adj[20:])
    # plt.show()
    # plt.plot(t1_adj[20:itrlim_high])
    # plt.plot(t2_adj[20:itrlim_high])
    # plt.show()
    # plt.plot(t1_adj[20:itrlim_high]-t2_adj[20:itrlim_high])
    # plt.show()
    # plt.plot(waveform1.TFp)
    # plt.plot(waveform2.TFp)
    # plt.show()

    # anchor time and phase offsets, can test those are as expected separately
    assert_allclose(waveform1.TFp[itrlim_low:itrlim_high], waveform2.TFp[itrlim_low:itrlim_high], atol=1.e-100, rtol=1.e-6)
    assert_allclose(t1_adj[itrlim_low:itrlim_high], t2_adj[itrlim_low:itrlim_high], atol=1.e-10, rtol=1.e-6)
    assert_allclose(phi1_adj[itrlim_low:itrlim_high], phi2_adj[itrlim_low:itrlim_high], atol=1.e-9, rtol=1.e-6)
    assert_allclose(waveform1.AF[itrlim_low_amp:itrlim_high_amp], waveform2.AF[itrlim_low_amp:itrlim_high_amp], atol=1.e-100, rtol=1.e-6)


@pytest.mark.parametrize('q', [0.4781820536061116])
@pytest.mark.parametrize('waveform_method1', ['old_phase', 'ampphasefull0', 'ampphasefull1', 'ampphasetc0', 'ampphasetc1', 'phasefull0', 'phasefull1'])
@pytest.mark.parametrize('waveform_method2', ['amp_phase_ins_ansatz1', 'amp_phase_series_ins_ansatz1', 'amp_phase_ins_ansatz0', 'amp_phase_series_ins_ansatz0', 'old_ins_ansatz'])
def test_imrphenomd_full_cross_inspiral(q: float, waveform_method1: str, waveform_method2: str) -> None:
    """Various checks for cross consistency of full regime methods, eliminating phase and time absolute setpoints"""
    # if waveform_method1 == waveform_method2:
    #    return
    m_tot_solar = 1242860.685 + 2599137.035
    m1_solar = m_tot_solar / (1 + q)
    m2_solar = m1_solar * q
    intrinsic, MfRef_in = setup_test_helper(m1_solar, m2_solar)

    NF: int = 16384 * 10

    DF: float = 0.99 * imrc.f_CUT / intrinsic.mass_total_detector_sec / NF
    freq: NDArray[np.floating] = np.arange(1, NF + 1) * DF
    Mfs: NDArray[np.floating] = freq * intrinsic.mass_total_detector_sec

    nf_lim: PixelGenericRange = PixelGenericRange(0, NF, DF, DF)

    ins_mode1, waveform1 = get_waveform(waveform_method1, intrinsic, freq, Mfs, nf_lim, MfRef_in)
    ins_mode2, waveform2 = get_waveform(waveform_method2, intrinsic, freq, Mfs, nf_lim, MfRef_in)
    assert ins_mode1 == ins_mode2
    assert ins_mode1 == 1

    # only check up to inspiral-merger transition
    itrfIntPhi = int(np.searchsorted(freq, imrc.PHI_fJoin_INS / intrinsic.mass_total_detector_sec))
    itrfIntAmp = int(np.searchsorted(freq, imrc.AMP_fJoin_INS / intrinsic.mass_total_detector_sec))
    itr_anchor = itrfIntPhi - 1
    itrlim_low = 0
    itrlim_high = itrfIntPhi
    itrlim_low_amp = 0
    itrlim_high_amp = itrfIntAmp

    phi1_adj = waveform1.PF - 2 * np.pi * waveform1.TF[itr_anchor] * (freq - freq[itr_anchor]) - waveform1.PF[itr_anchor]
    phi2_adj = waveform2.PF - 2 * np.pi * waveform2.TF[itr_anchor] * (freq - freq[itr_anchor]) - waveform2.PF[itr_anchor]

    t1_adj = waveform1.TF - waveform1.TF[itr_anchor]
    t2_adj = waveform2.TF - waveform2.TF[itr_anchor]

    # plt.plot(phi1_adj[20:])
    # plt.plot(phi2_adj[20:])
    # plt.show()
    # plt.plot(t1_adj[20:itrlim_high])
    # plt.plot(t2_adj[20:itrlim_high])
    # plt.show()
    # plt.plot(t1_adj[20:itrlim_high]-t2_adj[20:itrlim_high])
    # plt.show()
    # plt.plot(waveform1.TFp)
    # plt.plot(waveform2.TFp)
    # plt.show()

    # anchor time and phase offsets, can test those are as expected separately
    assert_allclose(waveform1.TFp[itrlim_low:itrlim_high], waveform2.TFp[itrlim_low:itrlim_high], atol=1.e-100, rtol=1.e-6)
    assert_allclose(t1_adj[itrlim_low:itrlim_high], t2_adj[itrlim_low:itrlim_high], atol=1.e-10, rtol=1.e-6)
    assert_allclose(phi1_adj[itrlim_low:itrlim_high], phi2_adj[itrlim_low:itrlim_high], atol=1.e-9, rtol=1.e-6)
    assert_allclose(waveform1.AF[itrlim_low_amp:itrlim_high_amp], waveform2.AF[itrlim_low_amp:itrlim_high_amp], atol=1.e-100, rtol=1.e-6)


@pytest.mark.parametrize('q', [0.4781820536061116])
@pytest.mark.parametrize('waveform_method1', ['taylorf2'])
@pytest.mark.parametrize('waveform_method2', ['amp_phase_ins_ansatz1', 'amp_phase_series_ins_ansatz1', 'amp_phase_ins_ansatz0', 'amp_phase_series_ins_ansatz0', 'old_ins_ansatz', 'old_phase', 'ampphasefull0', 'ampphasefull1', 'ampphasetc0', 'ampphasetc1', 'phasefull0', 'phasefull1'])
def test_taylorf2_cross_inspiral(q: float, waveform_method1: str, waveform_method2: str) -> None:
    """Various checks for cross consistency of taylorf2 and inspiral methods, eliminating phase and time absolute setpoints"""
    # if waveform_method1 == waveform_method2:
    #    return
    m_tot_solar = 1242860.685 + 2599137.035
    m1_solar = m_tot_solar / (1 + q)
    m2_solar = m1_solar * q
    # intrinsic, MfRef_in = setup_test_helper(m1_solar, m2_solar, chi1=0., chi2=0.)
    intrinsic, MfRef_in = setup_test_helper(m1_solar, m2_solar)

    NF: int = 16384 * 10

    DF: float = 0.99 * imrc.f_CUT / intrinsic.mass_total_detector_sec / NF
    freq: NDArray[np.floating] = np.arange(1, NF + 1) * DF
    Mfs: NDArray[np.floating] = freq * intrinsic.mass_total_detector_sec

    nf_lim: PixelGenericRange = PixelGenericRange(0, NF, DF, DF)

    ins_mode1, waveform1 = get_waveform(waveform_method1, intrinsic, freq, Mfs, nf_lim, MfRef_in)
    ins_mode2, waveform2 = get_waveform(waveform_method2, intrinsic, freq, Mfs, nf_lim, MfRef_in)
    assert ins_mode1 == ins_mode2
    assert ins_mode1 == 1

    # only check up to inspiral-merger transition
    itrfIntPhi = int(np.searchsorted(freq, imrc.PHI_fJoin_INS / intrinsic.mass_total_detector_sec))
    itrfIntAmp = int(np.searchsorted(freq, imrc.AMP_fJoin_INS / intrinsic.mass_total_detector_sec))
    itr_anchor = itrfIntPhi - 1
    itrlim_low = 0
    itrlim_high = itrfIntPhi
    itrlim_low_amp = 0
    itrlim_high_amp = itrfIntAmp

    phi1_adj = waveform1.PF - 2 * np.pi * waveform1.TF[itr_anchor] * (freq - freq[itr_anchor]) - waveform1.PF[itr_anchor]
    phi2_adj = waveform2.PF - 2 * np.pi * waveform2.TF[itr_anchor] * (freq - freq[itr_anchor]) - waveform2.PF[itr_anchor]

    t1_adj = waveform1.TF - waveform1.TF[itr_anchor]
    t2_adj = waveform2.TF - waveform2.TF[itr_anchor]

    # import matplotlib.pyplot as plt
    # plt.loglog(np.abs(phi1_adj[20:itrlim_high]))
    # plt.plot(np.abs(phi2_adj[20:itrlim_high]))
    # plt.show()
    # plt.loglog(np.abs(t1_adj[20:itrlim_high]))
    # plt.plot(np.abs(t2_adj[20:itrlim_high]))
    # plt.show()
    # plt.loglog(np.abs(waveform1.TFp[20:itrlim_high]))
    # plt.loglog(np.abs(waveform2.TFp[20:itrlim_high]))
    # plt.show()
    # plt.loglog(np.abs(waveform1.AF*imrc.CLIGHT))
    # plt.loglog(np.abs(waveform2.AF))
    # plt.show()

    atol_t = 1.e-3 * float(np.abs(t1_adj[itr_anchor - 10] - t1_adj[itr_anchor - 1]))
    atol_tp = 1.e-3 * float(np.abs(waveform1.TFp[itr_anchor - 10] - waveform1.TFp[itr_anchor - 1]))

    # anchor time and phase offsets, can test those are as expected separately
    # amplitude model is exactly the same currently
    assert_allclose(waveform1.AF[itrlim_low_amp:itrlim_high_amp], waveform2.AF[itrlim_low_amp:itrlim_high_amp], atol=1.e-100, rtol=1.e-10)
    # other models are slightly different currently
    assert_allclose(waveform1.TFp[itrlim_low:itrlim_high], waveform2.TFp[itrlim_low:itrlim_high], atol=atol_tp, rtol=1.3e-1)
    assert_allclose(t1_adj[itrlim_low:itrlim_high], t2_adj[itrlim_low:itrlim_high], atol=atol_t, rtol=1.3e-1)
    assert_allclose(phi1_adj[itrlim_low:itrlim_high], phi2_adj[itrlim_low:itrlim_high], atol=1.e-9, rtol=1.3e-1)


@pytest.mark.parametrize('q', [0.4781820536061116])
@pytest.mark.parametrize('waveform_method1', ['old_phase', 'ampphasefull0', 'ampphasefull1', 'ampphasetc0', 'ampphasetc1', 'phasefull0', 'phasefull1'])
@pytest.mark.parametrize('waveform_method2', ['old_phase', 'ampphasefull0', 'ampphasefull1', 'ampphasetc0', 'ampphasetc1', 'phasefull0', 'phasefull1'])
def test_imrphenomd_full_cross(q: float, waveform_method1: str, waveform_method2: str) -> None:
    """Various checks for cross consistency of full regime methods, eliminating phase and time absolute setpoints"""
    # if waveform_method1 == waveform_method2:
    #    return
    m_tot_solar = 1242860.685 + 2599137.035
    m1_solar = m_tot_solar / (1 + q)
    m2_solar = m1_solar * q
    intrinsic, MfRef_in = setup_test_helper(m1_solar, m2_solar)

    NF: int = 16384 * 10

    DF: float = 0.99 * imrc.f_CUT / intrinsic.mass_total_detector_sec / NF
    freq: NDArray[np.floating] = np.arange(1, NF + 1) * DF
    Mfs: NDArray[np.floating] = freq * intrinsic.mass_total_detector_sec

    nf_lim: PixelGenericRange = PixelGenericRange(0, NF, DF, DF)

    ins_mode1, waveform1 = get_waveform(waveform_method1, intrinsic, freq, Mfs, nf_lim, MfRef_in)
    ins_mode2, waveform2 = get_waveform(waveform_method2, intrinsic, freq, Mfs, nf_lim, MfRef_in)
    assert ins_mode1 == ins_mode2
    assert ins_mode1 == 1

    phi1_adj = waveform1.PF - 2 * np.pi * waveform1.TF[-1] * (freq - freq[-1]) - waveform1.PF[-1]
    phi2_adj = waveform2.PF - 2 * np.pi * waveform2.TF[-1] * (freq - freq[-1]) - waveform2.PF[-1]

    t1_adj = waveform1.TF - waveform1.TF[-1]
    t2_adj = waveform2.TF - waveform2.TF[-1]

    # import matplotlib.pyplot as plt
    # plt.plot(phi1_adj[20:])
    # plt.plot(phi2_adj[20:])
    # plt.show()
    # plt.plot(t1_adj)
    # plt.plot(t2_adj)
    # plt.show()
    # plt.plot(waveform1.TFp)
    # plt.plot(waveform2.TFp)
    # plt.show()

    # anchor time and phase offsets, can test those are as expected separately
    assert_allclose(t1_adj, t2_adj, atol=1.e-10, rtol=1.e-6)
    assert_allclose(phi1_adj, phi2_adj, atol=1.e-9, rtol=1.e-6)
    assert_allclose(waveform1.TFp, waveform2.TFp, atol=1.e-100, rtol=1.e-6)
    assert_allclose(waveform1.AF, waveform2.AF, atol=1.e-100, rtol=1.e-6)


@pytest.mark.parametrize('q', [0.4781820536061116])
@pytest.mark.parametrize('waveform_method1', ['amp_phase_mrd_ansatz1', 'amp_phase_mrd_ansatz0', 'old_mrd_ansatz'])
@pytest.mark.parametrize('waveform_method2', ['amp_phase_mrd_ansatz1', 'amp_phase_mrd_ansatz0', 'old_mrd_ansatz'])
def test_imrphenomd_mrd_only_cross(q: float, waveform_method1: str, waveform_method2: str) -> None:
    """Various checks for cross consistency of mrd regime methods, eliminating phase and time absolute setpoints"""
    # if waveform_method1 == waveform_method2:
    #    return
    m_tot_solar = 1242860.685 + 2599137.035
    m1_solar = m_tot_solar / (1 + q)
    m2_solar = m1_solar * q
    intrinsic, MfRef_in = setup_test_helper(m1_solar, m2_solar)

    NF: int = 16384 * 10

    DF: float = 0.99 * imrc.f_CUT / intrinsic.mass_total_detector_sec / NF
    freq: NDArray[np.floating] = np.arange(1, NF + 1) * DF
    Mfs: NDArray[np.floating] = freq * intrinsic.mass_total_detector_sec

    nf_lim: PixelGenericRange = PixelGenericRange(0, NF, DF, DF)

    ins_mode1, waveform1 = get_waveform(waveform_method1, intrinsic, freq, Mfs, nf_lim, MfRef_in)
    ins_mode2, waveform2 = get_waveform(waveform_method2, intrinsic, freq, Mfs, nf_lim, MfRef_in)
    assert ins_mode1 == ins_mode2
    assert ins_mode1 == 0

    phi1_adj = waveform1.PF - 2 * np.pi * waveform1.TF[-1] * (freq - freq[-1]) - waveform1.PF[-1]
    phi2_adj = waveform2.PF - 2 * np.pi * waveform2.TF[-1] * (freq - freq[-1]) - waveform2.PF[-1]

    t1_adj = waveform1.TF - waveform1.TF[-1]
    t2_adj = waveform2.TF - waveform2.TF[-1]

    # import matplotlib.pyplot as plt
    # plt.plot(phi1_adj[20:])
    # plt.plot(phi2_adj[20:])
    # plt.show()
    # plt.plot(t1_adj)
    # plt.plot(t2_adj)
    # plt.show()
    # plt.plot(waveform1.TFp)
    # plt.plot(waveform2.TFp)
    # plt.show()

    # anchor time and phase offsets, can test those are as expected separately
    assert_allclose(t1_adj, t2_adj, atol=1.e-10, rtol=1.e-6)
    assert_allclose(phi1_adj, phi2_adj, atol=1.e-9, rtol=1.e-6)
    assert_allclose(waveform1.TFp, waveform2.TFp, atol=1.e-100, rtol=1.e-6)
    assert_allclose(waveform1.AF, waveform2.AF, atol=1.e-100, rtol=1.e-6)


@pytest.mark.parametrize('q', [0.4781820536061116])
@pytest.mark.parametrize('waveform_method1', ['amp_phase_int_ansatz1', 'amp_phase_int_ansatz0', 'old_int_ansatz'])
@pytest.mark.parametrize('waveform_method2', ['amp_phase_int_ansatz1', 'amp_phase_int_ansatz0', 'old_int_ansatz'])
def test_imrphenomd_int_only_cross(q: float, waveform_method1: str, waveform_method2: str) -> None:
    """Various checks for cross consistency of intermediate regime methods, eliminating phase and time absolute setpoints"""
    # if waveform_method1 == waveform_method2:
    #    return
    m_tot_solar = 1242860.685 + 2599137.035
    m1_solar = m_tot_solar / (1 + q)
    m2_solar = m1_solar * q
    intrinsic, MfRef_in = setup_test_helper(m1_solar, m2_solar)

    NF: int = 16384 * 10

    DF: float = 0.99 * imrc.f_CUT / intrinsic.mass_total_detector_sec / NF
    freq: NDArray[np.floating] = np.arange(1, NF + 1) * DF
    Mfs: NDArray[np.floating] = freq * intrinsic.mass_total_detector_sec

    nf_lim: PixelGenericRange = PixelGenericRange(0, NF, DF, DF)

    ins_mode1, waveform1 = get_waveform(waveform_method1, intrinsic, freq, Mfs, nf_lim, MfRef_in)
    ins_mode2, waveform2 = get_waveform(waveform_method2, intrinsic, freq, Mfs, nf_lim, MfRef_in)
    assert ins_mode1 == ins_mode2
    assert ins_mode1 == 0

    phi1_adj = waveform1.PF - 2 * np.pi * waveform1.TF[-1] * (freq - freq[-1]) - waveform1.PF[-1]
    phi2_adj = waveform2.PF - 2 * np.pi * waveform2.TF[-1] * (freq - freq[-1]) - waveform2.PF[-1]

    t1_adj = waveform1.TF - waveform1.TF[-1]
    t2_adj = waveform2.TF - waveform2.TF[-1]

    # import matplotlib.pyplot as plt
    # plt.plot(phi1_adj[20:])
    # plt.plot(phi2_adj[20:])
    # plt.show()
    # plt.plot(t1_adj)
    # plt.plot(t2_adj)
    # plt.show()
    # plt.plot(waveform1.TFp)
    # plt.plot(waveform2.TFp)
    # plt.show()

    # anchor time and phase offsets, can test those are as expected separately
    assert_allclose(t1_adj, t2_adj, atol=1.e-10, rtol=1.e-6)
    assert_allclose(phi1_adj, phi2_adj, atol=1.e-9, rtol=1.e-6)
    assert_allclose(waveform1.TFp, waveform2.TFp, atol=1.e-100, rtol=1.e-6)
    assert_allclose(waveform1.AF, waveform2.AF, atol=1.e-100, rtol=1.e-6)


@pytest.mark.parametrize('q', [0.4781820536061116])
@pytest.mark.parametrize('waveform_method1', ['amp_phase_ins_ansatz1', 'amp_phase_series_ins_ansatz1', 'amp_phase_ins_ansatz0', 'amp_phase_series_ins_ansatz0', 'old_ins_ansatz'])
@pytest.mark.parametrize('waveform_method2', ['amp_phase_ins_ansatz1', 'amp_phase_series_ins_ansatz1', 'amp_phase_ins_ansatz0', 'amp_phase_series_ins_ansatz0', 'old_ins_ansatz'])
def test_imrphenomd_ins_only_cross(q: float, waveform_method1: str, waveform_method2: str) -> None:
    """Various checks for cross consistency of inspiral methods, eliminating phase and time absolute setpoints"""
    # if waveform_method1 == waveform_method2:
    #    return
    m_tot_solar = 1242860.685 + 2599137.035
    m1_solar = m_tot_solar / (1 + q)
    m2_solar = m1_solar * q
    intrinsic, MfRef_in = setup_test_helper(m1_solar, m2_solar)

    NF: int = 16384 * 10

    DF: float = 0.99 * imrc.f_CUT / intrinsic.mass_total_detector_sec / NF
    freq: NDArray[np.floating] = np.arange(1, NF + 1) * DF
    Mfs: NDArray[np.floating] = freq * intrinsic.mass_total_detector_sec

    nf_lim: PixelGenericRange = PixelGenericRange(0, NF, DF, DF)

    ins_mode1, waveform1 = get_waveform(waveform_method1, intrinsic, freq, Mfs, nf_lim, MfRef_in)
    ins_mode2, waveform2 = get_waveform(waveform_method2, intrinsic, freq, Mfs, nf_lim, MfRef_in)
    assert ins_mode1 == ins_mode2
    assert ins_mode1 == 1

    phi1_adj = waveform1.PF - 2 * np.pi * waveform1.TF[-1] * (freq - freq[-1]) - waveform1.PF[-1]
    phi2_adj = waveform2.PF - 2 * np.pi * waveform2.TF[-1] * (freq - freq[-1]) - waveform2.PF[-1]

    t1_adj = waveform1.TF - waveform1.TF[-1]
    t2_adj = waveform2.TF - waveform2.TF[-1]

    # import matplotlib.pyplot as plt
    # plt.plot(phi1_adj[20:])
    # plt.plot(phi2_adj[20:])
    # plt.show()
    # plt.plot(t1_adj)
    # plt.plot(t2_adj)
    # plt.show()
    # plt.plot(waveform1.TFp)
    # plt.plot(waveform2.TFp)
    # plt.show()

    # anchor time and phase offsets, can test those are as expected separately
    assert_allclose(t1_adj, t2_adj, atol=1.e-10, rtol=1.e-6)
    assert_allclose(phi1_adj, phi2_adj, atol=1.e-9, rtol=1.e-6)
    assert_allclose(waveform1.TFp, waveform2.TFp, atol=1.e-100, rtol=1.e-6)
    assert_allclose(waveform1.AF, waveform2.AF, atol=1.e-100, rtol=1.e-6)


@pytest.mark.parametrize('q', [0.4781820536061116])
@pytest.mark.parametrize('waveform_method', ['old_phase', 'old_mrd_ansatz', 'amp_phase_mrd_ansatz0', 'amp_phase_mrd_ansatz1', 'ampphasefull0', 'ampphasefull1', 'phasefull0', 'phasefull1', 'ampphasetc0', 'ampphasetc1'])
def test_imrphenomd_derivative_consistency_mrd(q: float, waveform_method: str) -> None:
    """Various checks for internal consistency of phase derivatives, restricted to mrd"""
    m_tot_solar = 1242860.685 + 2599137.035
    m1_solar = m_tot_solar / (1 + q)
    m2_solar = m1_solar * q
    intrinsic, MfRef_in = setup_test_helper(m1_solar, m2_solar)

    NF: int = 16384 * 10

    DF: float = 0.99 * imrc.f_CUT / intrinsic.mass_total_detector_sec / NF
    freq: NDArray[np.floating] = np.arange(1, NF + 1) * DF
    Mfs: NDArray[np.floating] = freq * intrinsic.mass_total_detector_sec

    nf_lim: PixelGenericRange = PixelGenericRange(0, NF, DF, DF)

    finspin: float = FinalSpin0815(intrinsic.symmetric_mass_ratio, intrinsic.chi_s, intrinsic.chi_a)  # FinalSpin0815 - 0815 is like a version number
    Mf_ringdown, _ = fringdown(intrinsic.symmetric_mass_ratio, intrinsic.chi_s, intrinsic.chi_a, finspin)
    MfMRDJoinPhi: float = Mf_ringdown / 2.0

    # only check up to inspiral-merger transition
    itrfMRDPhi = int(np.searchsorted(freq, MfMRDJoinPhi / intrinsic.mass_total_detector_sec))
    itr_anchor = itrfMRDPhi
    itrlim_low = itrfMRDPhi
    itrlim_high = NF

    MfRef_in = Mfs[itr_anchor]

    _, waveform = get_waveform(waveform_method, intrinsic, freq, Mfs, nf_lim, MfRef_in)

    dm = intrinsic.mass_total_detector_sec / (2 * np.pi)

    phi_adj = waveform.PF - 2 * np.pi * waveform.TF[itr_anchor] * (freq - freq[itr_anchor]) - waveform.PF[itr_anchor]

    t_adj = waveform.TF - waveform.TF[itr_anchor]

    t_adj_alt = gradient(phi_adj, Mfs, edge_order=2) * dm

    tp_adj_alt = gradient(t_adj, freq, edge_order=2)
    tp_alt = gradient(waveform.TF, freq, edge_order=2)
    # import matplotlib.pyplot as plt
    # plt.plot((t_adj[itrlim_low:itrlim_high]-t_adj_alt[itrlim_low:itrlim_high])/t_adj[itrlim_low:itrlim_high])
    # plt.plot(t_adj_alt[itrlim_low:itrlim_high])
    # plt.show()

    # plt.plot(waveform.TFp[itrlim_low:itrlim_high] - tp_adj_alt[itrlim_low:itrlim_high])
    # plt.show()

    atol_t = 1.e-3 * float(np.abs(t_adj_alt[itr_anchor + 10] - t_adj_alt[itr_anchor - 1]))
    atol_tp = 1.e-3 * float(np.abs(tp_adj_alt[itr_anchor + 10] - tp_adj_alt[itr_anchor - 1]))
    assert_allclose(t_adj[itrlim_low:itrlim_high] - t_adj[itr_anchor], t_adj_alt[itrlim_low:itrlim_high] - t_adj_alt[itr_anchor], atol=atol_t, rtol=5.e-5)
    assert_allclose(waveform.TFp[itrlim_low:itrlim_high], tp_adj_alt[itrlim_low:itrlim_high], atol=atol_tp, rtol=3.e-2)
    assert_allclose(waveform.TFp[itrlim_low:itrlim_high], tp_alt[itrlim_low:itrlim_high], atol=atol_tp, rtol=3.e-2)


@pytest.mark.parametrize('q', [0.4781820536061116])
@pytest.mark.parametrize('waveform_method', ['old_phase', 'old_int_ansatz', 'amp_phase_int_ansatz0', 'amp_phase_int_ansatz1', 'ampphasefull0', 'ampphasefull1', 'phasefull0', 'phasefull1', 'ampphasetc0', 'ampphasetc1'])
def test_imrphenomd_derivative_consistency_int(q: float, waveform_method: str) -> None:
    """Various checks for internal consistency of phase derivatives, restricted to int"""
    m_tot_solar = 1242860.685 + 2599137.035
    m1_solar = m_tot_solar / (1 + q)
    m2_solar = m1_solar * q
    intrinsic, MfRef_in = setup_test_helper(m1_solar, m2_solar)

    NF: int = 16384 * 10

    DF: float = 0.99 * imrc.f_CUT / intrinsic.mass_total_detector_sec / NF
    freq: NDArray[np.floating] = np.arange(1, NF + 1) * DF
    Mfs: NDArray[np.floating] = freq * intrinsic.mass_total_detector_sec

    nf_lim: PixelGenericRange = PixelGenericRange(0, NF, DF, DF)

    finspin: float = FinalSpin0815(intrinsic.symmetric_mass_ratio, intrinsic.chi_s, intrinsic.chi_a)  # FinalSpin0815 - 0815 is like a version number
    Mf_ringdown, _ = fringdown(intrinsic.symmetric_mass_ratio, intrinsic.chi_s, intrinsic.chi_a, finspin)
    MfMRDJoinPhi: float = Mf_ringdown / 2.0

    # only check up to inspiral-merger transition
    itrfMRDPhi = int(np.searchsorted(freq, MfMRDJoinPhi / intrinsic.mass_total_detector_sec))
    itrfIntPhi = int(np.searchsorted(freq, imrc.PHI_fJoin_INS / intrinsic.mass_total_detector_sec))
    itr_anchor = itrfMRDPhi - 1
    itrlim_low = itrfIntPhi
    itrlim_high = itrfMRDPhi

    MfRef_in = Mfs[itr_anchor]

    _, waveform = get_waveform(waveform_method, intrinsic, freq, Mfs, nf_lim, MfRef_in)

    dm = intrinsic.mass_total_detector_sec / (2 * np.pi)

    phi_adj = waveform.PF - 2 * np.pi * waveform.TF[itr_anchor] * (freq - freq[itr_anchor]) - waveform.PF[itr_anchor]

    t_adj = waveform.TF - waveform.TF[itr_anchor]

    t_adj_alt = gradient(phi_adj, Mfs, edge_order=2) * dm

    tp_adj_alt = gradient(t_adj, freq, edge_order=2)
    tp_alt = gradient(waveform.TF, freq, edge_order=2)
    # import matplotlib.pyplot as plt
    # plt.plot((t_adj[itrlim_low:itrlim_high]-t_adj_alt[itrlim_low:itrlim_high])/t_adj[itrlim_low:itrlim_high])
    # plt.plot(t_adj_alt[itrlim_low:itrlim_high])
    # plt.show()

    # plt.plot(waveform.TFp[itrlim_low:itrlim_high] - tp_adj_alt[itrlim_low:itrlim_high])
    # plt.show()

    atol_t = 1.e-3 * float(np.abs(t_adj_alt[itr_anchor - 10] - t_adj_alt[itr_anchor - 1]))
    atol_tp = 1.e-3 * float(np.abs(tp_adj_alt[itr_anchor - 10] - tp_adj_alt[itr_anchor - 1]))
    assert_allclose(t_adj[itrlim_low:itrlim_high] - t_adj[itr_anchor], t_adj_alt[itrlim_low:itrlim_high] - t_adj_alt[itr_anchor], atol=atol_t, rtol=5.e-5)
    assert_allclose(waveform.TFp[itrlim_low:itrlim_high], tp_adj_alt[itrlim_low:itrlim_high], atol=atol_tp, rtol=3.e-2)
    assert_allclose(waveform.TFp[itrlim_low:itrlim_high], tp_alt[itrlim_low:itrlim_high], atol=atol_tp, rtol=3.e-2)


@pytest.mark.parametrize('q', [0.4781820536061116])
@pytest.mark.parametrize('waveform_method', ['taylorf2', 'old_phase', 'old_ins_ansatz', 'amp_phase_ins_ansatz0', 'amp_phase_ins_ansatz1', 'amp_phase_series_ins_ansatz0', 'amp_phase_series_ins_ansatz1', 'ampphasefull0', 'ampphasefull1', 'phasefull0', 'phasefull1', 'ampphasetc0', 'ampphasetc1'])
def test_imrphenomd_derivative_consistency_ins(q: float, waveform_method: str) -> None:
    """Various checks for internal consistency of phase derivatives, restricted to inspiral"""
    m_tot_solar = 1242860.685 + 2599137.035
    m1_solar = m_tot_solar / (1 + q)
    m2_solar = m1_solar * q
    intrinsic, MfRef_in = setup_test_helper(m1_solar, m2_solar)

    NF: int = 16384 * 10

    DF: float = 0.99 * imrc.f_CUT / intrinsic.mass_total_detector_sec / NF
    freq: NDArray[np.floating] = np.arange(1, NF + 1) * DF
    Mfs: NDArray[np.floating] = freq * intrinsic.mass_total_detector_sec

    nf_lim: PixelGenericRange = PixelGenericRange(0, NF, DF, DF)

    # only check up to inspiral-merger transition
    itrfIntPhi = int(np.searchsorted(freq, imrc.PHI_fJoin_INS / intrinsic.mass_total_detector_sec))
    itr_anchor = itrfIntPhi - 1
    itrlim_low = 200
    itrlim_high = itrfIntPhi

    MfRef_in = Mfs[itr_anchor]

    _, waveform = get_waveform(waveform_method, intrinsic, freq, Mfs, nf_lim, MfRef_in)

    dm = intrinsic.mass_total_detector_sec / (2 * np.pi)

    phi_adj = waveform.PF - 2 * np.pi * waveform.TF[itr_anchor] * (freq - freq[itr_anchor]) - waveform.PF[itr_anchor]

    t_adj = waveform.TF - waveform.TF[itr_anchor]

    t_adj_alt = gradient(phi_adj, Mfs, edge_order=2) * dm

    tp_adj_alt = gradient(t_adj, freq, edge_order=2)
    tp_alt = gradient(waveform.TF, freq, edge_order=2)
    # import matplotlib.pyplot as plt
    # plt.plot((t_adj[itrlim_low:itrlim_high]-t_adj_alt[itrlim_low:itrlim_high])/t_adj[itrlim_low:itrlim_high])
    # plt.plot(t_adj_alt[itrlim_low:itrlim_high])
    # plt.show()

    # plt.plot(waveform.TFp[itrlim_low:itrlim_high] - tp_adj_alt[itrlim_low:itrlim_high])
    # plt.show()

    atol_t = 1.e-2 * float(np.abs(t_adj_alt[itr_anchor - 10] - t_adj_alt[itr_anchor - 1]))
    atol_tp = 1.e-3 * float(np.abs(tp_adj_alt[itr_anchor - 10] - tp_adj_alt[itr_anchor - 1]))
    assert_allclose(t_adj[itrlim_low:itrlim_high] - t_adj[itr_anchor], t_adj_alt[itrlim_low:itrlim_high] - t_adj_alt[itr_anchor], atol=atol_t, rtol=5.e-5)
    assert_allclose(waveform.TFp[itrlim_low:itrlim_high], tp_adj_alt[itrlim_low:itrlim_high], atol=atol_tp, rtol=3.e-2)
    assert_allclose(waveform.TFp[itrlim_low:itrlim_high], tp_alt[itrlim_low:itrlim_high], atol=atol_tp, rtol=3.e-2)


@pytest.mark.parametrize('q', [0.4781820536061116, 0.999])
@pytest.mark.parametrize('waveform_method', ['taylorf2', 'old_phase', 'old_mrd_ansatz', 'old_int_ansatz', 'old_ins_ansatz', 'amp_phase_mrd_ansatz0', 'amp_phase_mrd_ansatz1', 'amp_phase_ins_ansatz0', 'amp_phase_ins_ansatz1', 'amp_phase_int_ansatz0', 'amp_phase_int_ansatz1', 'amp_phase_series_ins_ansatz0', 'amp_phase_series_ins_ansatz1', 'ampphasefull0', 'ampphasefull1', 'phasefull0', 'phasefull1', 'ampphasetc0', 'ampphasetc1'])
def test_imrphenomd_derivative_consistency(q: float, waveform_method: str) -> None:
    """Various checks for internal consistency of phase derivatives"""
    m_tot_solar = 1242860.685 + 2599137.035
    m1_solar = m_tot_solar / (1 + q)
    m2_solar = m1_solar * q
    intrinsic, MfRef_in = setup_test_helper(m1_solar, m2_solar)

    NF: int = 16384 * 10

    DF: float = 0.99 * imrc.f_CUT / intrinsic.mass_total_detector_sec / NF
    freq: NDArray[np.floating] = np.arange(1, NF + 1) * DF
    Mfs: NDArray[np.floating] = freq * intrinsic.mass_total_detector_sec

    nf_lim: PixelGenericRange = PixelGenericRange(0, NF, DF, DF)

    ins_mode, waveform = get_waveform(waveform_method, intrinsic, freq, Mfs, nf_lim, MfRef_in)

    dm = intrinsic.mass_total_detector_sec / (2 * np.pi)

    time_alt = gradient(waveform.PF, Mfs, edge_order=2) * dm
    time_alt2 = InterpolatedUnivariateSpline(Mfs, waveform.PF * dm, k=3, ext=2).derivative(1)(Mfs)
    timep_alt = gradient(waveform.TF, Mfs, edge_order=2) * intrinsic.mass_total_detector_sec
    timep_alt2 = InterpolatedUnivariateSpline(Mfs, waveform.TF, k=3, ext=2).derivative(1)(Mfs) * intrinsic.mass_total_detector_sec
    # import matplotlib.pyplot as plt
    # plt.plot(waveform.TFp[NF // 4:]-timep_alt2[NF//4:])
    # plt.plot(waveform.TFp[NF // 4:]-timep_alt[NF//4:])
    # plt.show()

    # plt.plot(waveform.TF[NF // 4:]-time_alt2[NF//4:])
    # plt.plot(waveform.TF[NF // 4:]-time_alt[NF//4:])
    # plt.show()

    tp_atol = min(float(np.std(waveform.TFp[NF // 2:] - timep_alt2[NF // 2:])), float(np.std(waveform.TFp[NF // 2:] - timep_alt[NF // 2:])), 0.1)

    # check differences/integral
    assert_allclose(np.diff(waveform.PF)[100:], ((waveform.TF[1:] + waveform.TF[:-1:]) / 2. * np.diff(Mfs) / dm)[100:], atol=2.e-4, rtol=2.e-4)
    assert_allclose(np.diff(waveform.TF)[100:], ((waveform.TFp[1:] + waveform.TFp[:-1:]) / 2. * np.diff(Mfs) / intrinsic.mass_total_detector_sec)[100:], atol=5.e-8, rtol=5.e-3)

    # check phase derivatives
    assert_allclose(waveform.TF[20:], time_alt[20:], atol=1.e-6, rtol=2.7e-2)
    assert_allclose(waveform.TF[20:], time_alt2[20:], atol=1.e-5, rtol=2.4e-5)  # worse absolute accuracy at low frequency
    assert_allclose(waveform.TF[0.1 * waveform.TF[-1] < waveform.TF], time_alt2[0.1 * waveform.TF[-1] < waveform.TF], atol=1.e-10, rtol=2.e-6)  # more concerned about relative accuracy at high frequency
    assert_allclose(waveform.TF[0.1 * waveform.TF[-1] < waveform.TF], time_alt[0.1 * waveform.TF[-1] < waveform.TF], atol=1.e-10, rtol=3.e-4)
    assert_allclose(waveform.TF[20:] - waveform.TF[10], time_alt2[20:] - waveform.TF[10], atol=1.e-10, rtol=2.e-6)  # absorb absolute differences
    assert_allclose(waveform.TF[20:] - waveform.TF[10], time_alt[20:] - waveform.TF[10], atol=1.e-10, rtol=9.e-4)  # absorb absolute differences

    # check time derivatives
    assert_allclose(waveform.TFp[20:], timep_alt[20:], atol=1.e-6, rtol=6.e-2)
    assert_allclose(waveform.TFp[20:], timep_alt2[20:], atol=tp_atol, rtol=3.e-2)
    assert_allclose(waveform.TFp[NF // 2:], timep_alt2[NF // 2:], atol=10. * tp_atol, rtol=3.e-6)
    assert_allclose(waveform.TFp[NF // 2:], timep_alt[NF // 2:], atol=4. * tp_atol, rtol=3.e-4)
    assert np.abs(np.mean(waveform.TFp[NF // 2:] - timep_alt2[NF // 2:])) < 1.e-4
    # assert_allclose(waveform.TFp[NF // 2:], timep_alt2[NF // 2:], atol=1.e-10, rtol=3.e-6)
    # assert_allclose(waveform.TFp[NF // 2:], timep_alt[NF // 2:], atol=1.e-10, rtol=3.e-4)

    # use knowledge of low-frequency behavior
    if ins_mode == 1:
        time_alt3 = dm / Mfs ** (5 / 3) * (InterpolatedUnivariateSpline(Mfs, waveform.PF * Mfs ** (5 / 3), k=3, ext=2).derivative(1)(Mfs) - waveform.PF * Mfs ** (2 / 3) * (5 / 3))
        timep_alt3 = intrinsic.mass_total_detector_sec / Mfs ** (8 / 3) * (InterpolatedUnivariateSpline(Mfs, waveform.TF * Mfs ** (8 / 3), k=3, ext=2).derivative(1)(Mfs) - waveform.TF * Mfs ** (5 / 3) * (8 / 3))
        # assert np.sum(~np.isclose(waveform.TFp, timep_alt3, atol=1.e-10, rtol=1.e-4)) < 15
        assert np.sum(~np.isclose(waveform.TFp, timep_alt3, atol=1.e-10, rtol=2.e-2)) < 5
        assert_allclose(waveform.TFp, timep_alt3, atol=1.e-10, rtol=1.2e-1)
        assert_allclose(waveform.TF, time_alt3, atol=2.e-6, rtol=8.e-4)


@pytest.mark.parametrize('q', [0.4781820536061116, 0.999])
def test_imrphenomd_derivative_consistency2(q: float) -> None:
    """Checks for the consistency of derivatives with a finer DF grid"""
    m_tot_solar = 1242860.685 + 2599137.035
    m1_solar = m_tot_solar / (1 + q)
    m2_solar = m1_solar * q
    intrinsic, MfRef_in = setup_test_helper(m1_solar, m2_solar)

    amp0: float = 2. * np.sqrt(5. / (64. * np.pi)) * intrinsic.mass_total_detector_sec**2 / intrinsic.luminosity_distance_m  # * imrc.CLIGHT  # *imrc.MTSUN_SI**2#*imrc.MTSUN_SI**2*wc.CLIGHT

    NF: int = 16384 * 100

    freq: NDArray[np.floating] = np.linspace(1.e-4, 0.99, NF) * imrc.f_CUT / intrinsic.mass_total_detector_sec
    DF: float = float(freq[1] - freq[0])
    Mfs: NDArray[np.floating] = freq * intrinsic.mass_total_detector_sec

    nf_lim: PixelGenericRange = PixelGenericRange(0, NF, DF, DF)

    waveform = StationaryWaveformFreq(freq, np.zeros(NF), np.zeros(NF), np.zeros(NF), np.zeros(NF))
    _itrFCut, _imr_params = IMRPhenDAmpPhaseFI(waveform, intrinsic, nf_lim, MfRef_in=MfRef_in, phi0=intrinsic.phase_c, amp_mult=amp0, imr_default_t=1)

    dm = intrinsic.mass_total_detector_sec / (2 * np.pi)

    time_alt = gradient(waveform.PF, Mfs, edge_order=2) * dm
    time_alt2 = InterpolatedUnivariateSpline(Mfs, waveform.PF * dm, k=3, ext=2).derivative(1)(Mfs)
    timep_alt = gradient(waveform.TF, Mfs, edge_order=2) * intrinsic.mass_total_detector_sec
    timep_alt2 = InterpolatedUnivariateSpline(Mfs, waveform.TF * intrinsic.mass_total_detector_sec, k=3, ext=2).derivative(1)(Mfs)

    # check phase derivatives
    assert_allclose(waveform.TF[20:], time_alt[20:], atol=8.e-5, rtol=1.3e-2)
    assert_allclose(waveform.TF[20:], time_alt2[20:], atol=8.e-5, rtol=1.e-5)  # worse absolute accuracy at low frequency
    assert_allclose(waveform.TF[0.1 * waveform.TF[-1] < waveform.TF], time_alt2[0.1 * waveform.TF[-1] < waveform.TF], atol=8.e-5, rtol=2.e-6)  # more concerned about relative accuracy at high frequency
    assert_allclose(waveform.TF[0.1 * waveform.TF[-1] < waveform.TF], time_alt[0.1 * waveform.TF[-1] < waveform.TF], atol=8.e-5, rtol=3.e-5)
    assert_allclose(waveform.TF[20:] - waveform.TF[10], time_alt2[20:] - waveform.TF[10], atol=8.e-5, rtol=2.e-6)  # absorb absolute differences
    assert_allclose(waveform.TF[20:] - waveform.TF[10], time_alt[20:] - waveform.TF[10], atol=8.e-5, rtol=9.e-4)  # absorb absolute differences

    # check time derivatives
    assert_allclose(waveform.TFp[20:], timep_alt[20:], atol=5.e-4, rtol=5.e-2)
    assert_allclose(waveform.TFp[20:], timep_alt2[20:], atol=5.e-4, rtol=5.e-2)
    assert_allclose(waveform.TFp[NF // 2:], timep_alt2[NF // 2:], atol=5.e-4, rtol=3.e-6)
    assert_allclose(waveform.TFp[NF // 2:], timep_alt[NF // 2:], atol=5.e-4, rtol=3.e-4)

    # use knowledge of low-frequency behavior
    time_alt3 = dm / Mfs ** (5 / 3) * (InterpolatedUnivariateSpline(Mfs, waveform.PF * Mfs ** (5 / 3), k=3, ext=2).derivative(1)(Mfs) - waveform.PF * Mfs ** (2 / 3) * (5 / 3))
    timep_alt3 = intrinsic.mass_total_detector_sec / Mfs ** (8 / 3) * (InterpolatedUnivariateSpline(Mfs, waveform.TF * Mfs ** (8 / 3), k=3, ext=2).derivative(1)(Mfs) - waveform.TF * Mfs ** (5 / 3) * (8 / 3))
    assert np.sum(~np.isclose(waveform.TFp, timep_alt3, atol=1.e-10, rtol=1.e-4)) < 15
    assert_allclose(waveform.TFp, timep_alt3, atol=1.e-10, rtol=5.e-2)
    assert_allclose(waveform.TF, time_alt3, atol=6.e-5, rtol=8.e-4)


@pytest.mark.parametrize('q', [0.4781820536061116, 0.999])
def test_imrphenomd_internal_consistency(q: float) -> None:
    """Various checks for internal consistency of imrphenomd"""
    m_tot_solar = 1242860.685 + 2599137.035
    m1_solar = m_tot_solar / (1 + q)
    m2_solar = m1_solar * q
    intrinsic, MfRef_in = setup_test_helper(m1_solar, m2_solar)

    amp0: float = 2. * np.sqrt(5. / (64. * np.pi)) * intrinsic.mass_total_detector_sec**2 / intrinsic.luminosity_distance_m  # * imrc.CLIGHT  # *imrc.MTSUN_SI**2#*imrc.MTSUN_SI**2*wc.CLIGHT
    finspin: float = FinalSpin0815(intrinsic.symmetric_mass_ratio, intrinsic.chi_s, intrinsic.chi_a)  # FinalSpin0815 - 0815 is like a version number
    Mf_ringdown, Mf_damp = fringdown(intrinsic.symmetric_mass_ratio, intrinsic.chi_s, intrinsic.chi_a, finspin)
    amp0_use: float = amp0 * amp0Func(intrinsic.symmetric_mass_ratio)

    NF: int = 16384 * 10

    DF: float = 0.99 * imrc.f_CUT / intrinsic.mass_total_detector_sec / NF
    freq: NDArray[np.floating] = np.arange(1, NF + 1) * DF
    Mfs: NDArray[np.floating] = freq * intrinsic.mass_total_detector_sec

    nf_lim: PixelGenericRange = PixelGenericRange(0, NF, DF, DF)

    waveform_FI = StationaryWaveformFreq(freq, np.zeros(NF), np.zeros(NF), np.zeros(NF), np.zeros(NF))
    waveform_FI2 = StationaryWaveformFreq(freq, np.zeros(NF), np.zeros(NF), np.zeros(NF), np.zeros(NF))
    waveform_FI3 = StationaryWaveformFreq(freq, np.zeros(NF), np.zeros(NF), np.zeros(NF), np.zeros(NF))
    waveform_FI4 = StationaryWaveformFreq(freq, np.zeros(NF), np.zeros(NF), np.zeros(NF), np.zeros(NF))
    waveform_FI6 = StationaryWaveformFreq(freq, np.zeros(NF), np.zeros(NF), np.zeros(NF), np.zeros(NF))

    waveform1 = StationaryWaveformFreq(freq, np.zeros(NF), np.zeros(NF), np.zeros(NF), np.zeros(NF))
    waveform2 = StationaryWaveformFreq(freq, np.zeros(NF), np.zeros(NF), np.zeros(NF), np.zeros(NF))
    waveform3 = StationaryWaveformFreq(freq, np.zeros(NF), np.zeros(NF), np.zeros(NF), np.zeros(NF))
    waveform4 = StationaryWaveformFreq(freq, np.zeros(NF), np.zeros(NF), np.zeros(NF), np.zeros(NF))
    waveform5 = StationaryWaveformFreq(freq, np.zeros(NF), np.zeros(NF), np.zeros(NF), np.zeros(NF))
    waveform6 = StationaryWaveformFreq(freq, np.zeros(NF), np.zeros(NF), np.zeros(NF), np.zeros(NF))
    waveform_t = StationaryWaveformFreq(freq, np.zeros(NF), np.zeros(NF), np.zeros(NF), np.zeros(NF))
    waveform_imr = StationaryWaveformFreq(freq, np.zeros(NF), np.zeros(NF), np.zeros(NF), np.zeros(NF))
    waveform_tot = StationaryWaveformFreq(freq, np.zeros(NF), np.zeros(NF), np.zeros(NF), np.zeros(NF))

    _itrFCut_FI3, imr_params_FI3 = IMRPhenDAmpPhaseFI(waveform_FI3, intrinsic, nf_lim, MfRef_in=MfRef_in, phi0=intrinsic.phase_c, amp_mult=amp0, imr_default_t=0)

    imr_params_FI: IMRPhenomDParams = imr_params_FI3
    imr_params_FI2: IMRPhenomDParams = imr_params_FI3

    phic_use: float = PhiInsAnsatzInt(float(intrinsic.mass_total_detector_sec * intrinsic.frequency_i_hz), intrinsic.symmetric_mass_ratio, intrinsic.chi_s, intrinsic.chi_a, intrinsic.chi_postnewtonian) + float(intrinsic.frequency_i_hz * intrinsic.time_c_sec * 2 * np.pi + 2 * intrinsic.phase_c)
    _itrFCut_FI6, _imr_params_FI6 = IMRPhenDAmpPhase_tc(waveform_FI6, intrinsic, nf_lim, TTRef_in=intrinsic.time_c_sec, phi0=phic_use, amp_mult=amp0)

    do_amp_ins_test = True
    if do_amp_ins_test:
        do_deriv_ins_test = True

        _itrFCut4, imr_params4 = IMRPhenDAmpPhaseFI(waveform4, intrinsic, nf_lim, MfRef_in=MfRef_in, phi0=intrinsic.phase_c, amp_mult=amp0, imr_default_t=1)

        imr_params5 = imr_params4
        imr_params6 = imr_params_FI3

        AmpPhaseSeriesInsAnsatz(waveform5, imr_params5, nf_lim)
        AmpInsAnsatzInplace(waveform6, imr_params6, nf_lim)

        waveform1.AF[:] = AmpInsAnsatz(Mfs, intrinsic.symmetric_mass_ratio, intrinsic.chi_s, intrinsic.chi_a, intrinsic.chi_postnewtonian, amp0_use)

        assert_allclose(waveform6.AF, waveform5.AF, atol=1.e-14, rtol=1.e-3)
        assert_allclose(waveform6.AF[:30000], waveform5.AF[:30000], atol=1.e-16, rtol=1.e-3)
        assert_allclose(waveform5.AF[:30000], waveform4.AF[:30000], atol=1.e-16, rtol=1.e-6)
        assert_allclose(waveform1.AF, waveform6.AF, atol=1.e-30, rtol=1.e-6)

        waveform2.AF[:] = AmpIntAnsatz(Mfs, Mf_ringdown, Mf_damp, intrinsic.symmetric_mass_ratio, intrinsic.chi_s, intrinsic.chi_a, intrinsic.chi_postnewtonian, amp0_use)

        waveform3.AF[:] = AmpMRDAnsatz(Mfs, Mf_ringdown, Mf_damp, intrinsic.symmetric_mass_ratio, intrinsic.chi_postnewtonian, amp0_use)

        if do_deriv_ins_test:
            # assert np.sum(np.isclose(waveform3.AF, waveform2.AF, atol=1.e-30, rtol=1.e-3)) > 3000
            assert_allclose(waveform1.AF[:30000], waveform5.AF[:30000], atol=1.e-16, rtol=1.e-2)
            assert_allclose(waveform2.AF[:30000], waveform5.AF[:30000], atol=1.e-16, rtol=1.e-2)

        waveform_tot.AF[:] = IMRPhenDAmplitude(Mfs, intrinsic.symmetric_mass_ratio, intrinsic.chi_s, intrinsic.chi_a, NF, amp0)
        assert_allclose(waveform4.AF, waveform_tot.AF, atol=1.e-30, rtol=1.e-6)

    _itrFCut_FI3, imr_params_FI3 = IMRPhenDAmpPhaseFI(waveform_FI3, intrinsic, nf_lim, MfRef_in=MfRef_in, phi0=intrinsic.phase_c, amp_mult=amp0, imr_default_t=0)

    imr_params_FI = imr_params_FI3
    imr_params_FI2 = imr_params_FI3

    h22 = AmpPhaseFDWaveform(NF, waveform_imr.F, waveform_imr.AF, waveform_imr.PF, waveform_imr.TF, waveform_imr.TFp, 0., 0.)
    h22 = IMRPhenomDGenerateh22FDAmpPhase(h22, freq, intrinsic.phase_c, MfRef_in, intrinsic.mass_1_detector_kg, intrinsic.mass_2_detector_kg, intrinsic.chi_1z, intrinsic.chi_2z, intrinsic.luminosity_distance_m)

    assert_allclose(2. * np.sqrt(5. / (64. * np.pi)) * h22.amp / imrc.CLIGHT, waveform_FI3.AF, atol=1.e-30, rtol=1.e-10)
    # assert_allclose(h22.timep, waveform_FI3.TFp, atol=1.e-30, rtol=2.1e-10)

    waveform5.AF[:] = IMRPhenDAmplitude(Mfs, intrinsic.symmetric_mass_ratio, intrinsic.chi_s, intrinsic.chi_a, NF, amp0)

    AmpPhaseSeriesInsAnsatz(waveform_FI2, imr_params_FI2, nf_lim)
    PhiSeriesInsAnsatz(waveform_FI, imr_params_FI, nf_lim)
    PhiSeriesIntAnsatz(waveform_FI, imr_params_FI, nf_lim)
    PhiSeriesMRDAnsatz(waveform_FI, imr_params_FI, nf_lim)

    AmpInsAnsatzInplace(waveform_FI2, imr_params_FI, nf_lim)
    AmpMRDAnsatzInplace(waveform_FI, imr_params_FI, nf_lim)
    AmpIntAnsatzInplace(waveform_FI, imr_params_FI, nf_lim)

    imr_params4, _itrFCut4 = IMRPhenDPhaseFI(waveform4, intrinsic, nf_lim, MfRef_in, phi0=intrinsic.phase_c)
    IMRPhenDAmplitudeFI(waveform_FI4, intrinsic, nf_lim, amp_mult=amp0)

    TaylorF2_aligned_inplace(waveform_t, intrinsic, nf_lim, amplitude_pn_mode=2, include_pn_ss3=0)

    do_comp_test = True
    if do_comp_test:
        timep_FI3_alt = gradient(waveform_FI3.TF, Mfs) * intrinsic.mass_total_detector_sec
        assert_allclose(waveform_FI3.TFp[20:], timep_FI3_alt[20:], atol=1.e-30, rtol=1.e-1)

        time_FI3_alt2 = gradient(waveform_FI3.PF, Mfs) * intrinsic.mass_total_detector_sec / (2 * np.pi)
        time_FI3_alt = np.zeros((1, Mfs.size))
        stabilized_gradient_uniform_inplace(waveform_FI3.PF / (2 * np.pi) * intrinsic.mass_total_detector_sec, waveform_FI3.TF, np.array([waveform_FI3.PF / (2 * np.pi) * intrinsic.mass_total_detector_sec]), time_FI3_alt, Mfs[1] - Mfs[0])
        time_FI3_alt = time_FI3_alt[0]
        abs_scale = 1.e-7 * np.max(np.abs(np.diff(waveform_FI3.TF[20:])))
        assert_allclose(waveform_FI3.TF[20:], time_FI3_alt[20:], atol=1.e-30, rtol=1.e-14)
        assert_allclose(time_FI3_alt2[20:], time_FI3_alt[20:], atol=abs_scale, rtol=1.e-2)
        assert_allclose(waveform_FI3.TF[20:], time_FI3_alt[20:], atol=1.e-30, rtol=1.e-14)

        assert_allclose(waveform_t.TF, waveform_FI3.TF, atol=1.e-30, rtol=1.e-3)
        assert_allclose(waveform_t.PF, waveform_FI3.PF, atol=1.e1, rtol=1.e-3)

        ts_t_alt = gradient(waveform_t.PF, Mfs) * intrinsic.mass_total_detector_sec / (2 * np.pi)
        assert_allclose(waveform_t.TF[20:], ts_t_alt[20:], atol=abs_scale, rtol=1.e-2)


def test_QNMData_match() -> None:
    n_q = 10001
    qs = np.linspace(-1., 1., n_q)
    fring1 = InterpolatedUnivariateSpline(QNMData_a, QNMData_fring, k=5, ext=2)(qs)
    fdamp1 = InterpolatedUnivariateSpline(QNMData_a, QNMData_fdamp, k=5, ext=2)(qs)
    assert_allclose(fring1, fring_interp(qs), atol=1.e-3, rtol=1.e-8)
    assert_allclose(fdamp1, fdamp_interp(qs), atol=1.e-3, rtol=1.e-8)
