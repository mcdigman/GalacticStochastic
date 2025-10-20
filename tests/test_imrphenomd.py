"""test the IMRPhenomD module C 2021 Matthew Digman"""
from time import perf_counter

import numpy as np
import PyIMRPhenomD.IMRPhenomD_const as imrc
from numpy import gradient
from numpy.testing import assert_allclose
from numpy.typing import NDArray
from PyIMRPhenomD.IMRPhenomD import IMRPhenomDGenerateh22FDAmpPhase
from PyIMRPhenomD.IMRPhenomD_fring_helper import QNMData_a, QNMData_fdamp, QNMData_fring, fdamp_interp, fring_interp
from PyIMRPhenomD.IMRPhenomD_internals import AmpInsAnsatz, AmpIntAnsatz, AmpMRDAnsatz, AmpPhaseFDWaveform, DAmpInsAnsatz, DAmpMRDAnsatz, DDPhiInsAnsatzInt, DDPhiIntAnsatz, DDPhiMRD, DPhiInsAnsatzInt, DPhiIntAnsatz, DPhiMRD, FinalSpin0815, IMRPhenDAmplitude, IMRPhenDPhase, PhiInsAnsatzInt, PhiIntAnsatz, PhiMRDAnsatzInt, amp0Func, chiPN, fringdown
from scipy.interpolate import InterpolatedUnivariateSpline

from LisaWaveformTools.binary_params_manager import BinaryIntrinsicParams, BinaryIntrinsicParamsManager
from LisaWaveformTools.imrphenomd_waveform import AmpInsAnsatzInplace, AmpIntAnsatzInplace, AmpMRDAnsatzInplace, AmpPhaseSeriesInsAnsatz, IMRPhenDAmplitudeFI, IMRPhenDAmpPhase_tc, IMRPhenDAmpPhaseFI, IMRPhenDPhaseFI, IMRPhenomDParams, PhiSeriesInsAnsatz, PhiSeriesIntAnsatz, PhiSeriesMRDAnsatz
from LisaWaveformTools.source_params import ExtrinsicParams, SourceParams
from LisaWaveformTools.stationary_source_waveform import StationaryWaveformFreq
from LisaWaveformTools.taylorf2_helpers import TaylorF2_aligned_inplace
from WaveletWaveforms.sparse_waveform_functions import PixelGenericRange


def test_imrphenomd_internal_consistency() -> None:
    """Various checks for internal consistency of imrphenomd"""
    NF: int = 16384 * 10

    distance: float = 56.00578366287752 * 1.0e9 * imrc.PC_SI / imrc.CLIGHT
    chi1: float = 0.7534821857057837
    chi2: float = 0.6215875279643664
    m1_sec: float = 2599137.035 * imrc.MTSUN_SI
    m2_sec: float = 1242860.685 * imrc.MTSUN_SI
    Mt_sec: float = m1_sec + m2_sec
    Mt: float = Mt_sec / imrc.MTSUN_SI

    m1: float = m1_sec / imrc.MTSUN_SI
    m2: float = m2_sec / imrc.MTSUN_SI

    m1_SI: float = m1 * imrc.MSUN_SI
    m2_SI: float = m2 * imrc.MSUN_SI

    tc: float = 2.496000e+07

    eta: float = m1_sec * m2_sec / Mt_sec**2
    Mc: float = eta**(3 / 5) * Mt_sec
    assert_allclose(eta, (Mc / Mt_sec)**(5 / 3))

    DF: float = 0.99 * imrc.f_CUT / Mt_sec / NF

    freq: NDArray[np.floating] = np.arange(1, NF + 1) * DF

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

    amp0: float = 2. * np.sqrt(5. / (64. * np.pi)) * Mt_sec**2 / distance  # *imrc.MTSUN_SI**2#*imrc.MTSUN_SI**2*wc.CLIGHT
    finspin: float = FinalSpin0815(eta, chis, chia)  # FinalSpin0815 - 0815 is like a version number
    fRD, fDM = fringdown(eta, chis, chia, finspin)
    amp0_use: float = amp0 * amp0Func(eta)

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
    waveform_res = StationaryWaveformFreq(freq, np.zeros(NF), np.zeros(NF), np.zeros(NF), np.zeros(NF))
    waveform_tot = StationaryWaveformFreq(freq, np.zeros(NF), np.zeros(NF), np.zeros(NF), np.zeros(NF))

    _itrFCut_FI3, imr_params_FI3 = IMRPhenDAmpPhaseFI(waveform_FI3, intrinsic, nf_lim, MfRef_in=MfRef_in, phi0=phic, amp_mult=amp0, imr_default_t=0)

    imr_params_FI: IMRPhenomDParams = imr_params_FI3
    imr_params_FI2: IMRPhenomDParams = imr_params_FI3

    phic_use: float = PhiInsAnsatzInt(float(Mt_sec * FI), eta, chis, chia, chi) + float(FI * tc * 2 * np.pi + 2 * phic)
    _itrFCut_FI6, _imr_params_FI6 = IMRPhenDAmpPhase_tc(waveform_FI6, intrinsic, nf_lim, TTRef_in=tc, phi0=phic_use, amp_mult=amp0)

    Mfs = freq * Mt * imrc.MTSUN_SI

    do_phi_ins_test = True
    if do_phi_ins_test:
        Mfs = freq * Mt * imrc.MTSUN_SI
        do_deriv_ins_test = True
        if do_deriv_ins_test:
            dphi1 = DPhiInsAnsatzInt(Mfs, eta, chis, chia, chi)
            waveform1.PF[:] = PhiInsAnsatzInt(Mfs, eta, chis, chia, chi)
            dphi1_alt = gradient(waveform1.PF[:], Mfs)
            assert_allclose(np.abs(dphi1[20:]), np.abs(dphi1_alt[20:]), atol=1.e-30, rtol=1.e-2)

        do_deriv_int_test = True
        if do_deriv_int_test:
            waveform2.PF[:] = PhiIntAnsatz(Mfs, eta, chi)
            dphi2_alt = gradient(waveform2.PF, Mfs)
            dphi2 = DPhiIntAnsatz(Mfs, eta, chi)
            assert_allclose(np.abs(dphi2[20:]), np.abs(dphi2_alt[20:]), atol=1.e-30, rtol=1.e-2)

        do_deriv_mrd_test = True
        if do_deriv_mrd_test:
            waveform3.PF[:] = PhiMRDAnsatzInt(Mfs, fRD, fDM, eta, chi)
            dphi3_alt = gradient(waveform3.PF, Mfs)
            dphi3 = DPhiMRD(Mfs, fRD, fDM, eta, chi)
            assert_allclose(np.abs(dphi3[20:]), np.abs(dphi3_alt[20:]), atol=1.e-30, rtol=1.e-2)
        waveform1.PF[:], waveform1.TF[:], _t0, _MfRef, _itrFCut = IMRPhenDPhase(Mfs, Mt_sec, eta, chis, chia, NF, MfRef_in, 0.)
        _itrFCut2, _imr_params2 = IMRPhenDAmpPhaseFI(waveform2, intrinsic, nf_lim, MfRef_in=MfRef_in, phi0=phic, amp_mult=amp0, imr_default_t=1)
        assert_allclose(np.abs(waveform1.PF[:]), np.abs(waveform2.PF), atol=1.e1, rtol=1.e-6)
        time1_alt = gradient(waveform1.PF[:], Mfs) * imrc.MTSUN_SI * Mt / (2 * np.pi)
        assert_allclose(np.abs(waveform1.TF), np.abs(waveform2.TF), atol=1.e-30, rtol=1.e-6)
        assert_allclose(np.abs(waveform1.TF[20:]), np.abs(time1_alt[20:]), atol=1.e-30, rtol=1.e-2)

    do_dphi_ins_test = False
    if do_dphi_ins_test:
        Mfs = freq * Mt * imrc.MTSUN_SI

        dphi1 = DPhiInsAnsatzInt(Mfs, eta, chis, chia, chi)
        ddphi1 = DDPhiInsAnsatzInt(Mfs, eta, chis, chia, chi)
        ddphi1_alt = gradient(dphi1, Mfs)
        assert_allclose(ddphi1[20:], ddphi1_alt[20:], atol=1.e-30, rtol=1.e-2)

        dphi2 = DPhiIntAnsatz(Mfs, eta, chi)
        ddphi2 = DDPhiIntAnsatz(Mfs, eta, chi)
        ddphi2_alt = gradient(dphi2, Mfs)
        assert_allclose(ddphi2[20:], ddphi2_alt[20:], atol=1.e-30, rtol=1.e-1)

        dphi3 = DPhiMRD(Mfs, fRD, fDM, eta, chi)
        ddphi3_alt = gradient(dphi3, Mfs)
        ddphi3 = DDPhiMRD(Mfs, fRD, fDM, eta, chi)
        assert_allclose(ddphi3[20:], ddphi3_alt[20:], atol=1.e-30, rtol=1.e-2)

        assert np.isclose(np.abs(ddphi1), np.abs(ddphi2), atol=1.e-30, rtol=1.e-2).sum() > 100
        assert np.isclose(np.abs(ddphi1), np.abs(ddphi3), atol=1.e-30, rtol=1.e-2).sum() > 350
        assert np.isclose(np.abs(ddphi2), np.abs(ddphi3), atol=1.e-30, rtol=1.e-2).sum() > 10000

    do_amp_ins_test = False
    if do_amp_ins_test:
        Mfs = freq * Mt * imrc.MTSUN_SI
        do_deriv_ins_test = True

        _itrFCut4, imr_params4 = IMRPhenDAmpPhaseFI(waveform4, intrinsic, nf_lim, MfRef_in=MfRef_in, phi0=phic, amp_mult=amp0, imr_default_t=1)

        imr_params5 = imr_params4
        imr_params6 = imr_params_FI3

        AmpPhaseSeriesInsAnsatz(waveform5, imr_params5, nf_lim)
        AmpInsAnsatzInplace(waveform6, imr_params6, nf_lim)

        waveform1.AF[:] = AmpInsAnsatz(Mfs, eta, chis, chia, chi, amp0_use)

        assert_allclose(waveform6.AF, waveform5.AF, atol=1.e-14, rtol=1.e-3)
        assert_allclose(waveform6.AF[:30000], waveform5.AF[:30000], atol=1.e-16, rtol=1.e-3)
        assert_allclose(np.abs(waveform5.AF[:30000]), np.abs(waveform4.AF[:30000]), atol=1.e-16, rtol=1.e-6)
        assert_allclose(np.abs(waveform1.AF), np.abs(waveform6.AF), atol=1.e-30, rtol=1.e-6)

        damp1 = DAmpInsAnsatz(Mfs, eta, chis, chia, chi, amp_mult=amp0_use)
        damp1_alt1 = gradient(waveform1.AF * Mfs**(7 / 6), Mfs)
        damp1_alt2 = gradient(waveform5.AF * Mfs**(7 / 6), Mfs)
        damp1_alt3 = gradient(waveform6.AF * Mfs**(7 / 6), Mfs)
        if do_deriv_ins_test:
            assert_allclose(np.abs(damp1[20:]), np.abs(damp1_alt1[20:]), atol=1.e-30, rtol=1.e-2)
            assert_allclose(np.abs(damp1[20:]), np.abs(damp1_alt2[20:]), atol=1.e-30, rtol=1.e-2)
            assert_allclose(np.abs(damp1[20:]), np.abs(damp1_alt3[20:]), atol=1.e-30, rtol=1.e-2)

        waveform2.AF[:] = AmpIntAnsatz(Mfs, fRD, fDM, eta, chis, chia, chi, amp0_use)

        waveform3.AF[:] = AmpMRDAnsatz(Mfs, fRD, fDM, eta, chi, amp0_use)
        damp3 = DAmpMRDAnsatz(Mfs, fRD, fDM, eta, chi, amp0_use)
        damp3_alt = gradient(waveform3.AF * Mfs**(7 / 6), Mfs)
        if do_deriv_ins_test:
            assert_allclose(np.abs(damp3), np.abs(damp3_alt), atol=1.e-23, rtol=1.e-6)

        if do_deriv_ins_test:
            assert np.sum(np.isclose(waveform3.AF, waveform2.AF, atol=1.e-30, rtol=1.e-3)) > 3000
            assert_allclose(waveform1.AF[:30000], waveform5.AF[:30000], atol=1.e-16, rtol=1.e-2)
            assert_allclose(waveform2.AF[:30000], waveform5.AF[:30000], atol=1.e-16, rtol=1.e-2)

        waveform_tot.AF[:] = IMRPhenDAmplitude(Mfs, eta, chis, chia, NF, amp0)
        assert_allclose(waveform4.AF, waveform_tot.AF, atol=1.e-30, rtol=1.e-6)

    _itrFCut_FI3, imr_params_FI3 = IMRPhenDAmpPhaseFI(waveform_FI3, intrinsic, nf_lim, MfRef_in=MfRef_in, phi0=phic, amp_mult=amp0, imr_default_t=0)

    imr_params_FI = imr_params_FI3
    imr_params_FI2 = imr_params_FI3

    h22 = AmpPhaseFDWaveform(NF, waveform_imr.F, waveform_imr.AF, waveform_imr.PF, waveform_imr.TF, waveform_imr.TFp, 0., 0.)
    h22 = IMRPhenomDGenerateh22FDAmpPhase(h22, freq, phic, MfRef_in, m1_SI, m2_SI, chi1, chi2, distance * imrc.CLIGHT)

    assert_allclose(2. * np.sqrt(5. / (64. * np.pi)) * h22.amp, waveform_FI3.AF, atol=1.e-30, rtol=1.e-10)
    assert_allclose(h22.timep, waveform_FI3.TFp, atol=1.e-30, rtol=1.e-10)

    waveform5.AF[:] = IMRPhenDAmplitude(Mfs, eta, chis, chia, NF, amp0)

    AmpPhaseSeriesInsAnsatz(waveform_FI2, imr_params_FI2, nf_lim)
    PhiSeriesInsAnsatz(waveform_FI, imr_params_FI, nf_lim)
    PhiSeriesIntAnsatz(waveform_FI, imr_params_FI, nf_lim)
    PhiSeriesMRDAnsatz(waveform_FI, imr_params_FI, nf_lim)

    AmpInsAnsatzInplace(waveform_FI2, imr_params_FI, nf_lim)
    AmpMRDAnsatzInplace(waveform_FI, imr_params_FI, nf_lim)
    AmpIntAnsatzInplace(waveform_FI, imr_params_FI, nf_lim)

    imr_params4, _itrFCut4 = IMRPhenDPhaseFI(waveform4, intrinsic, nf_lim, MfRef_in, phi0=phic)
    IMRPhenDAmplitudeFI(waveform_FI4, intrinsic, nf_lim, amp_mult=amp0)

    TaylorF2_aligned_inplace(waveform_t, intrinsic, nf_lim, amplitude_pn_mode=2, include_pn_ss3=0)

    do_comp_test = True
    if do_comp_test:
        timep_FI3_alt = gradient(waveform_FI3.TF, Mfs) * Mt_sec
        assert_allclose(np.abs(waveform_FI3.TFp[20:]), np.abs(timep_FI3_alt[20:]), atol=1.e-30, rtol=1.e-1)

        time_FI3_alt = gradient(waveform_FI3.PF, Mfs) * Mt_sec / (2 * np.pi)
        assert_allclose(np.abs(waveform_FI3.TF[20:]), np.abs(time_FI3_alt[20:]), atol=1.e-30, rtol=1.e-2)

        assert_allclose(np.abs(waveform_t.TF), np.abs(waveform_FI3.TF), atol=1.e-30, rtol=1.e-3)
        assert_allclose(np.abs(waveform_t.PF), np.abs(waveform_FI3.PF), atol=1.e1, rtol=1.e-3)

        ts_t_alt = gradient(waveform_t.PF, Mfs) * Mt_sec / (2 * np.pi)
        assert_allclose(waveform_t.TF[20:], ts_t_alt[20:], atol=1.e-30, rtol=1.e-2)

    do_time_dphi_test = True
    if do_time_dphi_test:
        waveform_res.PF[:], waveform_res.TF[:], _t0, _MfRef, _itrFCut = IMRPhenDPhase(Mfs, Mt_sec, eta, chis, chia, NF, MfRef_in, 0.)
        time_res_alt = gradient(waveform_res.PF, Mfs) * imrc.MTSUN_SI * Mt / (2 * np.pi)
        assert_allclose(np.abs(waveform_res.TF[20:]), np.abs(time_res_alt[20:]), atol=1.e-30, rtol=1.e-2)

    do_interp_test = True
    if do_interp_test:
        n_q = 10001
        qs = np.linspace(-1., 1., n_q)
        fring1 = InterpolatedUnivariateSpline(QNMData_a, QNMData_fring, k=5, ext=2)(qs)
        fdamp1 = InterpolatedUnivariateSpline(QNMData_a, QNMData_fdamp, k=5, ext=2)(qs)
        assert_allclose(fring1, fring_interp(qs), atol=1.e-3, rtol=1.e-8)
        assert_allclose(fdamp1, fdamp_interp(qs), atol=1.e-3, rtol=1.e-8)

# if __name__=='__main__':
#    pytest.cmdline.main(['test_IMRPhenomD.py'])


if __name__ == '__main__':

    t_start = perf_counter()
    NF = 16384 * 10
    nitr = 1

    distance = 56.00578366287752 * 1.0e9 * imrc.PC_SI / imrc.CLIGHT
    chi1 = 0.7534821857057837
    chi2 = 0.6215875279643664
    m1_sec = 2599137.035 * imrc.MTSUN_SI
    m2_sec = 1242860.685 * imrc.MTSUN_SI
    Mt_sec = m1_sec + m2_sec
    Mt = Mt_sec / imrc.MTSUN_SI
    m1 = m1_sec / imrc.MTSUN_SI
    m2 = m2_sec / imrc.MTSUN_SI

    tc = 2.496000e+07
    DF = 0.99 * imrc.f_CUT / Mt_sec / NF

    Mt_sec = Mt * imrc.MTSUN_SI
    eta = m1_sec * m2_sec / Mt_sec**2
    Mc = eta**(3 / 5) * Mt_sec
    assert_allclose(eta, (Mc / Mt_sec)**(5 / 3))

    m1_SI = m1 * imrc.MSUN_SI
    m2_SI = m2 * imrc.MSUN_SI

    freq = np.arange(1, NF + 1) * DF
    MfRef_in = 0.  # 0.02*Mt_sec

    chis = (chi1 + chi2) / 2
    chia = (chi1 - chi2) / 2
    phic = 2.848705 / 2
    # FI = 3.4956509169372005e-05
    FI = 3.4956509169372e-05  # 3.4956509169372005e-05
    # determine FI from tc InterpolatedUnivariateSpline(DPhiInsAnsatzInt(Mfs,eta,chis,chia,chi)[:3196588]*Mt_sec/(2*np.pi),freq[:3196588],k=3,ext=2)(-tc)
    MfRef_in = FI * Mt_sec
    logD = np.log(distance)
    chi = chiPN(eta, chis, chia)
    amp0 = 2. * np.sqrt(5. / (64. * np.pi)) * Mt_sec**2 / distance  # *imrc.MTSUN_SI**2#*imrc.MTSUN_SI**2*wc.CLIGHT
    finspin = FinalSpin0815(eta, chis, chia)  # FinalSpin0815 - 0815 is like a version number
    fRD, fDM = fringdown(eta, chis, chia, finspin)
    amp0_use = amp0 * amp0Func(eta)

    chi_postnewtonian_norm = chi / (1. - 76. / 113. * eta)

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

    intrinsic_params_manager = BinaryIntrinsicParamsManager(intrinsic_params_packed)
    intrinsic = intrinsic_params_manager.params
    extrinsic = ExtrinsicParams(
        costh=0.1, phi=0.1, cosi=0.2, psi=0.3
    )

    source_params = SourceParams(intrinsic=intrinsic, extrinsic=extrinsic)

    nf_lim = PixelGenericRange(0, NF, DF, DF)

    waveform1 = StationaryWaveformFreq(freq, np.zeros(NF), np.zeros(NF), np.zeros(NF), np.zeros(NF))
    waveform2 = StationaryWaveformFreq(freq, np.zeros(NF), np.zeros(NF), np.zeros(NF), np.zeros(NF))
    waveform3 = StationaryWaveformFreq(freq, np.zeros(NF), np.zeros(NF), np.zeros(NF), np.zeros(NF))
    waveform4 = StationaryWaveformFreq(freq, np.zeros(NF), np.zeros(NF), np.zeros(NF), np.zeros(NF))
    waveform5 = StationaryWaveformFreq(freq, np.zeros(NF), np.zeros(NF), np.zeros(NF), np.zeros(NF))
    waveform6 = StationaryWaveformFreq(freq, np.zeros(NF), np.zeros(NF), np.zeros(NF), np.zeros(NF))
    waveform_t = StationaryWaveformFreq(freq, np.zeros(NF), np.zeros(NF), np.zeros(NF), np.zeros(NF))
    waveform_imr = StationaryWaveformFreq(freq, np.zeros(NF), np.zeros(NF), np.zeros(NF), np.zeros(NF))
    waveform_res = StationaryWaveformFreq(freq, np.zeros(NF), np.zeros(NF), np.zeros(NF), np.zeros(NF))
    waveform_tot = StationaryWaveformFreq(freq, np.zeros(NF), np.zeros(NF), np.zeros(NF), np.zeros(NF))

    waveform_FI = StationaryWaveformFreq(freq, np.zeros(NF), np.zeros(NF), np.zeros(NF), np.zeros(NF))
    waveform_FI2 = StationaryWaveformFreq(freq, np.zeros(NF), np.zeros(NF), np.zeros(NF), np.zeros(NF))
    waveform_FI3 = StationaryWaveformFreq(freq, np.zeros(NF), np.zeros(NF), np.zeros(NF), np.zeros(NF))
    waveform_FI4 = StationaryWaveformFreq(freq, np.zeros(NF), np.zeros(NF), np.zeros(NF), np.zeros(NF))
    waveform_FI6 = StationaryWaveformFreq(freq, np.zeros(NF), np.zeros(NF), np.zeros(NF), np.zeros(NF))
    waveform_FI7 = StationaryWaveformFreq(freq, np.zeros(NF), np.zeros(NF), np.zeros(NF), np.zeros(NF))

    print('2', Mt_sec, eta, chis, chia, NF, MfRef_in, phic, amp0)
    _itrFCut_FI3, imr_params_FI3 = IMRPhenDAmpPhaseFI(waveform_FI3, intrinsic, nf_lim, MfRef_in=MfRef_in, phi0=phic, amp_mult=amp0, imr_default_t=0)

    phic_use = PhiInsAnsatzInt(Mt_sec * FI, eta, chis, chia, chi) + FI * tc * 2 * np.pi + 2 * phic
    _itrFCut_FI6, imr_params_FI6 = IMRPhenDAmpPhase_tc(waveform_FI6, intrinsic, nf_lim, TTRef_in=tc, phi0=phic_use, amp_mult=amp0)

    _itrFCut_FI7, imr_params_FI6 = IMRPhenDAmpPhaseFI(waveform_FI7, intrinsic, nf_lim, MfRef_in=1 / (6**(3 / 2) * np.pi), phi0=phic, amp_mult=amp0, imr_default_t=0, t_offset=tc)
    time_FI7_alt = gradient(waveform_FI7.PF, Mt_sec * freq) * Mt_sec / (2 * np.pi)
    import matplotlib.pyplot as plt

    arglim = np.argmax(waveform_FI7.TFp <= 0.) - 100
    print(InterpolatedUnivariateSpline(waveform_FI7.TF[:arglim], freq[:arglim])(np.array([0.]))[0])
    print(InterpolatedUnivariateSpline(freq[:arglim], waveform_FI7.TF[:arglim])(np.array([1 / (6**(3 / 2) * np.pi * Mt_sec)]))[0])
    print(InterpolatedUnivariateSpline(waveform_FI7.TF[:arglim], waveform_FI7.PF[:arglim])(np.array([tc]))[0])

    Mt_solar = Mt

    Mfs = freq * Mt * imrc.MTSUN_SI
    do_timings = False
    if do_timings:
        chi = chiPN(eta, chis, chia)
        finspin = FinalSpin0815(eta, chis, chia)
        fRD, fDM = fringdown(eta, chis, chia, finspin)

        dphi1 = DPhiInsAnsatzInt(Mfs, eta, chis, chia, chi)
        dphi2 = DPhiIntAnsatz(Mfs, eta, chi)
        dphi3 = DPhiMRD(Mfs, fRD, fDM, eta, chi)

        waveform1.PF[0:NF] = PhiInsAnsatzInt(Mfs[0:NF], eta, chis, chia, chi)
        waveform2.PF[0:NF] = PhiIntAnsatz(Mfs[0:NF], eta, chi)
        waveform3.PF[0:NF] = PhiMRDAnsatzInt(Mfs[0:NF], fRD, fDM, eta, chi)

        waveform1.AF[0:NF] = AmpInsAnsatz(Mfs[0:NF], eta, chis, chia, chi, 1.)
        waveform2.AF[0:NF] = AmpIntAnsatz(Mfs[0:NF], fRD, fDM, eta, chis, chia, chi, 1.)
        waveform3.AF[0:NF] = AmpMRDAnsatz(Mfs[0:NF], fRD, fDM, eta, chi, 1.)

        n_amp_ins = 1000
        ti = perf_counter()
        for _itrn in range(n_amp_ins):
            waveform1.AF[0:NF] = AmpInsAnsatz(Mfs[0:NF], eta, chis, chia, chi, 1.)
        tf = perf_counter()
        if n_amp_ins > 0:
            print('calculating amp Ins took %9.7f seconds' % ((tf - ti) / n_amp_ins))

        n_amp_int = 100
        ti = perf_counter()
        for _itrn in range(n_amp_int):
            waveform2.AF[0:NF] = AmpIntAnsatz(Mfs[0:NF], fRD, fDM, eta, chis, chia, chi, 1.)
        tf = perf_counter()
        if n_amp_int > 0:
            print('calculating amp Int took %9.7f seconds' % ((tf - ti) / n_amp_int))

        n_amp_mrd = 100
        ti = perf_counter()
        for _itrn in range(n_amp_mrd):
            waveform3.AF[0:NF] = AmpMRDAnsatz(Mfs[0:NF], fRD, fDM, eta, chi, 1.)
        tf = perf_counter()
        if n_amp_mrd > 0:
            print('calculating amp mrd took %9.7f seconds' % ((tf - ti) / n_amp_mrd))

        n_dphi_ins = 100
        ti = perf_counter()
        for _itrn in range(n_dphi_ins):
            DPhiInsAnsatzInt(Mfs, eta, chis, chia, chi)
        tf = perf_counter()
        if n_dphi_ins > 0:
            print('calculating phase deriv ins   took %9.7f seconds' % ((tf - ti) / n_dphi_ins))

        n_dphi_int = 100
        ti = perf_counter()
        for _itrn in range(n_dphi_int):
            DPhiIntAnsatz(Mfs, eta, chi)
        tf = perf_counter()
        if n_dphi_int > 0:
            print('calculating phase deriv int   took %9.7f seconds' % ((tf - ti) / n_dphi_int))

        n_dphi_mrd = 100
        ti = perf_counter()
        for _itrn in range(n_dphi_mrd):
            DPhiMRD(Mfs, fRD, fDM, eta, chi)
        tf = perf_counter()
        if n_dphi_mrd > 0:
            print('calculating phase deriv ins   took %9.7f seconds' % ((tf - ti) / n_dphi_mrd))

        n_phi_ins = 10000
        ti = perf_counter()
        for _itrn in range(n_phi_ins):
            waveform1.PF[0:NF] = PhiInsAnsatzInt(Mfs[0:NF], eta, chis, chia, chi)
        tf = perf_counter()
        if n_phi_ins > 0:
            print('calculating phase ins        took %9.7f seconds' % ((tf - ti) / n_phi_ins))

        n_phi_int = 1000
        ti = perf_counter()
        for _itrn in range(n_phi_int):
            waveform2.PF[0:NF] = PhiIntAnsatz(Mfs[0:NF], eta, chi)
        tf = perf_counter()
        if n_phi_ins > 0:
            print('calculating phase int        took %9.7f seconds' % ((tf - ti) / n_phi_ins))

        n_phi_mrd = 1000
        ti = perf_counter()
        for _itrn in range(n_phi_mrd):
            waveform3.PF[0:NF] = PhiMRDAnsatzInt(Mfs[0:NF], fRD, fDM, eta, chi)
        tf = perf_counter()
        if n_phi_mrd > 0:
            print('calculating phase mrd        took %9.7f seconds' % ((tf - ti) / n_phi_mrd))

    do_phi_ins_test = False
    if do_phi_ins_test:
        import matplotlib.pyplot as plt
        Mfs = freq * Mt * imrc.MTSUN_SI
        chi = chiPN(eta, chis, chia)
        finspin = FinalSpin0815(eta, chis, chia)
        fRD, fDM = fringdown(eta, chis, chia, finspin)
        do_deriv_ins_test = True
        if do_deriv_ins_test:
            dphi1 = DPhiInsAnsatzInt(Mfs, eta, chis, chia, chi)
            waveform1.PF[:] = PhiInsAnsatzInt(Mfs, eta, chis, chia, chi)
            dphi1_alt = gradient(waveform1.PF[:], Mfs)
            plt.loglog(freq, np.abs(dphi1_alt))
            plt.loglog(freq, np.abs(dphi1))
            assert_allclose(np.abs(dphi1[20:]), np.abs(dphi1_alt[20:]), atol=1.e-30, rtol=1.e-2)
            plt.show()

        do_deriv_int_test = True
        if do_deriv_int_test:
            waveform2.PF[:] = PhiIntAnsatz(Mfs, eta, chi)
            dphi2_alt = gradient(waveform2.PF, Mfs)
            dphi2 = DPhiIntAnsatz(Mfs, eta, chi)
            plt.loglog(freq, np.abs(dphi2))
            plt.loglog(freq, np.abs(dphi2_alt))
            assert_allclose(np.abs(dphi2[20:]), np.abs(dphi2_alt[20:]), atol=1.e-30, rtol=1.e-2)
            plt.show()

        do_deriv_mrd_test = True
        if do_deriv_mrd_test:
            waveform3.PF[:] = PhiMRDAnsatzInt(Mfs, fRD, fDM, eta, chi)
            dphi3_alt = gradient(waveform3.PF[:], Mfs)
            dphi3 = DPhiMRD(Mfs, fRD, fDM, eta, chi)
            plt.loglog(freq, np.abs(dphi3))
            plt.loglog(freq, np.abs(dphi3_alt))
            assert_allclose(np.abs(dphi3[20:]), np.abs(dphi3_alt[20:]), atol=1.e-30, rtol=1.e-2)
            plt.show()

        waveform1.PF[:], waveform1.TF[:], t0, MfRef, itrFCut = IMRPhenDPhase(Mfs, Mt_sec, eta, chis, chia, NF, MfRef_in, 0.)
        IMRPhenDAmpPhaseFI(waveform2, intrinsic, nf_lim, MfRef_in=MfRef_in, phi0=phic, amp_mult=amp0, imr_default_t=1)
        plt.loglog(freq, np.abs(waveform1.PF[:]))
        plt.loglog(freq, np.abs(waveform2.PF[:]))
        assert_allclose(np.abs(waveform1.PF[:]), np.abs(waveform2.PF[:]), atol=1.e1, rtol=1.e-6)
        plt.show()
        time1_alt = gradient(waveform1.PF[:], Mfs) * imrc.MTSUN_SI * Mt / (2 * np.pi)
        plt.loglog(freq, np.abs(waveform1.TF))
        plt.loglog(freq, np.abs(waveform2.TF))
        plt.loglog(freq, np.abs(time1_alt))
        assert_allclose(np.abs(waveform1.TF), np.abs(waveform2.TF), atol=1.e-30, rtol=1.e-6)
        assert_allclose(np.abs(waveform1.TF[20:]), np.abs(time1_alt[20:]), atol=1.e-30, rtol=1.e-2)
        plt.show()

    do_dphi_ins_test = False
    if do_dphi_ins_test:
        import matplotlib.pyplot as plt
        Mfs = freq * Mt * imrc.MTSUN_SI
        chi = chiPN(eta, chis, chia)
        finspin = FinalSpin0815(eta, chis, chia)
        fRD, fDM = fringdown(eta, chis, chia, finspin)

        dphi1 = DPhiInsAnsatzInt(Mfs, eta, chis, chia, chi)
        ddphi1 = DDPhiInsAnsatzInt(Mfs, eta, chis, chia, chi)
        ddphi1_alt = gradient(dphi1, Mfs)
        plt.loglog(freq, np.abs(ddphi1_alt))
        plt.loglog(freq, np.abs(ddphi1))
        plt.show()

        dphi2 = DPhiIntAnsatz(Mfs, eta, chi)
        ddphi2 = DDPhiIntAnsatz(Mfs, eta, chi)
        ddphi2_alt = gradient(dphi2, Mfs)
        plt.loglog(freq, np.abs(ddphi2))
        plt.loglog(freq, np.abs(ddphi2_alt))
        plt.show()

        dphi3 = DPhiMRD(Mfs, fRD, fDM, eta, chi)
        ddphi3_alt = gradient(dphi3, Mfs)
        ddphi3 = DDPhiMRD(Mfs, fRD, fDM, eta, chi)
        plt.loglog(freq, np.abs(ddphi3))
        plt.loglog(freq, np.abs(ddphi3_alt))
        plt.show()

        plt.loglog(freq, np.abs(ddphi1))
        plt.loglog(freq, np.abs(ddphi2))
        plt.loglog(freq, np.abs(ddphi3))

        assert np.isclose(np.abs(ddphi1), np.abs(ddphi2), atol=1.e-30, rtol=1.e-2).sum() > 100
        assert np.isclose(np.abs(ddphi1), np.abs(ddphi3), atol=1.e-30, rtol=1.e-2).sum() > 350
        assert np.isclose(np.abs(ddphi2), np.abs(ddphi3), atol=1.e-30, rtol=1.e-2).sum() > 10000
        plt.show()
        waveform1.PF[:], waveform1.TF[:], t0, MfRef, itrFCut = IMRPhenDPhase(Mfs, Mt_sec, eta, chis, chia, NF, MfRef_in, 0.)
        plt.loglog(freq, np.abs(waveform1.PF[:]))
        plt.show()
        time1_alt = gradient(waveform1.PF[:], Mfs) * imrc.MTSUN_SI * Mt / (2 * np.pi)
        plt.loglog(freq, np.abs(waveform1.TF))
        plt.loglog(freq, np.abs(time1_alt))
        plt.show()

    do_amp_ins_test = False
    if do_amp_ins_test:
        import matplotlib.pyplot as plt
        Mfs = freq * Mt * imrc.MTSUN_SI
        chi = chiPN(eta, chis, chia)
        finspin = FinalSpin0815(eta, chis, chia)
        fRD, fDM = fringdown(eta, chis, chia, finspin)
        do_deriv_ins_test = True

        _itrFCut4, imr_params4 = IMRPhenDAmpPhaseFI(waveform4, intrinsic, nf_lim, MfRef_in=MfRef_in, phi0=phic, amp_mult=amp0, imr_default_t=1)

        imr_params5 = imr_params4
        imr_params6 = imr_params_FI3

        AmpPhaseSeriesInsAnsatz(waveform5, imr_params5, nf_lim)  # ,TTRef4,amp0_use,0,NF)
        AmpInsAnsatzInplace(waveform6, imr_params6, nf_lim)

        waveform1.AF[:] = AmpInsAnsatz(Mfs, eta, chis, chia, chi, amp0_use)

        plt.loglog(freq, np.abs(waveform1.AF))
        plt.loglog(freq, np.abs(waveform6.AF))
        plt.loglog(freq, np.abs(waveform5.AF))
        plt.loglog(freq, np.abs(waveform4.AF))
        assert_allclose(waveform6.AF, waveform5.AF, atol=1.e-14, rtol=1.e-3)
        assert_allclose(waveform6.AF[:30000], waveform5.AF[:30000], atol=1.e-16, rtol=1.e-3)
        assert_allclose(np.abs(waveform5.AF[:30000]), np.abs(waveform4.AF[:30000]), atol=1.e-16, rtol=1.e-6)
        assert_allclose(np.abs(waveform1.AF), np.abs(waveform6.AF), atol=1.e-30, rtol=1.e-6)
        plt.show()

        damp1 = DAmpInsAnsatz(Mfs, eta, chis, chia, chi, amp_mult=amp0_use)
        damp1_alt1 = gradient(waveform1.AF * Mfs**(7 / 6), Mfs)
        damp1_alt2 = gradient(waveform5.AF * Mfs**(7 / 6), Mfs)
        damp1_alt3 = gradient(waveform6.AF * Mfs**(7 / 6), Mfs)
        if do_deriv_ins_test:
            plt.loglog(freq, np.abs(damp1))
            plt.loglog(freq, np.abs(damp1_alt1))
            plt.loglog(freq, np.abs(damp1_alt2))
            plt.loglog(freq, np.abs(damp1_alt3))
            assert_allclose(np.abs(damp1[20:]), np.abs(damp1_alt1[20:]), atol=1.e-30, rtol=1.e-2)
            assert_allclose(np.abs(damp1[20:]), np.abs(damp1_alt2[20:]), atol=1.e-30, rtol=1.e-2)
            assert_allclose(np.abs(damp1[20:]), np.abs(damp1_alt3[20:]), atol=1.e-30, rtol=1.e-2)
            plt.show()

        waveform2.AF[:] = AmpIntAnsatz(Mfs, fRD, fDM, eta, chis, chia, chi, amp0_use)

        waveform3.AF[:] = AmpMRDAnsatz(Mfs, fRD, fDM, eta, chi, amp0_use)
        damp3 = DAmpMRDAnsatz(Mfs, fRD, fDM, eta, chi, amp0_use)
        damp3_alt = gradient(waveform3.AF * Mfs**(7 / 6), Mfs)
        if do_deriv_ins_test:
            plt.loglog(freq, np.abs(damp3))
            plt.loglog(freq, np.abs(damp3_alt))
            plt.show()
            assert_allclose(np.abs(damp3), np.abs(damp3_alt), atol=1.e-23, rtol=1.e-6)

        if do_deriv_ins_test:
            plt.loglog(freq, waveform1.AF)
            plt.loglog(freq, waveform5.AF)
            plt.loglog(freq, waveform2.AF)
            plt.loglog(freq, waveform3.AF)
            assert np.sum(np.isclose(waveform3.AF, waveform2.AF, atol=1.e-30, rtol=1.e-3)) > 3000
            assert_allclose(waveform1.AF[:30000], waveform5.AF[:30000], atol=1.e-16, rtol=1.e-2)
            assert_allclose(waveform2.AF[:30000], waveform5.AF[:30000], atol=1.e-16, rtol=1.e-2)
            plt.show()

        waveform_tot.AF[:] = IMRPhenDAmplitude(Mfs, eta, chis, chia, NF, amp0)
        plt.loglog(freq, waveform_tot.AF)
        plt.loglog(freq, waveform4.AF)
        assert_allclose(waveform4.AF, waveform_tot.AF, atol=1.e-30, rtol=1.e-6)
        plt.show()

    _itrFCut_FI3, imr_params_FI3 = IMRPhenDAmpPhaseFI(waveform_FI3, intrinsic, nf_lim, MfRef_in=MfRef_in, phi0=phic, amp_mult=amp0, imr_default_t=0)

    waveform5.AF[:] = IMRPhenDAmplitude(Mfs, eta, chis, chia, NF, amp0)

    ti = perf_counter()
    for _itrn in range(nitr):
        IMRPhenDAmpPhaseFI(waveform_FI3, intrinsic, nf_lim, MfRef_in=MfRef_in, phi0=phic, amp_mult=amp0, imr_default_t=0)
    tf = perf_counter()
    print('creating waveform complete          took %9.7f seconds' % ((tf - ti) / nitr))

    imr_params_FI = imr_params_FI3
    imr_params_FI2 = imr_params_FI3

    AmpPhaseSeriesInsAnsatz(waveform_FI2, imr_params_FI2, nf_lim)
    PhiSeriesInsAnsatz(waveform_FI, imr_params_FI, nf_lim)
    PhiSeriesIntAnsatz(waveform_FI, imr_params_FI, nf_lim)
    PhiSeriesMRDAnsatz(waveform_FI, imr_params_FI, nf_lim)

    AmpInsAnsatzInplace(waveform_FI2, imr_params_FI2, nf_lim)
    AmpMRDAnsatzInplace(waveform_FI, imr_params_FI, nf_lim)
    AmpIntAnsatzInplace(waveform_FI, imr_params_FI, nf_lim)

    ti = perf_counter()
    for _itrn in range(nitr):
        PhiSeriesMRDAnsatz(waveform_FI, imr_params_FI, nf_lim)
    tf = perf_counter()
    print('creating waveform phasing   mrd     took %9.7f seconds' % ((tf - ti) / nitr))

    ti = perf_counter()
    for _itrn in range(nitr):
        PhiSeriesIntAnsatz(waveform_FI, imr_params_FI, nf_lim)
    tf = perf_counter()
    print('creating waveform phasing   int     took %9.7f seconds' % ((tf - ti) / nitr))

    ti = perf_counter()
    for _itrn in range(nitr):
        PhiSeriesInsAnsatz(waveform_FI, imr_params_FI, nf_lim)
    tf = perf_counter()
    print('creating waveform phasing   ins     took %9.7f seconds' % ((tf - ti) / nitr))

    ti = perf_counter()
    for _itrn in range(nitr):
        AmpMRDAnsatzInplace(waveform_FI, imr_params_FI, nf_lim)
    tf = perf_counter()
    print('creating waveform amplitude mrd     took %9.7f seconds' % ((tf - ti) / nitr))

    ti = perf_counter()
    for _itrn in range(nitr):
        AmpIntAnsatzInplace(waveform_FI, imr_params_FI, nf_lim)
    tf = perf_counter()
    print('creating waveform amplitude Int     took %9.7f seconds' % ((tf - ti) / nitr))

    ti = perf_counter()
    for _itrn in range(nitr):
        AmpInsAnsatzInplace(waveform_FI, imr_params_FI, nf_lim)
    tf = perf_counter()
    print('creating waveform amplitude ins     took %9.7f seconds' % ((tf - ti) / nitr))

    ti = perf_counter()
    for _itrn in range(nitr):
        AmpPhaseSeriesInsAnsatz(waveform_FI2, imr_params_FI2, nf_lim)
    tf = perf_counter()
    print('creating waveform phasing numba ins took %9.7f seconds' % ((tf - ti) / nitr))

    imr_params4, _itrFCut4 = IMRPhenDPhaseFI(waveform4, intrinsic, nf_lim, MfRef_in, phi0=phic)
    IMRPhenDAmplitudeFI(waveform_FI4, intrinsic, nf_lim, amp_mult=amp0)

    TaylorF2_aligned_inplace(waveform_t, intrinsic, nf_lim, amplitude_pn_mode=2, include_pn_ss3=0)

    h22 = AmpPhaseFDWaveform(NF, waveform_imr.F, waveform_imr.AF, waveform_imr.PF, waveform_imr.TF, waveform_imr.TFp, 0., 0.)
    h22 = IMRPhenomDGenerateh22FDAmpPhase(h22, freq, phic, MfRef_in, m1_SI, m2_SI, chi1, chi2, distance * imrc.CLIGHT)

    plt.loglog(freq, 2. * np.sqrt(5. / (64. * np.pi)) * h22.amp)
    plt.loglog(freq, waveform_FI3.AF)
    plt.show()
    plt.loglog(freq, np.abs(h22.phase))
    plt.loglog(freq, np.abs(waveform_FI.PF))
    plt.show()
    plt.loglog(freq, np.abs(h22.time))
    plt.loglog(freq, np.abs(waveform_FI3.TF))
    plt.show()
    plt.loglog(freq, np.abs(h22.timep))
    plt.loglog(freq, np.abs(waveform_FI3.TFp))
    plt.show()

    assert_allclose(2. * np.sqrt(5. / (64. * np.pi)) * h22.amp, waveform_FI3.AF, atol=1.e-30, rtol=1.e-8)
    assert_allclose(h22.timep, waveform_FI3.TFp, atol=1.e-30, rtol=1.e-8)

    tf = perf_counter()
    print('compiled in %10.7f seconds' % (tf - t_start))

    ti = perf_counter()
    for _itrn in range(nitr):
        IMRPhenDAmplitudeFI(waveform_FI4, intrinsic, nf_lim, amp_mult=amp0)
        IMRPhenDPhaseFI(waveform4, intrinsic, nf_lim, MfRef_in, phi0=phic)
    tf = perf_counter()
    print('creating waveform phasing    took %9.7f seconds' % ((tf - ti) / nitr))

    ti = perf_counter()
    for _itrn in range(nitr):
        TaylorF2_aligned_inplace(waveform_t, intrinsic, nf_lim, amplitude_pn_mode=2, include_pn_ss3=0)
    tf = perf_counter()
    print('creating waveform inspiral   took %9.7f seconds' % ((tf - ti) / nitr))

    do_comp_test = True
    if do_comp_test:
        # print('f1 %.14e'%(InterpolatedUnivariateSpline(waveform_t.TF,freq)(np.array([0.]))[0]))
        # print('f2 %.14e'%(InterpolatedUnivariateSpline(waveform_FI.TF,freq)(np.array([0.]))[0]))
        import matplotlib.pyplot as plt

        timep_FI3_alt = gradient(waveform_FI3.TF, Mfs) * Mt_sec
        plt.loglog(freq, np.abs(waveform_FI3.TFp))
        plt.loglog(freq, np.abs(timep_FI3_alt))
        assert_allclose(np.abs(waveform_FI3.TFp[20:]), np.abs(timep_FI3_alt[20:]), atol=1.e-30, rtol=1.e-1)
        plt.show()

        time_FI3_alt = gradient(waveform_FI3.PF, Mfs) * Mt_sec / (2 * np.pi)
        plt.loglog(freq, np.abs(waveform_FI3.TF))
        plt.loglog(freq, np.abs(time_FI3_alt))
        assert_allclose(np.abs(waveform_FI3.TF[20:]), np.abs(time_FI3_alt[20:]), atol=1.e-30, rtol=1.e-2)
        plt.show()

        plt.loglog(freq, waveform_t.AF)
        plt.loglog(freq, waveform_FI3.AF)
        plt.show()

        plt.loglog(freq, np.abs(waveform_t.TF))
        plt.loglog(freq, np.abs(waveform_FI3.TF))
        assert_allclose(np.abs(waveform_t.TF), np.abs(waveform_FI3.TF), atol=1.e-30, rtol=1.e-3)
        plt.show()

        plt.loglog(freq, np.abs(waveform_t.PF))
        plt.loglog(freq, np.abs(waveform_FI3.PF))
        assert_allclose(np.abs(waveform_t.PF), np.abs(waveform_FI3.PF), atol=1.e1, rtol=1.e-3)

        plt.show()

        plt.semilogx(freq, waveform_FI.TF)
        plt.show()

        ts_t_alt = gradient(waveform_t.PF, Mfs) * Mt_sec / (2 * np.pi)
        plt.loglog(freq, np.abs(waveform_t.TF))
        plt.loglog(freq, np.abs(ts_t_alt))
        assert_allclose(waveform_t.TF[20:], ts_t_alt[20:], atol=1.e-30, rtol=1.e-2)
        plt.show()

    do_time_dphi_test = True
    if do_time_dphi_test:
        import matplotlib.pyplot as plt
        waveform_res.PF[:], waveform_res.TF[:], t0, MfRef, itrFCut = IMRPhenDPhase(Mfs, Mt_sec, eta, chis, chia, NF, MfRef_in, 0.)
        time_res_alt = gradient(waveform_res.PF, Mfs) * imrc.MTSUN_SI * Mt / (2 * np.pi)
        plt.loglog(freq, np.abs(waveform_res.TF[:]))
        plt.loglog(freq, np.abs(time_res_alt))
        assert_allclose(np.abs(waveform_res.TF[20:]), np.abs(time_res_alt[20:]), atol=1.e-30, rtol=1.e-2)
        plt.loglog(freq, np.abs(h22.time))
        plt.show()

    do_interp_test = True
    if do_interp_test:
        import matplotlib.pyplot as plt
        n_q = 10001
        qs = np.linspace(-1., 1., n_q)
        fring1 = InterpolatedUnivariateSpline(QNMData_a, QNMData_fring, k=5, ext=2)(qs)
        fdamp1 = InterpolatedUnivariateSpline(QNMData_a, QNMData_fdamp, k=5, ext=2)(qs)
        plt.plot(QNMData_a, QNMData_fring)
        plt.plot(qs, fring1)
        plt.plot(qs, fring_interp(qs))
        assert_allclose(fring1, fring_interp(qs), atol=1.e-3, rtol=1.e-8)
        plt.show()

        plt.plot(QNMData_a, QNMData_fdamp)
        plt.plot(qs, fdamp1)
        plt.plot(qs, fdamp_interp(qs))
        assert_allclose(fdamp1, fdamp_interp(qs), atol=1.e-3, rtol=1.e-8)
        plt.show()
