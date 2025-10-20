"""test the IMRPhenomD module C 2021 Matthew Digman"""

from typing import TYPE_CHECKING

import numpy as np
import PyIMRPhenomD.IMRPhenomD_const as imrc
from numpy import gradient
from numpy.testing import assert_allclose
from PyIMRPhenomD.IMRPhenomD import IMRPhenomDGenerateh22FDAmpPhase
from PyIMRPhenomD.IMRPhenomD_fring_helper import QNMData_a, QNMData_fdamp, QNMData_fring, fdamp_interp, fring_interp
from PyIMRPhenomD.IMRPhenomD_internals import AmpInsAnsatz, AmpIntAnsatz, AmpMRDAnsatz, AmpPhaseFDWaveform, DAmpInsAnsatz, DAmpMRDAnsatz, DDPhiInsAnsatzInt, DDPhiIntAnsatz, DDPhiMRD, DPhiInsAnsatzInt, DPhiIntAnsatz, DPhiMRD, FinalSpin0815, IMRPhenDAmplitude, IMRPhenDPhase, PhiInsAnsatzInt, PhiIntAnsatz, PhiMRDAnsatzInt, amp0Func, chiPN, fringdown
from scipy.interpolate import InterpolatedUnivariateSpline

from LisaWaveformTools.binary_params_manager import BinaryIntrinsicParams, BinaryIntrinsicParamsManager
from LisaWaveformTools.imrphenomd_waveform import AmpInsAnsatzInplace, AmpIntAnsatzInplace, AmpMRDAnsatzInplace, AmpPhaseSeriesInsAnsatz, IMRPhenDAmplitudeFI, IMRPhenDAmpPhase_tc, IMRPhenDAmpPhaseFI, IMRPhenDPhaseFI, IMRPhenomDParams, PhiSeriesInsAnsatz, PhiSeriesIntAnsatz, PhiSeriesMRDAnsatz
from LisaWaveformTools.stationary_source_waveform import StationaryWaveformFreq
from LisaWaveformTools.taylorf2_helpers import TaylorF2_aligned_inplace
from WaveletWaveforms.sparse_waveform_functions import PixelGenericRange

if TYPE_CHECKING:
    from numpy.typing import NDArray


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
        assert_allclose(np.abs(waveform1.TF[20:]), np.abs(time1_alt[20:]), atol=1.e-30, rtol=1.3e-2)

    do_dphi_ins_test = True
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

    do_amp_ins_test = True
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
            assert_allclose(np.abs(damp3), np.abs(damp3_alt), atol=1.e-23, rtol=3.e-5)

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
        assert_allclose(np.abs(waveform_res.TF[20:]), np.abs(time_res_alt[20:]), atol=1.e-30, rtol=1.3e-2)

    do_interp_test = True
    if do_interp_test:
        n_q = 10001
        qs = np.linspace(-1., 1., n_q)
        fring1 = InterpolatedUnivariateSpline(QNMData_a, QNMData_fring, k=5, ext=2)(qs)
        fdamp1 = InterpolatedUnivariateSpline(QNMData_a, QNMData_fdamp, k=5, ext=2)(qs)
        assert_allclose(fring1, fring_interp(qs), atol=1.e-3, rtol=1.e-8)
        assert_allclose(fdamp1, fdamp_interp(qs), atol=1.e-3, rtol=1.e-8)
