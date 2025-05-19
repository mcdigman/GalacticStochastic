"""functions to compute rigid adiabatic response in frequency domain"""

import numpy as np
from numba import njit, prange

from LisaWaveformTools.lisa_config import LISAConstants

CLIGHT = 2.99792458e8     # Speed of light in m/s
AU = 1.4959787e11         # Astronomical Unit in meters


@njit()
def get_tensor_basis(phi, costh):
    """Get the 3 dimensional tensor basis"""
    # Calculate cos and sin of sky position, inclination, polarization
    n_space = 3  # number of spatial dimensions (must be 3)

    sinth = np.sqrt(1.0 - costh**2)
    cosph = np.cos(phi)
    sinph = np.sin(phi)

    kv = np.zeros(n_space)
    u = np.zeros(n_space)
    v = np.zeros(n_space)

    kv[0] = -sinth * cosph
    kv[1] = -sinth * sinph
    kv[2] = -costh

    u[0] = sinph
    u[1] = -cosph
    u[2] = 0.

    v[0] = -costh * cosph
    v[1] = -costh * sinph
    v[2] = sinth

    eplus = np.zeros((n_space, n_space))
    ecross = np.zeros((n_space, n_space))

    for i in range(n_space):
        for j in range(n_space):
            eplus[i, j] = u[i] * u[j] - v[i] * v[j]
            ecross[i, j] = u[i] * v[j] + v[i] * u[j]
    return kv, eplus, ecross

# TODO eliminate need for constants by storing e.g. Larm_AU


@njit(fastmath=True)
def RAantenna_inplace(spacecraft_channels, cosi, psi, phi, costh, ts, FFs, nf_low, NTs, kdotx, lc: LISAConstants) -> None:
    """Get the waveform for LISA given polarization angle, spacecraft, tensor basis and Fs, channel order AET"""
    RRs = spacecraft_channels.RR
    IIs = spacecraft_channels.II

    kv, eplus, ecross = get_tensor_basis(phi, costh)

    n_space = 3  # number of spatial dimensions (must be 3)
    assert n_space == 3

    assert kv.shape == (n_space,)
    assert eplus.shape == (n_space, n_space)
    assert ecross.shape == (n_space, n_space)

    n_spacecraft = 3  # number of spacecraft (currently must be 3)
    assert n_spacecraft == 3

    n_arm = 3  # number of arms (currently must be 3)
    assert n_arm == 3

    nc_waveform = RRs.shape[0]  # number of channels in the output waveform
    assert IIs.shape[0] == nc_waveform

    nc_generate = 3  # number of combinations to generate internally (currently must be 3)
    assert nc_generate == 3
    assert nc_generate >= nc_waveform

    nc_michelson = 3  # number of michelson combinations (currently must be 3)
    assert nc_michelson == 3
    assert nc_michelson >= nc_generate

    dplus = np.zeros((n_arm, n_arm))
    dcross = np.zeros((n_arm, n_arm))

    TR = np.zeros((n_arm, n_arm))
    TI = np.zeros((n_arm, n_arm))

    # spacecraft x y and z coordinates
    x = np.zeros(n_spacecraft)
    y = np.zeros(n_spacecraft)
    z = np.zeros(n_spacecraft)

    # for projecting spacecraft arm vectors into tensor basis
    kdr = np.zeros((n_arm, n_arm))
    kdg = np.zeros(n_arm)

    # michelson combinations
    fprs = np.zeros(nc_michelson)
    fcrs = np.zeros(nc_michelson)
    fpis = np.zeros(nc_michelson)
    fcis = np.zeros(nc_michelson)

    # tdi combinations
    FpRs = np.zeros(nc_generate)
    FcRs = np.zeros(nc_generate)
    FpIs = np.zeros(nc_generate)
    FcIs = np.zeros(nc_generate)

    # spacecraft separation vectors
    r12 = np.zeros(n_space)
    r13 = np.zeros(n_space)
    r23 = np.zeros(n_space)

    # separation vectors of spacecraft and guiding center
    r10 = np.zeros(n_space)
    r20 = np.zeros(n_space)
    r30 = np.zeros(n_space)

    # quantities for computing spacecraft positions
    betas = np.zeros(n_spacecraft)
    sbs = np.zeros(n_spacecraft)
    cbs = np.zeros(n_spacecraft)
    for itrc in range(n_spacecraft):
        betas[itrc] = 2. / 3. * np.pi * itrc + lc.lambda0
        sbs[itrc] = (AU / lc.Larm * lc.ec) * np.sin(betas[itrc])
        cbs[itrc] = (AU / lc.Larm * lc.ec) * np.cos(betas[itrc])

    # polarization angle quantities
    cosps = np.cos(2 * psi)
    sinps = np.sin(2 * psi)

    # amplitude multipliers
    Aplus = (1. + cosi**2) / 2
    Across = -cosi

    # Main Loop
    for n in prange(NTs):
        # Barycenter time
        # pull out Larm scaling

        alpha = 2 * np.pi * lc.fm * ts[n + nf_low] + lc.kappa0
        sa = np.sin(alpha)
        ca = np.cos(alpha)
        for itrc in range(n_spacecraft):
            x[itrc] = (AU / lc.Larm) * ca + sa * ca * sbs[itrc] - (1. + sa * sa) * cbs[itrc]
            y[itrc] = (AU / lc.Larm) * sa + sa * ca * cbs[itrc] - (1. + ca * ca) * sbs[itrc]
            z[itrc] = -np.sqrt(3.) * (ca * cbs[itrc] + sa * sbs[itrc])

        # manually average over n_spacecraft to get coordinates of the guiding center
        xa = (x[0] + x[1] + x[2]) / n_spacecraft
        ya = (y[0] + y[1] + y[2]) / n_spacecraft
        za = (z[0] + z[1] + z[2]) / n_spacecraft

        # manual dot product with n_spacecraft
        kdotx[n] = (lc.Larm / CLIGHT) * (xa * kv[0] + ya * kv[1] + za * kv[2])

        # normalized frequency
        fr = 1 / (2 * lc.fstr) * FFs[n + nf_low]

        # Separation vector from spacecraft i to j
        r12[0] = x[1] - x[0]
        r12[1] = y[1] - y[0]
        r12[2] = z[1] - z[0]
        r13[0] = x[2] - x[0]
        r13[1] = y[2] - y[0]
        r13[2] = z[2] - z[0]
        r23[0] = x[2] - x[1]
        r23[1] = y[2] - y[1]
        r23[2] = z[2] - z[1]

        # Separation vector between spacecraft and guiding center (not unit vectors)
        r10[0] = xa - x[0]
        r10[1] = ya - y[0]
        r10[2] = za - z[0]
        r20[0] = xa - x[1]
        r20[1] = ya - y[1]
        r20[2] = za - z[1]
        r30[0] = xa - x[2]
        r30[1] = ya - y[2]
        r30[2] = za - z[2]

        kdr[:] = 0.
        kdg[:] = 0.
        for k in range(n_space):
            kdr[0, 1] += kv[k] * r12[k]
            kdr[0, 2] += kv[k] * r13[k]
            kdr[1, 2] += kv[k] * r23[k]
            kdg[0] += kv[k] * r10[k]
            kdg[1] += kv[k] * r20[k]
            kdg[2] += kv[k] * r30[k]

        for i in range(n_arm - 1):
            for j in range(i + 1, n_arm):
                q1 = fr * (1. - kdr[i, j])
                q2 = fr * (1. + kdr[i, j])
                q3 = -fr * (3. + kdr[i, j] - 2 * kdg[i])
                q4 = -fr * (1. + kdr[i, j] - 2 * kdg[i])  # TODO missing 1/sqrt(3) on kdg?
                q5 = -fr * (3. - kdr[i, j] - 2 * kdg[j])
                q6 = -fr * (1. - kdr[i, j] - 2 * kdg[j])  # TODO missing 1/sqrt(3) on kdg?
                sincq1 = np.sin(q1) / q1 / 2
                sincq2 = np.sin(q2) / q2 / 2
                # Real part of T from eq B9 in PhysRevD.101.124008
                TR[i, j] = sincq1 * np.cos(q3) + sincq2 * np.cos(q4)   # goes to 1 when f/fstar small
                # imaginary part of T
                TI[i, j] = sincq1 * np.sin(q3) + sincq2 * np.sin(q4)   # goes to 0 when f/fstar small
                # save ops computing other triangle simultaneously
                TR[j, i] = sincq2 * np.cos(q5) + sincq1 * np.cos(q6)   # goes to 1 when f/fstar small
                TI[j, i] = sincq2 * np.sin(q5) + sincq1 * np.sin(q6)   # goes to 0 when f/fstar small

        dplus[:] = 0.
        dcross[:] = 0.
        # Convenient quantities d+ & dx
        for i in range(n_space):
            for j in range(n_space):
                dplus[0, 1] += r12[i] * r12[j] * eplus[i, j]
                dcross[0, 1] += r12[i] * r12[j] * ecross[i, j]
                dplus[1, 2] += r23[i] * r23[j] * eplus[i, j]
                dcross[1, 2] += r23[i] * r23[j] * ecross[i, j]
                dplus[0, 2] += r13[i] * r13[j] * eplus[i, j]
                dcross[0, 2] += r13[i] * r13[j] * ecross[i, j]

        dplus[1, 0] = dplus[0, 1]
        dcross[1, 0] = dcross[0, 1]
        dplus[2, 1] = dplus[1, 2]
        dcross[2, 1] = dcross[1, 2]
        dplus[2, 0] = dplus[0, 2]
        dcross[2, 0] = dcross[0, 2]

        for i in range(nc_michelson):
            fprs[i] = -(
                        + (dplus[i, (i + 1) % 3] * cosps + dcross[i, (i + 1) % 3] * sinps) * TR[i, (i + 1) % 3]
                        - (dplus[i, (i + 2) % 3] * cosps + dcross[i, (i + 2) % 3] * sinps) * TR[i, (i + 2) % 3]
                       ) / 2
            fcrs[i] = -(
                        + (-dplus[i, (i + 1) % 3] * sinps + dcross[i, (i + 1) % 3] * cosps) * TR[i, (i + 1) % 3]
                        - (-dplus[i, (i + 2) % 3] * sinps + dcross[i, (i + 2) % 3] * cosps) * TR[i, (i + 2) % 3]
                       ) / 2
            fpis[i] = -(
                        + (dplus[i, (i + 1) % 3] * cosps + dcross[i, (i + 1) % 3] * sinps) * TI[i, (i + 1) % 3]
                        - (dplus[i, (i + 2) % 3] * cosps + dcross[i, (i + 2) % 3] * sinps) * TI[i, (i + 2) % 3]
                       ) / 2
            fcis[i] = -(
                        + (-dplus[i, (i + 1) % 3] * sinps + dcross[i, (i + 1) % 3] * cosps) * TI[i, (i + 1) % 3]
                        - (-dplus[i, (i + 2) % 3] * sinps + dcross[i, (i + 2) % 3] * cosps) * TI[i, (i + 2) % 3]
                       ) / 2

        FpRs[0] = (2 * fprs[0] - fprs[1] - fprs[2]) / 3. * Aplus
        FcRs[0] = (2 * fcrs[0] - fcrs[1] - fcrs[2]) / 3. * Across

        FpRs[1] = (fprs[2] - fprs[1]) / np.sqrt(3.) * Aplus
        FcRs[1] = (fcrs[2] - fcrs[1]) / np.sqrt(3.) * Across

        FpRs[2] = (fprs[0] + fprs[1] + fprs[2]) / 3. * Aplus
        FcRs[2] = (fcrs[0] + fcrs[1] + fcrs[2]) / 3. * Across

        FpIs[0] = (2 * fpis[0] - fpis[1] - fpis[2]) / 3. * Aplus
        FcIs[0] = (2 * fcis[0] - fcis[1] - fcis[2]) / 3. * Across

        FpIs[1] = (fpis[2] - fpis[1]) / np.sqrt(3) * Aplus
        FcIs[1] = (fcis[2] - fcis[1]) / np.sqrt(3) * Across

        FpIs[2] = (fpis[0] + fpis[1] + fpis[2]) / 3. * Aplus
        FcIs[2] = (fcis[0] + fcis[1] + fcis[2]) / 3. * Across

        for itrc in range(nc_waveform):
            RRs[itrc, n] = FpRs[itrc] - FcIs[itrc]
            IIs[itrc, n] = FcRs[itrc] + FpIs[itrc]


@njit()
def get_xis_inplace(kv, ts, xas, yas, zas, xis, lc: LISAConstants) -> None:
    """Get time adjusted to guiding center for tensor basis"""
    kdotx = (xas * kv[0] + yas * kv[1] + zas * kv[2]) * (lc.Larm / CLIGHT)
    xis[:] = ts - kdotx


@njit()
def spacecraft_vec(ts, lc: LISAConstants):
    """Calculate the spacecraft positions as a function of time, with Larm scaling pulled out"""
    n_spacecraft = 3  # number of spacecraft (currently must be 3)

    xs = np.zeros((n_spacecraft, ts.size))
    ys = np.zeros((n_spacecraft, ts.size))
    zs = np.zeros((n_spacecraft, ts.size))
    alpha = 2 * np.pi * lc.fm * ts + lc.kappa0

    sa = np.sin(alpha)
    ca = np.cos(alpha)

    for i in range(n_spacecraft):
        beta = i * 2 / 3 * np.pi + lc.lambda0
        sb = np.sin(beta)
        cb = np.cos(beta)
        xs[i] = AU / lc.Larm * ca + AU / lc.Larm * lc.ec * (sa * ca * sb - (1. + sa * sa) * cb)
        ys[i] = AU / lc.Larm * sa + AU / lc.Larm * lc.ec * (sa * ca * cb - (1. + ca * ca) * sb)
        zs[i] = -np.sqrt(3) * AU / lc.Larm * lc.ec * (ca * cb + sa * sb)

    # guiding center
    xas = (xs[0] + xs[1] + xs[2]) / n_spacecraft
    yas = (ys[0] + ys[1] + ys[2]) / n_spacecraft
    zas = (zs[0] + zs[1] + zs[2]) / n_spacecraft
    return xs, ys, zs, xas, yas, zas
