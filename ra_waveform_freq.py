"""functions to compute rigid adiabatic response in frequency domain"""

import numpy as np
from numba import njit, prange

import global_const as gc


@njit(fastmath=True)
def RAantenna_inplace(spacecraft_channels, cosi, psi, phi, costh, ts, FFs, nf_low, NTs, kdotx, lc):
    """get the waveform for LISA given polarization angle, spacecraft, tensor basis and Fs, channel order AET"""
    RRs = spacecraft_channels.RR
    IIs = spacecraft_channels.II

    kv, eplus, ecross = get_tensor_basis(phi, costh)

    dplus  = np.zeros((3, 3))
    dcross = np.zeros((3, 3))

    TR = np.zeros((3, 3))
    TI = np.zeros((3, 3))

    x = np.zeros(3)
    y = np.zeros(3)
    z = np.zeros(3)

    kdr = np.zeros((3, 3))

    kdg = np.zeros(3)

    fprs = np.zeros(3)
    fcrs = np.zeros(3)
    fpis = np.zeros(3)
    fcis = np.zeros(3)

    FpRs = np.zeros(3)
    FcRs = np.zeros(3)
    FpIs = np.zeros(3)
    FcIs = np.zeros(3)

    r12 = np.zeros(3)
    r13 = np.zeros(3)
    r23 = np.zeros(3)
    r10 = np.zeros(3)
    r20 = np.zeros(3)
    r30 = np.zeros(3)

    betas = np.zeros(3)
    sbs = np.zeros(3)
    cbs = np.zeros(3)
    for itrc in range(0, 3):
        betas[itrc] = 2/3*np.pi*itrc+lc.lambda0
        sbs[itrc] = (gc.AU/lc.Larm*lc.ec)*np.sin(betas[itrc])
        cbs[itrc] = (gc.AU/lc.Larm*lc.ec)*np.cos(betas[itrc])

    cosps = np.cos(2*psi)
    sinps = np.sin(2*psi)

    Aplus = (1.+cosi**2)/2
    Across = -cosi

    # Main Loop
    for n in prange(0, NTs):
        # Barycenter time
        # pull out Larm scaling

        alpha = 2*np.pi*lc.fm*ts[n+nf_low]+lc.kappa0
        sa = np.sin(alpha)
        ca = np.cos(alpha)
        for itrc in range(0, 3):
            x[itrc] = (gc.AU/lc.Larm)*ca + sa*ca*sbs[itrc] - (1. + sa*sa)*cbs[itrc]
            y[itrc] = (gc.AU/lc.Larm)*sa + sa*ca*cbs[itrc] - (1. + ca*ca)*sbs[itrc]
            z[itrc] = -np.sqrt(3)*(ca*cbs[itrc] + sa*sbs[itrc])

        xa = (x[0]+x[1]+x[2])/3
        ya = (y[0]+y[1]+y[2])/3
        za = (z[0]+z[1]+z[2])/3
        kdotx[n] = (lc.Larm/gc.CLIGHT)*(xa*kv[0]+ya*kv[1]+za*kv[2])

        fr = 1/(2*lc.fstr)*FFs[n+nf_low]

        # Unit separation vector from spacecraft i to j
        r12[0] = x[1]-x[0]
        r12[1] = y[1]-y[0]
        r12[2] = z[1]-z[0]
        r13[0] = x[2]-x[0]
        r13[1] = y[2]-y[0]
        r13[2] = z[2]-z[0]
        r23[0] = x[2]-x[1]
        r23[1] = y[2]-y[1]
        r23[2] = z[2]-z[1]

        # These are not unit vectors. Just pulling out the lc.Larm scaling
        r10[0] = xa-x[0]
        r10[1] = ya-y[0]
        r10[2] = za-z[0]
        r20[0] = xa-x[1]
        r20[1] = ya-y[1]
        r20[2] = za-z[1]
        r30[0] = xa-x[2]
        r30[1] = ya-y[2]
        r30[2] = za-z[2]

        kdr[:] = 0.
        kdg[:] = 0.
        for k in range(0, 3):
            kdr[0, 1] += kv[k]*r12[k]
            kdr[0, 2] += kv[k]*r13[k]
            kdr[1, 2] += kv[k]*r23[k]
            kdg[0] += kv[k]*r10[k]
            kdg[1] += kv[k]*r20[k]
            kdg[2] += kv[k]*r30[k]

        for i in range(0, 2):
            for j in range(i+1, 3):
                q1 = fr*(1.-kdr[i, j])
                q2 = fr*(1.+kdr[i, j])
                q3 = -fr*(3.+kdr[i, j]-2*kdg[i])
                q4 = -fr*(1.+kdr[i, j]-2*kdg[i])  # TODO missing 1/sqrt(3) on kdg?
                q5 = -fr*(3.-kdr[i, j]-2*kdg[j])
                q6 = -fr*(1.-kdr[i, j]-2*kdg[j])  # TODO missing 1/sqrt(3) on kdg?
                sincq1 = np.sin(q1)/q1/2
                sincq2 = np.sin(q2)/q2/2
                # Real part of T from eq B9 in PhysRevD.101.124008
                TR[i, j] = sincq1*np.cos(q3)+sincq2*np.cos(q4)   # goes to 1 when f/fstar small
                # imaginary part of T
                TI[i, j] = sincq1*np.sin(q3)+sincq2*np.sin(q4)   # goes to 0 when f/fstar small
                # save ops computing other triangle simultaneously
                TR[j, i] = sincq2*np.cos(q5)+sincq1*np.cos(q6)   # goes to 1 when f/fstar small
                TI[j, i] = sincq2*np.sin(q5)+sincq1*np.sin(q6)   # goes to 0 when f/fstar small

        dplus[:] = 0.
        dcross[:] = 0.
        # Convenient quantities d+ & dx
        for i in range(0, 3):
            for j in range(0, 3):
                dplus[0, 1]  += r12[i]*r12[j]*eplus[i, j]
                dcross[0, 1] += r12[i]*r12[j]*ecross[i, j]
                dplus[1, 2]  += r23[i]*r23[j]*eplus[i, j]
                dcross[1, 2] += r23[i]*r23[j]*ecross[i, j]
                dplus[0, 2]  += r13[i]*r13[j]*eplus[i, j]
                dcross[0, 2] += r13[i]*r13[j]*ecross[i, j]

        dplus[1, 0] = dplus[0, 1]
        dcross[1, 0] = dcross[0, 1]
        dplus[2, 1] = dplus[1, 2]
        dcross[2, 1] = dcross[1, 2]
        dplus[2, 0] = dplus[0, 2]
        dcross[2, 0] = dcross[0, 2]

        for i in range(0, 3):
            fprs[i] = -(
                        + (dplus[i, (i+1) % 3]*cosps + dcross[i, (i+1) % 3]*sinps)*TR[i, (i+1) % 3]
                        - (dplus[i, (i+2) % 3]*cosps + dcross[i, (i+2) % 3]*sinps)*TR[i, (i+2) % 3]
                       )/2
            fcrs[i] = -(
                        + (-dplus[i, (i+1) % 3]*sinps + dcross[i, (i+1) % 3]*cosps)*TR[i, (i+1) % 3]
                        - (-dplus[i, (i+2) % 3]*sinps + dcross[i, (i+2) % 3]*cosps)*TR[i, (i+2) % 3]
                       )/2
            fpis[i] = -(
                        + (dplus[i, (i+1) % 3]*cosps + dcross[i, (i+1) % 3]*sinps)*TI[i, (i+1) % 3]
                        - (dplus[i, (i+2) % 3]*cosps + dcross[i, (i+2) % 3]*sinps)*TI[i, (i+2) % 3]
                       )/2
            fcis[i] = -(
                        + (-dplus[i, (i+1) % 3]*sinps + dcross[i, (i+1) % 3]*cosps)*TI[i, (i+1) % 3]
                        - (-dplus[i, (i+2) % 3]*sinps + dcross[i, (i+2) % 3]*cosps)*TI[i, (i+2) % 3]
                       )/2

        FpRs[0] = (2*fprs[0]-fprs[1]-fprs[2])/3.*Aplus
        FcRs[0] = (2*fcrs[0]-fcrs[1]-fcrs[2])/3.*Across

        FpRs[1] = (fprs[2]-fprs[1])/np.sqrt(3)*Aplus
        FcRs[1] = (fcrs[2]-fcrs[1])/np.sqrt(3)*Across

        FpRs[2] = (fprs[0]+fprs[1]+fprs[2])/3.*Aplus
        FcRs[2] = (fcrs[0]+fcrs[1]+fcrs[2])/3.*Across

        FpIs[0] = (2*fpis[0]-fpis[1]-fpis[2])/3.*Aplus
        FcIs[0] = (2*fcis[0]-fcis[1]-fcis[2])/3.*Across

        FpIs[1] = (fpis[2]-fpis[1])/np.sqrt(3)*Aplus
        FcIs[1] = (fcis[2]-fcis[1])/np.sqrt(3)*Across

        FpIs[2] = (fpis[0]+fpis[1]+fpis[2])/3.*Aplus
        FcIs[2] = (fcis[0]+fcis[1]+fcis[2])/3.*Across

        for itrc in range(0, 3):
            RRs[itrc, n] = FpRs[itrc] - FcIs[itrc]
            IIs[itrc, n] = FcRs[itrc] + FpIs[itrc]


@njit()
def get_tensor_basis(phi, costh):
    """get tensor basis"""
    # Calculate cos and sin of sky position, inclination, polarization
    sinth = np.sqrt(1.0-costh**2)
    cosph = np.cos(phi)
    sinph = np.sin(phi)

    kv = np.zeros(3)
    u = np.zeros(3)
    v = np.zeros(3)

    kv[0] = -sinth*cosph
    kv[1] = -sinth*sinph
    kv[2] = -costh

    u[0] =  sinph
    u[1] = -cosph
    u[2] =  0.

    v[0] =  -costh*cosph
    v[1] =  -costh*sinph
    v[2] = sinth

    eplus = np.zeros((3, 3))
    ecross = np.zeros((3, 3))

    for i in range(0, 3):
        for j in range(0, 3):
            eplus[i, j]  = u[i]*u[j] - v[i]*v[j]
            ecross[i, j] = u[i]*v[j] + v[i]*u[j]
    return kv, eplus, ecross


@njit()
def get_xis_inplace(kv, ts, xas, yas, zas, xis, lc):
    """get time adjusted to guiding center for tensor basis"""
    kdotx = (xas*kv[0]+yas*kv[1]+zas*kv[2])*lc.Larm/gc.CLIGHT
    xis[:] = ts-kdotx


@njit()
def spacecraft_vec(ts, lc):
    """calculate the spacecraft positions as a function of time, with Larm scaling pulled out"""
    xs = np.zeros((3, ts.size))
    ys = np.zeros((3, ts.size))
    zs = np.zeros((3, ts.size))
    alpha = 2*np.pi*lc.fm*ts+lc.kappa0

    sa = np.sin(alpha)
    ca = np.cos(alpha)

    for i in range(0, 3):
        beta = i*2/3*np.pi+lc.lambda0
        sb = np.sin(beta)
        cb = np.cos(beta)
        xs[i] = gc.AU/lc.Larm*ca + gc.AU/lc.Larm*lc.ec*(sa*ca*sb - (1. + sa*sa)*cb)
        ys[i] = gc.AU/lc.Larm*sa + gc.AU/lc.Larm*lc.ec*(sa*ca*cb - (1. + ca*ca)*sb)
        zs[i] = -np.sqrt(3)*gc.AU/lc.Larm*lc.ec*(ca*cb + sa*sb)

    # guiding center
    xas = (xs[0]+xs[1]+xs[2])/3
    yas = (ys[0]+ys[1]+ys[2])/3
    zas = (zs[0]+zs[1]+zs[2])/3
    return xs, ys, zs, xas, yas, zas
