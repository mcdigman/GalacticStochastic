"""Unit tests for the functions in algebra_tools."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from LisaWaveformTools.algebra_tools import gradient_homog_2d_inplace


def test_gradient_homog_2d_inplace_raise_bad_shape1():
    """Check for an assertion error if the shape is wrong"""
    DT = 0.1
    x = np.zeros((3, 100))
    y = np.zeros((2, 100))
    with pytest.raises(AssertionError):
        gradient_homog_2d_inplace(x, y, DT)


def test_gradient_homog_2d_inplace_raise_bad_shape2():
    """Check for an assertion error if the shape is wrong"""
    DT = 0.1
    x = np.zeros((3, 100))
    y = np.zeros((3, 101))
    with pytest.raises(AssertionError):
        gradient_homog_2d_inplace(x, y, DT)


def test_gradient_homog_2d_inplace_raise_bad_shape3():
    """Check for an assertion error if the shape is wrong"""
    DT = 0.1
    x = np.zeros((3, 100))
    y = np.zeros((2, 101))
    with pytest.raises(AssertionError):
        gradient_homog_2d_inplace(x, y, DT)


def test_gradient_homog_2d_inplace_raise_bad_shape4():
    """Check for an assertion error if the shape is wrong"""
    DT = 0.1
    x = np.zeros((3, 100))
    y = np.zeros((3, 100, 1))
    with pytest.raises(AssertionError):
        gradient_homog_2d_inplace(x, y, DT)


@pytest.mark.parametrize('DT', [0.05, 1])
@pytest.mark.parametrize('t0', [-101.0, -50.0, 0.0, 50.0, 101.0])
@pytest.mark.parametrize('nt_loc', [0, 1, 2, 3, 4, 100, 101])
@pytest.mark.parametrize(
    't_scaling',
    [
        'const1',
        'lin1',
        'abs1',
        'quad1',
        'tri1',
        'quart1',
        'quint1',
        'sqrt1',
        'cbrt1',
        'osc1',
        'osc2',
        'heaviside1',
        'dirac1',
        'dirac2',
        'dirac3',
        'dirac4',
        'dirac5',
        'dirac6',
        'dirac8',
        'dirac9',
        'exp1',
        'log1',
    ],
)
def test_gradient_homog_2d_inplace(t_scaling, DT, t0, nt_loc):
    """Test the function gradient_homog_2d_implace produces expected results"""
    nc_waveform = 3
    T = np.arange(0.0, nt_loc) * DT + t0
    channel_scale_mult = np.array([0.9, 0.5, 0.3])

    if t_scaling == 'const1':
        channel_t_scale = np.full(nt_loc, 1.0)
    elif t_scaling == 'lin1':
        channel_t_scale = T
    elif t_scaling == 'abs1':
        if nt_loc < 1:
            return
        channel_t_scale = np.abs(T - T[nt_loc // 2])
    elif t_scaling == 'quad1':
        channel_t_scale = T**2
    elif t_scaling == 'tri1':
        channel_t_scale = T**3
    elif t_scaling == 'quart1':
        channel_t_scale = T**4
    elif t_scaling == 'quint1':
        channel_t_scale = T**5
    elif t_scaling == 'sqrt1':
        with np.errstate(invalid='ignore'):
            channel_t_scale = np.sqrt(T)
    elif t_scaling == 'cbrt1':
        channel_t_scale = np.cbrt(T)
    elif t_scaling == 'sin1':
        channel_t_scale = np.sin(2 * np.pi * T)
    elif t_scaling == 'osc1' or t_scaling == 'osc2':
        channel_t_scale = (-1) ** np.arange(0, nt_loc)
    elif t_scaling == 'heaviside1':
        channel_t_scale = np.heaviside(T - DT * nt_loc / 2, 0.5)
    elif t_scaling == 'dirac1':
        if nt_loc < 1:
            return
        channel_t_scale = np.zeros(nt_loc)
        channel_t_scale[nt_loc // 2] = 1.0
    elif t_scaling == 'dirac2':
        if nt_loc < 3:
            return
        channel_t_scale = np.zeros(nt_loc)
        channel_t_scale[nt_loc // 2 + 1] = 1.0
    elif t_scaling == 'dirac3':
        if nt_loc < 1:
            return
        channel_t_scale = np.zeros(nt_loc)
        channel_t_scale[0] = 1.0
    elif t_scaling == 'dirac4':
        if nt_loc < 1:
            return
        channel_t_scale = np.zeros(nt_loc)
        channel_t_scale[-1] = 1.0
    elif t_scaling == 'dirac5':
        if nt_loc < 2:
            return
        channel_t_scale = np.zeros(nt_loc)
        channel_t_scale[1] = 1.0
    elif t_scaling == 'dirac6':
        if nt_loc < 2:
            return
        channel_t_scale = np.zeros(nt_loc)
        channel_t_scale[-2] = 1.0
    elif t_scaling == 'dirac7':
        channel_t_scale = np.zeros(nt_loc)
        channel_t_scale[2] = 1.0
    elif t_scaling == 'dirac8':
        if nt_loc < 3:
            return
        channel_t_scale = np.zeros(nt_loc)
        channel_t_scale[-3] = 1.0
    elif t_scaling == 'dirac9':
        if nt_loc < 3:
            return
        channel_t_scale = np.zeros(nt_loc)
        channel_t_scale[nt_loc // 2] = 1.0
        channel_t_scale[nt_loc // 2 + 1] = -1.0

    elif t_scaling == 'exp1':
        channel_t_scale = np.exp(T)
    elif t_scaling == 'log1':
        with np.errstate(divide='ignore', invalid='ignore'):
            channel_t_scale = np.log(T)
    else:
        msg = 'unrecognized option for t_scaling'
        raise ValueError(msg)

    ys = np.outer(channel_scale_mult, channel_t_scale)

    result = np.zeros((nc_waveform, nt_loc))

    # check we get the same results if we iterate over every dimension separately
    result2 = np.zeros((nc_waveform, nt_loc))

    # check we raise an error if a gradient cannot be computed
    if nt_loc < 2:
        with pytest.raises(AssertionError) as msg:
            gradient_homog_2d_inplace(ys, result, DT)
        with pytest.raises(AssertionError):
            gradient_homog_2d_inplace(ys, result2[0:1], DT)
        return
    gradient_homog_2d_inplace(ys, result, DT)

    for itrc in range(nc_waveform):
        gradient_homog_2d_inplace(ys[itrc:itrc + 1], result2[itrc:itrc + 1], DT)

    assert_array_equal(result, result2)
    result2 = None

    gradient_exp = np.gradient(ys, DT, axis=1, edge_order=1)

    assert_allclose(result, gradient_exp, atol=1.0e-14, rtol=1.0e-14)
