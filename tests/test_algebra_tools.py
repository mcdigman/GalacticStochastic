"""Unit tests for the functions in algebra_tools."""

import numba
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from LisaWaveformTools.algebra_tools import gradient_uniform_inplace, stabilized_gradient_uniform_inplace


def test_gradient_uniform_inplace_raise_bad_shape1():
    """Check for an assertion error if the shape is wrong"""
    DT = 0.1
    x = np.zeros((3, 100))
    y = np.zeros((2, 100))
    with pytest.raises(AssertionError):
        gradient_uniform_inplace(x, y, DT)


def test_gradient_uniform_inplace_raise_bad_shape2():
    """Check for an assertion error if the shape is wrong"""
    DT = 0.1
    x = np.zeros((3, 100))
    y = np.zeros((3, 101))
    with pytest.raises(AssertionError):
        gradient_uniform_inplace(x, y, DT)


def test_gradient_uniform_inplace_raise_bad_shape3():
    """Check for an assertion error if the shape is wrong"""
    DT = 0.1
    x = np.zeros((3, 100))
    y = np.zeros((2, 101))
    with pytest.raises(AssertionError):
        gradient_uniform_inplace(x, y, DT)


def test_gradient_uniform_inplace_raise_bad_shape4():
    """Check for an assertion error if the shape is wrong"""
    DT = 0.1
    x = np.zeros((3, 100))
    y = np.zeros((3, 100, 1))
    with pytest.raises(AssertionError):
        gradient_uniform_inplace(x, y, DT)


def test_gradient_uniform_inplace_raise_bad_shape5():
    """Check for an assertion error if the shape is wrong"""
    DT = 0.1
    x = np.zeros((3, 1))
    y = np.zeros((3, 1))
    with pytest.raises(AssertionError):
        gradient_uniform_inplace(x, y, DT)


def test_gradient_uniform_inplace_raise_bad_shape6():
    """Check for an assertion error if the shape is wrong"""
    DT = 0.1
    x = np.zeros((3, 0))
    y = np.zeros((3, 0))
    with pytest.raises(AssertionError):
        gradient_uniform_inplace(x, y, DT)


def test_gradient_uniform_inplace_raise_bad_shape7():
    """Check for an assertion error if the shape is wrong"""
    DT = 0.1
    x = np.zeros(100)
    y = np.zeros(100)
    with pytest.raises((AssertionError, numba.errors.TypingError)):
        gradient_uniform_inplace(x, y, DT)


def test_gradient_uniform_inplace_raise_bad_shape8():
    """Check for an assertion error if the shape is wrong"""
    DT = 0.1
    x = np.zeros((3, 100, 1))
    y = np.zeros((3, 100, 1))
    with pytest.raises((AssertionError, numba.errors.TypingError)):
        gradient_uniform_inplace(x, y, DT)


def test_gradient_uniform_inplace_raise_bad_shape9():
    """Check for an assertion error if the shape is wrong"""
    DT = 0.1
    x = np.zeros((3, 100, 1))
    y = np.zeros((3, 100))
    with pytest.raises((AssertionError, numba.errors.TypingError)):
        gradient_uniform_inplace(x, y, DT)


def test_stabilized_gradient_uniform_inplace_raise_bad_shape1():
    """Check for an assertion error if the shape is wrong"""
    DT = 0.1
    x = np.zeros(100)
    dxdt = np.zeros(100)
    y = np.zeros((3, 100))
    dydt = np.zeros((3, 100, 1))
    with pytest.raises(AssertionError):
        stabilized_gradient_uniform_inplace(x, dxdt, y, dydt, DT)


def test_stabilized_gradient_uniform_inplace_raise_bad_shape2():
    """Check for an assertion error if the shape is wrong"""
    DT = 0.1
    x = np.zeros((100, 1))
    dxdt = np.zeros(100)
    y = np.zeros((3, 100))
    dydt = np.zeros((3, 100))
    with pytest.raises((AssertionError, numba.errors.TypingError)):
        stabilized_gradient_uniform_inplace(x, dxdt, y, dydt, DT)


def test_stabilized_gradient_uniform_inplace_raise_bad_shape3():
    """Check for an assertion error if the shape is wrong"""
    DT = 0.1
    x = np.zeros(100)
    dxdt = np.zeros(100)
    y = np.zeros((4, 100))
    dydt = np.zeros((3, 100))
    with pytest.raises(AssertionError):
        stabilized_gradient_uniform_inplace(x, dxdt, y, dydt, DT)


def test_stabilized_gradient_uniform_inplace_raise_bad_shape4():
    """Check for an assertion error if the shape is wrong"""
    DT = 0.1
    x = np.zeros(100)
    dxdt = np.zeros(100)
    y = np.zeros((3, 101))
    dydt = np.zeros((3, 100))
    with pytest.raises(AssertionError):
        stabilized_gradient_uniform_inplace(x, dxdt, y, dydt, DT)


def test_stabilized_gradient_uniform_inplace_raise_bad_shape5():
    """Check for an assertion error if the shape is wrong"""
    DT = 0.1
    x = np.zeros(101)
    dxdt = np.zeros(100)
    y = np.zeros((3, 100))
    dydt = np.zeros((3, 100))
    with pytest.raises(AssertionError):
        stabilized_gradient_uniform_inplace(x, dxdt, y, dydt, DT)


def test_stabilized_gradient_uniform_inplace_raise_bad_shape6():
    """Check for an assertion error if the shape is wrong"""
    DT = 0.1
    x = np.zeros(100)
    dxdt = np.zeros(101)
    y = np.zeros((3, 100))
    dydt = np.zeros((3, 100))
    with pytest.raises(AssertionError):
        stabilized_gradient_uniform_inplace(x, dxdt, y, dydt, DT)


def test_stabilized_gradient_uniform_inplace_raise_bad_shape7():
    """Check for an assertion error if the shape is wrong"""
    DT = 0.1
    x = np.zeros(100)
    dxdt = np.zeros(100)
    y = np.zeros((3, 100))
    dydt = np.zeros((3, 101))
    with pytest.raises(AssertionError):
        stabilized_gradient_uniform_inplace(x, dxdt, y, dydt, DT)


def test_stabilized_gradient_uniform_inplace_raise_bad_shape8():
    """Check for an assertion error if the shape is wrong"""
    DT = 0.1
    x = np.zeros(1)
    dxdt = np.zeros(1)
    y = np.zeros((3, 1))
    dydt = np.zeros((3, 1))
    with pytest.raises(AssertionError):
        stabilized_gradient_uniform_inplace(x, dxdt, y, dydt, DT)


def test_stabilized_gradient_uniform_inplace_raise_bad_shape9():
    """Check for an assertion error if the shape is wrong"""
    DT = 0.1
    x = np.zeros(0)
    dxdt = np.zeros(0)
    y = np.zeros((3, 0))
    dydt = np.zeros((3, 0))
    with pytest.raises(AssertionError):
        stabilized_gradient_uniform_inplace(x, dxdt, y, dydt, DT)


def test_stabilized_gradient_uniform_inplace_raise_bad_shape10():
    """Check for an assertion error if the shape is wrong"""
    DT = 0.1
    x = np.zeros(100)
    dxdt = np.zeros((100, 1))
    y = np.zeros((3, 100))
    dydt = np.zeros((3, 100))
    with pytest.raises((AssertionError, numba.errors.TypingError)):
        stabilized_gradient_uniform_inplace(x, dxdt, y, dydt, DT)


def test_stabilized_gradient_uniform_inplace_raise_bad_shape11():
    """Check for an assertion error if the shape is wrong"""
    DT = 0.1
    x = np.zeros(100)
    dxdt = np.zeros(100)
    y = np.zeros((3, 100, 1))
    dydt = np.zeros((3, 100))
    with pytest.raises((AssertionError, numba.errors.TypingError)):
        stabilized_gradient_uniform_inplace(x, dxdt, y, dydt, DT)


def test_stabilized_gradient_uniform_inplace_raise_bad_shape12():
    """Check for an assertion error if the shape is wrong"""
    DT = 0.1
    x = np.zeros(100)
    dxdt = np.zeros(100)
    y = np.zeros((3, 100, 1))
    dydt = np.zeros((3, 100, 1))
    with pytest.raises((AssertionError, numba.errors.TypingError)):
        stabilized_gradient_uniform_inplace(x, dxdt, y, dydt, DT)


def test_stabilized_gradient_uniform_inplace_raise_bad_shape13():
    """Check for an assertion error if the shape is wrong"""
    DT = 0.1
    x = np.zeros(101)
    dxdt = np.zeros(101)
    y = np.zeros((3, 100))
    dydt = np.zeros((3, 100))
    with pytest.raises((AssertionError, numba.errors.TypingError)):
        stabilized_gradient_uniform_inplace(x, dxdt, y, dydt, DT)


def test_stabilized_gradient_uniform_inplace_raise_bad_shape14():
    """Check for an assertion error if the shape is wrong"""
    DT = 0.1
    x = np.zeros(100)
    dxdt = np.zeros(100)
    y = np.zeros(100)
    dydt = np.zeros(100)
    with pytest.raises((AssertionError, numba.errors.TypingError)):
        stabilized_gradient_uniform_inplace(x, dxdt, y, dydt, DT)


def test_stabilized_gradient_uniform_inplace_raise_bad_shape15():
    """Check for an assertion error if the shape is wrong"""
    DT = 0.1
    x = np.zeros((3, 100))
    dxdt = np.zeros((3, 100))
    y = np.zeros(100)
    dydt = np.zeros(100)
    with pytest.raises((AssertionError, numba.errors.TypingError)):
        stabilized_gradient_uniform_inplace(x, dxdt, y, dydt, DT)


def get_scaling_test_case_helper(t_scaling, T, DT):
    """Get various cases of scaling for gradient computation tests"""
    nt_loc = T.size
    if t_scaling == 'const1':
        x = np.full(nt_loc, 1.0)
        dxdt = np.zeros(nt_loc)
    elif t_scaling == 'lin1':
        x = T
        dxdt = np.full(nt_loc, 1.0)
    elif t_scaling == 'abs1':
        if nt_loc < 1:
            return None, None
        x = np.abs(T - T[nt_loc // 2])
        dxdt = np.zeros(nt_loc)
        dxdt[nt_loc // 2] = 0.0
        dxdt[nt_loc // 2 + 1:] = 1.0
        dxdt[: nt_loc // 2] = -1.0
    elif t_scaling == 'quad1':
        x = T**2
        dxdt = 2 * T
    elif t_scaling == 'tri1':
        x = T**3
        dxdt = 3 * T**2
    elif t_scaling == 'quart1':
        x = T**4
        dxdt = 4 * T**3
    elif t_scaling == 'quint1':
        x = T**5
        dxdt = 5 * T**4
    elif t_scaling == 'sqrt1':
        with np.errstate(invalid='ignore', divide='ignore'):
            x = np.sqrt(T)
            dxdt = 1 / (2 * x)
    elif t_scaling == 'cbrt1':
        x = np.cbrt(T)
        with np.errstate(invalid='ignore', divide='ignore'):
            dxdt = 1 / (3 * x**2)
    elif t_scaling == 'sin1':
        x = np.sin(2 * np.pi * T)
        dxdt = 2 * np.pi * np.cos(2 * np.pi * T)
    elif t_scaling == 'osc1':
        x = (-1) ** np.arange(0, nt_loc)
        dxdt = np.zeros(nt_loc)
        if nt_loc < 2:
            return None, None
        dxdt[0] = -2 / DT
        dxdt[-1] = -2 / DT * (-1) ** (nt_loc)
    elif t_scaling == 'heaviside1':
        x = np.heaviside(T - DT * nt_loc / 2, 0.5)
        if nt_loc < 2:
            return x, None
        dxdt = np.gradient(x, DT)
    elif t_scaling == 'dirac1':
        if nt_loc < 1:
            return None, None
        x = np.zeros(nt_loc)
        x[nt_loc // 2] = 1.0
        if nt_loc < 2:
            return x, None
        dxdt = np.gradient(x, DT)
    elif t_scaling == 'dirac2':
        if nt_loc < 3:
            return None, None
        x = np.zeros(nt_loc)
        x[nt_loc // 2 + 1] = 1.0
        if nt_loc < 2:
            return x, None
        dxdt = np.gradient(x, DT)
    elif t_scaling == 'dirac3':
        if nt_loc < 1:
            return None, None
        x = np.zeros(nt_loc)
        x[0] = 1.0
        if nt_loc < 2:
            return x, None
        dxdt = np.gradient(x, DT)
    elif t_scaling == 'dirac4':
        if nt_loc < 1:
            return None, None
        x = np.zeros(nt_loc)
        x[-1] = 1.0
        if nt_loc < 2:
            return x, None
        dxdt = np.gradient(x, DT)
    elif t_scaling == 'dirac5':
        if nt_loc < 2:
            return None, None
        x = np.zeros(nt_loc)
        x[1] = 1.0
        if nt_loc < 2:
            return x, None
        dxdt = np.gradient(x, DT)
    elif t_scaling == 'dirac6':
        if nt_loc < 2:
            return None, None
        x = np.zeros(nt_loc)
        x[-2] = 1.0
        if nt_loc < 2:
            return x, None
        dxdt = np.gradient(x, DT)
    elif t_scaling == 'dirac7':
        x = np.zeros(nt_loc)
        x[2] = 1.0
        if nt_loc < 2:
            return x, None
        dxdt = np.gradient(x, DT)
    elif t_scaling == 'dirac8':
        if nt_loc < 3:
            return None, None
        x = np.zeros(nt_loc)
        x[-3] = 1.0
        if nt_loc < 2:
            return x, None
        dxdt = np.gradient(x, DT)
    elif t_scaling == 'dirac9':
        if nt_loc < 3:
            return None, None
        x = np.zeros(nt_loc)
        x[nt_loc // 2] = 1.0
        x[nt_loc // 2 + 1] = -1.0
        if nt_loc < 2:
            return x, None
        dxdt = np.gradient(x, DT)

    elif t_scaling == 'exp1':
        x = np.exp(T)
        dxdt = np.exp(T)
    elif t_scaling == 'log1':
        with np.errstate(divide='ignore', invalid='ignore'):
            x = np.log(T)
            dxdt = 1 / T
    else:
        msg = 'unrecognized option for t_scaling'
        raise ValueError(msg)

    return x, dxdt


@pytest.mark.parametrize('DT', [0.05, 1])
@pytest.mark.parametrize('t0', [-101.0, -50.0, 0.0, 50.0, 101.0])
@pytest.mark.parametrize('nt_loc', [2, 3, 4, 100, 101])
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
def test_gradient_uniform_inplace(t_scaling, DT, t0, nt_loc):
    """Test the function gradient_homog_2d_implace produces expected results"""
    nc_waveform = 3
    T = np.arange(0.0, nt_loc) * DT + t0
    channel_scale_mult = np.array([0.9, 0.5, 0.3])

    channel_t_scale, _ = get_scaling_test_case_helper(t_scaling, T, DT)

    if channel_t_scale is None:
        return

    ys = np.outer(channel_scale_mult, channel_t_scale)

    result = np.zeros((nc_waveform, nt_loc))

    # check we get the same results if we iterate over every dimension separately
    result2 = np.zeros((nc_waveform, nt_loc))

    gradient_uniform_inplace(ys, result, DT)

    for itrc in range(nc_waveform):
        gradient_uniform_inplace(ys[itrc:itrc + 1], result2[itrc:itrc + 1], DT)

    assert_array_equal(result, result2)

    del result2

    gradient_exp = np.gradient(ys, DT, axis=1, edge_order=1)

    assert_allclose(result, gradient_exp, atol=1.0e-14, rtol=1.0e-14)


@pytest.mark.parametrize('DT', [0.05, 1])
@pytest.mark.parametrize('t0', [-101.0, -51.0, 0.0, 50.0, 101.0])
@pytest.mark.parametrize('nt_loc', [4, 100, 101])
@pytest.mark.parametrize(
    't_scale_base',
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
@pytest.mark.parametrize(
    't_scale_perturb',
    [
        'const1',
        'lin1',
        'quad1',
        'tri1',
        'quart1',
        'quint1',
        'heaviside1',
        'dirac1',
        'exp1',
    ],
)
@pytest.mark.parametrize('perturb_mult', [-1.0, 0.0, 0.1, 1.0])
def test_stabilized_gradient_uniform_inplace(t_scale_base, t_scale_perturb, perturb_mult, DT, t0, nt_loc):
    """Test the function gradient_homog_2d_implace produces expected results"""
    nc_waveform = 3
    T = np.arange(0.0, nt_loc) * DT + t0
    x_scale_mult = np.array([1.0, 1.0, 1.0])
    x_perturb_scale_mult = np.array([0.95, 1.05, 1.0]) * perturb_mult

    x, dxdt = get_scaling_test_case_helper(t_scale_base, T, DT)
    if x is None or dxdt is None:
        return

    x_perturb, dxdt_perturb = get_scaling_test_case_helper(t_scale_perturb, T, DT)
    if (
        x_perturb is None
        or dxdt_perturb is None
        or not np.all(np.isfinite(dxdt_perturb))
        or not np.all(np.isfinite(x_perturb))
    ):
        return

    with np.errstate(invalid='ignore'):
        y_unperturbed = np.outer(x_scale_mult, x)
        dydt_unperturbed = np.outer(x_scale_mult, dxdt)

        y_perturber = np.outer(x_perturb_scale_mult, x_perturb)
        dydt_perturber = np.outer(x_perturb_scale_mult, dxdt_perturb)

    y_perturbed = y_unperturbed + y_perturber
    dydt_perturbed = dydt_unperturbed + dydt_perturber

    result = np.zeros((nc_waveform, nt_loc))

    # check we get the same results if we iterate over every dimension separately
    result2 = np.zeros((nc_waveform, nt_loc))

    # check we raise an error if a gradient cannot be computed
    stabilized_gradient_uniform_inplace(x, dxdt, y_perturbed, result, DT)

    for itrc in range(nc_waveform):
        stabilized_gradient_uniform_inplace(x, dxdt, y_perturbed[itrc:itrc + 1], result2[itrc:itrc + 1], DT)

    assert_array_equal(result, result2)
    del result2

    # the expected result
    gradient_perturber = np.gradient(y_perturber, DT, axis=1, edge_order=1)
    gradient_exp = dydt_unperturbed + gradient_perturber

    # the result without this method
    gradient_basic = np.gradient(y_perturbed, DT, axis=1, edge_order=1)

    for itrc in range(nc_waveform):
        # check locations where result is non-finite match;
        # shouldn't matter much if nan or inf as long as they are in the same places
        mask1 = np.isfinite(result[itrc])
        mask2 = np.isfinite(gradient_exp[itrc])
        mask3 = np.isfinite(gradient_basic[itrc])
        # check we don't produce a finite result any where we can't;
        # we will get a non-finite result if *either* result would produce a non-finite result
        assert np.all(mask1 <= mask2)
        assert np.all(mask1 <= mask3)
        assert_array_equal(mask1, (mask2 & mask3))

        # indices where finite except at the edge
        mask_ind = np.argwhere(mask1[1:nt_loc - 1]).flatten() + 1
        # check general closeness to expectation
        assert_allclose(result[itrc, mask1], gradient_exp[itrc, mask1], atol=1.0e-12, rtol=1.0e-12)

        # check result matches input if there is no perturber
        if perturb_mult == 0.0:
            assert_allclose(result[itrc, mask1], x_scale_mult[itrc] * dxdt[mask1], atol=1.0e-14, rtol=1.0e-14)

        if mask_ind.size > 0:
            # check both components separately
            assert_allclose(
                result[itrc, mask_ind] - gradient_perturber[itrc, mask_ind],
                dydt_unperturbed[itrc, mask_ind],
                atol=max(np.abs(result[itrc, mask_ind]).max() * 1.0e-11, 1.0e-14),
                rtol=1.0e-12,
            )
            assert_allclose(
                result[itrc, mask_ind] - dydt_unperturbed[itrc, mask_ind],
                gradient_perturber[itrc, mask_ind],
                atol=max(np.abs(result[itrc, mask_ind]).max() * 1.0e-11, 1.0e-14),
                rtol=1.0e-12,
            )
            # check that the approximation is an improvement on average
            # by asserting that it is not more than 10% worse in any test case
            if mask_ind.size > 4 and perturb_mult != 0.0 and perturb_mult != -1.0:
                nrm1 = np.linalg.norm(dydt_perturbed[itrc, mask_ind] - gradient_basic[itrc, mask_ind])
                nrm2 = np.linalg.norm(dydt_perturbed[itrc, mask_ind] - result[itrc, mask_ind])
                assert max(float(nrm1), 1.e-9) >= 0.9 * float(nrm2)
