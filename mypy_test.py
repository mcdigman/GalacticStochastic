# mypy: ignore-errors
from __future__ import annotations

from typing import overload

import numpy as np
from numpy.typing import NDArray


@overload
def test_type1(a: NDArray[np.floating]) -> NDArray[np.floating]:  ...
@overload
def test_type1(a: NDArray[np.complexfloating]) -> NDArray[np.complexfloating]: ...
def test_type1(a: NDArray[np.floating] | NDArray[np.complexfloating]) -> NDArray[np.floating] | NDArray[np.complexfloating]:
    return a


@overload
def test_type2(a: NDArray[np.complexfloating]) -> NDArray[np.complexfloating]: ...
@overload
def test_type2(a: NDArray[np.floating]) -> NDArray[np.floating]:  ...
def test_type2(a: NDArray[np.floating] | NDArray[np.complexfloating]) -> NDArray[np.floating] | NDArray[np.complexfloating]:
    return a


@overload
def test_type3(a: np.floating) -> np.floating:  ...
@overload
def test_type3(a: np.complexfloating) -> np.complexfloating: ...
def test_type3(a: np.floating | np.complexfloating) -> np.floating | np.complexfloating:
    return a


@overload
def test_type4(a: np.complexfloating) -> np.floating:  ...
@overload
def test_type4(a: np.floating) -> np.floating: ...
def test_type4(a: np.floating | np.complexfloating) -> np.floating | np.complexfloating:
    return a


@overload
def test_type5(a: np.float32) -> np.float32:  ...
@overload
def test_type5(a: np.complex128) -> np.complex128: ...
def test_type5(a: np.float32 | np.complex128) -> np.float32 | np.complex128:
    return a


@overload
def test_type6(a: float) -> float: ...
@overload
def test_type6(a: complex) -> float:  ...
def test_type6(a: float | complex) -> float | complex:
    # the only one that passes mypy
    return a
