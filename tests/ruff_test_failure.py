"""Test that ruff fails"""


import pytest

# whatever
# whatever 2

# whatever 3
@pytest.mark.skip
def test_ruff_fails() -> None:
    """A test case that causes ruff to have an infinite loop"""
    return
