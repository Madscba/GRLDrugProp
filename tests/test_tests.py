"""Inspiration script with a few basic unit tests."""
import typing as T

import pandas as pd
import pytest


def test_that_something_works():
    """Test that math is not broken."""
    a = 1
    b = 2
    a += 1
    assert a == b


def complex_processing_function(df: pd.DataFrame) -> pd.DataFrame:
    """A complex function."""
    df["a"] += 2
    return df


def test_that_pandas_works():
    """Example of .assert_frame_equal method."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [3, 4, 5]})
    output = complex_processing_function(df=df)
    expected = pd.DataFrame({"a": [3, 4, 5], "b": [3, 4, 5]})
    pd.testing.assert_frame_equal(output, expected, check_dtype=True, check_exact=True)


def fancy_model_function(some_array: T.List[int]) -> sum:
    """Dummy function that tries summing an array."""
    return sum(some_array)


def test_that_errors_are_raised():
    """Example of how to test for errors."""
    with pytest.raises(TypeError):
        fancy_model_function(["cant", "sum", "this"])


class FancyConfig:
    """Dummy config class that would be difficult to run tests with."""

    def __init__(self):
        """Default parameters."""
        self.half = "some value"
        self.space = "another value"
        self.store_model = True


@pytest.fixture
def config_for_tests():
    """Easier config class that we can run tests with."""
    config = FancyConfig()
    config.store_model = False
    return config


@pytest.fixture
def data_for_tests():
    """Example of getting a pytest fixture."""
    return pd.DataFrame({"a": [1, 2, 3]})


def test_pipeline(config_for_tests, data_for_tests):
    """Example of testing that loads fixtures."""
    if config_for_tests.store_model:
        raise ValueError("You broke me")
    assert data_for_tests.a.sum() == 6, "Bad data"
