"""Regression cases for pandas 3 Copy-on-Write NumPy exports."""
import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal

from curve_curator.models import LogisticModel
from curve_curator.data_simulator import simulate_decoys


def test_inverse_accepts_pandas_and_readonly_arrays_without_mutating_input():
    values = pd.Series([1., .75, .5, .25, 0., np.nan])
    original = values.copy()
    array = values.to_numpy(copy=True)
    array.flags.writeable = False
    for data in (values, array):
        result = LogisticModel.inverse_function(data, pec50=7., slope=1., front=1., back=0.)
        np.testing.assert_allclose(result, [np.nan, -7.477121254719663, -7., -6.522878745280337, np.nan, np.nan])
    assert_series_equal(values, original)


def test_decoy_noise_mask_accepts_readonly_input_without_mutation():
    noise = pd.Series([0., .1, .2, .3, .4])
    original = noise.copy()
    array = noise.to_numpy(copy=True)
    array.flags.writeable = False
    state = np.random.get_state()
    try:
        np.random.seed(42)
        expected = simulate_decoys(4, ['a', 'b', 'c', 'd', 'e'], noise)
        np.random.seed(42)
        actual = simulate_decoys(4, ['a', 'b', 'c', 'd', 'e'], array)
    finally:
        np.random.set_state(state)
    np.testing.assert_array_equal(actual, expected)
    assert np.isfinite(actual[['a', 'b', 'c', 'd', 'e']]).all().all()
    assert_series_equal(noise, original)
