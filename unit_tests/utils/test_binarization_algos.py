import pytest
import pandas as pd
import numpy as np
from llm_val.utils.binarization_algos import bin_by_spec_with_merge

@pytest.fixture
def negative_to_positive_series():
    """Серия с значениями от -10 до 10."""
    return pd.Series(range(-10, 10))

def test_case_negative_to_positive(negative_to_positive_series):
    """Проверка группировки отрицательных и положительных чисел."""
    _, bins_val, _ = bin_by_spec_with_merge(negative_to_positive_series, min_share=0.1)
    expected = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9]
    np.testing.assert_array_equal(bins_val, expected)

@pytest.mark.parametrize("data,expected_bins,expected_val", [
    (
        [99999, 2, 2, 2, 3, 4, 5, 9999, 9, 9, 9, 9, 99997, 0],
        [(0, 2), (3, 5), (9, 9), (9999, 99999)],
        [3, 0, 0, 0, 1, 1, 1, 3, 2, 2, 2, 2, 3, 0]
    ),
])
def test_large_values(data, expected_bins, expected_val):
    """Проверка обработки выбросов."""
    data_series = pd.Series(data)
    bins, bins_val, _ = bin_by_spec_with_merge(data_series, min_share=0.1)
    assert bins == expected_bins
    np.testing.assert_array_equal(bins_val, expected_val)