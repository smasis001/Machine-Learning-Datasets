"""Test Preprocess Functions"""
import pandas as pd
import numpy as np
import torchvision
from machine_learning_datasets import preprocess


def test_group_values_into_other_category():
    """Unit test that tests the function that groups values that do not meet the
       minimum number of records or are beyond the maximum number of dummies into
       the 'Other' category.

        Args:
            None

        Returns:
            None

        Raises:
            AssertionError: If the result is not a DataFrame
            AssertionError: If 'col1_A' is not in the result columns
            AssertionError: If 'col1_B' is not in the result columns
            AssertionError: If 'col1_Other' is not in the result columns
    """
    # Create a valid DataFrame with values that do not meet the minimum number of
    # records or are beyond the maximum number of dummies
    df = pd.DataFrame({'col1': ['A', 'A', 'A', 'B', 'B', 'B', 'B', 'H', 'I', 'J'],
                        'col2': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})

    # Call the function with valid arguments
    result = preprocess.make_dummies_with_limits(df, 'col1', min_recs=3,\
                                                 max_dummies=5)

    # Assert that the result is a DataFrame
    assert isinstance(result, pd.DataFrame)

    # Assert that values that do not meet the minimum number of records or are beyond
    # the maximum number of dummies are grouped into the 'Other' category
    assert 'col1_A' in result.columns
    assert 'col1_B' in result.columns
    assert 'col1_Other' in result.columns


def test_create_dummies_from_dict():
    """Unit test for the `make_dummies_from_dict` function.

    This test case checks if the function correctly creates dummy variables based on a
    dictionary or list of values.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If the result dataframe does not match the expected dataframe.
    """
    df = pd.DataFrame({'col1': ['blueberry', 'strawberry', 'taco']})
    match_dict = {'berry': 'berry', 'taco': 'taco'}
    expected_df = pd.DataFrame({'col1_berry': [1, 1, 0], 'col1_taco': [0, 0, 1]})
    result_df = preprocess.make_dummies_from_dict(df, 'col1', match_dict)

    pd.testing.assert_frame_equal(result_df, expected_df)

def test_apply_cmap_grayscale_default():
    """Unit test that tests the function that applies a colormap to a grayscale image.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If the shape of the result is not (100, 100, 3).
        AssertionError: If the minimum value of the result is not close to 0.
        AssertionError: If the maximum value of the result is not close to 1.
    """
    img = np.random.rand(100, 100)
    cmap = 'jet'
    result = preprocess.apply_cmap(img, cmap)
    assert result.shape == (100, 100, 3)
    assert np.allclose(result.min(), 0)
    assert np.allclose(result.max(), 1)


def test_no_normalization_required():
    """Unit test that tests the function that converts a tensor to an image when no
    normalization is required.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If any of the assertions fail.
    """
    # Create a tensor
    tensor = torchvision.utils.torch.randn(3, 256, 256)

    # Call the tensor_to_img function
    image = preprocess.tensor_to_img(tensor, to_numpy=True)

    # Check if the output is a numpy array
    assert isinstance(image, np.ndarray)

    # Check if the shape of the output is correct
    assert image.shape == (256, 256, 3)

    # Check if the dtype of the output is correct
    assert image.dtype == np.float32


def test_discretize_numeric_variable():
    """Unit test that tests the function that discretizes a numeric variable into a given
       number of intervals.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If the function does not produce the expected output.
    """
    v = np.array([1, 2, 3, 4, 5])
    v_intervals = 2
    expected_v = [0, 0, 0, 1, 1]
    expected_bins = ['(1.0, 3.0]', '(3.0, 5.0]']

    v, bins = preprocess.discretize(v, v_intervals)

    assert v.tolist() == expected_v
    assert bins.tolist() == expected_bins


def test_calculate_threshold():
    """Unit test for the calculate_threshold function.

    This test verifies the correctness of the calculate_threshold function by
    comparing the calculated threshold with the expected threshold. The function
    takes an array of values and a percentile as input and returns the cumulative
    sum threshold.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If the function does not produce the expected output.
    """
    values = np.array([1, 2, 3, 4, 5])
    percentile = 50
    expected_threshold = 4
    assert preprocess.cumulative_sum_threshold(values, percentile) == expected_threshold
