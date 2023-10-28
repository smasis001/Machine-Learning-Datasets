"""Test Interpret Functions"""
# pylint: disable=C0103
import numpy as np
from machine_learning_datasets import interpret

def test_compare_confusion_matrices():
    """Unit test that tests the function that compares two confusion matrices
       side by side and computes the false positive rate based on that.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If the computed false positive rate is not equal to 1.0
    """
    y_true_1 = [0, 1, 0, 1]
    y_pred_1 = [0, 1, 1, 0]
    y_true_2 = [1, 0, 1, 0]
    y_pred_2 = [1, 0, 0, 1]
    group_1 = "Group 1"
    group_2 = "Group 2"

    fpr = interpret.compare_confusion_matrices(y_true_1, y_pred_1, y_true_2, y_pred_2,\
                                        group_1, group_2, plot=False, compare_fpr=True)

    # Assert that the fpr is of the right type
    assert isinstance(fpr, float)
    # Assert that the fpr is of the right value
    assert fpr == 1.0

def test_profits_by_thresh():
    """Unit test that tests the function that calculates profits, costs,
       and ROI for given threshold values.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If any of the expected outputs don't match.
    """
    y_profits = np.array([10, 20, 30, 40, 50])
    y_pred = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    threshs = [0.3, 0.6, 0.9]
    var_costs = 2
    min_profit = 15
    fixed_costs = 5

    result = interpret.profits_by_thresh(y_profits, y_pred, threshs, var_costs,\
                                         min_profit, fixed_costs)

    assert result.loc[0.3, 'revenue'] == 140
    assert result.loc[0.3, 'costs'] == 13
    assert result.loc[0.3, 'profit'] == 127
    assert result.loc[0.3, 'roi'] == 9.76923076923077

    assert result.loc[0.6, 'revenue'] == 90
    assert result.loc[0.6, 'costs'] == 9
    assert result.loc[0.6, 'profit'] == 81
    assert result.loc[0.6, 'roi'] == 9.000000

    assert result.loc[0.9, 'revenue'] == 50
    assert result.loc[0.9, 'costs'] == 7
    assert result.loc[0.9, 'profit'] == 43
    assert result.loc[0.9, 'roi'] == 6.142857142857143
