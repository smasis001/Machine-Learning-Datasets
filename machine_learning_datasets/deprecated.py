"""Common Utility Functions"""
# pylint: disable=E1101,W0212,C0302,C0103,C0415,C0121
from typing import Tuple, Union, Dict, Optional
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .common import ArrayLike

def plot_3dim_decomposition(
        Z:Union[pd.DataFrame, np.ndarray],
        y_labels:ArrayLike,
        y_names:Dict,
        save_name:Optional[str] = None
    ) -> None:
    """Plots a 3-dimensional decomposition of the given data.

    Parameters:
        Z (Union[pd.DataFrame, np.ndarray]): The input data.
        y_labels (ArrayLike): The labels for the data points.
        y_names (Dict): A dictionary mapping label indices to their names.
        save_name (Optional[str], optional): The name to save the plot as. Defaults to None.

    Returns:
        None
    """
    warnings.warn('This method is deprecated.', DeprecationWarning, stacklevel=2)
    if len(y_names) > 2:
        cmap = 'plasma_r'
    else:
        cmap = 'viridis'
    fig, axs = plt.subplots(1, 3, figsize = (16,4))
    fig.subplots_adjust(hspace=0, wspace=0.3)
    scatter = axs[0].scatter(Z[:,0], Z[:,1],\
                             c=y_labels, alpha=0.5, cmap=cmap)
    legend = axs[0].legend(*scatter.legend_elements(), loc='best')
    for n in y_names.keys():
        legend.get_texts()[n].set_text(y_names[n])
    axs[0].set_xlabel('x', fontsize = 12)
    axs[0].set_ylabel('y', fontsize = 12)
    scatter = axs[1].scatter(Z[:,1], Z[:,2],\
                   c=y_labels, alpha=0.5, cmap=cmap)
    legend = axs[1].legend(*scatter.legend_elements(), loc='best')
    for n in y_names.keys():
        legend.get_texts()[n].set_text(y_names[n])
    axs[1].set_xlabel('y', fontsize = 12)
    axs[1].set_ylabel('z', fontsize = 12)
    axs[2].scatter(Z[:,0], Z[:,2],\
                   c=y_labels, alpha=0.5, cmap=cmap)
    legend = axs[2].legend(*scatter.legend_elements(), loc='best')
    for n in y_names.keys():
        legend.get_texts()[n].set_text(y_names[n])
    axs[2].set_xlabel('x', fontsize = 12)
    axs[2].set_ylabel('z', fontsize = 12)
    if save_name is not None:
        plt.savefig(save_name+'_3dim.png', dpi=300, bbox_inches="tight")
    plt.show()

def encode_classification_error_vector(
        y_true:Union[pd.Series, np.ndarray],
        y_pred:Union[pd.Series, np.ndarray]
    ) -> Tuple[np.ndarray, Dict]:
    """Encodes the classification error vector.

    Args:
        y_true (Union[pd.Series, np.ndarray]): The true classification labels.
        y_pred (Union[pd.Series, np.ndarray]): The predicted classification labels.

    Returns:
        Tuple[np.ndarray, Dict]: A tuple containing the encoded error vector and the error labels.

    Example:
        >>> y_true = np.array([1, 0, 1, 0])
        >>> y_pred = np.array([0, 0, 1, 1])
        >>> encode_classification_error_vector(y_true, y_pred)
        (array([3, 4, 1, 2]), {0: 'FP', 1: 'FN', 2: 'TP', 3: 'TN'})
    """
    warnings.warn('This method is deprecated.', DeprecationWarning, stacklevel=2)
    error_vector = (y_true * 2) - y_pred
    error_vector = np.where(error_vector==0, 4, error_vector + 1)
    error_vector = np.where(error_vector==3, 0, error_vector - 1)
    error_vector = np.where(error_vector==3, error_vector, error_vector + 1)
    error_labels = {0:'FP', 1:'FN', 2:'TP', 3:'TN'}
    return error_vector, error_labels
