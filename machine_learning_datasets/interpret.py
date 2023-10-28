"""Common Utility Functions"""
# pylint: disable=E1101,W0212,C0302,C0103,C0415,C0121
from typing import Tuple, Union, Dict, List, Optional,\
                   Literal, Callable
import math
import pandas as pd
import numpy as np
from sklearn import preprocessing, metrics
from matplotlib import ticker
import matplotlib.ticker as plticker
import matplotlib.pyplot as plt
from matplotlib.axes._base import _AxesBase
from matplotlib.colors import Colormap
import seaborn as sns
from .common import ArrayLike, BaseModelProtocol, BaseTransformerProtocol,\
                    BaseAIF360DSProtocol, BaseAIF360MetricProtocol, BaseAlibiExplanationProtocol
from .preprocess import discretize, img_np_from_fig, apply_cmap, find_closest_datapoint_idx

def compare_confusion_matrices(
        y_true_1:ArrayLike,
        y_pred_1:ArrayLike,
        y_true_2:ArrayLike,
        y_pred_2:ArrayLike,
        group_1:str,
        group_2:str,
        plot:Optional[bool] = True,
        compare_fpr:Optional[bool] = False,
        save_name:Optional[str] = None
    ) -> Optional[float]:
    """Compare confusion matrices for two different groups.

    Args:
        y_true_1 (ArrayLike): True labels for group 1.
        y_pred_1 (ArrayLike): Predicted labels for group 1.
        y_true_2 (ArrayLike): True labels for group 2.
        y_pred_2 (ArrayLike): Predicted labels for group 2.
        group_1 (str): Name of group 1.
        group_2 (str): Name of group 2.
        plot (Optional[bool], default=True): Whether to plot the confusion matrices.
        compare_fpr (Optional[bool], default=False): Whether to compare the False Positive
                                                     Rates (FPR) of the two groups.
        save_name (Optional[str], default=None): Name to save the plot if `plot` is True.

    Returns:
        Optional[float]: The ratio between the FPRs of the two groups if `compare_fpr` is True,
                         otherwise None.
    """
    #Create confusion matrices for two different groups.
    conf_matrix_1 = metrics.confusion_matrix(y_true_1, y_pred_1)
    conf_matrix_2 = metrics.confusion_matrix(y_true_2, y_pred_2)

    #Plot both confusion matrices side-by-side.
    if plot:
        _, ax = plt.subplots(1,2,figsize=(12,5))
        sns.heatmap(conf_matrix_1/np.sum(conf_matrix_1), annot=True,\
                    fmt='.2%', cmap='Blues', annot_kws={'size':16}, ax=ax[0])
        ax[0].set_title(group_1 + ' Confusion Matrix', fontsize=14)
        sns.heatmap(conf_matrix_2/np.sum(conf_matrix_2), annot=True,\
                    fmt='.2%', cmap='Blues', annot_kws={'size':16}, ax=ax[1])
        ax[1].set_title(group_2 + ' Confusion Matrix', fontsize=14)
        if save_name is not None:
            plt.savefig(save_name+'_cm.png', dpi=300, bbox_inches="tight")
        plt.show()

    #Calculate False Positive Rates (FPR) for each Group.
    tn, fp, _, _ = conf_matrix_1.ravel()
    fpr_1 = fp/(fp+tn)
    tn, fp, _, _ = conf_matrix_2.ravel()
    fpr_2 = fp/(fp+tn)

    #Print the FPRs and the ratio between them.
    if compare_fpr:
        if fpr_2 > fpr_1:
            print(f"\t{group_2} FPR:\t{fpr_2:.1%}")
            print(f"\t{group_1} FPR:\t\t{fpr_1:.1%}")
            print(f"\tRatio FPRs:\t\t{fpr_2/fpr_1:.2f} x")
            return fpr_2/fpr_1
        else:
            print(f"\t{group_1} FPR:\t{fpr_1:.1%}")
            print(f"\t{group_2} FPR:\t\t{fpr_2:.1%}")
            print(f"\tRatio FPRs:\t\t{fpr_1/fpr_2:.2f} x")
            return fpr_1/fpr_2

def plot_prob_progression(
        x:Union[str, pd.Series, np.ndarray],
        y:Union[str, pd.Series, np.ndarray],
        x_intervals:Optional[int] = 7,
        use_quantiles:Optional[bool] = False,
        xlabel:Optional[str] = None,
        ylabel:Optional[str] = 'Observations',
        title:Optional[str] = None,
        model:Optional[BaseModelProtocol] = None,
        X_df:Optional[str] = None,
        x_col:Optional[str] = None,
        mean_line:Optional[bool] = False,
        figsize:Optional[Tuple] = None,
        x_margin:Optional[float] = 0.01,
        save_name:Optional[str] = None
    ) -> None:
    """Plots the progression of probabilities for a given dataset.

    Args:
        x (Union[str, pd.Series, np.ndarray]): The input data for the x-axis. Can be a string,
                                               pandas series, or numpy array.
        y (Union[str, pd.Series, np.ndarray]): The input data for the y-axis. Can be a string,
                                               pandas series, or numpy array.
        x_intervals (Optional[int]): The number of intervals to divide the x-axis into. Default
                                     is 7.
        use_quantiles (Optional[bool]): Whether to use quantiles for dividing the x-axis intervals.
                                        Default is False.
        xlabel (Optional[str]): The label for the x-axis. Default is None.
        ylabel (Optional[str]): The label for the y-axis. Default is 'Observations'.
        title (Optional[str]): The title of the plot. Default is None.
        model (Optional[BaseModelProtocol]): The model used for predictions. Default is None.
        X_df (Optional[str]): The dataset used for predictions. Default is None.
        x_col (Optional[str]): The column name for the x-axis data in the dataset.
                               Default is None.
        mean_line (Optional[bool]): Whether to plot a dashed line representing the mean of the
                                    y-axis data. Default is False.
        figsize (Optional[Tuple]): The size of the figure. Default is None.
        x_margin (Optional[float]): The margin for the x-axis. Default is
                                    0.01.
        save_name (Optional[str]): The name to save the plot as. Default is
                                   None.

    Returns:
        None

    Raises:
        TypeError: If x and y are not lists, pandas series, or numpy arrays.
        ValueError: If x and y do not have a single dimension, x_intervals is less than 2,
                    or y dimension is not a list, pandas series, or numpy array of integers
                    or floats.
        ValueError: If y dimension has less than two values, or if it has two values but
                    the max is not 1 or the min is not 0, or if it has more than two values
                    but the range is not between 0 and 1.
    """
    if figsize is None:
        figsize = (12,6)
    if isinstance(x, list) is True:
        x = np.array(x)
    if isinstance(y, list) is True:
        y = np.array(y)
    if (not isinstance(x, (str, pd.core.series.Series, np.ndarray))) or\
        (not isinstance(y, (str, pd.core.series.Series, np.ndarray))):
        raise TypeError("x and y must be either lists, pandas series or numpy arrays. "
                        "x can be string when dataset is provided seperately")
    if (isinstance(x, (pd.core.series.Series, np.ndarray)) and (len(x.shape) != 1)) or\
        ((isinstance(y, (pd.core.series.Series, np.ndarray))) and (len(y.shape) != 1)):
        raise ValueError("x and y must have a single dimension")
    if (isinstance(x_intervals, (int)) and (x_intervals < 2)) or\
        (isinstance(x_intervals, (list, np.ndarray)) and (len(x_intervals) < 2)):
        raise ValueError("there must be at least two intervals to plot")
    if not np.isin(y.dtype, [int, float, 'int8', 'int16', 'int32', 'float16', 'float32']):
        raise ValueError("y dimension must be a list, pandas series or numpy array of integers "
                         "or floats")
    if max(y) == min(y):
        raise ValueError("y dimension must have at least two values")
    elif len(np.unique(y)) == 2 and ((max(y) != 1) or (min(y) != 0)):
        raise ValueError("y dimension if has two values must have a max of exactly 1 and min of "
                         "exactly zero")
    elif len(np.unique(y)) > 2 and ((max(y) <= 1) or (min(y) >= 0)):
        raise ValueError("y dimension if has more than two values must have range between 0-1")
    x_use_continuous_bins = (model is not None) and (isinstance(x_intervals, (list, np.ndarray)))
    x, x_bins = discretize(x, x_intervals, use_quantiles, x_use_continuous_bins)
    x_range = [*range(len(x_bins))]
    plot_df = pd.DataFrame({'x':x_range})
    if (model is not None) and (X_df is not None) and (x_col is not None):
        preds = model.predict(X_df).squeeze()
        if len(np.unique(preds)) <= 2:
            preds = model.predict_proba(X_df)[:,1]
        x_, _ = discretize(X_df[x_col], x_intervals, use_quantiles, x_use_continuous_bins)
        xy_df = pd.DataFrame({'x':x_, 'y':preds})
    else:
        xy_df = pd.DataFrame({'x':x,'y':y})
    probs_df = xy_df.groupby(['x']).mean().reset_index()
    probs_df = pd.merge(plot_df, probs_df, how='left', on='x').fillna(0)

    sns.set()
    x_bin_cnt = len(x_bins)
    l_width = 0.933
    r_width = 0.05
    w, _ = figsize
    wp = (w-l_width-r_width)/9.27356902357
    xh_margin = ((wp-(x_margin*2))/(x_bin_cnt*2))+x_margin
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=figsize,\
                                   gridspec_kw={'height_ratios': [3, 1]})
    if title is not None:
        fig.suptitle(title, fontsize=21)
        plt.subplots_adjust(top = 0.92, bottom=0.01, hspace=0.001, wspace=0.001)
    else:
        plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.001, wspace=0.001)
    ax0.minorticks_on()
    sns.lineplot(data=probs_df, x='x', y='y', ax=ax0)
    ax0.set_ylabel('Probability', fontsize=15)
    ax0.set_xlabel('')
    ax0.grid(visible=True, axis='x', which='minor', color='w', linestyle=':')
    #ax0.set_xticks([], [])
    ax0.margins(x=xh_margin)
    if mean_line:
        ax0.axhline(y=xy_df.y.mean(), c='red', linestyle='dashed', label="mean")
        ax0.legend()
    sns.histplot(xy_df, x="x", stat='probability', bins=np.arange(x_bin_cnt+1)-0.5, ax=ax1)
    ax1.set_ylabel(ylabel, fontsize=15)
    ax1.set_xlabel(xlabel, fontsize=15)
    ax1.xaxis.set_major_locator(plticker.MultipleLocator(base=1.0))
    ax1.set_xticklabels([''] + list(x_bins))
    ax1.margins(x=x_margin)
    if save_name is not None:
        plt.savefig(save_name+'_prog.png', dpi=300, bbox_inches="tight")
    plt.show()

def plot_prob_contour_map(
        x:Union[str, pd.Series, np.ndarray],
        y:Union[str, pd.Series, np.ndarray],
        z:Union[str, pd.Series, np.ndarray],
        x_intervals:Optional[int] = 7,
        y_intervals:Optional[int] = 7,
        use_quantiles:Optional[bool] = False,
        plot_type:Optional[Literal['contour','grid']] = 'contour',
        xlabel:Optional[str] = None,
        ylabel:Optional[str] = None,
        title:Optional[str] = None,
        model:Optional[BaseModelProtocol] = None,
        X_df:Optional[pd.DataFrame] = None,
        x_col:Optional[str] = None,
        y_col:Optional[str] = None,
        cmap:Optional[Union[str, Colormap]] = None,
        diff_to_mean:Optional[bool] = False,
        annotate:Optional[bool] = False,
        color:Optional[str] = "w",
        save_name:Optional[str] = None
    ) -> None:
    """Plots a probability contour map.

    Args:
        x (Union[str, pd.Series, np.ndarray]): The x-axis values. Can be a string, pandas series,
                                               or numpy array.
        y (Union[str, pd.Series, np.ndarray]): The y-axis values. Can be a string, pandas series,
                                               or numpy array.
        z (Union[str, pd.Series, np.ndarray]): The z-axis values. Can be a string, pandas series,
                                               or numpy array.
        x_intervals (Optional[int]): The number of intervals to divide the x-axis into. Default
                                     is 7.
        y_intervals (Optional[int]): The number of intervals to divide the y-axis into. Default
                                     is 7.
        use_quantiles (Optional[bool]): Whether to use quantiles for discretizing the x and y
                                        values. Default is False.
        plot_type (Optional[Literal['contour','grid']]): The type of plot to generate. Default
                                                         is 'contour'.
        xlabel (Optional[str]): The label for the x-axis. Default is None.
        ylabel (Optional[str]): The label for the y-axis. Default is None.
        title (Optional[str]): The title of the plot. Default is None.
        model (Optional[BaseModelProtocol]): The model used for prediction. Default is None.
        X_df (Optional[pd.DataFrame]): The dataset used for prediction. Default is None.
        x_col (Optional[str]): The column name for the x values in the dataset. Default is None.
        y_col (Optional[str]): The column name for the y values in the dataset. Default is None.
        cmap (Optional[Union[str, Colormap]]): The colormap to use for the plot. Default is None.
        diff_to_mean (Optional[bool]): Whether to subtract the mean value from the z values.
                                       Default is False.
        annotate (Optional[bool]): Whether to annotate the grid plot with the z values. Default is
                                   False.
        color (Optional[str]): The color of the annotations. Default is "w".
        save_name (Optional[str]): The name of the file to save the plot. Default is None.

    Returns:
        None
    """
    if isinstance(x, list) is True:
        x = np.array(x)
    if isinstance(y, list) is True:
        y = np.array(y)
    if isinstance(z, list) is True:
        z = np.array(z)
    if (not isinstance(x, (str, pd.core.series.Series, np.ndarray))) or\
        (not isinstance(y, (str, pd.core.series.Series, np.ndarray))) or\
            (not isinstance(z, (pd.core.series.Series, np.ndarray))):
        raise TypeError("x, y and z must be either lists, pandas series or numpy arrays. "
                        "x and y can be strings when dataset is provided seperately")
    if (isinstance(x, (pd.core.series.Series, np.ndarray)) and (len(x.shape) != 1)) or\
        ((isinstance(y, (pd.core.series.Series, np.ndarray))) and (len(y.shape) != 1)) or\
            (len(z.shape) != 1):
        raise ValueError("x, y and z must have a single dimension")
    if (isinstance(x_intervals, (int)) and (x_intervals < 2)) or\
        (isinstance(x_intervals, (list, np.ndarray)) and (len(x_intervals) < 2)) or\
            (isinstance(y_intervals, (int)) and (y_intervals < 2)) or\
                (isinstance(y_intervals, (list, np.ndarray)) and (len(y_intervals) < 2)):
        raise ValueError("there must be at least two intervals to contour")
    if not np.isin(z.dtype, [int, float, 'int8', 'int16', 'int32', 'float16', 'float32']):
        raise ValueError("z dimension must be a list, pandas series or numpy array of integers "
                         "or floats")
    if max(z) == min(z):
        raise ValueError("z dimension must have at least two values")
    elif len(np.unique(z)) == 2 and ((max(z) != 1) or (min(z) != 0)):
        raise ValueError("z dimension if has two values must have a max of exactly 1 and min of "
                         "exactly zero")
    elif len(np.unique(z)) > 2 and ((max(z) <= 1) or (min(z) >= 0)):
        raise ValueError("z dimension if has more than two values must have range between 0-1")
    x_use_continuous_bins = (model is not None) and (isinstance(x_intervals, (list, np.ndarray)))
    y_use_continuous_bins = (model is not None) and (isinstance(y_intervals, (list, np.ndarray)))
    x, x_bins = discretize(x, x_intervals, use_quantiles, x_use_continuous_bins)
    y, y_bins = discretize(y, y_intervals, use_quantiles, y_use_continuous_bins)
    x_range = [*range(len(x_bins))]
    #if isinstance(y_intervals, (int)):
    y_range = [*range(len(y_bins))]
    #else:
    #y_range = y_intervals
    x_grid, y_grid = np.meshgrid(x_range, y_range)
    positions = np.vstack([x_grid.ravel(), y_grid.ravel()])
    plot_df = pd.DataFrame(positions.T, columns=['x', 'y'])

    if (model is not None) and (X_df is not None) and (x_col is not None) and (y_col is not None):
        preds = model.predict(X_df).squeeze()
        if len(np.unique(preds)) <= 2:
            preds = model.predict_proba(X_df)[:,1]
        x_, _ = discretize(X_df[x_col], x_intervals, use_quantiles, x_use_continuous_bins)
        y_, _ = discretize(X_df[y_col], y_intervals, use_quantiles, y_use_continuous_bins)
        xyz_df = pd.DataFrame({'x':x_, 'y':y_, 'z':preds})
    else:
        xyz_df = pd.DataFrame({'x':x,'y':y,'z':z})
    probs_df = xyz_df.groupby(['x','y']).mean().reset_index()
    probs_df = pd.merge(plot_df, probs_df, how='left', on=['x','y']).fillna(0)
    if diff_to_mean:
        expected_value = xyz_df.z.mean()
        probs_df['z'] = probs_df['z'] - expected_value
        if cmap is None:
            cmap = plt.cm.RdYlBu
    elif cmap is None:
        cmap = plt.cm.viridis
    grid_probs = np.reshape(probs_df.z.to_numpy(), x_grid.shape)

    x_bin_cnt = len(x_bins)
    y_bin_cnt = len(y_bins)
    fig, (ax_top, ax_bottom) = plt.subplots(2, 2, figsize=(12,9),\
                                   gridspec_kw={'height_ratios': [1, 7], 'width_ratios': [6, 1]})
    if title is not None:
        fig.suptitle(title, fontsize=21)
        plt.subplots_adjust(top = 0.95, bottom=0.01, hspace=0.001, wspace=0.001)
    else:
        plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.001, wspace=0.001)

    sns.set_style(None)
    sns.set_style({'axes.facecolor':'white', 'grid.color': 'white'})
    sns.histplot(xyz_df, x='x', stat='probability', bins=np.arange(x_bin_cnt+1)-0.5,\
                 color=('dimgray'), ax=ax_top[0])
    ax_top[0].set_xticks([])
    ax_top[0].set_yticks([])
    ax_top[0].set_xlabel('')
    ax_top[0].set_ylabel('')
    ax_top[1].set_visible(False)

    if plot_type == 'contour':
        ax_bottom[0].contour(
            x_grid,
            y_grid,
            grid_probs,
            colors=('w',)
        )
        mappable = ax_bottom[0].contourf(
            x_grid,
            y_grid,
            grid_probs,
            cmap=cmap
        )
    else:
        mappable = ax_bottom[0].imshow(grid_probs, cmap=cmap,\
                                      interpolation='nearest', aspect='auto') #plt.cm.viridis
        if annotate:
            for i in range(y_bin_cnt):
                for j in range(x_bin_cnt):
                    _ = ax_bottom[0].text(j, i, f"{grid_probs[i, j]:.1%}", fontsize=16,
                                             ha="center", va="center", color=color, weight="bold")
            ax_bottom[0].grid(False)

    ax_bottom[0].xaxis.set_major_locator(plticker.MultipleLocator(base=1.0))
    ax_bottom[0].set_xticklabels([''] + list(x_bins))
    ax_bottom[0].yaxis.set_major_locator(plticker.MultipleLocator(base=1.0))
    ax_bottom[0].set_yticklabels([''] + list(y_bins))
    #ax_bottom[0].margins(x=0.04, y=0.04)

    if xlabel is not None:
        ax_bottom[0].set_xlabel(xlabel, fontsize=15)

    if ylabel is not None:
        ax_bottom[0].set_ylabel(ylabel, fontsize=15)

    cbar = plt.colorbar(mappable, ax=ax_bottom[1])
    cbar.ax.set_ylabel('Probability', fontsize=13)
    cbar.ax.tick_params(labelsize=11)

    sns.set_style(None)
    sns.set_style({'axes.facecolor':'white', 'grid.color': 'white'})
    sns.histplot(xyz_df, y="y", stat='probability', bins=np.arange(y_bin_cnt+1)-0.5,\
                 color=('dimgray'), ax=ax_bottom[1])
    ax_bottom[1].set_xticks([])
    ax_bottom[1].set_yticks([])
    ax_bottom[1].set_xlabel('')
    ax_bottom[1].set_ylabel('')
    sns.set_style(None)

    if save_name is not None:
        plt.savefig(save_name+'_contour.png', dpi=300, bbox_inches="tight")
    plt.show()

def create_decision_plot(
        X:ArrayLike,
        y:ArrayLike,
        model:BaseModelProtocol,
        feature_index:Union[list, tuple],
        feature_names:Union[list, tuple],
        X_highlight:np.ndarray,
        filler_feature_values:Dict,
        filler_feature_ranges:Optional[Dict] =None,
        ax:Optional[_AxesBase] = None,
        add_constant:Optional[bool] = True
    ) -> _AxesBase:
    """Create a decision plot.

    Args:
        X (ArrayLike): The input feature matrix.
        y (ArrayLike): The target variable.
        model (BaseModelProtocol): The trained model.
        feature_index (Union[list, tuple]): The indices of the features to be used in the plot.
        feature_names (Union[list, tuple]): The names of the features.
        X_highlight (np.ndarray): The data points to be highlighted in the plot.
        filler_feature_values (Dict): The values to fill the non-selected features.
        filler_feature_ranges (Optional[Dict], optional): The ranges of the filler features.
                                                          Defaults to None.
        ax (Optional[_AxesBase], optional): The matplotlib axes to plot on. Defaults to None.
        add_constant (Optional[bool], optional): Whether to add a constant feature. Defaults
                                                 to True.

    Returns:
        _AxesBase: The matplotlib axes object containing the decision plot.
    """
    try:
        from mlxtend.plotting import plot_decision_regions
        import statsmodels.api as sm
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("`statsmodels` and `mlxtend` must be installed to execute "
                                  "this function") from exc
    if feature_names is None:
        feature_names=feature_index
    if add_constant:
        con=1
    else:
        con=0
    if list(np.arange(X.shape[1]+con))!=list(filler_feature_values.keys()):
        if add_constant:
            fillers= {0:1}
        else:
            fillers= {}
        fillers.update({X.columns.get_loc(key)+con:vals\
                        for key, vals in filler_feature_values.items()})
        filler_feature_values = fillers
    filler_feature_keys = list(filler_feature_values.keys())
    if sum([True for k in feature_index if k not in filler_feature_keys])==2:
        feature_index = [X.columns.get_loc(k)+con for k in feature_index\
                         if k not in filler_feature_keys]
    filler_values = dict((k, filler_feature_values[k]) for k in filler_feature_values.keys()\
                         if k not in feature_index)
    X_arr = sm.add_constant(X).to_numpy()
    if filler_feature_ranges is None:
        filler_vals = np.array(list(filler_feature_values.values()))
        filler_rangs = np.vstack([np.abs(np.amax(X_arr, axis=0) - filler_vals),\
                                   np.abs(np.amin(X_arr, axis=0) - filler_vals)])
        filler_rangemax = np.amax(filler_rangs, axis=0)
        filler_rangemax = list(np.where(filler_rangemax==0, 1, filler_rangemax))
        filler_feature_ranges = {i:v for i,v in enumerate(filler_rangemax)}
    filler_ranges = dict((k, filler_feature_ranges[k]) for k in filler_feature_ranges.keys()\
                         if k not in feature_index)
    ax = plot_decision_regions(X_arr, y.to_numpy(), clf=model,
                          feature_index=feature_index,
                          X_highlight=X_highlight,
                          filler_feature_values=filler_values,
                          filler_feature_ranges=filler_ranges,
                          scatter_kwargs = {'s': 48, 'edgecolor': None, 'alpha': 0.7},
                          contourf_kwargs = {'alpha': 0.2}, legend=2, ax=ax)
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    return ax

def describe_cf_instance(
        X:ArrayLike,
        explanation:BaseAlibiExplanationProtocol,
        class_names:Union[list, Dict],
        cat_vars_ohe:Optional[BaseTransformerProtocol] = None,
        category_map:Optional[Dict]=None,
        feature_names:list = None,
        eps:Optional[float] = 1e-2
    ) -> None:
    """Describes the counterfactual instance and its perturbations.

    Args:
        X (ArrayLike): The original instance.
        explanation (BaseAlibiExplanationProtocol): The explanation object containing the
                                                    counterfactual instance.
        class_names (Union[list, Dict]): The names of the classes.
        cat_vars_ohe (Optional[BaseTransformerProtocol]): The transformer for one-hot encoded
                                                          categorical variables. Default is None.
        category_map (Optional[Dict]): The mapping of categorical variables. Default is None.
        feature_names (list): The names of the features. Default is None.
        eps (Optional[float]): The threshold for numerical feature perturbations. Default is 1e-2.

    Raises:
        ModuleNotFoundError: If `alibi` is not installed.

    Returns:
        None

    Note:
        This function requires the `alibi` library to be installed.
    """
    try:
        from alibi.utils.mapping import ohe_to_ord
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("`alibi` must be installed to execute this function") from exc
    print("Instance Outcomes and Probabilities")
    print("-" * 48)
    max_len = len(max(feature_names, key=len))
    print(f"{'original'.rjust(max_len)}:  {class_names[explanation.orig_class]}\r\n"
          f"{' ' * max_len}   {explanation.orig_proba[0]}")
    if explanation.cf is not None:
        print(f"{'counterfactual'.rjust(max_len)}:  {class_names[explanation.cf['class']]}\r\n"
              f"{' ' * max_len}   {explanation.cf['proba'][0]}")
        print("\r\nCategorical Feature Counterfactual Perturbations")
        print("-" * 48)
        X_orig_ord = ohe_to_ord(X, cat_vars_ohe)[0]
        try:
            X_cf_ord = ohe_to_ord(explanation.cf['X'], cat_vars_ohe)[0]
        except (AssertionError, TypeError, ValueError):
            X_cf_ord = ohe_to_ord(explanation.cf['X'].transpose(), cat_vars_ohe)[0].transpose()
        delta_cat = {}
        for _, (i, v) in enumerate(category_map.items()):
            cat_orig = v[int(X_orig_ord[0, i])]
            cat_cf = v[int(X_cf_ord[0, i])]
            if cat_orig != cat_cf:
                delta_cat[feature_names[i]] = [cat_orig, cat_cf]
        if delta_cat:
            for k, v in delta_cat.items():
                print(f"\t{k.rjust(max_len)}:  {v[0]}  -->  {v[1]}")
        print("\r\nNumerical Feature Counterfactual Perturbations")
        print("-" * 48)
        num_idxs = [i for i in list(range(0,len(feature_names)))\
                    if i not in category_map.keys()]
        delta_num = X_cf_ord[0, num_idxs] - X_orig_ord[0, num_idxs]
        for i in range(delta_num.shape[0]):
            if np.abs(delta_num[i]) > eps:
                print(f"\t{feature_names[i].rjust(max_len)}:  {X_orig_ord[0,i]:.2f}  "
                      f"-->  {X_cf_ord[0,i]:.2f}")
    else:
        print("\tNO COUNTERFACTUALS")

def compare_img_pred_viz(
        img_np:np.ndarray,
        viz_np:np.ndarray,
        y_true:Union[str,int,float],
        y_pred:Union[str,int,float],
        probs_s:Optional[pd.Series] = None,
        title:Optional[str] = None,
        save_name:Optional[str] = None
    ) -> None:
    """Compare and visualize the image, predicted label, and probability distribution.

    Args:
        img_np (np.ndarray): The original image as a NumPy array.
        viz_np (np.ndarray): The visualization image as a NumPy array.
        y_true (Union[str, int, float]): The true label of the image.
        y_pred (Union[str, int, float]): The predicted label of the image.
        probs_s (Optional[pd.Series], optional): The probability distribution as a
                                                 Pandas Series. Defaults to None.
        title (Optional[str], optional): The title of the plot. Defaults to None.
        save_name (Optional[str], optional): The name to save the plot. Defaults to None.

    Returns:
        None
    """
    if isinstance(probs_s, (pd.core.series.Series)):
        p_df = probs_s.sort_values(ascending=False)[0:4].to_frame().reset_index()
        p_df.columns = ['class', 'prob']
        fig = plt.figure(figsize=(5, 2))
        ax = sns.barplot(x="prob", y="class", data=p_df)
        ax.set_xlim(0, 120)
        for p in ax.patches:
            ax.annotate(format(p.get_width(), '.2f')+'%',
                           (p.get_x() + p.get_width() + 1.2, p.get_y()+0.6),
                           size=13)
        ax.set(xticklabels=[], ylabel=None, xlabel=None)
        ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize = 16)
        fig.tight_layout()
        barh_np = img_np_from_fig(fig)
        plt.close(fig)
    else:
        barh_np = [[]]

    fig = plt.figure(figsize=(15, 7.15))
    gridspec = plt.GridSpec(3, 5, wspace=0.5,\
                            hspace=0.4, figure=fig)
    orig_img_ax = plt.subplot(gridspec[:2, :2])
    orig_img_ax.grid(False)
    orig_img_ax.imshow(img_np, interpolation='lanczos')
    orig_img_ax.set_title("Actual Label: " + r"$\bf{" + str(y_true) + "}$")
    viz_img_ax = plt.subplot(gridspec[:3, 2:])
    viz_img_ax.grid(False)
    viz_img_ax.imshow(viz_np, interpolation='spline16')
    pred_ax = plt.subplot(gridspec[2, :2])
    pred_ax.set_title("Predicted Label: " + r"$\bf{" + str(y_pred) + "}$")
    pred_ax.imshow(barh_np, interpolation='spline36')
    pred_ax.axis('off')
    pred_ax.axes.get_xaxis().set_visible(False)
    pred_ax.axes.get_yaxis().set_visible(False)
    if title is not None:
        fig.suptitle(title, fontsize=18, weight='bold', x=0.65)
        plt.subplots_adjust(bottom=0, top=0.92)
    if save_name is not None:
        plt.savefig(save_name+'_compare_pred.png', dpi=300, bbox_inches="tight")
    plt.show()

def create_attribution_grid(
        attribution:np.ndarray,
        cmap:Optional[Union[str,Colormap]] = None,
        cmap_norm:Optional[Union[str,Colormap]] = None
    ) -> np.ndarray:
    """Create an attribution grid from a given attribution map.

    Args:
        attribution (np.ndarray): The attribution map with shape (n, w, h), where n is the number
                                  of images and w, h are the width and height of each image.
        cmap (Optional[Union[str,Colormap]], optional): The colormap to be applied to the grid
                                                        image. Defaults to None.
        cmap_norm (Optional[Union[str,Colormap]], optional): The normalization to be applied to
                                                             the colormap. Defaults to None.

    Returns:
        np.ndarray: The attribution grid image with shape (grid_size * w, grid_size * h), where
                    grid_size is the size of the grid calculated as the ceiling of the square
                    root of n.

    Raises:
        ValueError: If the attribution map does not have 3 dimensions.
    """
    if len(attribution.shape) == 4:
        attribution = attribution.squeeze()
    if len(attribution.shape) != 3:
        raise ValueError("Attribution map should have 3 dimensions")
    n, w, h = attribution.shape
    grid_size = math.ceil(math.sqrt(n))

    # Create an empty array for the grid
    grid_image = np.zeros((grid_size * w, grid_size * h), dtype=float)

    # Fill the grid with images
    for i in range(grid_size):
        for j in range(grid_size):
            p = i * grid_size + j
            if p < n:
                current_filter = preprocessing.MinMaxScaler().fit_transform(attribution[p])
                grid_image[i * w:(i + 1) * w, j * h:(j + 1) * h] = current_filter

    if cmap is not None:
        grid_image = apply_cmap(grid_image, cmap, cmap_norm)

    return grid_image

def approx_predict_ts(
        X:ArrayLike,
        X_df:pd.DataFrame,
        gen_X:np.ndarray,
        ts_mdl:BaseModelProtocol,
        dist_metric:Optional[str] = 'euclidean',
        lookback:Optional[int] = 0,
        filt_fn:Optional[Callable] = None,
        X_scaler:Optional[BaseTransformerProtocol] = None,
        y_scaler:Optional[BaseTransformerProtocol] = None,
        progress_bar:Optional[bool] = False,
        no_info:Optional[np.ndarray] = None
    ) -> np.ndarray:
    """Approximately predicts time series values.

    Args:
        X (ArrayLike): The input time series data.
        X_df (pd.DataFrame): The input time series data as a DataFrame.
        gen_X (np.ndarray): The generated time series data.
        ts_mdl (BaseModelProtocol): The time series model used for prediction.
        dist_metric (Optional[str], optional): The distance metric used for finding closest
                                               datapoint. Defaults to 'euclidean'.
        lookback (Optional[int], optional): The number of previous time steps to consider for
                                            prediction. Defaults to 0.
        filt_fn (Optional[Callable], optional): The function used for filtering the input data.
                                                Defaults to None.
        X_scaler (Optional[BaseTransformerProtocol], optional): The scaler used for scaling the
                                                                input data. Defaults to None.
        y_scaler (Optional[BaseTransformerProtocol], optional): The scaler used for scaling the
                                                                output data. Defaults to None.
        progress_bar (Optional[bool], optional): Whether to display a progress bar during
                                                 prediction. Defaults to False.
        no_info (Optional[np.ndarray], optional): The value to return if no predictions are made.
                                                  Defaults to None.

    Returns:
        np.ndarray: The predicted time series values.

    Raises:
        ModuleNotFoundError: If `tqdm` is not installed.
    """
    try:
        from tqdm.notebook import trange
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("`tqdm` must be installed to execute this function") from exc
    if no_info is None:
        no_info = np.array([[0]])
    b_size = gen_X[0][0].shape[0]
    preds = None
    if progress_bar:
        rng = trange(X.shape[0], desc='Predicting')
    else:
        rng = range(X.shape[0])
    for i in rng:
        x = X[i]
        if filt_fn is not None:
            X_filt_df, x = filt_fn(X_df, x, lookback)
        else:
            X_filt_df = X_df
        idx = find_closest_datapoint_idx(x, X_filt_df, dist_metric, find_exact_first=1,\
                                         scaler=X_scaler)

        nidx = idx - lookback
        pred = ts_mdl.predict(gen_X[nidx//b_size][0])[nidx%b_size].reshape(1,-1)
        if i==0:
            preds = pred
        else:
            preds = np.vstack((preds,pred))
    if preds is not None:
        if y_scaler is not None:
            return y_scaler.inverse_transform(preds)
        else:
            return preds
    else:
        return no_info

def profits_by_thresh(
        y_profits:np.ndarray,
        y_pred:np.ndarray,
        threshs:Union[list, tuple, np.ndarray],
        var_costs:Optional[Union[int, float]] = 1,
        min_profit:Optional[Union[int, float]] = None,
        fixed_costs:Optional[Union[int, float]] = 0
    ) -> pd.DataFrame:
    """Calculate profits, costs, and return on investment (ROI) based on a given threshold.

    Args:
        y_profits (np.ndarray): Array of profits.
        y_pred (np.ndarray): Array of predicted values.
        threshs (Union[list, tuple, np.ndarray]): List, tuple, or array of threshold values.
        var_costs (Optional[Union[int, float]], default=1): Variable costs per unit.
        min_profit (Optional[Union[int, float]], default=None): Minimum profit threshold.
        fixed_costs (Optional[Union[int, float]], default=0): Fixed costs.

    Returns:
        pd.DataFrame: DataFrame containing profits, costs, and ROI for each threshold.

    Example:
        y_profits = np.array([10, 20, 30, 40, 50])
        y_pred = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
        threshs = [0.3, 0.6, 0.9]
        var_costs = 2
        min_profit = 15
        fixed_costs = 5

        profits_by_thresh(y_profits, y_pred, threshs, var_costs, min_profit, fixed_costs)

        Output:
            revenue  costs  profit  roi
            0.3       70     15     55   3.666667
            0.6       90     20     70   3.500000
            0.9      100     25     75   3.000000
    """
    profits_dict = {}
    for thresh in threshs:
        profits_dict[thresh] = {}
        profits_dict[thresh]["revenue"] = sum(y_profits[y_pred > thresh])
        profits_dict[thresh]["costs"] = (sum(y_pred > thresh)*var_costs) + fixed_costs
        profits_dict[thresh]["profit"] = profits_dict[thresh]["revenue"] -\
                                         profits_dict[thresh]["costs"]
        if profits_dict[thresh]["costs"] > 0:
            profits_dict[thresh]["roi"] = profits_dict[thresh]["profit"]/\
                                          profits_dict[thresh]["costs"]
        else:
            profits_dict[thresh]["roi"] = 0

    profits_df = pd.DataFrame.from_dict(profits_dict, 'index')
    if min_profit is not None:
        profits_df = profits_df[profits_df.profit >= min_profit]
    return profits_df

def compare_df_plots(
        df1:pd.DataFrame,
        df2:pd.DataFrame,
        title1:Optional[str] = None,
        title2:Optional[str] = None,
        y_label:Optional[str] = None,
        x_label:Optional[str] = None,
        y_formatter:Optional[ticker.Formatter] = None,
        x_formatter:Optional[ticker.Formatter] = None,
        plot_args:Optional[Dict] = None,
        save_name:Optional[str] = None
    ) -> None:
    """Compare and plot two DataFrames side by side.

    Args:
        df1 (pd.DataFrame): The first DataFrame to compare.
        df2 (pd.DataFrame): The second DataFrame to compare.
        title1 (Optional[str], optional): Title for the first plot. Defaults to None.
        title2 (Optional[str], optional): Title for the second plot. Defaults to None.
        y_label (Optional[str], optional): Label for the y-axis. Defaults to None.
        x_label (Optional[str], optional): Label for the x-axis. Defaults to None.
        y_formatter (Optional[ticker.Formatter], optional): Formatter for the y-axis tick labels.
                                                            Defaults to None.
        x_formatter (Optional[ticker.Formatter], optional): Formatter for the x-axis tick labels.
                                                            Defaults to None.
        plot_args (Optional[Dict], optional): Additional arguments for the plot. Defaults to None.
        save_name (Optional[str], optional): Name to save the plot as. Defaults to None.

    Returns:
        None
    """
    if plot_args is None:
        plot_args = {}
    if y_formatter is None:
        y_formatter = plt.FuncFormatter(lambda x, loc: f"${x/1000:,}K")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5), sharey=True)
    df1.plot(ax=ax1, fontsize=13, **plot_args)
    if title1 is not None:
        ax1.set_title(title1, fontsize=20)
    if y_label is not None:
        ax1.set_ylabel(y_label, fontsize=14)
    if x_label is not None:
        ax1.set_xlabel(x_label, fontsize=14)
    if 'secondary_y' in plot_args:
        ax1.get_legend().set_bbox_to_anchor((0.7, 0.99))
    if y_formatter is not None:
        ax1.yaxis.set_major_formatter(y_formatter)
    if x_formatter is not None:
        ax1.xaxis.set_major_formatter(x_formatter)
    ax1.grid(visible=True)
    ax1.right_ax.grid(False)
    df2.plot(ax=ax2, fontsize=13, **plot_args)
    if title2 is not None:
        ax2.set_title(title2, fontsize=20)
    if x_label is not None:
        ax2.set_xlabel(x_label, fontsize=14)
    if 'secondary_y' in plot_args:
        ax2.get_legend().set_bbox_to_anchor((0.7, 0.99))
    if x_formatter is not None:
        ax2.xaxis.set_major_formatter(x_formatter)
    ax2.grid(True)
    ax2.right_ax.grid(False)
    fig.tight_layout()
    if save_name is not None:
        plt.savefig(save_name+'_compare_df.png', dpi=300, bbox_inches="tight")
    plt.show()

def compute_aif_metrics(
        dataset_true:BaseAIF360DSProtocol,
        dataset_pred:BaseAIF360DSProtocol,
        unprivileged_groups:List[Dict],
        privileged_groups:List[Dict],
        ret_eval_dict:Optional[bool] = True
    ) -> Tuple[Optional[Dict], BaseAIF360MetricProtocol]:
    """Compute various fairness metrics using the AIF360 library.

    Args:
        dataset_true (BaseAIF360DSProtocol): The true dataset.
        dataset_pred (BaseAIF360DSProtocol): The predicted dataset.
        unprivileged_groups (List[Dict]): List of unprivileged groups.
        privileged_groups (List[Dict]): List of privileged groups.
        ret_eval_dict (Optional[bool], optional): Whether to return the evaluation dictionary.
                                                  Defaults to True.

    Returns:
        Tuple[Optional[Dict], BaseAIF360MetricProtocol]: A tuple containing the evaluation
                                                         dictionary (if ret_eval_dict is True)
                                                         and the AIF360 metric object.

    Raises:
        ModuleNotFoundError: If the `aif360` library is not installed.

    Note:
        This function requires the `aif360` library to be installed in order to execute.

    Example:
        dataset_true = ...
        dataset_pred = ...
        unprivileged_groups = ...
        privileged_groups = ...
        metrics_dict, metrics_cls = compute_aif_metrics(dataset_true, dataset_pred,\
                                                        unprivileged_groups, privileged_groups)
    """
    try:
        from aif360.metrics import ClassificationMetric
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("`aif360` must be installed to execute this function") from exc
    metrics_cls = ClassificationMetric(dataset_true, dataset_pred,\
                                        unprivileged_groups=unprivileged_groups,
                                        privileged_groups=privileged_groups)
    metrics_dict = {}
    metrics_dict["BA"] = 0.5*(metrics_cls.true_positive_rate()+
                                             metrics_cls.true_negative_rate())
    metrics_dict["SPD"] = metrics_cls.statistical_parity_difference()
    metrics_dict["DI"] = metrics_cls.disparate_impact()
    metrics_dict["AOD"] = metrics_cls.average_odds_difference()
    metrics_dict["EOD"] = metrics_cls.equal_opportunity_difference()
    metrics_dict["DFBA"] = metrics_cls.differential_fairness_bias_amplification()
    metrics_dict["TI"] = metrics_cls.theil_index()

    if ret_eval_dict:
        return metrics_dict, metrics_cls
    else:
        return metrics_cls

def compare_image_predictions(
        X_mod:np.ndarray,
        X_orig:np.ndarray,
        y_mod:Union[np.ndarray, list, tuple],
        y_orig:Union[np.ndarray, list, tuple],
        y_mod_prob:Optional[Union[np.ndarray, list, tuple]] = None,
        y_orig_prob:Optional[Union[np.ndarray, list, tuple]] = None,
        num_samples:Optional[int] = 3,
        title_mod_prefix:Optional[str] = "Modified: ",
        title_orig_prefix:Optional[str] = "Original: ",
        calc_difference:Optional[bool] = True,
        title_difference_prefix:Optional[str] = "Average difference: ",
        max_width:Optional[int] = 14,
        use_misclass:Optional[bool] = True,
        save_name:Optional[str] = None
    ) -> None:
    """Compare image predictions.

    Args:
        X_mod (np.ndarray): Modified images.
        X_orig (np.ndarray): Original images.
        y_mod (Union[np.ndarray, list, tuple]): Modified labels.
        y_orig (Union[np.ndarray, list, tuple]): Original labels.
        y_mod_prob (Optional[Union[np.ndarray, list, tuple]]): Probabilities of modified
                                                               labels. Default is None.
        y_orig_prob (Optional[Union[np.ndarray, list, tuple]]): Probabilities of original
                                                                labels. Default is None.
        num_samples (Optional[int]): Number of samples to display. Default is 3.
        title_mod_prefix (Optional[str]): Prefix for modified image titles. Default is
                                          "Modified: ".
        title_orig_prefix (Optional[str]): Prefix for original image titles. Default is
                                           "Original: ".
        calc_difference (Optional[bool]): Whether to calculate and display the average
                                          difference between modified and original images.
                                          Default is True.
        title_difference_prefix (Optional[str]): Prefix for the average difference title.
                                                 Default is "Average difference: ".
        max_width (Optional[int]): Maximum width of the displayed images. Default is 14.
        use_misclass (Optional[bool]): Whether to use misclassified samples for display.
                                       Default is True.
        save_name (Optional[str]): Name to save the comparison image. Default is None.

    Returns:
        None
    """
    if calc_difference:
        X_difference = np.mean(np.abs((X_mod - X_orig)))
        diff_title = (title_difference_prefix + '{:4.3f}').format(X_difference)
    if num_samples > 0:
        if use_misclass:
            misclass_idx = np.unique(np.where(y_orig != y_mod)[0])
        else:
            misclass_idx = np.unique(np.where(y_orig == y_mod)[0])
        if misclass_idx.shape[0] > 0:
            if misclass_idx.shape[0] < num_samples:
                num_samples = misclass_idx.shape[0]
                samples_idx = misclass_idx
            else:
                np.random.shuffle(misclass_idx)
                samples_idx = misclass_idx[0:num_samples]
            if num_samples > 2:
                width = max_width
                lg = math.log(num_samples)
            elif num_samples == 2:
                width = round(max_width*0.6)
                lg = 0.6
            else:
                width = round(max_width*0.3)
                lg = 0.3
            img_ratio = X_mod.shape[1]/X_mod.shape[2]
            height = round((((width - ((num_samples - 1)*(0.75 / lg)))/num_samples)*img_ratio))*2
            plt.subplots(figsize=(width,height))
            for i, s in enumerate(samples_idx, start=1):
                plt.subplot(2, num_samples, i)
                plt.imshow(X_mod[s])
                plt.grid(False)
                if num_samples > 3:
                    plt.axis('off')
                if y_mod_prob is None:
                    plt.title(f"{title_mod_prefix}{y_mod[s]}")
                else:
                    plt.title(f"{title_mod_prefix}{y_mod[s]} ({y_mod_prob[s]:.1%})")
            for i, s in enumerate(samples_idx, start=1):
                plt.subplot(2, num_samples, i+num_samples)
                plt.imshow(X_orig[s])
                plt.grid(False)
                if num_samples > 3:
                    plt.axis('off')
                if y_orig_prob is None:
                    plt.title(f"{title_orig_prefix}{y_orig[s]}")
                else:
                    plt.title(f"{title_orig_prefix}{y_orig[s]} ({y_orig_prob[s]:.1%})")
            if calc_difference:
                plt.subplots_adjust(bottom=0, top=0.88)
                fs = 21 - num_samples
                plt.suptitle(diff_title, fontsize=fs)
            if save_name is not None:
                plt.savefig(save_name+'_compare.png', dpi=300, bbox_inches="tight")
            plt.show()
        else:
            if calc_difference:
                print(diff_title)
            print("No Different Classifications")
