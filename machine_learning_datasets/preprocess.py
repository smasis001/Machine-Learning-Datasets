"""Common Utility Functions"""
# pylint: disable=E1101,W0212,C0302,C0103,C0415,C0121
from typing import Any, Tuple, Union, Dict,\
                   List, Optional, Callable
import io
import copy
import warnings
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap, Colormap
from matplotlib import cm
from scipy.spatial import distance
import cv2
import torchvision
from .common import ArrayLike, Tensor, BaseTransformerProtocol

def make_dummies_with_limits(
        df_:pd.DataFrame,
        colname:str,
        min_recs:Optional[Union[int, float]] = 0.005,
        max_dummies:Optional[int] = 20,
        defcatname:Optional[str] = 'Other',
        nospacechr:Optional[str] = '_'
    ) -> pd.DataFrame:
    """Make dummies with limits.

    Args:
        df_ (pd.DataFrame): The input DataFrame.
        colname (str): The name of the column to create dummies for.
        min_recs (Optional[Union[int, float]], default=0.005): The minimum number of repeated
                                                               records.
        max_dummies (Optional[int], default=20): The maximum number of dummies to create.
        defcatname (Optional[str], default='Other'): The name for the 'Other' category.
        nospacechr (Optional[str], default='_'): The character to replace spaces in the column name.

    Returns:
        pd.DataFrame: The DataFrame with dummies created.

    Note:
        - If min_recs is less than 1, it is interpreted as a fraction of the total number of
          records.
        - Dummies are created for the top values in the specified column, up to the maximum
          number of dummies.
        - Values that do not meet the minimum number of records or are beyond the maximum
          number of dummies are grouped into the 'Other' category.
        - Spaces in the column name are replaced with the specified character.
    """
    df = df_.copy()
    # min_recs is the number of repeated recalls
    if min_recs < 1:
        min_recs = df.shape[0]*min_recs
    topvals_df = df.groupby(colname).size().reset_index(name="counts").\
                    sort_values(by="counts", ascending=False).reset_index()
    other_l = topvals_df[(topvals_df.index > max_dummies) |\
                         (topvals_df.counts < min_recs)][colname].to_list()
    # Set the column name to the other_l if the column is in other_l
    if len(other_l):
        df.loc[df[colname].isin(other_l), colname] = defcatname
    # Remove nospacechr characters from the column name.
    if len(nospacechr) > 0:
        df[colname] = df[colname].str.replace(' ',\
                                                  nospacechr, regex=False)
    return pd.get_dummies(df, prefix=[colname], columns=[colname])

def make_dummies_from_dict(
        df:pd.DataFrame,
        colname:str,
        match_dict:Union[Dict, List],
        drop_orig:Optional[bool] = True,
        nospacechr:Optional[str] = '_'
    ) -> pd.DataFrame:
    """Creates dummy variables based on a dictionary or list of values.

    Args:
        df (pd.DataFrame): The input DataFrame.
        colname (str): The name of the column to create dummies from.
        match_dict (Union[Dict, List]): A dictionary or list of values to match against
                                        in the column.
        drop_orig (Optional[bool]): Whether to drop the original column after creating
                                    dummies. Defaults to True.
        nospacechr (Optional[str]): The character to replace spaces with in the dummy
                                    variable names. Defaults to '_'.

    Returns:
        pd.DataFrame: The DataFrame with dummy variables created.

    Example:
        >>> df = pd.DataFrame({'col1': ['apple', 'banana', 'orange']})
        >>> match_dict = {'apple': 'fruit', 'banana': 'fruit'}
        >>> make_dummies_from_dict(df, 'col1', match_dict)
           col1_fruit  col1_orange
        0           1            0
        1           1            0
        2           0            1
    """
    if isinstance(match_dict, list) is True:
        if len(nospacechr) > 0:
            match_dict = {match_key:match_key.\
                              replace(' ', nospacechr)\
                              for match_key in match_dict }
        else:
            match_dict = {match_key:match_key\
                              for match_key in match_dict}
    for match_key in match_dict.keys():
        df[colname+'_'+match_dict[match_key]] =\
                    np.where(df[colname].str.contains(match_key), 1, 0)
    if drop_orig:
        return df.drop([colname], axis=1)
    else:
        return df

def apply_cmap(
        img:np.ndarray,
        cmap:Union[str,Colormap],
        cmap_norm:Optional[Union[str,Colormap]] = None,
        alwaysscale:Optional[bool] = False,
        overlay_bg:Optional[np.ndarray] = None,
        **kwargs:Any
    ) -> np.ndarray:
    """Apply a colormap to an image.

    Args:
        img (np.ndarray): The input image.
        cmap (Union[str,Colormap]): The colormap to apply. Can be a string representing the name
                                    of the colormap or a Colormap object.
        cmap_norm (Optional[Union[str,Colormap]]): The normalization to apply to the image before
                                                   applying the colormap. Can be a string
                                                   representing the name of the normalization or
                                                   a Colormap object. Defaults to None.
        alwaysscale (Optional[bool]): Whether to always scale the image before applying the
                                      colormap. Defaults to False.
        overlay_bg (Optional[np.ndarray]): The background image to overlay on the colormap.
                                           Defaults to None.
        **kwargs (Any): Additional keyword arguments to pass to the normalize_heatmap function.

    Returns:
        np.ndarray: The image with the applied colormap.

    Note:
        - If the input image has 3 channels, it will be converted to grayscale before
          applying the colormap.
        - If cmap_norm is provided, the image will be normalized using the normalize_heatmap
          function before applying the colormap.
        - If alwaysscale is True or the image values are outside the range [0, 1], the
          image will be scaled using MinMaxScaler before applying the colormap.
        - The alpha channel of the colormap image will be removed.
        - If overlay_bg is provided, it will be overlaid on the colormap image using the
          heatmap_overlay function.
    """
    if len(img.shape) == 3:
        img = np.mean(img, axis=2) #cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) #
    inf_dec = 1e-5
    if cmap_norm is not None:
        img, _, _, _ = normalize_heatmap(img, cmap_norm,\
                                         **kwargs)
    elif alwaysscale or (img.min() < 0 - inf_dec) or (img.max() > 1 + inf_dec):
        img = preprocessing.MinMaxScaler().fit_transform(img)
    colormap = plt.get_cmap(cmap)
    img = np.delete(colormap(img), 3, 2)
    if overlay_bg is not None:
        if len(overlay_bg.shape) == 3:
            if overlay_bg.shape[0] == 3:
                overlay_bg = np.transpose(overlay_bg, (1,2,0))
            overlay_bg = cv2.cvtColor(overlay_bg, cv2.COLOR_RGB2GRAY)
        if len(overlay_bg.shape) == 2:
            overlay_bg = np.stack((overlay_bg,)*3, axis=-1)
        img = heatmap_overlay(img, overlay_bg) / 255
    return img

def tensor_to_img(
        tensor:Tensor,
        norm_std:Optional[Tuple] = None,
        norm_mean:Optional[Tuple] = None,
        to_numpy:Optional[bool] = False,
        cmap_norm:Optional[Union[str,Colormap]] = None,
        cmap:Optional[Union[str,Colormap]] = None,
        cmap_alwaysscale:Optional[bool] = False,
        overlay_bg:Optional[np.ndarray] = None,
        **kwargs:Any
    ) -> Optional[np.ndarray]:
    """Converts a tensor to an image.

    Args:
        tensor (Tensor): The input tensor.
        norm_std (Optional[Tuple]): The standard deviation for normalization. Default is None.
        norm_mean (Optional[Tuple]): The mean for normalization. Default is None.
        to_numpy (Optional[bool]): Whether to convert the tensor to a numpy array. Default is False.
        cmap_norm (Optional[Union[str,Colormap]]): The normalization method for the colormap.
                                                   Default is None.
        cmap (Optional[Union[str,Colormap]]): The colormap to apply to the image. Default is None.
        cmap_alwaysscale (Optional[bool]): Whether to always scale the colormap. Default is False.
        overlay_bg (Optional[np.ndarray]): The background image to overlay. Default is None.
        **kwargs (Any): Additional keyword arguments.

    Returns:
        Optional[np.ndarray]: The converted image as a numpy array, or None if the conversion fails.
    """
    if norm_std is not None and norm_mean is not None:
        tensor_ = copy.deepcopy(tensor)
        for t, s, m in zip(tensor_, norm_std, norm_mean):
            t.mul_(s).add_(m)
    else:
        tensor_ = tensor

    if to_numpy:
        img_ = tensor_.cpu().detach().numpy()
        if (len(img_.shape) == 3) and (img_.shape[0] == 3):
            img_ = np.transpose(img_, (1,2,0))
            #img_ = np.where(img_ > 0, img_, 0)
        if cmap_norm is not None:
            img_, default_cmap, _, _ = normalize_heatmap(img_, cmap_norm,\
                                                            **kwargs)
            if cmap is None:
                cmap = default_cmap
        if cmap is not None:
            img_ = apply_cmap(img_, cmap, alwaysscale=cmap_alwaysscale, overlay_bg=overlay_bg)
    else:
        img_ = torchvision.transforms.ToPILImage()(tensor_)

    return img_

def discretize(
        v:Union[str, pd.Series, np.ndarray],
        v_intervals:Union[int, list, np.ndarray],
        use_quantiles:Optional[bool] = False,
        use_continuous_bins:Optional[bool]=  False
    ) -> Tuple[Union[str, pd.Series, np.ndarray], np.ndarray]:
    """Discretize a variable into intervals.

    Args:
        v (Union[str, pd.Series, np.ndarray]): The variable to be discretized.
        v_intervals (Union[int, list, np.ndarray]): The intervals to discretize the variable into.
        use_quantiles (Optional[bool], default=False): Whether to use quantiles for discretization.
        use_continuous_bins (Optional[bool], default=False): Whether to use continuous bins for
                                                             discretization.

    Returns:
        Tuple[Union[str, pd.Series, np.ndarray], np.ndarray]: The discretized variable and the bins.

    Raises:
        ValueError: If the length of the interval does not match the number of unique items in
                    the array.

    Note:
        - If `v` is a string and `v_intervals` is a list or array, the function returns `v` and
          `v_intervals` as is.
        - If `v` is numeric and `v_intervals` is an integer, the function discretizes `v` into
         `v_intervals` bins.
        - If `v` is an object or a category, the function converts `v` into a string and assigns
          a numerical value to each unique item.

    Examples:
        >>> v = [1, 2, 3, 4, 5]
        >>> v_intervals = 2
        >>> discretize(v, v_intervals)
        ([0, 0, 1, 1, 1], array([1, 3, 5]))

        >>> v = pd.Series(['A', 'B', 'C', 'A', 'B'])
        >>> v_intervals = ['A', 'B', 'C']
        >>> discretize(v, v_intervals)
        (0    0
        1    1
        2    2
        3    0
        4    1
        dtype: object, array(['A', 'B', 'C'], dtype=object))
    """
    if isinstance(v, (pd.core.series.Series, np.ndarray)) and\
        isinstance(v_intervals, (list, np.ndarray)) and len(np.unique(v)) != len(v_intervals):
        raise ValueError("length of interval must match unique items in array")

    if isinstance(v, (str)) and isinstance(v_intervals, (list, np.ndarray)):
        #name of variable instead of array and list of intervals used
        if isinstance(v_intervals, list) is True:
            v_intervals = np.array(v_intervals)
        return v, v_intervals

    if (np.isin(v.dtype, [int, float, 'int8', 'int16', 'int32', 'float16', 'float32'])) and\
        (isinstance(v_intervals, (int))) and (len(np.unique(v)) >= v_intervals) and\
            (max(v) > min(v)):
        #v is discretizable, otherwise assumed to be already discretized
        if use_continuous_bins:
            if use_quantiles:
                v, bins = pd.qcut(v, v_intervals, duplicates='drop', retbins=True,\
                                  labels=True, precision=2)
            else:
                v, bins = pd.cut(v, v_intervals, duplicates='drop', retbins=True,\
                                 labels=True, precision=2)
        else:
            if use_quantiles:
                v = pd.qcut(v, v_intervals, duplicates='drop', precision=2)
            else:
                v = pd.cut(v, v_intervals, duplicates='drop', precision=2)

    if np.isin(v.dtype, [object, 'category']):
        if not isinstance(v, (pd.core.series.Series)):
            v = pd.Series(v)
        bins = np.sort(np.unique(v)).astype(str)
        v = v.astype(str)
        bin_dict = {bins[i]:i for i in range(len(bins))}
        v = v.replace(bin_dict)
    else:
        bins = np.unique(v)

    if isinstance(v_intervals, (list, np.ndarray)) and len(bins) == len(v_intervals):
        bins = v_intervals

    return v, bins

def normalize_heatmap(
        heatmap:np.ndarray,
        sign:str,
        outlier_perc:Optional[int] = 2,
        reduction_axis:Optional[int] = None
    ) -> Tuple[np.ndarray, Union[str,Colormap], int, int]:
    """Normalize the heatmap based on the given sign type and outlier percentage.

    Args:
        heatmap (np.ndarray): The input heatmap.
        sign (str): The sign type for normalization. Possible values are "all", "positive",
                    "negative", and "absolute_value".
        outlier_perc (Optional[int]): The percentage of outliers to remove. Default is 2.
        reduction_axis (Optional[int]): The axis along which to reduce the heatmap. Default is None.

    Returns:
        Tuple[np.ndarray, Union[str,Colormap], int, int]: A tuple containing the normalized heatmap,
                                                          the colormap, vmin, and vmax.

    Raises:
        AssertionError: If the sign type is not valid.
    """
    heatmap_combined = heatmap
    if reduction_axis is not None:
        heatmap_combined = np.sum(heatmap, axis=reduction_axis)

    # Choose appropriate signed values and rescale, removing given outlier percentage.
    default_cmap = "jet"
    if sign == "all":
        threshold = cumulative_sum_threshold(np.abs(heatmap_combined), 100 - outlier_perc)
        default_cmap = LinearSegmentedColormap.from_list(
            "RdWhGn", ["red", "white", "green"]
        )
        vmin, vmax = -1, 1
    elif sign == "positive":
        heatmap_combined = (heatmap_combined > 0) * heatmap_combined
        threshold = cumulative_sum_threshold(heatmap_combined, 100 - outlier_perc)
        #default_cmap = "Greens"
        vmin, vmax = 0, 1
    elif sign == "negative":
        heatmap_combined = (heatmap_combined < 0) * heatmap_combined
        threshold = -1 * cumulative_sum_threshold(
            np.abs(heatmap_combined), 100 - outlier_perc
        )
        #default_cmap = "Reds"
        vmin, vmax = 0, 1
    elif sign == "absolute_value":
        heatmap_combined = np.abs(heatmap_combined)
        threshold = cumulative_sum_threshold(heatmap_combined, 100 - outlier_perc)
        #default_cmap = "Blues"
        vmin, vmax = 0, 1
    else:
        raise AssertionError("Heatmap normalization sign type is not valid.")

    heatmap_scaled = normalize_scale(heatmap_combined, threshold)
    if (vmin == -1) and (vmax == 1):
        heatmap_scaled = minmax_scale_img_posneg(heatmap_scaled)
    return heatmap_scaled, default_cmap, vmin, vmax

def cumulative_sum_threshold(
        values:np.ndarray,
        percentile:int
    ) -> float:
    """Calculate the cumulative sum threshold.

    This function calculates the cumulative sum threshold of a given array
    of values based on a specified percentile.

    Args:
        values (np.ndarray): The array of values.
        percentile (int): The percentile for thresholding. Must be between 0 and 100 inclusive.

    Returns:
        float: The threshold value.

    Raises:
        AssertionError: If the percentile is not between 0 and 100 inclusive.
    """
    # given values should be non-negative
    assert percentile >= 0 and percentile <= 100, (
        "Percentile for thresholding must be " "between 0 and 100 inclusive."
    )
    sorted_vals = np.sort(values.flatten())
    cum_sums = np.cumsum(sorted_vals)
    threshold_id = np.where(cum_sums >= cum_sums[-1] * 0.01 * percentile)[0][0]
    return sorted_vals[threshold_id]

def normalize_scale(
        heatmap:np.ndarray,
        scale_factor:float
    ) -> np.ndarray:
    """Normalize the given heatmap by dividing it by the specified scale factor.

    Parameters:
    - heatmap (np.ndarray): The input heatmap to be normalized.
    - scale_factor (float): The scale factor to divide the heatmap by.

    Returns:
    - np.ndarray: The normalized heatmap.

    Raises:
    - UserWarning: If the scale_factor is equal to 0, a warning is raised indicating that
                   normalization is not possible.
    - UserWarning: If the absolute value of the scale_factor is less than 1e-5, a warning
                   is raised indicating that the heatmap values are close to 0 and the
                   visualized results may be misleading.

    Note:
    - The normalized heatmap is obtained by dividing the input heatmap by the scale_factor.
    - The resulting normalized heatmap is clipped between -1 and 1 using np.clip() function.
    """
    if scale_factor == 0:
        warnings.warn("Cannot normalize by scale factor = 0")
        heatmap_norm = heatmap
    else:
        if abs(scale_factor) < 1e-5:
            warnings.warn(
                "Attempting to normalize by value approximately 0, visualized results"
                "may be misleading. This likely means that heatmap values are all"
                "close to 0."
            )
        heatmap_norm = heatmap / scale_factor
    return np.clip(heatmap_norm, -1, 1)

def minmax_scale_img(
        img:np.ndarray
    ) -> np.ndarray:
    """Scales the input image to the range [0, 1].

    Args:
        img (np.ndarray): The input image.

    Returns:
        np.ndarray: The scaled image.
    """
    if img.max() != img.min():
        img = (img - img.min()) / (img.max() - img.min())
    return img

def minmax_scale_img_posneg(
        img:np.ndarray
    ) -> np.ndarray:
    """Scales the input image to the range [0, 1] by performing min-max scaling
       separately for positive and negative values.

    Args:
        img (np.ndarray): The input image.

    Returns:
        np.ndarray: The scaled image.
    """
    img_pos = np.where(img > 0, img, 0)
    img_pos = np.where(img > 0, (minmax_scale_img(img_pos) / 2) + 0.5, 0.5)
    img_neg = np.where(img < 0, img, 0)
    img_neg = np.where(img < 0, (minmax_scale_img(img_neg) / 2), 0.5)
    img = np.where(img==0, 0.5, np.where(img > 0, img_pos, img_neg))
    return img

def img_np_from_fig(
        fig:Figure,
        dpi:Optional[int] = 14
    ) -> np.ndarray:
    """Converts a matplotlib figure to a NumPy array representing an image.

    Args:
        fig (Figure): The matplotlib figure to convert.
        dpi (Optional[int]): The resolution of the image in dots per inch. Default is 14.

    Returns:
        np.ndarray: The NumPy array representing the image.

    Example:
        fig = plt.figure()
        # ... create and modify the figure ...
        img = img_np_from_fig(fig)
    """
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi)
    buffer.seek(0)
    img_np = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
    buffer.close()
    img_np = cv2.imdecode(img_np, 1)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    return img_np

def heatmap_overlay(
        bg_img:np.ndarray,
        overlay_img:np.ndarray,
        cmap:Optional[Union[str,Colormap]] = 'jet'
    ) -> np.ndarray:
    """Overlay a heatmap on top of an image.

    Args:
        bg_img (np.ndarray): The background image.
        overlay_img (np.ndarray): The heatmap image to overlay.
        cmap (Optional[Union[str,Colormap]], optional): The colormap to use for the heatmap.
                                                        Defaults to 'jet'.

    Returns:
        np.ndarray: The resulting image with the heatmap overlay.
    """
    img = np.uint8(bg_img[..., :3] * 255)
    if len(overlay_img.shape) == 2:
        overlay_img = cm.get_cmap(cmap)(overlay_img)
    heatmap = np.uint8(overlay_img[..., :3] * 255)
    return cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)

def find_closest_datapoint_idx(
        point:ArrayLike,
        points:ArrayLike,
        metric_or_fn:Optional[Union[str, Callable]] = 'euclidean',
        find_exact_first:Optional[int] = 0,
        distargs:Optional[Dict] = None,
        scaler:Optional[BaseTransformerProtocol] = None
    ) -> int:
    """Find the index of the closest datapoint to a given point.

    Args:
        point (ArrayLike): The point for which to find the closest datapoint index.
        points (ArrayLike): The array of datapoints to search for the closest index.
        metric_or_fn (Optional[Union[str, Callable]], default='euclidean'): The distance metric or
                                         function to use for calculating distances between points.
        find_exact_first (Optional[int], default=0): Determines the behavior when multiple closest
                                                     datapoints are found.
            - 0: Return the index of the last closest datapoint found.
            - 1: Return the index of the last closest datapoint found where the sum of the features
                 of the datapoint matches the sum of the features of the point.
            - 2: Return the index of the last closest datapoint found where all the features of the
                 datapoint match all the features of the point.
        distargs (Optional[Dict], default=None): Additional arguments to pass to the distance metric
                                                 or function.
        scaler (Optional[BaseTransformerProtocol], default=None): A scaler object to transform
                                                                  the point and points before
                                                                  calculating distances.

    Returns:
        int: The index of the closest datapoint to the given point.

    Raises:
        ValueError: If the point is not 1-dimensional, the points are not 2-dimensional,
                    or the number of features in the point and points do not match.
        ValueError: If `metric_or_fn` is not a string or a callable object.

    Note:
        - If `find_exact_first` is set to 1, the function will first check for datapoints where
          the sum of the features matches the sum of the features of the point.
        - If `find_exact_first` is set to 2, the function will check for datapoints where all
          the features match all the features of the point.
        - If `scaler` is provided, the point and points will be transformed before calculating
          distances.
        - If `metric_or_fn` is a string, the function will use the specified distance metric
             from the `scipy.spatial.distance` module.
        - If `metric_or_fn` is a callable object, the function will use the provided distance
          function to calculate distances.
    """
    if distargs is None:
        distargs = {}
    if len(point.shape)!=1 or len(points.shape)!=2 or point.shape[0]!=points.shape[1]:
        raise ValueError("point must be a 1d and points 2d where their number of features match")
    closest_idx = None
    if find_exact_first==1:
        sums_pts = np.sum(points, axis=1)
        sum_pt = np.sum(point, axis=0)
        s = sums_pts==sum_pt
        if isinstance(s, pd.core.series.Series):
            closest_idxs = s[s==True].index.to_list() #TODO: check how to solve C0121
        else:
            closest_idxs = s.nonzero()[0]
        if len(closest_idxs) > 0:
            closest_idx = closest_idxs[-1]
    elif find_exact_first==2:
        if isinstance(points, pd.core.frame.DataFrame):
            for i in reversed(range(points.shape[0])):
                if np.allclose(point, points.iloc[i]):
                    closest_idx = points.iloc[i].name
                    break
        else:
            for i in reversed(range(points.shape[0])):
                if np.allclose(point, points[i]):
                    closest_idx = i
                    break
    if closest_idx is None:
        if scaler is not None:
            point_ = scaler.transform([point])
            #points_ = scaler.transform(points)
        else:
            point_ = [point]
            #points_ = points
        if isinstance(metric_or_fn, str):
            closest_idx = distance.cdist(point_, points, metric=metric_or_fn, **distargs).argmin()
        elif callable(metric_or_fn):
            dists = []
            if isinstance(points, pd.core.frame.DataFrame):
                for i in range(points.shape[0]):
                    dists.append(metric_or_fn(point_[0], points.iloc[i], **distargs))
            else:
                for i in range(points.shape[0]):
                    dists.append(metric_or_fn(point_[0], points[i], **distargs))
            closest_idx = np.array(dists).argmin()
        else:
            raise ValueError("`metric_or_fn` must be a string of a distance metric or valid "
                             "distance function")
        if isinstance(points, pd.core.frame.DataFrame):
            closest_idx = points.iloc[closest_idx].name

    return closest_idx
