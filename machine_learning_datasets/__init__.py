"""Load Functionality"""
import os
import sys
import warnings
from pathlib import Path
from .config import init, load
from .common import runcmd, plot_polar
from .preprocess import make_dummies_with_limits, make_dummies_from_dict, apply_cmap,\
                         tensor_to_img, discretize, normalize_heatmap, img_np_from_fig,\
                         heatmap_overlay, find_closest_datapoint_idx
from .evaluate import  evaluate_class_mdl, evaluate_reg_mdl, evaluate_multiclass_mdl,\
                       evaluate_class_metrics_mdl, evaluate_reg_metrics_mdl,\
                        evaluate_multiclass_metrics_mdl
from .interpret import compare_confusion_matrices, plot_prob_progression, plot_prob_contour_map,\
                       create_decision_plot, describe_cf_instance, compare_img_pred_viz,\
                       create_attribution_grid, approx_predict_ts, profits_by_thresh,\
                       compare_df_plots, compute_aif_metrics, compare_image_predictions
from .deprecated import plot_3dim_decomposition, encode_classification_error_vector
from .sources.kaggle import Kaggle
from .sources.url import URL

if sys.version_info < (3, 9):
    warnings.warn("`mldatasets` only supports Python 3.9 and above!")

__version__ = '0.1.23'

init(os.path.join(Path().parent.absolute(), 'data'))
