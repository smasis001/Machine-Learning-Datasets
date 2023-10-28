"""Common Utility Functions"""
# pylint: disable=E1101,W0212,C0302,C0103,C0415,C0121
from typing import Tuple, Union, Dict, Optional
import pandas as pd
import numpy as np
from sklearn import preprocessing, metrics
import matplotlib.pyplot as plt
from scipy import sparse
import torchvision
import seaborn as sns
from .common import ArrayLike, BaseModelProtocol, BaseTransformerProtocol

def evaluate_class_mdl(
        fitted_model:BaseModelProtocol,
        X_train:ArrayLike,
        X_test:ArrayLike,
        y_train:ArrayLike,
        y_test:ArrayLike,
        plot_roc:Optional[bool] = True,
        plot_conf_matrix:Optional[bool] = False,
        pct_matrix:Optional[bool] = True,
        predopts:Optional[Dict] = None,
        show_summary:Optional[bool] = True,
        ret_eval_dict:Optional[bool] = False,
        save_name:Optional[str] = None
    ) -> Union[Dict,Tuple[ArrayLike, ArrayLike, ArrayLike]]:
    """Evaluate the performance of a classification model.

    Args:
        fitted_model (BaseModelProtocol): The fitted classification model.
        X_train (ArrayLike): The training data.
        X_test (ArrayLike): The testing data.
        y_train (ArrayLike): The labels for the training data.
        y_test (ArrayLike): The labels for the testing data.
        plot_roc (Optional[bool], default=True): Whether to plot the ROC curve.
        plot_conf_matrix (Optional[bool], default=False): Whether to plot the confusion matrix.
        pct_matrix (Optional[bool], default=True): Whether to display the confusion matrix as
                                                   percentages.
        predopts (Optional[Dict], default=None): Additional options for predicting probabilities.
        show_summary (Optional[bool], default=True): Whether to display the evaluation summary.
        ret_eval_dict (Optional[bool], default=False): Whether to return the evaluation metrics
                                                       as a dictionary.
        save_name (Optional[str], default=None): The name to save the plots.

    Returns:
        Union[Dict, Tuple[ArrayLike, ArrayLike, ArrayLike]]: If `ret_eval_dict` is True,
                                    returns a dictionary of evaluation metrics. Otherwise,
                                    returns a tuple of the predicted labels for the training
                                    data, the predicted probabilities for the testing data,
                                    and the predicted labels for the testing data.
    """
    if predopts is None:
        predopts = {}
    y_train_pred = fitted_model.predict(X_train, **predopts).squeeze()
    if len(np.unique(y_train_pred)) > 2:
        y_train_pred = np.where(y_train_pred > 0.5, 1, 0)
        y_test_prob = fitted_model.predict(X_test, **predopts).squeeze()
        y_test_pred = np.where(y_test_prob > 0.5, 1, 0)
    else:
        y_test_prob = fitted_model.predict_proba(X_test, **predopts)[:,1]
        y_test_pred = np.where(y_test_prob > 0.5, 1, 0)

    roc_auc = metrics.roc_auc_score(y_test, y_test_prob)
    if plot_roc:
        plt.figure(figsize = (12,12))
        plt.tick_params(axis = 'both', which = 'major', labelsize = 12)
        fpr, tpr, _ = metrics.roc_curve(y_test, y_test_prob)
        plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:0.2f}')
        plt.plot([0, 1], [0, 1], 'k--')  # coin toss line
        plt.xlabel('False Positive Rate', fontsize = 14)
        plt.ylabel('True Positive Rate', fontsize = 14)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.legend(loc="lower right")
        if save_name is not None:
            plt.savefig(save_name+'_roc.png', dpi=300, bbox_inches="tight")
        plt.show()

    if plot_conf_matrix:
        cf_matrix = metrics.confusion_matrix(y_test,\
                                             y_test_pred)
        plt.figure(figsize=(6, 5))
        if pct_matrix:
            sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True,\
                        fmt='.2%', cmap='Blues', annot_kws={'size':16})
        else:
            sns.heatmap(cf_matrix, annot=True,\
                        fmt='d',cmap='Blues', annot_kws={'size':16})
        if save_name is not None:
            plt.savefig(save_name+'_cm.png', dpi=300, bbox_inches="tight")
        plt.show()

    if show_summary:
        print(f"Accuracy_train:  {metrics.accuracy_score(y_train, y_train_pred):.4f}\t\t"
              f"Accuracy_test:   {metrics.accuracy_score(y_test, y_test_pred):.4f}")
        print(f"Precision_test:  {metrics.precision_score(y_test,y_test_pred,zero_division=0):.4f}"
            f"\t\tRecall_test:     {metrics.recall_score(y_test,y_test_pred,zero_division=0):.4f}")
        print(f"ROC-AUC_test:    {roc_auc:.4f}\t\t"
              f"F1_test:         {metrics.f1_score(y_test, y_test_pred, zero_division=0):.4f}\t\t"
              f"MCC_test: {metrics.matthews_corrcoef(y_test, y_test_pred):.4f}")
    if ret_eval_dict:
        return evaluate_class_metrics_mdl(fitted_model, y_train_pred, y_test_prob, y_test_pred,\
                                          y_train, y_test)
    else:
        return y_train_pred, y_test_prob, y_test_pred

def evaluate_reg_mdl(
        fitted_model:BaseModelProtocol,
        X_train:ArrayLike,
        X_test:ArrayLike,
        y_train:ArrayLike,
        y_test:ArrayLike,
        scaler:Optional[BaseTransformerProtocol] = None,
        plot_regplot:Optional[bool] = True,
        y_truncate:Optional[bool] = False,
        show_summary:Optional[bool] = True,
        ret_eval_dict:Optional[bool] = False,
        predopts:Optional[Dict] = None,
        save_name:Optional[str] = None
    ) -> Union[Dict,Tuple[ArrayLike, ... ]]:
    """Evaluate a regression model.

    Args:
        fitted_model (BaseModelProtocol): The fitted regression model.
        X_train (ArrayLike): The training data features.
        X_test (ArrayLike): The testing data features.
        y_train (ArrayLike): The training data target variable.
        y_test (ArrayLike): The testing data target variable.
        scaler (Optional[BaseTransformerProtocol]): The scaler to use for inverse transforming
                                                    the target variables. Default is None.
        plot_regplot (Optional[bool]): Whether to plot a regression plot. Default is True.
        y_truncate (Optional[bool]): Whether to truncate the target variables to match the
                                     predicted values. Default is False.
        show_summary (Optional[bool]): Whether to print the evaluation summary. Default is True.
        ret_eval_dict (Optional[bool]): Whether to return the evaluation metrics as a dictionary.
                                        Default is False.
        predopts (Optional[Dict]): Additional options for the predict method. Default is None.
        save_name (Optional[str]): The name to use for saving the regression plot. Default is None.

    Returns:
        Union[Dict, Tuple[ArrayLike, ...]]: If ret_eval_dict is True, returns the evaluation
        metrics as a dictionary.
        If y_truncate is True, returns the predicted values and target variables for both
        training and testing data.
        Otherwise, returns only the predicted values for both training and testing data.
    """
    if predopts is None:
        predopts = {}
    y_train_ = y_train.copy()
    y_test_ = y_test.copy()
    if not isinstance(X_train, (np.ndarray, tuple, list)) or X_train.shape[1] != y_train.shape[1]:
        y_train_pred = fitted_model.predict(X_train, **predopts)
    else:
        y_train_pred = X_train.copy()
    if not isinstance(X_test, (np.ndarray, tuple, list)) or X_test.shape[1] != y_test.shape[1]:
        y_test_pred = fitted_model.predict(X_test, **predopts)
    else:
        y_test_pred = X_test.copy()
    if y_truncate:
        y_train_ = y_train_[-y_train_pred.shape[0]:]
        y_test_ = y_test_[-y_test_pred.shape[0]:]
    if scaler is not None:
        y_train_ = scaler.inverse_transform(y_train_)
        y_test_ = scaler.inverse_transform(y_test_)
        y_train_pred = scaler.inverse_transform(y_train_pred)
        y_test_pred = scaler.inverse_transform(y_test_pred)

    if plot_regplot:
        plt.figure(figsize = (12,12))
        plt.ylabel('Predicted', fontsize = 14)
        plt.scatter(y_test_, y_test_pred)
        sns.regplot(x=y_test_, y=y_test_pred, color="g")
        plt.xlabel('Observed', fontsize = 14)
        if save_name is not None:
            plt.savefig(save_name+'_regplot.png', dpi=300, bbox_inches="tight")
        plt.show()

    if show_summary:
        print(f"RMSE_train: {metrics.mean_squared_error(y_train_, y_train_pred, squared=False):.4f}"
              f"\tRMSE_test: {metrics.mean_squared_error(y_test_, y_test_pred, squared=False):.4f}"
              f"\tr2: {metrics.r2_score(y_test_, y_test_pred):.4f}")

    if ret_eval_dict:
        return evaluate_reg_metrics_mdl(fitted_model, y_train_pred, y_test_pred, y_train_, y_test_)
    elif y_truncate:
        return y_train_pred, y_test_pred, y_train_, y_test_
    else:
        return y_train_pred, y_test_pred

def evaluate_multiclass_mdl(
        fitted_model:BaseModelProtocol,
        X:Union[ArrayLike,torchvision.datasets.DatasetFolder],
        y:Optional[ArrayLike] = None,
        class_l:Optional[list] = None,
        ohe:Optional[BaseTransformerProtocol] = None,
        plot_roc:Optional[bool] = False,
        plot_roc_class:Optional[bool] = True,
        plot_conf_matrix:Optional[bool] = True,
        pct_matrix:Optional[bool] = True,
        plot_class_report:Optional[bool] = True,
        ret_eval_dict:Optional[bool] = False,
        predopts:Optional[Dict] = None,
        save_name:Optional[str] = None
    ) -> Union[Dict,Tuple[ArrayLike, ArrayLike]]:
    """
    Evaluate a multiclass classification model.

    Parameters:
        fitted_model (BaseModelProtocol): The fitted model to evaluate.
        X (Union[ArrayLike,torchvision.datasets.DatasetFolder]): The input data for evaluation.
        y (Optional[ArrayLike]): The true labels for evaluation. Default is None.
        class_l (Optional[list]): The list of class labels. Default is None.
        ohe (Optional[BaseTransformerProtocol]): The one-hot encoder for labels. Default is None.
        plot_roc (Optional[bool]): Whether to plot ROC curves. Default is False.
        plot_roc_class (Optional[bool]): Whether to plot ROC curves for each class. Default is True.
        plot_conf_matrix (Optional[bool]): Whether to plot the confusion matrix. Default is True.
        pct_matrix (Optional[bool]): Whether to display the confusion matrix as percentages.
                                     Default is True.
        plot_class_report (Optional[bool]): Whether to print the classification report. Default is
                                            True.
        ret_eval_dict (Optional[bool]): Whether to return evaluation metrics as a dictionary.
                                        Default is False.
        predopts (Optional[Dict]): Additional options for prediction. Default is None.
        save_name (Optional[str]): The name to save the plots. Default is None.

    Returns:
        Union[Dict,Tuple[ArrayLike, ArrayLike]]: The evaluation metrics or predicted labels
                                                 and probabilities.

    Raises:
        TypeError: If the data is not in the right format.
        TypeError: If sklearn one-hot encoder is not provided when labels aren't already encoded.
        ValueError: If the labels don't have dimensions that match the classes.
        ValueError: If the list of classes provided doesn't match the dimensions of model
                    predictions.
    """
    if predopts is None:
        predopts = {}
    if isinstance(X, (torchvision.datasets.DatasetFolder)):
        y = np.array([l for _, l in X])
    elif not isinstance(X, (list, tuple, np.ndarray, pd.DataFrame)) or\
        not isinstance(y, (list, tuple, np.ndarray, pd.Series)):
        raise TypeError("Data is not in the right format")
    if class_l is None:
        class_l = list(np.unique(class_l))
    n_classes = len(class_l)
    y = np.array(y)
    if len(y.shape)==1:
        y = np.expand_dims(y, axis=1)
    if y.shape[1] == 1:
        if isinstance(ohe, (preprocessing._encoders.OneHotEncoder)):
            y_ohe = ohe.transform(y)
        else:
            raise TypeError("Sklearn one-hot encoder is a required parameter "
                            "when labels aren't already encoded")
        if y.dtype.kind in ['i','u']:
            y = np.array([[class_l[o]] for o in y.reshape(-1)])
    elif y.shape[1] == n_classes:
        y_ohe = y.copy()
        y = np.array([[class_l[o]] for o in np.argmax(y_ohe, axis=1)])
    else:
        raise ValueError("Labels don't have dimensions that match the classes")
    y_prob = fitted_model.predict(X, **predopts)
    if isinstance(y_prob, sparse.csc_matrix):
        y_prob = y_prob.toarray()
    if len(y_prob.shape)==1:
        y_prob = np.expand_dims(y_prob, axis=1)
    if y_prob.shape[1] == 1:
        y_prob = fitted_model.predict_proba(X, **predopts)
    if y_prob.shape[1] == n_classes:
        y_pred_ord = np.argmax(y_prob, axis=1)
        y_pred = [class_l[o] for o in y_pred_ord]
    else:
        raise ValueError("List of classes provided doesn't match "
                         "the dimensions of model predictions")
    if plot_roc:
        #Compute FPR/TPR metrics for each class
        fpr = {}
        tpr = {}
        roc_auc = {}
        for i in range(n_classes):
            fpr[i], tpr[i], _ = metrics.roc_curve(y_ohe[:, i], y_prob[:, i])
            roc_auc[i] = metrics.auc(fpr[i], tpr[i])
        fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_ohe.ravel(), y_prob.ravel())
        roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

        # Compute interpolated macro and micro
        fpr["macro"] = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        tpr["macro"] = np.zeros_like(fpr["macro"])
        for i in range(n_classes):
            tpr["macro"] += np.interp(fpr["macro"], fpr[i], tpr[i])
        tpr["macro"] /= n_classes
        roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

        # Plot all ROCs
        plt.figure(figsize = (12,12))
        plt.tick_params(axis = 'both', which = 'major', labelsize = 12)
        plt.plot(fpr["micro"], tpr["micro"],
                 label=f'micro-average ROC curve (area = {roc_auc["micro"]:0.2f})',
                 color='navy', linestyle=':', linewidth=4)
        plt.plot(fpr["macro"], tpr["macro"],
                 label=f'macro-average ROC curve (area = {roc_auc["macro"]:0.2f})',
                 color='deeppink', linestyle=':', linewidth=4)
        if plot_roc_class:
            for i in range(n_classes):
                plt.plot(fpr[i], tpr[i],
                         label=f'ROC for class {class_l[i]} (area = {roc_auc[i]:0.2f})')
        plt.plot([0, 1], [0, 1], 'k--') # coin toss line
        plt.xlabel('False Positive Rate', fontsize = 14)
        plt.ylabel('True Positive Rate', fontsize = 14)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.legend(loc="lower right")
        if save_name is not None:
            plt.savefig(save_name+'_roc.png', dpi=300, bbox_inches="tight")
        plt.show()

    if plot_conf_matrix:
        conf_matrix = metrics.confusion_matrix(y, y_pred, labels=class_l)
        plt.figure(figsize=(12, 11))
        if pct_matrix:
            sns.heatmap(conf_matrix/np.sum(conf_matrix), annot=True, xticklabels=class_l,\
                        yticklabels=class_l, fmt='.1%', cmap='Blues', annot_kws={'size':12})
        else:
            sns.heatmap(conf_matrix, annot=True, xticklabels=class_l, yticklabels=class_l,\
                        cmap='Blues', annot_kws={'size':12})
        if save_name is not None:
            plt.savefig(save_name+'_cm.png', dpi=300, bbox_inches="tight")
        plt.show()

    if plot_class_report:
        print(metrics.classification_report(y, y_pred, digits=3, zero_division=0))

    if ret_eval_dict:
        return evaluate_multiclass_metrics_mdl(fitted_model, y_prob, y_pred, y, ohe)
    else:
        return y_pred, y_prob

def evaluate_class_metrics_mdl(
        fitted_model:BaseModelProtocol,
        y_train_pred:ArrayLike,
        y_test_prob:ArrayLike,
        y_test_pred:ArrayLike,
        y_train:ArrayLike,
        y_test:ArrayLike
    ) -> Dict:
    """Evaluate the classification metrics for a fitted model.

    Args:
        fitted_model (BaseModelProtocol): The fitted model.
        y_train_pred (ArrayLike): The predicted labels for the training set.
        y_test_prob (ArrayLike): The predicted probabilities for the test set. Can be None.
        y_test_pred (ArrayLike): The predicted labels for the test set.
        y_train (ArrayLike): The true labels for the training set.
        y_test (ArrayLike): The true labels for the test set.

    Returns:
        dict: A dictionary containing the evaluation metrics:
            - 'fitted': The fitted model.
            - 'preds_train': The predicted labels for the training set.
            - 'probs_test' (optional): The predicted probabilities for the test set.
            - 'preds_test': The predicted labels for the test set.
            - 'accuracy_train': The accuracy score for the training set.
            - 'accuracy_test': The accuracy score for the test set.
            - 'precision_train': The precision score for the training set.
            - 'precision_test': The precision score for the test set.
            - 'recall_train': The recall score for the training set.
            - 'recall_test': The recall score for the test set.
            - 'f1_train': The F1 score for the training set.
            - 'f1_test': The F1 score for the test set.
            - 'mcc_train': The Matthews correlation coefficient for the training set.
            - 'mcc_test': The Matthews correlation coefficient for the test set.
            - 'roc-auc_test' (optional): The ROC AUC score for the test set.
    """
    eval_dict = {}
    eval_dict['fitted'] = fitted_model
    eval_dict['preds_train'] = y_train_pred
    if y_test_prob is not None:
        eval_dict['probs_test'] = y_test_prob
    eval_dict['preds_test'] = y_test_pred
    eval_dict['accuracy_train'] = metrics.accuracy_score(y_train, y_train_pred)
    eval_dict['accuracy_test'] = metrics.accuracy_score(y_test, y_test_pred)
    eval_dict['precision_train'] = metrics.precision_score(y_train, y_train_pred, zero_division=0)
    eval_dict['precision_test'] = metrics.precision_score(y_test, y_test_pred, zero_division=0)
    eval_dict['recall_train'] = metrics.recall_score(y_train, y_train_pred, zero_division=0)
    eval_dict['recall_test'] = metrics.recall_score(y_test, y_test_pred, zero_division=0)
    eval_dict['f1_train'] = metrics.f1_score(y_train, y_train_pred, zero_division=0)
    eval_dict['f1_test'] = metrics.f1_score(y_test, y_test_pred, zero_division=0)
    eval_dict['mcc_train'] = metrics.matthews_corrcoef(y_train, y_train_pred)
    eval_dict['mcc_test'] = metrics.matthews_corrcoef(y_test, y_test_pred)
    if y_test_prob is not None:
        eval_dict['roc-auc_test'] = metrics.roc_auc_score(y_test, y_test_prob)
    return eval_dict

def evaluate_reg_metrics_mdl(
        fitted_model:BaseModelProtocol,
        y_train_pred:ArrayLike,
        y_test_pred:ArrayLike,
        y_train:ArrayLike,
        y_test:ArrayLike
    ) -> Dict:
    """Evaluates regression metrics for a fitted model.

    Args:
        fitted_model (BaseModelProtocol): The fitted model.
        y_train_pred (ArrayLike): Predicted values for the training set.
        y_test_pred (ArrayLike): Predicted values for the test set.
        y_train (ArrayLike): True values for the training set.
        y_test (ArrayLike): True values for the test set.

    Returns:
        dict: A dictionary containing the following evaluation metrics:
            - 'fitted': The fitted model.
            - 'preds_train': Predicted values for the training set.
            - 'preds_test': Predicted values for the test set.
            - 'trues_train': True values for the training set.
            - 'trues_test': True values for the test set.
            - 'rmse_train': Root mean squared error for the training set.
            - 'rmse_test': Root mean squared error for the test set.
            - 'r2_train': R-squared score for the training set.
            - 'r2_test': R-squared score for the test set.
    """
    eval_dict = {}
    eval_dict['fitted'] = fitted_model
    eval_dict['preds_train'] = y_train_pred
    eval_dict['preds_test'] = y_test_pred
    eval_dict['trues_train'] = y_train
    eval_dict['trues_test'] = y_test
    eval_dict['rmse_train'] = metrics.mean_squared_error(y_train, y_train_pred, squared=False)
    eval_dict['rmse_test'] = metrics.mean_squared_error(y_test, y_test_pred, squared=False)
    eval_dict['r2_train'] = metrics.r2_score(y_train, y_train_pred)
    eval_dict['r2_test'] = metrics.r2_score(y_test, y_test_pred)

    return eval_dict

def evaluate_multiclass_metrics_mdl(
        fitted_model:BaseModelProtocol,
        y_test_prob:ArrayLike,
        y_test_pred:ArrayLike,
        y_test:ArrayLike,
        ohe:Optional[BaseTransformerProtocol] = None
    ) -> Dict:
    """Evaluate multiclass classification metrics for a fitted model.

    Args:
        fitted_model (BaseModelProtocol): The fitted model to evaluate.
        y_test_prob (ArrayLike): The predicted probabilities for each class.
        y_test_pred (ArrayLike): The predicted class labels.
        y_test (ArrayLike): The true class labels.
        ohe (Optional[BaseTransformerProtocol], optional): The one-hot encoder transformer.
                                                           Defaults to None.

    Returns:
        Dict: A dictionary containing the evaluation metrics.
    """
    eval_dict = {}
    eval_dict['fitted'] = fitted_model
    if y_test_prob is not None:
        eval_dict['probs'] = y_test_prob
    eval_dict['preds'] = y_test_pred
    eval_dict['accuracy'] = metrics.accuracy_score(y_test, y_test_pred)
    eval_dict['precision'] = metrics.precision_score(y_test, y_test_pred,\
                                                     zero_division=0, average='micro')
    eval_dict['recall'] = metrics.recall_score(y_test, y_test_pred,\
                                               zero_division=0, average='micro')
    eval_dict['f1'] = metrics.f1_score(y_test, y_test_pred, zero_division=0, average='micro')
    eval_dict['mcc'] = metrics.matthews_corrcoef(y_test, y_test_pred)
    if y_test_prob is not None:
        if ohe is not None:
            eval_dict['roc-auc'] = metrics.roc_auc_score(ohe.transform(y_test), y_test_prob,\
                                                         multi_class="ovr")
        else:
            eval_dict['roc-auc'] = metrics.roc_auc_score(y_test, y_test_prob, multi_class="ovr")
    return eval_dict
