"""Test Evaluate Functions"""
# pylint: disable=C0103
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from machine_learning_datasets import evaluate

def test_evaluate_class_mdl():
    """Test functionality that evaluates the performance of a binary
       classification model.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If the evaluation metrics are not stored in a dictionary.
        AssertionError: If the accuracy score is not computed correctly.
        AssertionError: If the precision score is not computed correctly.
        AssertionError: If the recall score is not computed correctly.
        AssertionError: If the f1 score is not computed correctly.
        AssertionError: If the mcc score is not computed correctly.
        AssertionError: If the roc-auc score is not computed correctly.
    """
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,\
                                                        random_state=42)
    fitted_model = LogisticRegression(solver="liblinear", random_state=42).\
                                                            fit(X_train, y_train)
    evaluation_metrics = evaluate.evaluate_class_mdl(fitted_model, X_train, X_test,\
                                                y_train, y_test, plot_roc=False,\
                                                show_summary=False, ret_eval_dict=True)
    # Assert that the metrics are stored in a dictionary
    assert isinstance(evaluation_metrics, dict)
    # Assert that the accuracy score is computed correctly
    assert evaluation_metrics['accuracy_test'] == 0.9574468085106383
    # Assert that the precision score is computed correctly
    assert evaluation_metrics['precision_test'] == 0.9669421487603306
    # Assert that the recall score is computed correctly
    assert evaluation_metrics['recall_test'] == 0.9669421487603306
    # Assert that the f1 score is computed correctly
    assert evaluation_metrics['f1_test'] == 0.9669421487603306
    # Assert that the mcc score is computed correctly
    assert evaluation_metrics['mcc_test'] == 0.9072406562230172
    # Assert that the roc-auc score is computed correctly
    assert evaluation_metrics['roc-auc_test'] == 0.9967928950289873

def test_evaluate_reg_mdl():
    """Test functionality that evaluates the performance of a
       regression model.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If the evaluation metrics are not stored in a dictionary
        AssertionError: If the the RMSE score is not computed correctly.
        AssertionError: If the the R2 score is not computed correctly.
    """
    X, y = load_diabetes(return_X_y=True)
    y = y.reshape(-1,1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,\
                                                        random_state=42)
    fitted_model = LinearRegression().fit(X_train, y_train)
    evaluation_metrics = evaluate.evaluate_reg_mdl(fitted_model, X_train, X_test,\
                                                   y_train, y_test, plot_regplot=False,\
                                                   show_summary=False, ret_eval_dict=True)
    # Assert that the metrics are stored in a dictionary
    assert isinstance(evaluation_metrics, dict)
    # Assert that the accuracy score is computed correctly
    assert evaluation_metrics['rmse_test'] == 53.08303210274999
    # Assert that the precision score is computed correctly
    assert evaluation_metrics['r2_test'] == 0.5103942572821248

def test_evaluate_multiclass_mdl():
    """Test functionality that evaluates the performance of a multiclass
       classification model.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If the evaluation metrics are not stored in a dictionary.
        AssertionError: If the accuracy score is not computed correctly.
        AssertionError: If the precision score is not computed correctly.
        AssertionError: If the recall score is not computed correctly.
        AssertionError: If the f1 score is not computed correctly.
        AssertionError: If the mcc score is not computed correctly.
        AssertionError: If the roc-auc score is not computed correctly.
    """
    # Example Initialization
    cls = [0,1,2]
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,\
                                                        random_state=42)
    fitted_model = LogisticRegression(solver="liblinear", random_state=42).\
                                                            fit(X_train, y_train)
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    y_ohe = ohe.fit_transform(y_test.reshape(-1,1))

    # Invoke the function with simple model
    evaluation_metrics = evaluate.evaluate_multiclass_mdl(fitted_model, X_test, y_ohe, class_l=cls,\
                                            ohe=ohe, plot_roc_class=False, plot_conf_matrix=False,\
                                            plot_class_report=False, ret_eval_dict=True)
    # Assert that the metrics are stored in a dictionary
    assert isinstance(evaluation_metrics, dict)
    # Assert that the accuracy score is computed correctly
    assert evaluation_metrics['accuracy'] == 1.0
    # Assert that the precision score is computed correctly
    assert evaluation_metrics['precision'] == 1.0
    # Assert that the recall score is computed correctly
    assert evaluation_metrics['recall'] == 1.0
    # Assert that the f1 score is computed correctly
    assert evaluation_metrics['f1'] == 1.0
    # Assert that the mcc score is computed correctly
    assert evaluation_metrics['mcc'] == 1.0
    # Assert that the roc-auc score is computed correctly
    assert evaluation_metrics['roc-auc'] == 0.9993650793650793
