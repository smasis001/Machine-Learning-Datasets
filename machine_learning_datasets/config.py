"""Initialize Dataset Configuration Functions"""
# pylint: disable=W0603,C0103
from typing import Optional, Tuple, Dict, Any
import os
import json
import tempfile
from pathlib import Path
import warnings
from .sources.kaggle import Kaggle
from .sources.url import URL

dssave_path = os.path.join(tempfile.gettempdir(), 'data')
dsconfig_path = os.path.join(Path(__file__).parent.absolute(), 'dsconfig.json')
dsconfig = None
dsconfig_dscount = 0

def init(
        new_dssave_path:Optional[str] = None,
        new_dsconfig_path:Optional[str] = None
    ) -> Tuple[str, int, Dict]:
    """Initializes the paths and configuration for the dataset.

    Args:
        new_dssave_path (Optional[str]): The new path to save the dataset. Defaults to None.
        new_dsconfig_path (Optional[str]): The new path to the dataset configuration file.
                                           Defaults to None.

    Returns:
        Tuple[str, int, Dict]: A tuple containing the dataset configuration path, the dataset
                               count, and the dataset configuration.
    """
    global dssave_path, dsconfig_path, dsconfig, dsconfig_dscount

    if new_dssave_path is not None:
        dssave_path = new_dssave_path

    if new_dsconfig_path is not None:
        dsconfig_path = new_dsconfig_path

    if os.path.exists(dsconfig_path):
        with open(dsconfig_path, encoding="utf8") as dsconfig_file:
            dsconfig = json.load(dsconfig_file)
        dsconfig_dscount = len(dsconfig['datasets'])
    else:
        dsconfig = False

    return dsconfig_path, dsconfig_dscount, dsconfig

def load(
        name:Optional[str] = None,
        **kwargs:Any
    ) -> Any:
    """Load a dataset from a source.

    Args:
        name (Optional[str]): The name of the dataset to load. If not provided, all keyword
                              arguments will be used as parameters.
        **kwargs (Any): Additional keyword arguments that will be used as parameters if
                        `name` is not provided.

    Returns:
        Any: The loaded dataset.

    Raises:
        Warning: If the dataset specified by `name` is not found in the `dsconfig` JSON.
        Warning: If the `source` parameter is not defined.

    Note:
        This function assumes that the `dsconfig` variable is defined and contains a JSON
        object with a `datasets` key, which is a list of dataset configurations.

    Example:
        dataset = load(name='my_dataset', source='Kaggle', file_path='data.csv')
    """
    kparams = locals()['kwargs']
    params = {}
    dataset = None
    if name is not None:
        for i in range(len(dsconfig['datasets'])):
            if dsconfig['datasets'][i]["name"] == name:
                params = dsconfig['datasets'][i].copy()
                del params['name']
                break
        if not bool(params):
            warnings.warn("Dataset not found in dsconfig JSON")
    params.update(kparams)
    if 'source' not in params:
        warnings.warn("Source not defined")
    else:
        if params['source'] == 'Kaggle':
            source = Kaggle()
        else:
            source = URL()

        params1 = source.fetch(**params)
        params2 = source.extract(**params1)
        params3 = source.parse(**params2)

        #TODO add code that differenciates the x, y, train, test
        dataset = source.prepare(**params3)

    return dataset
