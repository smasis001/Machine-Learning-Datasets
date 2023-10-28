# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['machine_learning_datasets', 'machine_learning_datasets.sources']

package_data = \
{'': ['*']}

install_requires = \
['matplotlib>=3.7.1,<4.0.0',
 'numpy>=1.23.5,<2.0.0',
 'opencv-python>=4.8.0.76,<5.0.0',
 'pandas>=1.5.3,<2.0.0',
 'scikit-learn>=1.2.2,<2.0.0',
 'scipy>=1.11.3,<2.0.0',
 'seaborn>=0.12.2,<0.13.0',
 'torchvision>=0.16.0,<0.17.0']

setup_kwargs = {
    'name': 'machine-learning-datasets',
    'version': '0.1.23',
    'description': 'A simple library for loading machine learning datasets and performing some common machine learning interpretation functions. Built for the book "Interpretable Machine Learning with Python".',
    'long_description': None,
    'author': 'Serg Masís',
    'author_email': 'smasis@hawk.iit.edu',
    'maintainer': None,
    'maintainer_email': None,
    'url': None,
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.9,<4.0',
}


setup(**setup_kwargs)
