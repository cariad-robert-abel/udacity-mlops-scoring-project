#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import pickle
import timeit
import json
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from .ingestion import ingestion
from .training import training
from .utils import add_default_arguments, Configuration

if TYPE_CHECKING:
    from sklearn.linear_model import LogisticRegression

__all__ = (
    'model_predictions',
    'dataframe_summary',
    'execution_time',
    'outdated_packages_list',
    'diagnostics',
)

logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                    format='[%(asctime)s][%(levelname)-8s] %(message)s', datefmt='%d %b %Y %H:%M:%S')
logger = logging.getLogger(__name__)


def model_predictions(model: 'LogisticRegression', df: pd.DataFrame) -> list[int]:
    """Run Model Inference on Data

    Args:
        model: The trained Logistic Regression model.
        df:    The dataframe.

    Returns:
        List of model predictions, one per record in the dataframe.
    """
    # extract features only
    X = df.select_dtypes(include='number').drop('exited', axis=1)
    logger.info(f'Running inference on {len(df)} records with {X.shape[1]} features...')

    # run inference
    y_pred = model.predict(X)

    # return model prediction values as a list
    return y_pred.tolist()


def dataframe_summary(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Calculate Summary Statistics

    Calculate summary statistics for the input data.
    Measures to calculate include mean, median, and standard devication for
    every numerical column.

    Args:
        df: the dataframe.
    
    Returns:
        Map of summary statistics for each numerical column keyed by column name
        and then by statistic name.
    """
    logger.info(f'Calculating summary statistics for {len(df)} records...')

    # keys to extract
    keys = ('mean', 'median', 'std')
    # description for numerical columns only
    description = df.describe(include='number')

    # rename '50%' quantile index to 'median'
    description.rename(index={'50%': 'median'}, inplace=True)

    # transform into dictionary
    result = {
        col: {key: value for key, value in data.items() if key in keys}
        for col, data in description.items()
    }

    # print glorious debug output
    if logger.isEnabledFor(logging.DEBUG):
        for col, stats in result.items():
            logger.debug(f'\tColumn {col}: {", ".join(f"{key}: {value:.3f}" for key, value in stats.items())}')

    return result

def dataframe_missing(df: pd.DataFrame) -> dict[str, float]:
    """Calculate Missing Data Percentages

    Calculate the percentage of missing data for each column in the input data.

    Args:
        df: the dataframe.
    
    Returns:
        Map of missing data percentages for each column keyed by column name.
    """
    logger.info(f'Calculating missing statistics for {len(df)} records...')

    # NaN counts as percentage of total per column
    isna_pct = df.isna().mean(axis=0) * 100.0

    # transform into dictionary
    result = {
        col: data
        for col, data in isna_pct.items()
    }

    # print glorious debug output
    if logger.isEnabledFor(logging.DEBUG):
        for col, missing in result.items():
            logger.debug(f'\tColumn {col}: missing: {missing:.2f}%')

    return result


def execution_time(config: Configuration) -> tuple[float, float]:
    """Measure Execution Time for Important Tasks

    Args:
        config: Parsed JSON Configuration

    Returns:
        Execution time in seconds for data ingestion and model training.
    """
    logger.info(f'Measuring execution time for data ingestion and model training...')

    # measure execution time for data ingestion
    ingestion_time = timeit.timeit('ingestion(config)', globals={**globals(), 'config': config},
                                   number=1)

    # measure execution time for model training
    training_time = timeit.timeit('training(config)', globals={**globals(), 'config': config},
                                  number=1)

    logger.info(f'Execution time for data ingestion: {ingestion_time:.6f} seconds')
    logger.info(f'Execution time for model training: {training_time:.6f} seconds')

    return ingestion_time, training_time


################## Function to check dependencies
def outdated_packages_list() -> dict[str, dict[str, str]]:
    """Check for Outdated Packages

    Returns:
        Map of package name to version information ('version', 'latest') for all
        installed packages.
    """
    
    logger.info(f'Checking installed Python packages using pip...')
    proc = subprocess.run([sys.executable, '-m', 'pip', 'list', '--format=json'],
                          check=True, capture_output=True)
    
    # convert output to map of package name to installed version
    # format is list of {name, version} dicts
    installed_packages = json.loads(proc.stdout)
    installed_packages = {pkg['name']: pkg['version'] for pkg in installed_packages}
    logger.info(f'Found {len(installed_packages)} installed packages...')
    
    logger.info(f'Checking outdated Python packages using pip...')
    proc = subprocess.run([sys.executable, '-m', 'pip', 'list', '--outdated', '--format=json'],
                          check=True, capture_output=True)
    
    # convert output to map of package name to installed version
    # format is list of {name, version, latest_version} dicts
    outdated_packages = json.loads(proc.stdout)
    outdated_packages = {pkg['name']: pkg['latest_version'] for pkg in outdated_packages}
    logger.info(f'Found {len(installed_packages)} outdated packages...')

    result = {package: {'installed': version, 'latest': outdated_packages.get(package, version)}
              for package, version in installed_packages.items()
              }
    
    # print glorious debug output
    if logger.isEnabledFor(logging.DEBUG):
        for package, versions in result.items():
            icon = '✅' if versions['installed'] == versions['latest'] else '❌'
            logger.debug(f'\t{icon} Package {package}: installed: {versions["installed"]} latest: {versions["latest"]}')

    return result

def diagnostics(config: Configuration):
    """Run diagnostics for Model Training and Execution
    
    Args:
        config: Parsed JSON Configuration
    """
    logger.info(f'Reading train data from {config.train}...')
    df = pd.read_csv(config.train / 'finaldata.csv', low_memory=False)
    logger.info(f'Loading pre-trained model from {config.deploy}...')
    with open(config.deploy / 'trainedmodel.pkl', 'rb') as f:
        model = pickle.load(f)

    # run predictions
    model_predictions(model, df)
    # calculate statistics
    dataframe_summary(df)
    # calculate missing
    dataframe_missing(df)

    # measure execution time
    execution_time(config)
    # create out-dated package list
    outdated_packages_list()


def main():
    # sub-command-specific parser
    parser = argparse.ArgumentParser(description='Produce diagnostics for the trained Prediction Model')
    add_default_arguments(parser)

    # parse command-line arguments
    args = parser.parse_args()

    # modify log level
    logger.setLevel(args.loglevel)

    # parse configuration
    config = Configuration.from_json(args.config)

    # run sub-command
    diagnostics(config)
    return 0


if __name__ == '__main__':
    sys.exit(main())
