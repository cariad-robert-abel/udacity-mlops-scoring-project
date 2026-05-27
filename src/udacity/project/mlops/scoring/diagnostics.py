#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import timeit
import sys

import pandas as pd

from .utils import add_default_arguments, Configuration

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


################## Function to get model predictions
def model_predictions():
    # Read the deployed model and a test dataset, calculate predictions
    return  # Return value should be a list containing all predictions


################## Function to get summary statistics
def dataframe_summary():
    # Calculate summary statistics here
    return  # Return value should be a list containing all summary statistics


##################Function to get timings
def execution_time():
    # Calculate timing of training.py and ingestion.py
    return  # Return a list of 2 timing values in seconds


################## Function to check dependencies
def outdated_packages_list():
    # Get a list of
    pass


def diagnostics(config: Configuration):
    raise NotImplementedError('diagnostics() is not implemented yet')
    model_predictions()
    dataframe_summary()
    execution_time()
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
