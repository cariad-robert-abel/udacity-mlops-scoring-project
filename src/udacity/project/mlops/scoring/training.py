#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import pickle
import sys

import pandas as pd
from sklearn.linear_model import LogisticRegression

from .utils import add_default_arguments, Configuration

__all__ = (
    'train_model',
    'training',
)

logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                    format='[%(asctime)s][%(levelname)-8s] %(message)s', datefmt='%d %b %Y %H:%M:%S')
logger = logging.getLogger(__name__)


################# Function for training the model
def train_model():

    # use this logistic regression for training
    LogisticRegression(
        C=1.0,
        class_weight=None,
        dual=False,
        fit_intercept=True,
        intercept_scaling=1,
        l1_ratio=None,
        max_iter=100,
        n_jobs=None,
        penalty='l2',
        random_state=0,
        solver='liblinear',
        tol=0.0001,
        verbose=0,
        warm_start=False,
    )

    # Fit the logistic regression to your data

    # Write the trained model to your workspace in a file called trainedmodel.pkl
    pass


def training(config: Configuration):
    raise NotImplementedError('training() is not implemented yet')
    train_model()


def main():
    # sub-command-specific parser
    parser = argparse.ArgumentParser(description='Train the Prediction Model')
    add_default_arguments(parser)

    # parse command-line arguments
    args = parser.parse_args()

    # modify log level
    logger.setLevel(args.loglevel)

    # parse configuration
    config = Configuration.from_json(args.config)

    # run sub-command
    training(config)
    return 0


if __name__ == '__main__':
    sys.exit(main())
