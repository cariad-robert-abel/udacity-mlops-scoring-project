#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import pickle
import sys
from pathlib import Path

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


def train_model(df: pd.DataFrame, outdir: Path, *,
                random_state: int = 0x1234567) -> LogisticRegression:
    """Train a Logistic Regression Model

    Train on provided data that must have at least a 'exited' column as the
    target variable. The trained model is written to the output directory as
    'trainedmodel.pkl'.

    Args:
        df:           The training dataframe.
        outdir:       Directory to write the trained model to.
        random_state: Random state to use for reproducibility.
    
    Returns:
        The trained Logistic Regression model.
    """
    # configure logistic regression
    # remove deprecated params
    lr = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True, 
                            intercept_scaling=1, l1_ratio=0, max_iter=100,
                            random_state=random_state, solver='liblinear', tol=1e-4,
                            verbose=0, warm_start=False)

    # extract features and label
    X = df.select_dtypes(include='number').drop('exited', axis=1)
    y = df['exited']
    logger.info(f'Training logistic regression model on {len(df)} records with {X.shape[1]} features...')

    # fit the model
    model = lr.fit(X, y)

    # create output directory (if not present)
    outdir.mkdir(parents=True, exist_ok=True)

    # write the trained model to outdir
    filename = outdir / 'trainedmodel.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f'Wrote trained model to {filename}...')

    # return the model for immeditate re-use
    return model


def training(config: Configuration):
    """Perform Training of the Machine Learning Model

    Args:
        config: Parsed JSON Configuration
    """
    logger.info(f'Reading training data from {config.train}...')
    df = pd.read_csv(config.train / 'finaldata.csv', low_memory=False)
    train_model(df, config.model, random_state=config.random)


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
