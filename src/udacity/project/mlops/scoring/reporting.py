#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import pickle
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

from .utils import add_default_arguments, Configuration

if TYPE_CHECKING:
    from sklearn.linear_model import LogisticRegression

__all__ = (
    'create_confusion_matrix',
    'reporting',
)

logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                    format='[%(asctime)s][%(levelname)-8s] %(message)s', datefmt='%d %b %Y %H:%M:%S')
logger = logging.getLogger(__name__)


def create_confusion_matrix(model: 'LogisticRegression', df: pd.DataFrame, outdir: Path):
    """Create a confusion matrix for the given model and data.

    Args:
        model: pre-trained model
        df: test data
        outdir: the output directory of the confusion matrix
    """
    # extract features and label
    X = df.select_dtypes(include='number').drop('exited', axis=1)
    y = df['exited']
    logger.info(f'Creating confusion matrix for model on {len(df)} records...')

    # run inference
    y_pred = model.predict(X)

    plt.figure(figsize=(8, 6))

    # create confusion matrix display
    ConfusionMatrixDisplay.from_predictions(y, y_pred,
                                            ax=plt.gca(),
                                            cmap='Blues',
                                            colorbar=False)

    # save confusion matrix to file
    filename = outdir / 'confusionmatrix.png'
    logger.info(f'Saving confusion matrix to {filename}...')
    plt.tight_layout()
    plt.savefig(filename)


def reporting(config: Configuration):
    """Produce Report about Pre-Trained Model on Test Data

    Args:
        config: Parsed JSON Configuration
    """
    logger.info(f'Reading test data from {config.test}...')
    df = pd.read_csv(config.test / 'testdata.csv', low_memory=False)
    logger.info(f'Loading pre-trained model from {config.deploy}...')
    with open(config.deploy / 'trainedmodel.pkl', 'rb') as f:
        model = pickle.load(f)

    create_confusion_matrix(model, df, config.deploy)


def main():
    # sub-command-specific parser
    parser = argparse.ArgumentParser(description='Produce a report about the trained Prediction Model')
    add_default_arguments(parser)

    # parse command-line arguments
    args = parser.parse_args()

    # modify log level
    logger.setLevel(args.loglevel)

    # parse configuration
    config = Configuration.from_json(args.config)

    # run sub-command
    reporting(config)
    return 0


if __name__ == '__main__':
    sys.exit(main())
