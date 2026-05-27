#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import pickle
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from sklearn.metrics import f1_score

from .utils import add_default_arguments, Configuration

if TYPE_CHECKING:
    from sklearn.linear_model import LogisticRegression

__all__ = (
    'score_model',
    'scoring',
)

logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                    format='[%(asctime)s][%(levelname)-8s] %(message)s', datefmt='%d %b %Y %H:%M:%S')
logger = logging.getLogger(__name__)


def score_model(model: 'LogisticRegression', df: pd.DataFrame, outdir: Path) -> float:
    """Calculate the F1 score of a trained model on a given dataset.

    Args:
        model:  The trained Logistic Regression model.
        df:     The test dataframe.
        outdir: Directory to write the model score to.

    Returns:
        The F1 score of the model on the test data.
    """
    # extract features and label
    X = df.select_dtypes(include='number').drop('exited', axis=1)
    y = df['exited']
    logger.info(f'Scoring model on {len(df)} records with {X.shape[1]} features...')

    # run inference
    y_pred = model.predict(X)

    # calculate F1 score
    f1 = f1_score(y, y_pred)
    logger.info(f'Model F1 score: {f1:.6f}...')

    filename = outdir / 'latestscore.txt'
    logger.info(f'Writing current model score to {filename}...')
    filename.write_text(f'{f1}')

    # return the score for immediate re-use
    return f1


def scoring(config: Configuration):
    """Score a pre-trained Prediction Model

    Calculate the F1 score of the pre-trained model against the test dataset.
    The F1 score is written to the output directory as 'latestscore.txt'.

    Args:
        config: Parsed JSON Configuration
    """
    logger.info(f'Reading test data from {config.test}...')
    df = pd.read_csv(config.test / 'testdata.csv', low_memory=False)
    logger.info(f'Loading pre-trained model from {config.model}...')
    with open(config.model / 'trainedmodel.pkl', 'rb') as f:
        model = pickle.load(f)

    # actuall score model
    score_model(model, df, config.model)


def main():
    # sub-command-specific parser
    parser = argparse.ArgumentParser(description='Score the trained Prediction Model')
    add_default_arguments(parser)

    # parse command-line arguments
    args = parser.parse_args()

    # modify log level
    logger.setLevel(args.loglevel)

    # parse configuration
    config = Configuration.from_json(args.config)

    # run sub-command
    scoring(config)
    return 0


if __name__ == '__main__':
    sys.exit(main())
