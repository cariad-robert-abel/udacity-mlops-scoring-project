#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import pickle
import sys

import pandas as pd
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression

from .utils import add_default_arguments, Configuration

__all__ = (
    'score_model',
    'scoring',
)

logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                    format='[%(asctime)s][%(levelname)-8s] %(message)s', datefmt='%d %b %Y %H:%M:%S')
logger = logging.getLogger(__name__)


################# Function for model scoring
def score_model():
    # This function should take a trained model, load test data,  and
    # calculate an F1 score for the model relative to the test data.
    # It should write the result to the latestscore.txt file
    pass


def scoring(config: Configuration):
    raise NotImplementedError('scoring() is not implemented yet')
    score_model()


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
