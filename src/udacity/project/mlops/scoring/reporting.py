#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import pickle
import sys

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from .utils import add_default_arguments, Configuration

__all__ = (
    'reporting',
)

logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                    format='[%(asctime)s][%(levelname)-8s] %(message)s', datefmt='%d %b %Y %H:%M:%S')
logger = logging.getLogger(__name__)


############## Function for reporting
def reporting(config: Configuration):
    raise NotImplementedError('reporting() is not implemented yet')


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
