#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import sys

import pandas as pd

from .utils import add_default_arguments, Configuration

__all__ = (
    'merge_multiple_dataframe',
    'ingestion',
)

logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                    format='[%(asctime)s][%(levelname)-8s] %(message)s', datefmt='%d %b %Y %H:%M:%S')
logger = logging.getLogger(__name__)


############# Function for data ingestion
def merge_multiple_dataframe():
    # Check for datasets, compile them together, and write to an output file
    pass


def ingestion(config: Configuration):
    raise NotImplementedError('ingestion() is not implemented yet')
    merge_multiple_dataframe()


def main():
    # sub-command-specific parser
    parser = argparse.ArgumentParser(description='Consolidate and pre-process training data')
    add_default_arguments(parser)

    # parse command-line arguments
    args = parser.parse_args()

    # modify log level
    logger.setLevel(args.loglevel)

    # parse configuration
    config = Configuration.from_json(args.config)

    # run sub-command
    ingestion(config)
    return 0


if __name__ == '__main__':
    sys.exit(main())
