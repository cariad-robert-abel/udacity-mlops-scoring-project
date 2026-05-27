#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import os
import sys

from .utils import add_default_arguments, Configuration

__all__ = (
    'store_model_into_pickle',
    'deployment',
)

logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                    format='[%(asctime)s][%(levelname)-8s] %(message)s', datefmt='%d %b %Y %H:%M:%S')
logger = logging.getLogger(__name__)


#################### Function for deployment
def store_model_into_pickle():
    # Copy the latest pickle file, the latestscore.txt value,
    # and the ingestfiles.txt file into the deployment directory
    pass


def deployment(config: Configuration):
    raise NotImplementedError('deployment() is not implemented yet')
    store_model_into_pickle()


def main():
    # sub-command-specific parser
    parser = argparse.ArgumentParser(description='Deploy the trained Prediction Model')
    add_default_arguments(parser)

    # parse command-line arguments
    args = parser.parse_args()

    # modify log level
    logger.setLevel(args.loglevel)

    # parse configuration
    config = Configuration.from_json(args.config)

    # run sub-command
    deployment(config)
    return 0


if __name__ == '__main__':
    sys.exit(main())
