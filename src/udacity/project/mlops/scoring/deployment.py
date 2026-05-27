#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import shutil
import sys
from pathlib import Path

from .utils import add_default_arguments, Configuration

__all__ = (
    'store_model_into_pickle',
    'deployment',
)

logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                    format='[%(asctime)s][%(levelname)-8s] %(message)s', datefmt='%d %b %Y %H:%M:%S')
logger = logging.getLogger(__name__)


def store_model_into_pickle(indir: Path, modeldir: Path, outdir: Path):
    """Deploy Ingestion Record and Model to Deployment Directory

    Args:
        indir:    Input directory where the Ingestion Record and Model are stored
        modeldir: Directory where the trained model is stored
        outdir:   Output directory where the Ingestion Record and Model will be deployed
    """

    logger.info(f'Deploying Ingestion Record and Model from {indir} and {modeldir} to {outdir}...')

    # create output directory (if not present)
    outdir.mkdir(parents=True, exist_ok=True)

    # high-level copy operation on file paths
    # notice that Path.copy is only available from Python 3.14
    for directory, filename in ((indir, 'ingestedfiles.txt'), (modeldir, 'latestscore.txt'), (modeldir, 'trainedmodel.pkl')):
        shutil.copy2(str(directory / filename), str(outdir / filename))


def deployment(config: Configuration):
    """Deploy the trained Prediction Model

    Args:
        config: Parsed JSON Configuration
    """
    store_model_into_pickle(config.train, config.model, config.deploy)


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
