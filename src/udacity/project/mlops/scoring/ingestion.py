#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from .utils import add_default_arguments, Configuration

__all__ = (
    'merge_multiple_dataframe',
    'ingestion',
)

logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                    format='[%(asctime)s][%(levelname)-8s] %(message)s', datefmt='%d %b %Y %H:%M:%S')
logger = logging.getLogger(__name__)


def merge_multiple_dataframe(indir: Path, outdir: Path):
    """Merge multiple input files into a single output file

    Record, merge, and de-duplicate input files.

    Args:
        indir: Directory containing input files
        outdir: Directory to write output files to.
    """
    ingested_files = []
    result = pd.DataFrame()

    logger.info(f'Ingesting data from {indir} to {outdir}...')

    for file in indir.rglob('*.csv'):
        # append to list of ingested files
        ingested_files.append(str(file.relative_to(indir)))
        logger.debug(f'Ingesting file {file}')

        # merge datasets together
        df = pd.read_csv(file, low_memory=False)

        # concatenate the new dataframe and existing dataframe
        result = pd.concat([result, df], ignore_index=True)

    logger.info(f'Merged {len(ingested_files)} files with a total of {len(result)} records (before de-duplication)...')

    # run de-duplication
    result.drop_duplicates(inplace=True)

    logger.info(f'Merged {len(ingested_files)} files with a total of {len(result)} records (after de-duplication)...')

    # create output directory (if not present)
    outdir.mkdir(parents=True, exist_ok=True)

    # write the merged result to outdir
    resultfile = outdir / 'finaldata.csv'
    result.to_csv(resultfile, index=False)
    logger.info(f'Wrote merged data to {resultfile}...')

    # write the list of ingested files
    recordfile = outdir / 'ingestedfiles.txt'
    recordfile.write_text('\n'.join(ingested_files))
    logger.info(f'Wrote record of ingested files to {recordfile}...')


def ingestion(config: Configuration):
    """Perform Data Ingestion
    
    Merge files in input directory to training directory.

    Args:
        config: Parsed JSON Configuration
    """
    merge_multiple_dataframe(config.input, config.train)


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
