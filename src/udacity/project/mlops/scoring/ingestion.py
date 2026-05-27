#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from .utils import add_default_arguments, Configuration

__all__ = (
    'merge_multiple_dataframe',
    'ingestion',
)

logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                    format='[%(asctime)s][%(levelname)-8s] %(message)s', datefmt='%d %b %Y %H:%M:%S')
logger = logging.getLogger(__name__)


def merge_multiple_dataframe(indir: Path, outdir: Path) -> pd.DataFrame:
    """Merge multiple input files into a single output file

    Record, merge, and de-duplicate input files.

    Args:
        indir: Directory containing input files
        outdir: Directory to write output files to.
    
    Returns:
        The merged dataframe.
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

    # return the merged dataframe for immediate re-use
    return result


def _check_for_new_files(indir: Path, outdir: Path) -> bool:
    """Check for new files in the input directory

    Args:
        indir: Directory containing input files
        outdir: Directory containing result files.

    Returns:
        True if new files are present, else False.
    """
    logger.info(f'Checking for updated files to ingest in {indir} against previous output in {outdir}...')

    resultfile = outdir / 'finaldata.csv'
    recordfile = outdir / 'ingestedfiles.txt'

    if (not resultfile.exists() or not recordfile.exists()):
        logger.info(f'No existing result or record files found, treating all files as new...')
        return True

    # read previous records of ingested files
    ingested_files = set(file.strip() for file in recordfile.read_text().splitlines())
    present_files = set(str(file.relative_to(indir)) for file in indir.rglob('*.csv'))

    new_files = present_files - ingested_files

    if (new_files):
        logger.info(f'Found {len(new_files)} new file(s) to ingest...')
        return True

    logger.info('No new files found to ingest...')
    return False


def ingestion(config: Configuration, check: bool = False) -> Optional[pd.DataFrame]:
    """Perform Data Ingestion
    
    Merge files in input directory to training directory.

    Args:
        config: Parsed JSON Configuration
        check:  Whether to check for new files before or not.

    Returns:
        New merged dataframe or None, if no new data was present.
    """
    if (not check or _check_for_new_files(config.input, config.train)):
        result = merge_multiple_dataframe(config.input, config.train)
        return result

    # default to no new data written
    return None


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
