#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import re
import sys
import time
from multiprocessing import Process
from pathlib import Path

import pandas as pd

from .ingestion import *
from .training import *
from .scoring import *
from .deployment import *
from .diagnostics import *
from .reporting import *
from .app import *
from .apicalls import *
from .utils import add_subcommand, add_wsgi_subcommand, Configuration


logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                    format='[%(asctime)s][%(levelname)-8s] %(message)s', datefmt='%d %b %Y %H:%M:%S')
logger = logging.getLogger(__package__)


def process(config: Configuration):
    # run the ingestion step and only continue if there is new data
    train_df = ingestion(config, check=True)

    if (train_df is None):
        return

    # train a new model
    model = train_model(train_df, config.model, random_state=config.random)

    # For some reason, we now score against the new training data and never
    # change the test dataset. This seems wrong, but it's what the project
    # rubric requires...
    # Notice how we have a static test dataset to begin with and never actually
    # perform any meaningful test/train split anyway...
    f1_score = score_model(model, train_df, config.model)

    # read the previous deployed score
    filename = config.deploy / 'latestscore.txt'
    if (filename.exists() and (prev_f1_score := float(filename.read_text().strip())) >= f1_score):
        logger.info(f'New model F1 score {f1_score:.6f} did not improve over previous score {prev_f1_score:.6f}')
        return

    # re-deploy updated model, score, and ingestion record
    store_model_into_pickle(config.train, config.model, config.deploy)

    # load additional test data
    logger.info(f'Reading test data from {config.test}...')
    test_df = pd.read_csv(config.test / 'testdata.csv', low_memory=False)

    # create confusion matrix
    create_confusion_matrix(model, test_df, config.deploy)

    # create the server as a separate process, so it doesn't block the rest of
    # the testing code
    proc = Process(target=server, args=('localhost', 8080, config), name='model-server', daemon=True)
    proc.start()

    try:
        # wait 100ms until the server lives
        time.sleep(0.1)
        # run the client application
        client('localhost', 8080, config)
    finally:
        # terminate the server process
        proc.terminate()
        proc.join(timeout=5)


def main():
    # global parser for all sub-commands
    parser = argparse.ArgumentParser(description='Machine Learning Model Scoring and Monitoring: Dynamic Risk Assessment System')

    # determine whether we're called via a well-known project.scripts shim
    prog = None
    name = Path(sys.argv[0]).name
    if (len(sys.argv) > 0 and (match := re.match(r'udacity-project-mlops-scoring-(?P<command>.+)(\.exe)?', name)) is not None):
        # modify program name in usage and prepend command
        prog = name
        sys.argv = [sys.argv[0], match.group('command'), *sys.argv[1:]]

    subparser = parser.add_subparsers(dest='command', required=True, title='Commands')
    add_subcommand(subparser, 'ingest', 'Consolidate and pre-process training data', prog=prog)
    add_subcommand(subparser, 'train', 'Train the Prediction Model', prog=prog)
    add_subcommand(subparser, 'score', 'Score the trained Prediction Model', prog=prog)
    add_subcommand(subparser, 'deploy', 'Deploy the trained Prediction Model', prog=prog)
    add_subcommand(subparser, 'diagnose', 'Produce diagnostics for the trained Prediction Model', prog=prog)
    add_subcommand(subparser, 'report', 'Produce a report about the trained Prediction Model', prog=prog)
    add_subcommand(subparser, 'process', 'Run the full pipeline processing cycle', prog=prog)
    add_wsgi_subcommand(subparser, 'server', 'Serve the trained Prediction Model', prog=prog)
    add_wsgi_subcommand(subparser, 'client', 'Test the served Prediction Model APIs', prog=prog)

    # parse command-line arguments
    args = parser.parse_args()

    # modify log level
    logger.setLevel(args.loglevel)

    # parse configuration
    config = Configuration.from_json(args.config)

    match (args.command):
        case 'ingest':
            ingestion(config)
        case 'train':
            training(config)
        case 'score':
            scoring(config)
        case 'deploy':
            deployment(config)
        case 'diagnose':
            diagnostics(config)
        case 'report':
            reporting(config)
        case 'process':
            process(config)
        case 'server':
            server(args.host, args.port, config)
        case 'client':
            client(args.host, args.port, config)
        case _:
            raise RuntimeError(f'Unknown command: {args.command}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
