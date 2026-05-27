#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import re
import sys
from pathlib import Path

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
    raise NotImplementedError('process() is not implemented yet')


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
