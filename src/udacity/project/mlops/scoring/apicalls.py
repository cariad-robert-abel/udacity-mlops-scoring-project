#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import sys

import requests

from .utils import add_default_arguments, add_wsgi_subcommand, Configuration

__all__ = (
    'client',
)

logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                    format='[%(asctime)s][%(levelname)-8s] %(message)s', datefmt='%d %b %Y %H:%M:%S')
logger = logging.getLogger(__name__)

def client(host: str, port: int, config: Configuration):
    raise NotImplementedError('client() is not implemented yet')
    # Specify a URL that resolves to your workspace
    URL = "http://127.0.0.1/"



    # Call each API endpoint and store the responses
    response1 = None # Put an API call here
    response2 = None # Put an API call here
    response3 = None # Put an API call here
    response4 = None # Put an API call here

    # Combine all API responses
    responses = None #combine reponses here

    # Write the responses to your workspace


def main():
    # sub-command-specific parser
    parser = argparse.ArgumentParser(description='Test the served Prediction Model APIs')
    add_default_arguments(parser)
    add_wsgi_subcommand(parser)

    # parse command-line arguments
    args = parser.parse_args()

    # modify log level
    logger.setLevel(args.loglevel)

    # parse configuration
    config = Configuration.from_json(args.config)

    # run sub-command
    client(args.host, args.port, config)
    return 0


if __name__ == '__main__':
    sys.exit(main())
