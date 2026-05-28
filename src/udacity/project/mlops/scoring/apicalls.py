#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
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
    """Test the Server API Endpoints
    
    Args:
        host:   host address of the server
        port:   port number of the server
        config: Parsed JSON Configuration
    """

    url = f"http://{host}:{port}"

    logger.info(f'Testing API endpoints at {url}...')

    # Call each API endpoint and store the responses
    response1 = requests.post(f'{url}/prediction', params={'filename': 'data/testdata/testdata.csv'})
    response2 = requests.get(f'{url}/scoring')
    response3 = requests.get(f'{url}/summarystats')
    response4 = requests.get(f'{url}/diagnostics')

    responses = {
        **response1.json(),
        **response2.json(),
        **response3.json(),
        **response4.json(),
    }

    filename = config.deploy / 'apireturns.txt'
    logger.info(f'Writing API responses to {filename}...')

    filename.write_text(json.dumps(responses, indent=2))


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
