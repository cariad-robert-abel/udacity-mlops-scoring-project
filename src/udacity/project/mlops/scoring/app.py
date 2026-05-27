#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import pickle
import sys

import pandas as pd

from flask import Flask, request

from .diagnostics import model_predictions, dataframe_summary, execution_time
from .scoring import score_model
from .utils import add_default_arguments, add_wsgi_subcommand, Configuration

__all__ = (
    'server',
)

logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                    format='[%(asctime)s][%(levelname)-8s] %(message)s', datefmt='%d %b %Y %H:%M:%S')
logger = logging.getLogger(__name__)


###################### Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

####################### Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():
    # Call the prediction function you created in Step 3
    return # Add return value for prediction outputs

####################### Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def scoring():
    # Check the score of the deployed model
    return # Add return value (a single F1 score number)

####################### Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def summarystats():
    # Check means, medians, and modes for each column
    return # Return a list of all calculated summary statistics

####################### Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostics():
    # Check timing and percent NA values
    return # Add return value for all diagnostics


def server(host: str, port: int, config: Configuration):
    raise NotImplementedError('server() is not implemented yet')
    app.run(host=host, port=port, debug=True, threaded=True)


def main():
    # sub-command-specific parser
    parser = argparse.ArgumentParser(description='Serve the trained Prediction Model')
    add_default_arguments(parser)
    add_wsgi_subcommand(parser)

    # parse command-line arguments
    args = parser.parse_args()

    # modify log level
    logger.setLevel(args.loglevel)

    # parse configuration
    config = Configuration.from_json(args.config)

    # run sub-command
    server(args.host, args.port, config)
    return 0


if __name__ == '__main__':
    sys.exit(main())
