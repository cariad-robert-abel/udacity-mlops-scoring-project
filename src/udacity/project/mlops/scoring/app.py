#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import pickle
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from flask import Flask, request

from .diagnostics import model_predictions, dataframe_summary, dataframe_missing, execution_time, outdated_packages_list
from .scoring import score_model
from .utils import add_default_arguments, add_wsgi_subcommand, Configuration

if TYPE_CHECKING:
    from sklearn.linear_model import LogisticRegression

__all__ = (
    'server',
)

logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                    format='[%(asctime)s][%(levelname)-8s] %(message)s', datefmt='%d %b %Y %H:%M:%S')
logger = logging.getLogger(__name__)


class ModelReportingApp(Flask):

    SECRET_KEY = '1652d576-484a-49fd-913a-6879acfa6ba4'

    def __init__(self, model: 'LogisticRegression', config: Configuration,
                 test_df: pd.DataFrame,
                 train_df: pd.DataFrame):
        """Initialize the Model Reporting App

        Args:
            model:    pre-trained deployed model
            config:   parsed JSON configuration
            test_df:  test dataframe
            train_df: train dataframe
        """
        super().__init__('Model Reporting App')
        self._model = model
        self._config = config
        self._test_df = test_df
        self._train_df = train_df
    
        # add the necessary routes
        self.route("/prediction", methods=['GET', 'POST'])(self.predict)
        self.route("/scoring", methods=['GET'])(self.scoring)
        self.route("/summarystats", methods=['GET'])(self.summarystats)
        self.route("/diagnostics", methods=['GET'])(self.diagnostics)
    
    def predict(self):
        """Make Predictions using the Deployed Model"""
        
        try:
            filename = request.args['filename']
            filepath = (Path('.') / filename).absolute()

            # we like really shouldn't be doing this in production code
            if (not filepath.exists() or not filepath.is_file()):
                return f'File "{filename}" does not exist', 404

            logger.info(f'Reading data from {filepath}...')

            # read the data
            df = pd.read_csv(filepath)

            # return predictions from deployed model
            preds = model_predictions(self._model, df)
            return {'predictions': preds}

        except KeyError:
            return 'Missing "filename" query parameter', 422

    def scoring(self):
        """Return the F1 Score of the Deployed Model on Test Data"""
        # return the score of the deployed model
        with tempfile.TemporaryDirectory() as tmpdir:
            score = score_model(self._model, self._test_df, Path(tmpdir))
        return {'f1-score': score}

    def summarystats(self):
        """Return the Summary Statistics of the Train Data"""

        stats = dataframe_summary(self._train_df)
        return {'summary-stats': stats}

    def diagnostics(self):
        """Report Execution Time, Missing Data, and Dependency Checks"""

        # compute result
        result = {
            'execution-times': execution_time(self._config),
            'missing-data': dataframe_missing(self._train_df),
            'package-list': outdated_packages_list()
        }

        # serve result
        return result


def server(host: str, port: int, config: Configuration):
    """Serve the Deployed Pre-Trained Model REST-ful API

    Args:
        host:   host address to serve the app
        port:   port number to serve the app
        config: Parsed JSON Configuration
    """
    logger.info(f'Reading test data from {config.test}...')
    test_df = pd.read_csv(config.test / 'testdata.csv', low_memory=False)
    logger.info(f'Reading train data from {config.train}...')
    train_df = pd.read_csv(config.train / 'finaldata.csv', low_memory=False)
    logger.info(f'Loading pre-trained model from {config.deploy}...')
    with open(config.deploy / 'trainedmodel.pkl', 'rb') as f:
        model = pickle.load(f)

    # instantiate the app
    app = ModelReportingApp(model, config, test_df, train_df)

    # run the app until CTRL+C is pressed
    app.run(host=host, port=port, threaded=True)


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
