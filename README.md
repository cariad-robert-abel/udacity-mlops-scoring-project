# Machine Learning Model Scoring and Monitoring: Dynamic Risk Assessment System

This repository contains the project associated with "Machine Learning Model Scoring and Monitoring" Udacity course.
It's a fork of Udacity's [Starter Kit](https://github.com/udacity/cd0583-project-starter-file).

This GitHub.com project is located at [cariad-robert-abel/udacity-mlops-scoring-project](https://github.com/cariad-robert-abel/udacity-mlops-scoring-project).

## Pre-Requisites

Install this application before using it, for example by running the following inside a virtual environment:

    pip install .

This will also install the necessary `udacity-project-mlops-scoring*` shims for each step and make them available.

## Configuration

All commands accept the `--loglevel` parameter to control the log level (defaults to `INFO`) as well as the `--config`
parameter, that needs to point to a configuration file.  
The configuration defaults to `cfg/practice.json` and we assume code is run from the root of the repository.

## Step 1: Data Ingestion

Data is ingested from configured `input_folder_path` directory, de-duplicated, and stored to configured
`output_folder_path` directory.

Run ingestion either on its own using any of the following commands:

    udacity-project-mlops-scoring ingest [-h] [--loglevel {DEBUG,INFO,WARNING,ERROR,CRITICAL}] [--config CONFIG]
    udacity-project-mlops-scoring-ingest [-h] [--loglevel {DEBUG,INFO,WARNING,ERROR,CRITICAL}] [--config CONFIG]
    python -m udacity.project.mlops.scoring.ingestion [-h] [--loglevel {DEBUG,INFO,WARNING,ERROR,CRITICAL}] [--config CONFIG]

## License

Original files Copyright 2012–2026 Udacity, Inc.
My additions to documentation and code are [MIT](https://spdx.org/licenses/MIT).
See [LICENSE-Udacity](LICENSE-Udacity) resp. [LICENSE](LICENSE).
