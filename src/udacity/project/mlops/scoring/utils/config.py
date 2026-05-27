#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__all__ = (
    'Configuration',
)

import json
from dataclasses import dataclass, field, fields
from pathlib import Path


@dataclass
class Configuration:
    """Configuration for Individual Process Steps

    Contains basically all the configuration paths as absolute.
    """
    input: Path  = field(            metadata={'key': 'input_folder_path'})
    """Absolute Path to the Input Data Directory"""
    train: Path  = field(repr=False, metadata={'key': 'output_folder_path'})
    """Absolute Path to the Training Data Directory"""
    test: Path   = field(repr=False, metadata={'key': 'test_data_path'})
    """Absolute Path to the Test Data Directory"""
    model: Path  = field(            metadata={'key': 'output_model_path'})
    """Absolute Path to the Model Directory"""
    deploy: Path = field(repr=False, metadata={'key': 'prod_deployment_path'})
    """Absolute Path to the Deployment Directory"""
    random: int  = field(            metadata={'key': 'random-state'})
    """Random State for Reproducibility"""

    @classmethod
    def from_json(cls, config_path: Path) -> 'Configuration':
        """Create Configuration from JSON File

        Args:
            config_path: path to the JSON configuration file
        
        Raises:
            FileNotFoundError: if the config file does not exist
            KeyError: if any of the required keys are missing in the config file

        Returns:
            Configuration object with absolute paths
        """
        with open(config_path, 'r') as f:
            config_data = json.load(f)

        base = Path('.')
        kwargs = {}
        # parse Path fields
        for field in fields(cls):

            # JSON key mapped to attribute
            key = field.metadata['key']

            # extract value
            if key not in config_data:
                raise KeyError(f'Missing required key "{key}" in configuration file')
            value = config_data[key]

            # pre-process by type
            match (field.type):
                case Path():
                    kwargs[field.name] = (base / value).resolve()
                case _:
                    kwargs[field.name] = field.type(value)

        # construct instance
        return cls(**kwargs)
