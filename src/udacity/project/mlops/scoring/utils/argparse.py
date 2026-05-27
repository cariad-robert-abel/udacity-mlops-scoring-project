#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__all__ = (
    'add_default_arguments',
    'add_subcommand',
    'add_wsgi_subcommand',
)

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse


def add_default_arguments(parser: 'argparse.ArgumentParser'):
    """Add default arguments to the parser

    Adds the default arguments (--loglevel, --config).
    Args:
        parser: argument parser object
    """
    parser.add_argument('--loglevel', type=str, choices=('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'),
                        default='INFO', help='Log Level')
    parser.add_argument('--config', type=Path, default='cfg/practice.json', help='Path to the config file')


def add_wsgi_arguments(parser: 'argparse.ArgumentParser'):
    """Add WSGI arguments to the parser

    Adds the WSGI arguments (--host, --port).
    Args:
        parser: argument parser object
    """
    parser.add_argument('--host', type=str, default='localhost', help='Host for the server')
    parser.add_argument('--port', type=int, default=8000, help='Port for the server')


def add_subcommand(subparser: 'argparse._SubParsersAction[argparse.ArgumentParser]', command: str, help: str,
                   prog: str | None = None
                   ) -> 'argparse.ArgumentParser':
    """Add a Command to the Sub-Parser

    Adds the command and default arguments (--loglevel, --config).
    Args:
        subparser: sub-parser object
        command: command name
        help: command description
        prog: command program name (optional)

    Returns:
        Sub-parser for the command.
    """
    # create command parser
    parser = subparser.add_parser(command, help=help, prog=prog)
    # add default arguments
    add_default_arguments(parser)
    # return to add more arguments
    return parser


def add_wsgi_subcommand(subparser: 'argparse._SubParsersAction[argparse.ArgumentParser]', command: str, help: str,
                        prog: str | None = None
                        ) -> 'argparse.ArgumentParser':
    """Add a WSGI Command to the Sub-Parser

    Adds the command and default arguments (--loglevel, --config, --host, --port).
    Args:
        subparser: sub-parser object
        command: command name
        help: command description
        prog: command program name (optional)

    Returns:
        Sub-parser for the command.
    """
    parser = add_subcommand(subparser, command, help, prog=prog)
    add_wsgi_arguments(parser)
    return parser
