"""
Set up and manage logging with ``loguru``.

Copyright (C) 2024  Alexander S. Brunmayr  <asb219@ic.ac.uk>

This file is part of the ``evaluate_SOC_models`` python package, subject to
the GNU General Public License v3 (GPLv3). You should have received a copy
of GPLv3 along with this file. If not, see <https://www.gnu.org/licenses/>.
"""

import sys
import os
from loguru import logger

from evaluate_SOC_models.config import get_config
from evaluate_SOC_models.path import LOGFILEPATH


__all__ = [
    'get_console_handler_id',
    'get_logfile_handler_id',
    'get_console_logging_status',
    'get_logfile_logging_status',
    'initialize_logging',
    'enable_console_logging',
    'disable_console_logging',
    'enable_logfile_logging',
    'disable_logfile_logging'
]


console_handler_id = None
logfile_handler_id = None

def get_console_handler_id():
    return console_handler_id

def get_logfile_handler_id():
    return logfile_handler_id

def get_console_logging_status():
    return 'disabled' if console_handler_id is None else 'enabled'

def get_logfile_logging_status():
    return 'disabled' if logfile_handler_id is None else 'enabled'


def initialize_logging():

    config = get_config()
    log_console = config['log']['console'].strip().lower()
    log_logfile = config['log']['logfile'].strip().lower()

    if log_console == 'enabled':
        enable_console_logging()
    elif log_console == 'disabled':
        disable_console_logging()
    elif log_console == 'default':
        pass # keep default handler
    else:
        raise ValueError

    if log_logfile == 'enabled':
        enable_logfile_logging()
    elif log_logfile == 'disabled':
        disable_logfile_logging()
    else:
        raise ValueError


def _remove_default_handler():
    try:
        logger.remove(0)
    except ValueError:
        pass # already removed before


def enable_console_logging():
    global console_handler_id

    _remove_default_handler()
    logger.debug('enabling console logging ...')

    if console_handler_id is not None:
        logger.info('console logging already enabled')
        return

    console_handler_id = logger.add(
        sys.stderr,
        level='INFO',
        format="<level>{message}</level>",
        colorize=True
    )
    logger.success('console logging enabled')


def disable_console_logging():
    global console_handler_id
    _remove_default_handler()
    logger.debug('disabling console logging ...')
    if console_handler_id is None:
        logger.info('console logging already disabled')
    else:
        logger.remove(console_handler_id)
        console_handler_id = None
        logger.success('console logging disabled')


def enable_logfile_logging():
    global logfile_handler_id

    logger.debug('enabling logfile logging ...')

    if logfile_handler_id is not None:
        logger.info('logfile logging already enabled')
        return

    config = get_config()

    logfile_handler_id = logger.add(
        str(LOGFILEPATH),
        level='DEBUG', # 'TRACE'
        format="{time:YYYY-MM-DD HH:mm:ss} | {name} | {level} :: {message}",
        opener=lambda file, flags: os.open(file, flags, 0o600),
        backtrace=True, # for when logger.exception is used
        enqueue=True, # for compatibility with multiprocessing
        rotation=config['log']['logfile_rotation'],
        retention=config['log']['logfile_retention']
    )
    logger.success('logfile logging enabled')


def disable_logfile_logging():
    global logfile_handler_id
    logger.debug('disabling logfile logging ...')
    if logfile_handler_id is None:
        logger.info('logfile logging already disabled')
    else:
        logger.remove(logfile_handler_id)
        logfile_handler_id = None
        logger.success('logfile logging disabled')
