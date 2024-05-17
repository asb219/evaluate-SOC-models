import sys
import os
from pathlib import Path
from loguru import logger

from evaluate_SOC_models.config import get_config


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
        pass # had already been removed before


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

    config = get_config()
    logpath = Path(config['path']['log']).expanduser().resolve()
    logfilename = config['log']['filename']

    if logfile_handler_id is not None:
        logger.info('logfile logging already enabled')
        return

    logfile_handler_id = logger.add(
        str(logpath / logfilename),
        level='DEBUG', # 'TRACE'
        format="{time:YYYY-MM-DD HH:mm:ss} | {name} | {level} :: {message}",
        opener=lambda file, flags: os.open(file, flags, 0o600),
        backtrace=True, # for when logger.exception is used
        enqueue=True, # for compatibility with multiprocessing
        rotation="500 KB",
        retention="30 days"
    )
    logger.success('logfile logging enabled')


def disable_logfile_logging():
    logger.debug('disabling logfile logging ...')
    global logfile_handler_id
    if logfile_handler_id is None:
        logger.info('logfile logging already disabled')
    else:
        logger.remove(logfile_handler_id)
        logfile_handler_id = None
        logger.success('logfile logging disabled')
