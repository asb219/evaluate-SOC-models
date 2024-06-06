"""
Get and set configurations in config files of ``evaluate_SOC_data`` package.

Copyright (C) 2024  Alexander S. Brunmayr  <asb219@ic.ac.uk>

This file is part of the ``evaluate_SOC_models`` python package, subject to
the GNU General Public License v3 (GPLv3). You should have received a copy
of GPLv3 along with this file. If not, see <https://www.gnu.org/licenses/>.
"""

from pathlib import Path
from configparser import ConfigParser
from loguru import logger

from data_manager import DataFile


__all__ = [
    'ConfigFile',
    'DefaultConfigFile',
    'CustomConfigFile',
    'get_config',
    'print_config'
]


class ConfigFile(DataFile):
    """Configuration file managed by :py:mod:`configparser`."""

    def _read(self):
        config = ConfigParser()
        config.read(self.path)
        return config

    def _write(self, config, mode='w'):
        with self.path.open(mode) as config_file:
            return config.write(config_file)


class DefaultConfigFile(ConfigFile):
    """Default configuration file for the ``evaluate_SOC_models`` package."""

    def __init__(self):
        filename = 'config_defaults.ini'
        savedir = Path(__file__).resolve().parent # path of current file
        super().__init__(savedir / filename, readonly=True)


class CustomConfigFile(ConfigFile):
    """Custom configuration file for the ``evaluate_SOC_models`` package.
    
    Create and modify this file using functions :py:meth:`enable_log`,
    :py:meth:`disable_log`, and :py:meth:`set_...`.
    
    Calling :py:meth:`reset_defaults` removes this file.
    """

    def __init__(self, **kwargs):
        filename = 'config.ini'
        savedir = Path(__file__).resolve().parent # path of current file
        super().__init__(savedir / filename, **kwargs)


def get_config():
    """Get the current configurations of the ``evaluate_SOC_models`` package.
    Reads :py:class:`CustomConfigFile` if it exists,
    :py:class:`DefaultConfigFile` otherwise.
    
    Returns
    -------
    configparser.ConfigParser
    """
    try:
        return CustomConfigFile().read()
    except FileNotFoundError:
        return DefaultConfigFile().read()


def print_config(*, raw=False):
    config = get_config()
    print('')
    _packagepath = Path(__file__).resolve().parent
    print(f'# Note: Paths are relative to "{_packagepath}", unless absolute.')
    print('')
    for section in config.sections():
        print(f'[{section}]')
        for name, value in config.items(section, raw=raw):
            print(f'{name} = {value}')
        print('')


def reset_defaults():
    """Reset default configs by removing :py:class:`CustomConfigFile`."""
    logger.info('Resetting default configurations for evaluate_SOC_models ...')
    config_file = CustomConfigFile()
    if config_file.exists():
        config_file.remove(ask=False)
        logger.success('Configurations reset to default.')
        _ask_to_restart_kernel()
    else:
        logger.info('Configurations already default.')


def set_log_filename(filename):
    filename = str(filename)
    if not filename.strip():
        raise ValueError(f'Cannot set log_filename "{filename}"')
    _set_config('log', 'filename', str(filename))

def enable_log():
    _set_config('log', 'logfile', 'enabled')

def disable_log():
    _set_config('log', 'logfile', 'disabled')

def set_quiet():
    _set_config('log', 'console', 'disabled')

def set_verbose():
    _set_config('log', 'console', 'enabled')

def set_dump_path(path):
    _set_path('path', 'dump', path)


def _set_path(name, path):
    path = str(path)
    if not path.strip():
        raise ValueError(f'Cannot set path "{path}"')
    _set_config('path', name, path)


def _set_config(section, name, value):
    config = get_config()
    config[section][name] = value
    CustomConfigFile().write(config)
    logger.success(f'Set [{section}] {name} = {value}')
    _ask_to_restart_kernel()


ASK_TO_RESTART_KERNEL = True

def _ask_to_restart_kernel():
    if ASK_TO_RESTART_KERNEL:
        logger.warning('Please restart the kernel for changes to take effect.')


def main():

    import sys
    import argparse

    import evaluate_SOC_models.logging
    from evaluate_SOC_models.path import DUMPPATH, LOGFILEPATH

    parser = argparse.ArgumentParser(
        prog='config',
        description='Manage configurations for `evaluate_SOC_models` package.',
        epilog=''
    )
    parser.add_argument('-print', action='store_true',
        help='print configurations and exit')
    parser.add_argument('-reset', action='store_true',
        help='remove config.ini file and exit')
    parser.add_argument('-get-dump', action='store_true',
        help='print absolute path of file storage dump and exit')
    parser.add_argument('-set-dump', metavar='DUMP',
        help='set path of file storage dump')
    parser.add_argument('-log-status', action='store_true',
        help='print status of console and logfile logging and exit')
    parser.add_argument('-get-logfile', action='store_true',
        help='print absolute path of logfile and exit')
    parser.add_argument('-disable-log', action='store_true',
        help='disable logging to logfile')
    parser.add_argument('-enable-log', action='store_true',
        help='enable logging to logfile')
    parser.add_argument('-set-quiet', action='store_true',
        help='disable logging to console')
    parser.add_argument('-set-verbose', action='store_true',
        help='enable logging to console')

    if len(sys.argv) == 1: # `config` was run without arguments
        parser.print_help()
        return

    args = parser.parse_args()

    logger.disable('evaluate_SOC_models.logging')
    evaluate_SOC_models.logging.enable_console_logging()
    logger.enable('evaluate_SOC_models.logging')

    ASK_TO_RESTART_KERNEL = False

    if args.print:
        print_config()
    elif args.reset:
        reset_defaults()
    elif args.get_dump:
        print(DUMPPATH)
    elif args.log_status:
        status = get_config()['log']
        print('Logging to console is '+status['console'].strip().lower()+'.')
        print('Logging to logfile is '+status['logfile'].strip().lower()+'.')
    elif args.get_logfile:
        print(LOGFILEPATH)
    else:
        if args.set_dump:
            set_dump_path(args.set_dump)
        if args.set_quiet:
            set_quiet()
        if args.set_verbose:
            set_verbose()
        if args.disable_log:
            disable_log()
        if args.enable_log:
            enable_log()

    ASK_TO_RESTART_KERNEL = True


if __name__ == '__main__':
    main()
