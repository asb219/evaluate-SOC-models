from pathlib import Path
from configparser import ConfigParser
from loguru import logger

from evaluate_SOC_models.data_manager import DataFile


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
        savedir = Path(__file__).resolve().parent.parent # path of repository
        super().__init__(savedir / filename, readonly=True)


class CustomConfigFile(ConfigFile):
    """Custom configuration file for the ``evaluate_SOC_models`` package.
    
    Create and modify this file using functions :py:meth:`enable_log`,
    :py:meth:`disable_log`, and :py:meth:`set_...`.
    
    Calling :py:meth:`reset_defaults()` removes this file.
    """

    def __init__(self, **kwargs):
        filename = 'config.ini'
        savedir = Path(__file__).resolve().parent.parent # path of repository
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
