from pathlib import Path
from configparser import ConfigParser

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
    
    Create and modify this file using the tools provided in the
    :py:mod:`evaluate_SOC_models.config` module.
    """

    def __init__(self, **kwargs):
        filename = 'config.ini'
        savedir = Path(__file__).resolve().parent.parent # path of repository
        super().__init__(savedir / filename, **kwargs)


def get_config():
    """Get the current configurations of the ``evaluat_SOC_models`` package.
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
