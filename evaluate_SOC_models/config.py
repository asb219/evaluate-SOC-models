import sys
import argparse
from pathlib import Path
from loguru import logger

from evaluate_SOC_models.path import DUMPPATH
from evaluate_SOC_models.config_file import \
    CustomConfigFile, get_config, print_config


def reset_defaults():
    logger.info('Resetting default configurations for evaluate_SOC_models ...')
    config_file = CustomConfigFile()
    if config_file.exists():
        config_file.remove(ask=False)
        logger.success('Configurations reset to default.')
        _ask_to_restart_kernel()
    else:
        logger.info('Configurations already default.')


def set_log_filename(filename):
    _set_config('log', 'filename', filename)

def enable_log():
    _set_config('log', 'logfile', 'enabled')

def disable_log():
    _set_config('log', 'logfile', 'disabled')

def set_quiet():
    _set_config('log', 'console', 'disabled')

def set_verbose():
    _set_config('log', 'console', 'enabled')

def set_dump_path(path):
    _set_path('dump', path)

def _set_path(name, path):
    path = str(Path(path).expanduser())
    _set_config('path', name, path)

def _set_config(section, name, value):
    config = get_config()
    config[section][name] = value
    CustomConfigFile().write(config)
    logger.success(f'Set [{section}] {name} = {value}')
    _ask_to_restart_kernel()


def _ask_to_restart_kernel():
    if __name__ != '__main__':
        logger.warning('Please restart the kernel for changes to take effect.')


def main():

    parser = argparse.ArgumentParser(
        prog='evaluate_SOC_models.config',
        description='Set configurations in the config.ini file',
        epilog=''
    )
    parser.add_argument('-print', action='store_true',
        help='print configurations and exit')
    parser.add_argument('-reset', action='store_true',
        help='remove config.ini file and exit')
    parser.add_argument('-get-dump', action='store_true',
        help='print path of file storage dump and exit')
    parser.add_argument('-set-dump', help='set path of file storage dump')
    parser.add_argument('-set-quiet', action='store_true',
        help='disable logging to console')
    parser.add_argument('-set-verbose', action='store_true',
        help='enable logging to console')
    parser.add_argument('-disable-log', action='store_true',
        help='disable logging to logfile')
    parser.add_argument('-enable-log', action='store_true',
        help='enable logging to logfile')
    # parser.add_argument('-logfile', help='set filename of logfile')

    if len(sys.argv) == 1:
        # python -m evaluate_SOC_models.config was run without arguments
        parser.print_help()
        return

    args = parser.parse_args()

    if args.enable_log and args.disable_log:
        raise ValueError('Contradicting arguments: -disable-log -enable-log')
    if args.set_quiet and args.set_verbose:
        raise ValueError('Contradicting arguments: -set-quiet -set-verbose')

    if args.print:
        print_config()
        return
    if args.reset:
        reset_defaults()
        return
    if args.get_dump:
        print(DUMPPATH)
        return
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
    # if args.logfile:
    #     set_log_filename(args.logfile)


if __name__ == '__main__':

    import evaluate_SOC_models.logging

    logger.disable('evaluate_SOC_models.logging')
    evaluate_SOC_models.logging.enable_console_logging()

    main()
