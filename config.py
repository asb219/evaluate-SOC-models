"""
Manage the configurations of the ``evaluate_SOC_models`` package.

Copyright (C) 2024  Alexander S. Brunmayr  <asb219@ic.ac.uk>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

if __name__ == '__main__':

    import sys
    import argparse
    from loguru import logger

    import evaluate_SOC_models.logging
    from evaluate_SOC_models import config
    from evaluate_SOC_models.path import DUMPPATH, LOGFILEPATH


    parser = argparse.ArgumentParser(
        prog='python -m config',
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

    if len(sys.argv) == 1: # `python -m config` was run without arguments
        parser.print_help()
        sys.exit()

    args = parser.parse_args()


    logger.disable('evaluate_SOC_models.logging')
    evaluate_SOC_models.logging.enable_console_logging()
    logger.enable('evaluate_SOC_models.logging')

    config.ASK_TO_RESTART_KERNEL = False

    if args.print:
        config.print_config()
    elif args.reset:
        config.reset_defaults()
    elif args.get_dump:
        print(DUMPPATH)
    elif args.log_status:
        status = config.get_config()['log']
        print('Logging to console is '+status['console'].strip().lower()+'.')
        print('Logging to logfile is '+status['logfile'].strip().lower()+'.')
    elif args.get_logfile:
        print(LOGFILEPATH)
    else:
        if args.set_dump:
            config.set_dump_path(args.set_dump)
        if args.set_quiet:
            config.set_quiet()
        if args.set_verbose:
            config.set_verbose()
        if args.disable_log:
            config.disable_log()
        if args.enable_log:
            config.enable_log()

    config.ASK_TO_RESTART_KERNEL = True
