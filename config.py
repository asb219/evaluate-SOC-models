"""
Manage the configurations of the ``evaluate_SOC_models`` package.
"""


if __name__ == '__main__':

    import sys
    import argparse
    from loguru import logger

    from evaluate_SOC_models.path import DUMPPATH
    from evaluate_SOC_models import config
    import evaluate_SOC_models.logging


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

    if len(sys.argv) == 1: # `python -m config` was run without arguments
        parser.print_help()
        return

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
        # if args.logfile:
        #     config.set_log_filename(args.logfile)

    config.ASK_TO_RESTART_KERNEL = True
