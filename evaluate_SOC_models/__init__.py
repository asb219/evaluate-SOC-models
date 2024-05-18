from loguru import logger
from evaluate_SOC_models.logging import initialize_logging

logger.disable('evaluate_SOC_models')
initialize_logging()
logger.enable('evaluate_SOC_models') # enable log by default

del logger, initialize_logging

from evaluate_SOC_models.models import *
from evaluate_SOC_models.data import *
