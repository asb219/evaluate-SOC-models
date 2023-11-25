from loguru import logger

logger.disable('evaluate_SOC_models')
from evaluate_SOC_models.logging import initialize_logging
initialize_logging()
logger.enable('evaluate_SOC_models') # enable log by default

del logger, initialize_logging

import evaluate_SOC_models.models
import evaluate_SOC_models.data_sources
import evaluate_SOC_models.forcing_data
import evaluate_SOC_models.observed_data

from evaluate_SOC_models.models import *
