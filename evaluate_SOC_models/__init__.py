"""
``evaluate_SOC_models``

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

from loguru import logger
from evaluate_SOC_models.logging import initialize_logging

logger.disable('evaluate_SOC_models')
initialize_logging()
logger.enable('evaluate_SOC_models') # enable log by default

del logger, initialize_logging

from evaluate_SOC_models.models import *
from evaluate_SOC_models.data import *
