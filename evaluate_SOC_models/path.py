from pathlib import Path
from evaluate_SOC_models.config import get_config

config = get_config()

REPOSITORYPATH = Path(__file__).resolve().parent.parent # git repository path
MENDREPOSITORYPATH = REPOSITORYPATH / 'MEND'
MIMICS2021REPOSITORYPATH = REPOSITORYPATH / 'MIMICS2021'

absolute_path = lambda x: REPOSITORYPATH / Path(config['path'][x]).expanduser()

DUMPPATH = absolute_path('dump')
DOWNLOADPATH = absolute_path('downloads')
DATAPATH = absolute_path('data')
LOGPATH = absolute_path('log')
RESULTSPATH = absolute_path('results')

TOPSOIL_MIN_DEPTH = int(config['topsoil']['min_depth'])
TOPSOIL_MAX_DEPTH = int(config['topsoil']['max_depth'])
SAVEPATH = RESULTSPATH / \
    f'topsoil_min{TOPSOIL_MIN_DEPTH}cm_max{TOPSOIL_MAX_DEPTH}cm_depth'
SAVEOUTPUTPATH = SAVEPATH / 'model-output'
SAVEALLDATAPATH = SAVEPATH / 'data'

del config, absolute_path
