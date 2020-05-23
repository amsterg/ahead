from src.utils import skip_run
from subprocess import call
from src.data.data_create import create_interim_files,create_processed_data
from yaml import safe_load

with open('src/config.yaml', 'r') as f:
    config = safe_load(f.read())

with skip_run('run', 'Create Interim Files') as check, check():
    for game in config['VALID_ACTIONS']:
        create_interim_files(game)

with skip_run('skip', 'Create Processed Data') as check, check():
    for game in config['VALID_ACTIONS']:
        create_processed_data(stack=config['STACK_SIZE'],
                              game=game,
                              till_ix=-1,
                              data_types=config['DATA_TYPES'])
