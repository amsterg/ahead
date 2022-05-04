from src.utils import skip_run
from subprocess import call
from src.data.data_create import create_interim_files,create_processed_data
from yaml import safe_load

with open('src/config.yaml', 'r') as f:
    config = safe_load(f.read())

with skip_run('skip', 'Create Interim Files') as check, check():
    for game in config['VALID_ACTIONS']:
        create_interim_files(game)

with skip_run('skip', 'Create Processed Data') as check, check():
    for game in config['VALID_ACTIONS']:
        print(game)
        create_processed_data(stack=config['STACK_SIZE'],
                              game=game,
                              till_ix=-1,
                              data_types=config['DATA_TYPES'])

with skip_run('skip', 'Train Gaze Pred') as check, check():
    for game in config['VALID_ACTIONS']:
        print(game)
        call('python src/features/gaze_pred.py --game={}'.format(game).split(' '))
        # call('mv models_/{}/combined models_/{}/combinqed'.format(game,game).split(' '))

with skip_run('run', 'Train SGAZED') as check, check():
    for game in config['VALID_ACTIONS']:
        print(game)
        call('python src/features/selective_gazed_act_pred.py --game={} --gaze_net_cpt 14'.format(game).split(' '))
        