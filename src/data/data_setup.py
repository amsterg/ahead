import os
from yaml import safe_load
from subprocess import call

with open('src/config.yaml', 'r') as f:
    config_data = safe_load(f.read())


RAW_DATA_DIR = config_data['RAW_DATA_DIR']
PROC_DATA_DIR = config_data['PROC_DATA_DIR']
INTERIM_DATA_DIR = config_data['INTERIM_DATA_DIR']
MODEL_SAVE_DIR = config_data['MODEL_SAVE_DIR']
CMP_FMT = config_data['CMP_FMT']
OVERWRITE_INTERIM_GAZE = config_data['OVERWRITE_INTERIM_GAZE']
VALID_ACTIONS = config_data['VALID_ACTIONS']

# Create required directories
DIRS_REQD = [RAW_DATA_DIR, PROC_DATA_DIR, INTERIM_DATA_DIR, MODEL_SAVE_DIR]
for entry in DIRS_REQD:
    if not os.path.exists(entry):
        os.makedirs(entry)
games = VALID_ACTIONS.keys()

url_ph = 'https://zenodo.org/record/3451402/files/'
urls = [url_ph+'action_enums.txt']
urls += [url_ph+game+'.zip' for game in games]

# download data using wget
for url in urls:
    file_name = url.split('/')[-1]
    file_out_name = os.path.join(RAW_DATA_DIR, file_name)
    if not os.path.exists(file_out_name):
        args_string = 'wget {} -O {}'.format(url, file_out_name)
        args = args_string.split(' ')
        call(args)
        # unzips files at RAW_DATA_DIR
        if file_out_name.__contains__('.zip'):
            unzip_str = 'unzip -n {} -d {}'.format(file_out_name, RAW_DATA_DIR)
            unzip_args = unzip_str.split(' ')
            call(unzip_args)


# create interim and processed data
preprocess_str = "python src/data/data_create.py"
call(preprocess_str.split(' '))
