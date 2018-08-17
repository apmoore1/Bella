'''
Functions that are used throughout the package:

1. :py:func:`bella.helper.read_config` -- Given a file location to a Yaml file
   that stores only keys and relative or full path locations as values. Returns
   those paths as a fully resolved String given the key.
2. :py:func:`download_file` -- Given a file path and the URL address of the
   data, downloads the data from the URL to the file path location.
3. :py:func:`download_model` -- Downloads the specified model trained on the
   dataset from the model zoo and returns it as a trained model to use.
'''
from pathlib import Path

import requests
from ruamel.yaml import YAML

import bella


BELLA_MODEL_DIR = Path.home().joinpath('.Bella', 'Models')


def read_config(key: str, config_file_path: Path) -> str:
    '''
    Given a file location to a Yaml file that stores only keys and relative
    or full path locations as values. Returns those paths as a fully resolved
    String given the key.

    :param key: The key to the Path within the config file.
    :param config_file_path: File path to the YAML configuration file.
    :return: Full path as a String to the relative file location that is
             stored in the key of the config file.
    '''

    config_parent = config_file_path.parent
    with config_file_path.open('r') as config_file:
        yaml = YAML()
        config_data = yaml.load(config_file)
        if key in config_data:
            file_path = config_data[key]
            key_file_path = config_parent.joinpath(file_path)
            return str(key_file_path.resolve())
        else:
            raise ValueError(f'This key {key} does not exist in the '
                             f'config file {config_file_path}')


def download_file(file_path: Path, url: str) -> None:
    '''
    Given a file path and the URL address of the data, downloads the data
    from the URL to the file path location.

    :param file_path: Path to save the downloaded data to.
    :param url: URL location of the data to be downloaded.
    :return: Nothing
    '''
    with file_path.open('wb') as a_file:
        request = requests.get(url, stream=True)
        for chunk in request.iter_content(chunk_size=128):
            a_file.write(chunk)


def download_model(model: 'bella.models.base.BaseModel',
                   dataset_name: str) -> 'bella.models.base.BaseModel':
    '''
    Downloads the specified model trained on the dataset from the
    model zoo and returns it as a trained model to use.

    The model zoo `URL <https://delta.lancs.ac.uk/mooreap/bella-models>`_

    The model zoo came from the following `paper <https://aclanthology.coli.un\
    i-saarland.de/papers/C18-1097/c18-1097>`_ and results for those models
    on the relevant dataset are within the Mass Evaluation section of the paper

    :param model: Class of model you want to download
    :param dataset_name: Name of the dataset that the model has been trained
                         on.
    :return: An instance of the model class you gave as an argument trained
             on the dataset specified.
    '''

    BELLA_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    base_url = 'https://delta.lancs.ac.uk/mooreap/bella-models/raw/master/'

    model_name = model.name()
    model_file_name = f'{model_name} {dataset_name}'
    model_path = BELLA_MODEL_DIR.joinpath(model_file_name)
    if 'LSTM' in model_name:
        meta_data_path = model_path.with_suffix('.pkl')
        model_data_path = model_path.with_suffix('.h5')
        data_paths = [meta_data_path, model_data_path]
        for data_path in data_paths:
            if not data_path.is_file():
                data_url = base_url + data_path.name
                data_url = data_url.replace(' ', '%20')
                download_file(data_path, data_url)

    else:
        if not model_path.is_file():
            model_url = base_url + model_file_name
            model_url = model_url.replace(' ', '%20')
            download_file(model_path, model_url)
    return model.load(model_path)
