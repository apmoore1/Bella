'''
Functions that are used throughout the package:

1. :py:func:`bella.helper.read_config` -- Given a key returns the value \
associated to that key in the given YAML configuration file.
'''

from typing import Any
from pathlib import Path

from ruamel.yaml import YAML


def read_config(key: str, config_file_path: Path) -> Any:
    '''
    :param key: The key to the value you would like to get from the config file
    :param config_file_path: File path to the YAML configuration file.
    :return: The value stored at the key within the file.
    '''

    with config_file_path.open('r') as config_file:
        yaml = YAML()
        config_data = yaml.load(config_file)
        if key in config_data:
            return config_data[key]
        else:
            raise ValueError(f'This key {key} does not exist in the '
                             f'config file {config_file_path}')
