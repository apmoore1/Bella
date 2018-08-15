'''
Functions that are used throughout the package:

1. :py:func:`bella.helper.read_config` -- Given a file location to a Yaml file
   that stores only keys and relative or full path locations as values. Returns
   those paths as a fully resolved String given the key.
'''
from pathlib import Path

from ruamel.yaml import YAML


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
