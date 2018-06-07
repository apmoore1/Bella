'''
Functions that are used throughout the package:

1. :py:func:`bella.helper.read_config` -- Given a key returns the value \
associated to that key in the given YAML configuration file.
'''

from typing import Any
from pathlib import Path

from ruamel.yaml import YAML

#def package_dir():
#    '''
#    Returns the directory of the git repository.
#
#    :returns: The path to the directory of the package
#    :rtype: String
#    '''
#
#    return os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))


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


#def _read_config(config_key, config_file_name='config.yaml'):
#    '''
#    Returns a value within the configuarion file given the key.
#
#    :param config_key: key to a value stored in the config.yaml file
#    :param config_file_name: Name of the config file within the package directory
#    :type config_key: String
#    :type config_file_name: String Default "config.yaml"
#    :returns: Value stored at the keys location
#    :rtype: Python type of the value e.g. if 5 it will be int
#    '''
#
#    config_file_path = os.path.join(package_dir(), config_file_name)
#    if not os.path.isfile(config_file_path):
#        raise FileNotFoundError('The following config file does not exist {}'\
#                                .format(config_file_path))
#    with open(config_file_path, 'r') as config_file:
#        yaml = YAML()
#        config_data = yaml.load(config_file)
#        if config_key in config_data:
#            return config_data[config_key]
#        else:
#            raise ValueError('This key {} does not exist in the config file {}'\
#                             .format(config_key, config_file_path))

#def full_path(relative_path):
#    '''
#    Given a relative path that is relative to the package directory it returns
#    the full absolute path.
#
#    :param relative_path: Path that is relative to the package directory
#    :type relative_path: String
#    :returns: Absolute path of the relative path
#    :rtype: String
#    '''

#    package_path = package_dir()
#    return os.path.abspath(os.path.join(package_path, relative_path))
