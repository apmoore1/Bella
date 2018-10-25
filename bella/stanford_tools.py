import json
from pathlib import Path
from typing import Tuple

from nltk.tree import Tree
from ruamel.yaml import YAML
from stanfordcorenlp import StanfordCoreNLP

BELLA_CONFIG_FP = Path.home().joinpath('.Bella', 'config.yaml')

class StanfordNlp(object):
    '''
    Singleton Class instance
    '''

    instance = None

    @staticmethod
    def get_config() -> Tuple[str, int]:
        hostname = 'http://localhost'
        port = 9000
        yaml = YAML()
        config_data = {}
        if BELLA_CONFIG_FP.exists():
            with BELLA_CONFIG_FP.open('r') as config_file:
                config_data = yaml.load(config_file)
                if 'stanford_core_nlp' in config_data:
                    stanford_config = config_data['stanford_core_nlp']
                    if 'hostname' in stanford_config:
                        hostname = stanford_config['hostname']
                    if 'port' in stanford_config:
                        port = stanford_config['port']
        
        config_data['stanford_core_nlp'] = {}
        config_data['stanford_core_nlp']['hostname'] = hostname
        config_data['stanford_core_nlp']['port'] = port
        with BELLA_CONFIG_FP.open('w') as config_file:
            yaml.dump(config_data, config_file)
        return hostname, port

    def __new__(cls):
        if StanfordNlp.instance is None:
            hostname, port = cls.get_config()
            StanfordNlp.instance = StanfordCoreNLP(hostname, port)
        return StanfordNlp.instance

    def __getattr__(self, name):
        return getattr(self.instance, name)

    def __setattr__(self, name, value):
        return setattr(self.instance, name, value)


def tokenise(text):

    stanford_nlp = StanfordNlp()
    output_dict = stanford_nlp.annotate(text, {'annotators': 'ssplit,tokenize',
                                               'tokenize.language': 'English',
                                               'outputFormat': 'json'})
    output_dict = json.loads(output_dict, strict=False)
    tokens = [token['originalText'] for s in output_dict['sentences']
              for token in s['tokens']]
    return tokens


def constituency_parse(text):
    '''
    :param text: The text you want to parse
    :type text: String
    :returns: A list of parse trees where each tree is represented by \
    nltk.tree.Tree. Each parse tree is associated to a sentence in the text. \
    If one sentence in the text then the list will be of length 1.
    :rtype: list
    '''

    if text.strip() == '':
        raise ValueError('There has to be some text to parse. Text given {}'
                         .format(text))
    stanford_nlp = StanfordNlp()
    output_dict = stanford_nlp.annotate(text, {'annotators': 'pos,parse',
                                               'tokenize.language': 'English',
                                               'outputFormat': 'json'})
    output_dict = json.loads(output_dict)
    parse_trees = [Tree.fromstring(sent['parse'])
                   for sent in output_dict['sentences']]
    return parse_trees


def dependency_parse(text, dep_type='basicDependencies'):
    '''
    :param text: The text you want to parse
    :param dep_type: The dependency type to use either: 'basicDependencies', \
    'enhancedDependencies', or 'enhancedPlusPlusDependencies'. See for more \
    details \
    https://nlp.stanford.edu/~sebschu/pubs/schuster-manning-lrec2016.pdf
    :type text: String
    :type dep_type: String. Default basicDependencies
    '''

    if text.strip() == '':
        raise ValueError('There has to be some text to parse. Text given {}'
                         .format(text))

    stanford_nlp = StanfordNlp()
    output_dict = stanford_nlp.annotate(text, {'annotators': 'pos,depparse',
                                               'tokenize.language': 'English',
                                               'outputFormat': 'json'})
    sentences = json.loads(output_dict)['sentences']
    tokens_dicts = []
    dep_dicts = []
    for sentence in sentences:
        tokens_dict = {token_data['index']: token_data
                       for token_data in sentence['tokens']}
        dep_dict = {}
        for dep_data in sentence[dep_type]:
            dep_rel = dep_data['dep']
            dep_token_index = dep_data['governor']
            current_token_index = dep_data['dependent']
            dep_dict[current_token_index] = (dep_rel, dep_token_index)
        tokens_dicts.append(tokens_dict)
        dep_dicts.append(dep_dict)
    return dep_dicts, tokens_dicts
