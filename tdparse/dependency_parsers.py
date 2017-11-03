'''
Contains functions that perform depdency parsing.
'''
from collections import defaultdict
import os
import subprocess
import tempfile

from tdparse.helper import read_config

def tweebo_install(tweebo_func):
    '''
    Python decorator that ensures that
    `TweeboParser <https://github.com/ikekonglp/TweeboParser>`_ is installed,
    before running the function it wraps. Returns the given function.

    :param tweebo_func: A function that uses the Tweebo Parser.
    :type tweebo_func: function
    :returns: The given function
    :rtype: function
    '''

    tweebo_dir = os.path.abspath(read_config('depdency_parsers')['tweebo_dir'])
    # If the models file exists then Tweebo has been installed or failed to
    # install
    tweebo_models = os.path.join(tweebo_dir, 'pretrained_models.tar.gz')
    if not os.path.isfile(tweebo_models):
        install_script = os.path.join(tweebo_dir, 'install.sh')
        subprocess.run(['bash', install_script])
    return tweebo_func

def get_tweebo_dependencies(token_dep_sentence):
    def dep_search(dep_index, sentence, dep_info):
        head_index = sentence[dep_index][1]
        if head_index == -1 or head_index == -2:
            return dep_info
        prev_dep_info = dep_info[dep_index]
        head_deps = {dep_level + 1: deps
                     for dep_level, deps in prev_dep_info.items()}
        head_deps[1] = dep_index
        dep_info[head_index] = head_deps
        return dep_search(head_index, sentence, dep_info)
    dep_results = defaultdict(lambda: defaultdict(set))
    for index, _ in enumerate(token_dep_sentence):
        dep_result = dep_search(index, token_dep_sentence,
                                defaultdict(lambda: dict()))
        for token_index, dependencies in dep_result.items():
            for dep_level, dependent in dependencies.items():
                dep_results[token_index][dep_level].add(dependent)
    return dep_results


def tweebo_post_process(processed_text):
    tokens = processed_text.split('\n')
    sentences = []
    last_token = None
    sentence = []
    for token in tokens:
        token = token.strip()
        if last_token == '' and token == '':
            continue
        elif token == '':
            sentences.append(sentence)
            sentence = []
        else:
            token = token.split('\t')
            token_dep_index = int(token[6]) - 1
            token_text = token[1]
            sentence.append([token_text, token_dep_index])
        last_token = token
    return sentences

@tweebo_install
def tweebo(text, batch=False):
    '''
    Given a String will tokenise, pos tag and then dependecy parse the text using
    `Tweebo <https://github.com/ikekonglp/TweeboParser>`_ a Tweet specific parser.

    The Tweebo parser cannot handle no strings therefore a special empty string
    symbol is required.

    :param text: The text that is to be parsed
    :param batch: If too process the text in batch mode. If so the text has to be
    a list of Strings.
    :type text: String if batch==False else list
    :type batch: Boolean Default False
    :returns: Dependency parsed text
    :rtype: String
    '''

    def no_text(text):
        '''
        Given a String checks if it is empty if so returns an empty_token else
        the text that was given.

        :param text: Text to be checked
        :type text: String
        :returns: The text if it is not empty or empty token if it is.
        :rtype: String
        '''

        empty_token = '$$$EMPTY$$$'
        if text.strip() == '':
            return empty_token
        return text


    with tempfile.TemporaryDirectory() as temp_dir:
        text_file_path = os.path.join(temp_dir, 'text_file.txt')
        result_file_path = os.path.join(temp_dir, 'text_file.txt.predict')
        tweebo_dir = os.path.abspath(read_config('depdency_parsers')['tweebo_dir'])
        with open(text_file_path, 'w+') as text_file:
            if batch:
                for a_text in text:
                    a_text = no_text(a_text)
                    text_file.write(a_text)
                    text_file.write('\n')
            else:
                text = no_text(text)
                text_file.write(text)
        run_script = os.path.join(tweebo_dir, 'run.sh')
        if subprocess.run(['bash', run_script, text_file_path]):
            with open(result_file_path, 'r') as result_file:
                return result_file.read()
        else:
            raise SystemError('Could not run the Tweebo run script {} for text {}'\
                              .format(run_script, text))
