'''
Contains functions that perform depdency parsing.
'''
from functools import wraps
import os
import subprocess
import tempfile

from tdparse.helper import read_config

def tweebo_install(tweebo_func):
    tweebo_dir = os.path.abspath(read_config('depdency_parsers')['tweebo_dir'])
    # If the models file exists then Tweebo has been installed or failed to
    # install
    tweebo_models = os.path.join(tweebo_dir, 'pretrained_models.tar.gz')
    if not os.path.isfile(tweebo_models):
        install_script = os.path.join(tweebo_dir, 'install.sh')
        subprocess.run(['bash', install_script])
    return tweebo_func
@tweebo_install
def tweebo(text):
    with tempfile.TemporaryDirectory() as temp_dir:
        text_file_path = os.path.join(temp_dir, 'text_file.txt')
        result_file_path = os.path.join(temp_dir, 'text_file.txt.predict')
        tweebo_dir = os.path.abspath(read_config('depdency_parsers')['tweebo_dir'])
        with open(text_file_path, 'w+') as text_file:
            text_file.write(text)
        run_script = os.path.join(tweebo_dir, 'run.sh')
        if subprocess.run(['bash', run_script, text_file_path]):
            with open(result_file_path, 'r') as result_file:
                return result_file.read()


    return None
