from collections import defaultdict
import linecache
import math
from pathlib import Path
from typing import Dict, Callable

import numpy as np
from tqdm import tqdm
import requests
import zipfile

from bella.initializers import random_uniform

BELLA_VEC_DIR = Path.home().joinpath('.Bella', 'Vectors')

class GloveCommonEmbedding():
    def __init__(self, version: int, token_index: Dict[str, int] = None,
                 initializer: Callable[[int, int], np.ndarray] = random_uniform):
        self._version = version
        self._embedding_path = self._download()
        self.token_index = token_index
        self._initializer = initializer
        self.embedding = self.create_embeddings()

    def create_embeddings(self) -> np.ndarray:
        '''
        :return: The embedding matrix size = [num_words, embedding_dimension]
        '''

        # num words is all of the words plus the <pad> token as index 0
        num_words = len(self.token_index) + 1
        embedding_dim = self._embedding_dim()
        embedding_matrix = self._initializer(num_words, embedding_dim)

        with self._embedding_path.open('r') as embedding_file:
            for line in embedding_file:
                line_split = line.split()
                word = line_split[:-embedding_dim]
                word = ' '.join(word)
                vector = line_split[-embedding_dim:]

                if word in self.token_index:
                    index = self.token_index[word]
                    embedding_matrix[index] = vector
        embedding_matrix = embedding_matrix.astype(np.float32, copy=False)
        return embedding_matrix

    def _embedding_dim(self) -> int:
        '''
        :return: The embedding dimension size e.g. 300
        '''
        with self._embedding_path.open('r') as embedding_file:
            for line in embedding_file:
                line_split = line.split()
                vector = line_split[1:]
                return len(vector)

    
    def _download(self) -> Path:
        '''
        Downloads the GloveCommonCrawl embeddings from the `Stanford website \
        <https://nlp.stanford.edu/projects/glove/>`_ and returns the Path to 
        the downloaded embeddings. If they already exist returns the existing 
        Path.

        They are stored in the `.Bella/Vectors/glove_common_crawl` directory 
        within the current user directory.
        
        :returns: The filepath to the Glove Common Crawl Embeddings.
        '''

        glove_folder = BELLA_VEC_DIR.joinpath(f'glove_common_crawl')
        glove_folder.mkdir(parents=True, exist_ok=True)
        glove_file_name = f'glove.{self._version}B.300d.txt'
        glove_fp = glove_folder.joinpath(glove_file_name)
        if glove_fp.is_file():
            return glove_fp
        
        zip_file_name = f'glove.{self._version}B.300d.zip'
        download_link = f'http://nlp.stanford.edu/data/{ zip_file_name}'

        glove_zip_path = glove_folder.joinpath(zip_file_name)

        # Reference:
        # http://docs.python-requests.org/en/master/user/quickstart/#raw-response-content
        with glove_zip_path.open('wb') as glove_zip_file:
            request = requests.get(download_link, stream=True)
            total_size = int(request.headers.get('content-length', 0))
            print(f'Downloading Glove {self._version}B vectors')
            for chunk in tqdm(request.iter_content(chunk_size=128),
                                total=math.ceil(total_size//128)):
                glove_zip_file.write(chunk)
        print('Unzipping word vector download')
        glove_zip_path = str(glove_zip_path.resolve())
        with zipfile.ZipFile(glove_zip_path, 'r') as glove_zip_file:
            glove_zip_file.extractall(path=glove_folder)

        glove_folder_files = list(glove_folder.iterdir())
        if not glove_fp.is_file():
            raise Exception('Error in either downloading the glove vectors'
                            ' or file path names. Files in the glove '
                            f'folder {glove_folder_files} and where the '
                            f'golve file should be {str(glove_fp)}')
        return glove_fp

