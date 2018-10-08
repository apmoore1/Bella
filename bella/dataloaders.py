import json
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import Sequence
import numpy as np

from bella.contexts import context

class TargetSequence(Sequence):

    def __init__(self, json_fp: Path, batch_size: int, tokeniser: Tokenizer,  
                 tokeniser_function: Callable[[str], List[str]] = str.split,
                 n_classes: int = 3, sort_field: str = 'text'):
        '''
        :param json_fp: File path to the Target Dependent Sentiment data 
                        that has a json encoded sample per new line in the 
                        file.
        :param batch_size: size of batches to return e.g. 32 
        :param tokeniser: Maps tokens to indexs that map onto the embeddings
        :param tokeniser_function: Tokeniser like `moses` to use to split the 
                                   text into tokens. Default white space
        :param n_classes: Number of class labels. Default = 3
        :param sort_field: Field within the json encoded data to sort the 
                           data by. Used to make the model fitting process 
                           quicker. It is a basic form of sentence bucketing
        '''
        data = []
        self.label_mapper = {-1: 0, 0: 1, 1: 2}
        self.inv_label_mapper = {0: -1, 1: 0, 2: 1}
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.tokeniser = tokeniser
        self.tokeniser_function = tokeniser_function
    
        with json_fp.open('r') as json_file:
            for line in json_file:
                data.append(json.loads(line))
        data = sorted(data, key=self.sort_by(sort_field))
        self.texts = []
        self.targets = []
        self.labels = []
        for target in data:
            self.texts.append(target['text'])
            self.targets.append(target['target'])
            self.labels.append(target['sentiment'])
    
    def __len__(self):
        return int(np.ceil(len(self.texts) / float(self.batch_size)))

    def __getitem__(self, idx: int) -> Tuple[Tuple[np.ndarray, np.ndarray], 
                                             np.ndarray]:
        batch_texts = self.process_texts(self.texts[idx * self.batch_size:
                                                    (idx + 1) * self.batch_size])
        batch_targets = self.process_texts(self.targets[idx * self.batch_size:
                                                        (idx + 1) * self.batch_size])
        batch_labels = to_categorical(self.labels[idx * self.batch_size:
                                                  (idx + 1) * self.batch_size],
                                      num_classes=self.n_classes)
        return ([batch_texts, batch_targets], batch_labels)

    def sort_by(self, field: str) -> Callable[[Dict[str, str]], int]:
        '''
        Returns a function that outputs the length of a String that has been 
        split where the String is the value from a dictionary where the 
        key is the argument to this function. The Returned function input is 
        thus a dictionary that has to contain a key that is the String input 
        to this function.

        :param field: A key within the dinctionary that is input to the 
                      returned argument, where the value of the key 
                      sorts the list of dictionaries that are being sorted.
        :return: A function that outputs the length of a String that has been 
                 split where the String is the value from a dictionary where the 
                 key is the argument to this function. The Returned function 
                 input is thus a dictionary that has to contain a key that is 
                 the String input to this function.
        '''
        def field_sort(target: Dict[str, str]) -> int:
            return len(target[field].split())
        return field_sort

    def process_texts(self, texts: List[str], **pad_kwargs) -> np.ndarray:
        all_tokens = [self.tokeniser_function(text) for text in texts]
        tokenised_texts = [' '.join(tokens) for tokens in all_tokens]
        token_ids = self.tokeniser.texts_to_sequences(tokenised_texts)
        padded_token_ids = pad_sequences(token_ids, **pad_kwargs)
        return padded_token_ids

class LeftRightTargetSequence(TargetSequence):

    def __init__(self, json_fp: Path, batch_size: int, tokeniser: Tokenizer,  
                 tokeniser_function: Callable[[str], List[str]] = str.split,
                 n_classes: int = 3, sort_field: str = 'text',
                 include_target_in_sequence: bool = True,
                 include_target_in_batches: bool = True):
        '''
        :param json_fp: File path to the Target Dependent Sentiment data 
                        that has a json encoded sample per new line in the 
                        file.
        :param batch_size: size of batches to return e.g. 32 
        :param tokeniser: Maps tokens to indexs that map onto the embeddings
        :param tokeniser_function: Tokeniser like `moses` to use to split the 
                                   text into tokens. Default white space
        :param n_classes: Number of class labels. Default = 3
        :param sort_field: Field within the json encoded data to sort the 
                           data by. Used to make the model fitting process 
                           quicker. It is a basic form of sentence bucketing
        :param include_target_in_sequence: Whether the right and left texts 
                                           should include the target word(s)
                                           in the text
        :param include_target_in_batches: Whether to generate a seperate 
                                          list for the targets in each batch
                                          e.g. ([left_texts, right_texts, 
                                                 targets], labels) or if False 
                                         will it would be: 
                                         ([left_texts, right_texts], labels)                 
        '''
        data = []
        self.label_mapper = {-1: 0, 0: 1, 1: 2}
        self.inv_label_mapper = {0: -1, 1: 0, 2: 1}
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.tokeniser = tokeniser
        self.tokeniser_function = tokeniser_function
        self.include_target_in_sequence = include_target_in_sequence
        self.include_target_in_batches = include_target_in_batches
    
        with json_fp.open('r') as json_file:
            for line in json_file:
                data.append(json.loads(line))
        data = sorted(data, key=self.sort_by(sort_field))
        self.left_texts = []
        self.right_texts = []
        if self.include_target_in_batches:
            self.targets = []
        self.labels = []
        for target in data:
            left_text = context(target, 'left', 
                                self.include_target_in_sequence)[0].strip()
            right_text = context(target, 'right', 
                                 self.include_target_in_sequence)[0].strip()
            self.left_texts.append(left_text)
            self.right_texts.append(right_text)
            if self.include_target_in_batches:
                self.targets.append(target['target'])
            self.labels.append(target['sentiment'])
    
    def __getitem__(self, idx: int) -> Tuple[Tuple[np.ndarray, np.ndarray], 
                                             np.ndarray]:
        batch_left_texts = self.left_texts[idx * self.batch_size:
                                           (idx + 1) * self.batch_size]
        batch_left_texts = self.process_texts(batch_left_texts)

        batch_right_texts = self.right_texts[idx * self.batch_size:
                                             (idx + 1) * self.batch_size]
        batch_right_texts = self.process_texts(batch_right_texts, 
                                               padding='post', 
                                               truncating='post')
            
        batch_labels = to_categorical(self.labels[idx * self.batch_size:
                                                  (idx + 1) * self.batch_size],
                                      num_classes=self.n_classes)
        if self.include_target_in_batches:
            batch_targets = self.targets[idx * self.batch_size:
                                         (idx + 1) * self.batch_size]
            batch_targets = self.process_texts(batch_targets)
            return ([batch_left_texts, batch_right_texts, batch_targets], 
                    batch_labels)
    
        return ([batch_left_texts, batch_right_texts], batch_labels)
        

